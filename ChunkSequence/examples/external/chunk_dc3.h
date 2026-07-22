// Out-of-core suffix array construction (DC3 / skew algorithm).
//
// A streaming, sort-based port of the Kärkkäinen–Sanders "skew" algorithm.
// Where the sibling prefix-doubling example (chunk_suffix_array.h) does ~2
// external sorts per round over O(log n) rounds, DC3 recurses on a problem that
// shrinks by 2/3 each level, so its total sort-work is geometric (O(n)) rather
// than O(n log n).  This mirrors the classic external-memory result (Dementiev–
// Kärkkäinen–Mehnert–Sanders, "Better External Memory Suffix Array
// Construction").
//
// Two things make DC3 fit this library's streaming primitives:
//
//   * Random rank gathers become sort-joins.  DC3 needs the sample rank of the
//     suffix one/two positions to the right of each position; those O(n) random
//     gathers are realized by scattering (target_pos, srank) records and pairing
//     them against a per-position base stream (a slot-keyed sort, the same trick
//     the prefix-doubling example uses for rank[p+offset]).
//
//   * The final MERGE becomes one comparator sort.  Classic DC3 finishes by
//     merging the sorted sample suffixes (pos % 3 != 0) with the sorted
//     non-sample suffixes (pos % 3 == 0).  This library has no merge primitive,
//     but the DC3 merge comparison is a valid total order over *all* suffixes:
//     after <=2 characters every pair reaches a sample position whose relative
//     order is one rank comparison.  So we build one fixed-width record per
//     suffix carrying (chars, sample ranks) and hand direct_sample_sort the DC3
//     comparator — the sorted pos column is the suffix array.  No merge.
//
// Per recursion level (text length m, ~2/3 the parent):
//   1. triples : for every sample R-index r emit (s[i], s[i+1], s[i+2]) with the
//                position remapped to r (2-char forward halo per chunk).
//   2. sort + name : direct_sample_sort the triples, then assign_ranks (shared
//                    two-level max-scan) gives each r a lexicographic name.
//   3. recurse : if names are not yet unique, R = names in r-order; recurse to
//                get its suffix array; else the SA is read off the names.
//   4. scatter srank : from the recursive SA, scatter each sample's rank to its
//                original position, pair against a per-position base stream to
//                get one dense (chars, self-rank) record per position.
//   5. assemble + sort : a 2-position forward halo turns each position's record
//                into a full suffix record (self, next, next-next ranks); one
//                direct_sample_sort with the DC3 comparator yields the SA.
//
// A DRAM base case (shrink-until-it-fits, like chunk_convex_hull.h) finishes any
// level whose text fits a byte budget with an in-memory Kärkkäinen–Sanders
// suffix array — so only the top levels touch disk.  The budget is overridable
// via DC3_DRAM_BUDGET_BYTES (used by the test to force the out-of-core path at
// small n).
//
// All passes are ExternalTransform / RemoveWorker / direct_sample_sort — all
// density- and element-size-generic — so the dense-except-last chunk_seqs that
// the sort produces flow through with no repack.  Constraints (as upstream and
// the sibling): n < 2^32, characters treated as unsigned.  Records divide
// CHUNK_SIZE (16 or 32 B) so every chunk stays O_DIRECT-aligned.

#ifndef CHUNK_DC3_H
#define CHUNK_DC3_H

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "absl/log/check.h"
#include "parlay/primitives.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_seq_reader.h"
#include "ChunkSequence/external_engine.h"
#include "ChunkSequence/examples/external/chunk_sa_common.h"
#include "ChunkSequence/examples/external/direct_samplesort.h"
#include "utils/file_utils.h"
#include "configs.h"

namespace ChunkSequenceOps {
namespace dc3_detail {

using sa_detail::elem_prefix;
using sa_detail::sweep;

// ── record types (all divide CHUNK_SIZE) ──────────────────────────────────────

// A sample position's character triple, keyed to its R-index r.  Sort key =
// (t0,t1,t2); r carried but uncompared so the duplicate-heavy triple sort's
// dedup assigner engages (see [[direct-samplesort-dedup-comparator]]).
struct Triple {
    uint32_t t0, t1, t2, r;
    bool operator<(const Triple& o) const {
        if (t0 != o.t0) return t0 < o.t0;
        if (t1 != o.t1) return t1 < o.t1;
        return t2 < o.t2;
    }
    bool operator==(const Triple& o) const {
        return t0 == o.t0 && t1 == o.t1 && t2 == o.t2;
    }
};
static_assert(CHUNK_SIZE % sizeof(Triple) == 0, "Triple must divide CHUNK_SIZE");

// The naming output PR{pos,rank} = (r, name), reinterpreted for a sort by r
// (ByR: build R) or by name (ByName: read SA off unique names).  Both share
// PR's layout; the ==/< key is the sort field ONLY.
struct ByR {
    uint32_t r, name;
    bool operator<(const ByR& o) const { return r < o.r; }
    bool operator==(const ByR& o) const { return r == o.r; }
};
struct ByName {
    uint32_t r, name;
    bool operator<(const ByName& o) const { return name < o.name; }
    bool operator==(const ByName& o) const { return name == o.name; }
};
static_assert(sizeof(ByR) == sizeof(sa_detail::PR) && sizeof(ByName) == sizeof(sa_detail::PR),
              "ByR/ByName reinterpret PR");

// Position-keyed join record.  slot 0 = BASE (one per position, carries chars),
// slot 1 = VAL (one per sample, carries its srank).  (target,slot) is unique, so
// the sort has no duplicate keys.  p0/p1 are a payload union: BASE=(c0,c1),
// VAL=(srank,0).
struct Join {
    uint32_t target, slot, p0, p1;
    bool operator<(const Join& o) const {
        return target != o.target ? target < o.target : slot < o.slot;
    }
    bool operator==(const Join& o) const { return target == o.target && slot == o.slot; }
};
static_assert(CHUNK_SIZE % sizeof(Join) == 0, "Join must divide CHUNK_SIZE");

// One dense per-position record after the srank join: chars + self sample rank
// (0 if this position is non-sample).  res = pos % 3 is derived.
struct Pos {
    uint32_t pos, c0, c1, self;
};
static_assert(CHUNK_SIZE % sizeof(Pos) == 0, "Pos must divide CHUNK_SIZE");

// One suffix's DC3 comparison record.  operator< IS the DC3 merge comparator, a
// valid total order over all suffixes; pos is carried (the SA column) but only
// compared as the final unreachable tie-break (distinct suffixes never tie).
struct SufRec {
    uint32_t c0, c1, self, r1, r2, pos, pad0, pad1;
    bool operator<(const SufRec& b) const {
        const int ra = pos % 3, rb = b.pos % 3;
        if (ra != 0 && rb != 0) return self < b.self;   // both sample: recursion rank
        if (ra == 2 || rb == 2) {                        // mod0 vs mod2: 2 chars then rank
            if (c0 != b.c0) return c0 < b.c0;
            if (c1 != b.c1) return c1 < b.c1;
            return r2 < b.r2;
        }
        if (c0 != b.c0) return c0 < b.c0;                // mod0 vs mod0/mod1: 1 char then rank
        return r1 < b.r1;
    }
    bool operator==(const SufRec& b) const { return !(*this < b) && !(b < *this); }
};
static_assert(CHUNK_SIZE % sizeof(SufRec) == 0, "SufRec must divide CHUNK_SIZE");

// ── DC3 index arithmetic (mirrors the Kärkkäinen–Sanders reference) ───────────
// Sample R-indices r in [0, n02): the left half [0, n0) are pos%3==1 suffixes,
// the right half [n0, n02) are pos%3==2.  A dummy left-half entry at r in
// [n1, n0) (present iff n%3==1) maps to the pad position n.
static inline uint32_t pos_of_r(uint32_t r, uint32_t n0) {
    return r < n0 ? 3u * r + 1u : 3u * (r - n0) + 2u;
}
static inline uint32_t r_of_pos(uint32_t i, uint32_t n0) {
    return (i % 3 == 1) ? i / 3 : i / 3 + n0;   // i % 3 == 2 for the right half
}

// ── forward halo: first two elements of every chunk (for a 2-element lookahead)
template <typename T>
struct Halo2 {
    std::vector<std::array<T, 2>> first;
    std::vector<uint8_t> cnt;
};
template <typename T>
inline Halo2<T> forward_halo2(const chunk_seq& seq) {
    const size_t nc = seq.chunks.size();
    Halo2<T> h;
    h.first.assign(nc, std::array<T, 2>{});
    h.cnt.assign(nc, 0);
    RemoveWorker<T>(seq, 10, [&](ChunkSequenceReader<T>& r) {
        while (true) {
            auto [p, cnt, idx] = r.Poll();
            if (p == nullptr) break;
            if (cnt >= 1) { h.first[idx][0] = p[0]; h.cnt[idx] = 1; }
            if (cnt >= 2) { h.first[idx][1] = p[1]; h.cnt[idx] = 2; }
            r.allocator.Free(p);
        }
        return 0;
    });
    return h;
}

// ── concat two chunk_seqs (order/density irrelevant: only used as sort input) ─
inline chunk_seq concat(chunk_seq a, const chunk_seq& b) {
    const size_t base = a.chunks.size();
    for (chunk c : b.chunks) { c.index += base; a.chunks.push_back(c); }
    return a;
}

// ── read a chunk_seq<uint32> fully into DRAM (base case) ──────────────────────
inline void read_u32(const chunk_seq& seq, size_t n, std::vector<int>& out) {
    std::vector<const chunk*> ord;
    for (const auto& c : seq.chunks) ord.push_back(&c);
    std::sort(ord.begin(), ord.end(),
              [](const chunk* a, const chunk* b) { return a->index < b->index; });
    void* buf = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
    CHECK(buf != nullptr) << "dc3: base-case readback alloc failed";
    size_t j = 0;
    for (const chunk* c : ord) {
        if (c->used == 0) continue;
        int fd = open(c->filename.c_str(), O_RDONLY | O_DIRECT);
        SYSCALL(fd);
        SYSCALL(pread(fd, buf, AlignUp(c->used), (off_t)c->begin_addr));
        close(fd);
        const auto* e = (const uint32_t*)buf;
        const size_t cnt = c->used / sizeof(uint32_t);
        for (size_t i = 0; i < cnt; i++) out[j++] = (int)e[i];
    }
    free(buf);
    CHECK(j == n) << "dc3: base-case element count " << j << " != " << n;
}

// ── in-memory Kärkkäinen–Sanders suffix array (the DRAM base case) ────────────
// The reference "simple linear work" DC3 on an int array s[0..n-1] with s[i] > 0
// and three trailing zeros (s[n]=s[n+1]=s[n+2]=0); writes SA[0..n-1].
inline bool ks_leq(int a1, int a2, int b1, int b2) {
    return a1 < b1 || (a1 == b1 && a2 <= b2);
}
inline bool ks_leq(int a1, int a2, int a3, int b1, int b2, int b3) {
    return a1 < b1 || (a1 == b1 && ks_leq(a2, a3, b2, b3));
}
inline void ks_radix(const int* a, int* b, const int* r, int n, int K) {
    std::vector<int> c(K + 1, 0);
    for (int i = 0; i < n; i++) c[r[a[i]]]++;
    for (int i = 0, sum = 0; i <= K; i++) { int t = c[i]; c[i] = sum; sum += t; }
    for (int i = 0; i < n; i++) b[c[r[a[i]]]++] = a[i];
}
inline void ks_suffix_array(int* s, int* SA, int n, int K) {
    const int n0 = (n + 2) / 3, n1 = (n + 1) / 3, n2 = n / 3, n02 = n0 + n2;
    std::vector<int> s12(n02 + 3, 0), SA12(n02 + 3, 0), s0(n0 > 0 ? n0 : 1), SA0(n0 > 0 ? n0 : 1);
    for (int i = 0, j = 0; i < n + (n0 - n1); i++)
        if (i % 3 != 0) s12[j++] = i;
    ks_radix(s12.data(), SA12.data(), s + 2, n02, K);
    ks_radix(SA12.data(), s12.data(), s + 1, n02, K);
    ks_radix(s12.data(), SA12.data(), s, n02, K);
    int name = 0, c0 = -1, c1 = -1, c2 = -1;
    for (int i = 0; i < n02; i++) {
        if (s[SA12[i]] != c0 || s[SA12[i] + 1] != c1 || s[SA12[i] + 2] != c2) {
            name++; c0 = s[SA12[i]]; c1 = s[SA12[i] + 1]; c2 = s[SA12[i] + 2];
        }
        if (SA12[i] % 3 == 1) s12[SA12[i] / 3] = name;
        else s12[SA12[i] / 3 + n0] = name;
    }
    if (name < n02) {
        ks_suffix_array(s12.data(), SA12.data(), n02, name);
        for (int i = 0; i < n02; i++) s12[SA12[i]] = i + 1;
    } else {
        for (int i = 0; i < n02; i++) SA12[s12[i] - 1] = i;
    }
    for (int i = 0, j = 0; i < n02; i++)
        if (SA12[i] < n0) s0[j++] = 3 * SA12[i];
    ks_radix(s0.data(), SA0.data(), s, n0, K);
    for (int p = 0, t = n0 - n1, k = 0; k < n; k++) {
        const int i = SA12[t] < n0 ? SA12[t] * 3 + 1 : (SA12[t] - n0) * 3 + 2;
        const int j = SA0[p];
        const bool le = SA12[t] < n0
            ? ks_leq(s[i], s12[SA12[t] + n0], s[j], s12[j / 3])
            : ks_leq(s[i], s[i + 1], s12[SA12[t] - n0 + 1], s[j], s[j + 1], s12[j / 3 + n0]);
        if (le) {
            SA[k] = i;
            if (++t == n02) for (k++; p < n0; p++, k++) SA[k] = SA0[p];
        } else {
            SA[k] = j;
            if (++p == n0)
                for (k++; t < n02; t++, k++)
                    SA[k] = SA12[t] < n0 ? SA12[t] * 3 + 1 : (SA12[t] - n0) * 3 + 2;
        }
    }
}

inline size_t dram_budget() {
    const size_t phys = (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
    size_t b = std::min<size_t>(size_t(4) << 30, phys / 8);
    if (const char* e = getenv("DC3_DRAM_BUDGET_BYTES")) b = std::stoull(e);
    return b;
}

// Compute the suffix array of a uint32 text in DRAM, return it as a chunk_seq.
inline chunk_seq base_case(const chunk_seq& text, size_t n, const std::string& out_prefix) {
    if (n == 1) return tabulate<uint32_t>(1, out_prefix, [](size_t) { return uint32_t{0}; });
    std::vector<int> s(n + 3, 0);
    read_u32(text, n, s);   // fills s[0..n-1]; s[n..n+2] stay 0
    int K = 0;
    for (size_t i = 0; i < n; i++) K = std::max(K, s[i]);
    std::vector<int> SA(n);
    ks_suffix_array(s.data(), SA.data(), (int)n, K);
    return tabulate<uint32_t>(n, out_prefix,
                              [&SA](size_t i) { return (uint32_t)SA[i]; });
}

// ── step 1: character triples for every sample R-index ────────────────────────
inline chunk_seq make_triples(const chunk_seq& text, size_t n, uint32_t n0, uint32_t n1,
                              const std::string& pfx) {
    auto pre = elem_prefix(text, sizeof(uint32_t));
    Halo2<uint32_t> halo = forward_halo2<uint32_t>(text);
    const size_t nc = text.chunks.size();
    // Each input uint32 chunk yields ~2/3 as many Triples; a Triple (16 B) holds
    // 1/4 as many per block, so blocks/input = ceil((2/3)*4) = 3.  FAN=4 is a
    // safe upper bound for any CHUNK_SIZE (the ratio is CHUNK-independent).
    constexpr size_t FAN = 4;
    chunk_seq trips = ExternalTransform<uint32_t, Triple>(text, pfx,
        [&pre, &halo, n, n0, nc](const uint32_t* in, size_t nl, size_t idx,
                                 const ChunkEmitter<Triple>& emit) {
            const size_t cap = emit.out_cap();
            const size_t start = pre[idx];
            auto nb = [&](size_t g) -> uint32_t {
                if (g >= n) return 0;
                if (g < start + nl) return in[g - start];
                const size_t off = g - (start + nl);   // 0 or 1 into the next chunk
                return (idx + 1 < nc) ? halo.first[idx + 1][off] : 0;
            };
            Triple* out = emit.alloc();
            size_t fill = 0, sub = 0;
            auto put = [&](const Triple& t) {
                if (fill == cap) {
                    emit.emit(out, cap, idx * FAN + sub);
                    out = emit.alloc(); fill = 0; sub++;
                }
                out[fill++] = t;
            };
            for (size_t j = 0; j < nl; j++) {
                const size_t g = start + j;
                if (g % 3 == 0) continue;   // non-sample
                put(Triple{in[j], nb(g + 1), nb(g + 2), r_of_pos((uint32_t)g, n0)});
            }
            if (fill > 0) {
                memset((char*)out + fill * sizeof(Triple), 0, CHUNK_SIZE - fill * sizeof(Triple));
                emit.emit(out, fill, idx * FAN + sub);
            } else {
                free(out);
            }
        },
        /*max_out_per_input=*/FAN);

    // The dummy sample (pad position n) when n % 3 == 1: an all-zero triple at
    // r = n1, so R's left half is full size n0 and the recursion stays valid.
    if (n0 > n1) {
        chunk_seq dummy = tabulate<Triple>(1, pfx + "_dummy_",
            [n1](size_t) { return Triple{0, 0, 0, n1}; });
        trips = concat(std::move(trips), dummy);
    }
    return trips;
}

// ── step 4a: scatter each sample's recursive rank to its original position ────
// SA_R[g] = r (the g-th sample suffix in sorted order); srank of sample r is
// g + 1.  Emit a VAL record delivering that rank to pos_of_r(r).
inline chunk_seq scatter_vals(const chunk_seq& sa_r, uint32_t n0, const std::string& pfx) {
    auto pre = elem_prefix(sa_r, sizeof(uint32_t));
    // uint32 -> Join (16 B) is a 4x fanout, so one input chunk emits up to 4.
    constexpr size_t FAN = sizeof(Join) / sizeof(uint32_t);
    return ExternalTransform<uint32_t, Join>(sa_r, pfx,
        [&pre, n0](const uint32_t* in, size_t nl, size_t idx, const ChunkEmitter<Join>& emit) {
            const size_t cap = emit.out_cap();
            const size_t g0 = pre[idx];
            Join* out = emit.alloc();
            size_t fill = 0, sub = 0;
            for (size_t j = 0; j < nl; j++) {
                if (fill == cap) {
                    emit.emit(out, cap, idx * FAN + sub);
                    out = emit.alloc(); fill = 0; sub++;
                }
                out[fill++] = Join{pos_of_r(in[j], n0), /*slot=*/1,
                                   /*srank=*/(uint32_t)(g0 + j + 1), 0};
            }
            if (fill > 0) {
                memset((char*)out + fill * sizeof(Join), 0, CHUNK_SIZE - fill * sizeof(Join));
                emit.emit(out, fill, idx * FAN + sub);
            } else {
                free(out);
            }
        },
        /*max_out_per_input=*/FAN);
}

// ── step 4b: one BASE record per position (carries chars) ─────────────────────
inline chunk_seq make_base(const chunk_seq& text, size_t n, const std::string& pfx) {
    auto pre = elem_prefix(text, sizeof(uint32_t));
    Halo2<uint32_t> halo = forward_halo2<uint32_t>(text);
    const size_t nc = text.chunks.size();
    const size_t cap = CHUNK_SIZE / sizeof(Join);
    const size_t FAN = (CHUNK_SIZE / sizeof(uint32_t) + cap - 1) / cap;   // in/out cap ratio
    return ExternalTransform<uint32_t, Join>(text, pfx,
        [&pre, &halo, n, nc, FAN](const uint32_t* in, size_t nl, size_t idx,
                                  const ChunkEmitter<Join>& emit) {
            const size_t cap2 = emit.out_cap();
            const size_t start = pre[idx];
            auto nb = [&](size_t g) -> uint32_t {
                if (g >= n) return 0;
                if (g < start + nl) return in[g - start];
                return (idx + 1 < nc) ? halo.first[idx + 1][0] : 0;
            };
            Join* out = emit.alloc();
            size_t fill = 0, sub = 0;
            for (size_t j = 0; j < nl; j++) {
                if (fill == cap2) {
                    emit.emit(out, cap2, idx * FAN + sub);
                    out = emit.alloc(); fill = 0; sub++;
                }
                const size_t g = start + j;
                out[fill++] = Join{(uint32_t)g, /*slot=*/0, /*c0=*/in[j], /*c1=*/nb(g + 1)};
            }
            if (fill > 0) {
                memset((char*)out + fill * sizeof(Join), 0, CHUNK_SIZE - fill * sizeof(Join));
                emit.emit(out, fill, idx * FAN + sub);
            } else {
                free(out);
            }
        },
        /*max_out_per_input=*/FAN);
}

// ── step 4c: pair BASE with its VAL → one dense Pos per position ──────────────
inline chunk_seq pair_pos(const chunk_seq& sorted_join, const std::string& pfx) {
    const size_t nc = sorted_join.chunks.size();
    // Forward halo of one: the first Join of every chunk, so a BASE ending a
    // chunk can see the VAL that opens the next.
    std::vector<Join> first(nc);
    std::vector<char> has_first(nc, 0);
    RemoveWorker<Join>(sorted_join, 10, [&](ChunkSequenceReader<Join>& r) {
        while (true) {
            auto [p, cnt, idx] = r.Poll();
            if (p == nullptr) break;
            if (cnt > 0) { first[idx] = p[0]; has_first[idx] = 1; }
            r.allocator.Free(p);
        }
        return 0;
    });
    return ExternalTransform<Join, Pos>(sorted_join, pfx,
        [nc, &first, &has_first](const Join* in, size_t nl, size_t idx,
                                 const ChunkEmitter<Pos>& emit) {
            Pos* out = emit.alloc();
            size_t fill = 0;
            for (size_t i = 0; i < nl; i++) {
                if (in[i].slot != 0) continue;   // VAL: consumed by its BASE
                Join nxt{}; bool has_nxt = false;
                if (i + 1 < nl) { nxt = in[i + 1]; has_nxt = true; }
                else if (idx + 1 < nc && has_first[idx + 1]) { nxt = first[idx + 1]; has_nxt = true; }
                const uint32_t self =
                    (has_nxt && nxt.target == in[i].target && nxt.slot == 1) ? nxt.p0 : 0;
                out[fill++] = Pos{in[i].target, in[i].p0, in[i].p1, self};
            }
            memset((char*)out + fill * sizeof(Pos), 0, CHUNK_SIZE - fill * sizeof(Pos));
            emit.emit(out, fill, idx);
        },
        /*max_out_per_input=*/1);
}

// ── step 5: 2-position halo → one SufRec per position ─────────────────────────
inline chunk_seq make_sufrecs(const chunk_seq& pos_seq, size_t n, const std::string& pfx) {
    auto pre = elem_prefix(pos_seq, sizeof(Pos));
    Halo2<Pos> halo = forward_halo2<Pos>(pos_seq);
    const size_t nc = pos_seq.chunks.size();
    constexpr size_t FAN = 4;   // Pos(16B) in, SufRec(32B) out: <=2 blocks, 4 safe
    return ExternalTransform<Pos, SufRec>(pos_seq, pfx,
        [&pre, &halo, n, nc](const Pos* in, size_t nl, size_t idx,
                             const ChunkEmitter<SufRec>& emit) {
            const size_t cap = emit.out_cap();
            const size_t start = pre[idx];
            // self rank at global position g (pad / non-sample -> 0).
            auto self_at = [&](size_t g) -> uint32_t {
                if (g >= n) return 0;
                if (g < start + nl) return in[g - start].self;
                const size_t off = g - (start + nl);
                return (idx + 1 < nc) ? halo.first[idx + 1][off].self : 0;
            };
            SufRec* out = emit.alloc();
            size_t fill = 0, sub = 0;
            for (size_t j = 0; j < nl; j++) {
                if (fill == cap) {
                    emit.emit(out, cap, idx * FAN + sub);
                    out = emit.alloc(); fill = 0; sub++;
                }
                const size_t g = start + j;
                out[fill++] = SufRec{in[j].c0, in[j].c1, in[j].self,
                                     self_at(g + 1), self_at(g + 2), (uint32_t)g, 0, 0};
            }
            if (fill > 0) {
                memset((char*)out + fill * sizeof(SufRec), 0, CHUNK_SIZE - fill * sizeof(SufRec));
                emit.emit(out, fill, idx * FAN + sub);
            } else {
                free(out);
            }
        },
        /*max_out_per_input=*/FAN);
}

// ── extract the pos column of a DC3-sorted SufRec seq (== the suffix array) ────
inline chunk_seq extract_pos(const chunk_seq& sorted, const std::string& pfx) {
    return ExternalTransform<SufRec, uint32_t>(sorted, pfx,
        [](const SufRec* in, size_t nl, size_t idx, const ChunkEmitter<uint32_t>& emit) {
            uint32_t* out = emit.alloc();
            for (size_t i = 0; i < nl; i++) out[i] = in[i].pos;
            memset((char*)out + nl * sizeof(uint32_t), 0, CHUNK_SIZE - nl * sizeof(uint32_t));
            emit.emit(out, nl, idx);
        },
        /*max_out_per_input=*/1);
}

// ── extract a column (name) of a ByR-sorted (r,name) seq → R = names in r-order
inline chunk_seq extract_names(const chunk_seq& by_r, const std::string& pfx) {
    return ExternalTransform<ByR, uint32_t>(by_r, pfx,
        [](const ByR* in, size_t nl, size_t idx, const ChunkEmitter<uint32_t>& emit) {
            uint32_t* out = emit.alloc();
            for (size_t i = 0; i < nl; i++) out[i] = in[i].name;
            memset((char*)out + nl * sizeof(uint32_t), 0, CHUNK_SIZE - nl * sizeof(uint32_t));
            emit.emit(out, nl, idx);
        },
        /*max_out_per_input=*/1);
}

// ── extract the r column of a ByName-sorted (r,name) seq → SA of R directly ────
inline chunk_seq extract_r(const chunk_seq& by_name, const std::string& pfx) {
    return ExternalTransform<ByName, uint32_t>(by_name, pfx,
        [](const ByName* in, size_t nl, size_t idx, const ChunkEmitter<uint32_t>& emit) {
            uint32_t* out = emit.alloc();
            for (size_t i = 0; i < nl; i++) out[i] = in[i].r;
            memset((char*)out + nl * sizeof(uint32_t), 0, CHUNK_SIZE - nl * sizeof(uint32_t));
            emit.emit(out, nl, idx);
        },
        /*max_out_per_input=*/1);
}

// ── the recursive out-of-core DC3 on a uint32 text of length n ────────────────
inline chunk_seq dc3_rec(const chunk_seq& text, size_t n, const std::string& pfx) {
    if (n <= 2 || n * sizeof(uint32_t) <= dram_budget())
        return base_case(text, n, pfx + "_sa_");

    const uint32_t n0 = (uint32_t)((n + 2) / 3);
    const uint32_t n1 = (uint32_t)((n + 1) / 3);
    const uint32_t n2 = (uint32_t)(n / 3);
    const size_t n02 = (size_t)n0 + n2;

    // 1-2. triples -> sort -> name.
    const std::string tp = pfx + "_trip_", tsp = pfx + "_trips_";
    chunk_seq trips = make_triples(text, n, n0, n1, tp);
    chunk_seq sorted_t = direct_sample_sort<Triple>(trips, std::less<>{}, tsp);
    sweep(tp);
    const std::string np = pfx + "_name_";
    sa_detail::RankResult nm = sa_detail::assign_ranks<Triple>(
        sorted_t, np, [](const Triple& a, const Triple& b) { return a == b; },
        [](const Triple& t) { return t.r; });
    sweep(tsp);
    CHECK(nm.n == n02) << "dc3: triple count " << nm.n << " != n02 " << n02;

    // 3. build SA of R (recurse if names not yet unique).
    chunk_seq sa_r;
    if (nm.groups == n02) {
        // Names are already a permutation of 1..n02: SA read off by sorting on name.
        const std::string sp = pfx + "_byname_";
        chunk_seq by_name = direct_sample_sort<ByName>(nm.pr, std::less<>{}, sp);
        sa_r = extract_r(by_name, pfx + "_sar_");
        sweep(sp);
        sweep(np);
    } else {
        const std::string sp = pfx + "_byr_", rp = pfx + "_R_";
        chunk_seq by_r = direct_sample_sort<ByR>(nm.pr, std::less<>{}, sp);
        sweep(np);
        chunk_seq R = extract_names(by_r, rp);
        sweep(sp);
        sa_r = dc3_rec(R, n02, pfx + "r");
        sweep(rp);
    }

    // 4. scatter recursive ranks to positions, pair against a per-position base.
    const std::string vp = pfx + "_val_", bp = pfx + "_base_", jp = pfx + "_jsort_";
    chunk_seq vals = scatter_vals(sa_r, n0, vp);
    sweep(pfx + "_sar_");
    chunk_seq base = make_base(text, n, bp);
    chunk_seq joined = concat(std::move(base), vals);
    chunk_seq sorted_j = direct_sample_sort<Join>(joined, std::less<>{}, jp);
    sweep(vp);
    sweep(bp);
    const std::string pp = pfx + "_pos_";
    chunk_seq pos_seq = pair_pos(sorted_j, pp);
    sweep(jp);

    // 5. assemble suffix records and sort them by the DC3 comparator.
    const std::string rp2 = pfx + "_rec_", fp = pfx + "_fsort_";
    chunk_seq recs = make_sufrecs(pos_seq, n, rp2);
    sweep(pp);
    chunk_seq sorted_s = direct_sample_sort<SufRec>(recs, std::less<>{}, fp);
    sweep(rp2);
    chunk_seq sa = extract_pos(sorted_s, pfx);
    sweep(fp);
    return sa;
}

}  // namespace dc3_detail

/**
 * Out-of-core suffix array of `text` (a chunk_seq<char>) via DC3 / skew.
 * Returns an index-ordered chunk_seq<uint32_t> of the n suffix start positions
 * in lexicographic order.  `out_prefix` names the result files; all
 * intermediates are swept off the drives before return.  n = text length must
 * be < 2^32.
 */
inline chunk_seq ChunkDC3(const chunk_seq& text, const std::string& out_prefix) {
    namespace d = dc3_detail;

    size_t n = 0;
    for (const chunk& c : text.chunks) n += c.used;   // chars
    CHECK(n < (size_t{1} << 32)) << "dc3: n must be < 2^32";
    if (n == 0) return {};

    // Map chars to non-zero uint32 (c + 1), reserving 0 as the beyond-end pad
    // sentinel; the +1 is order-preserving so the suffix array is unchanged.
    // char -> uint32 is a 4x fanout (a full char chunk holds sizeof(uint32)x as
    // many elements as a uint32 chunk), so one input chunk emits up to 4 blocks.
    const std::string tp = out_prefix + "_dc3text_";
    constexpr size_t TFAN = sizeof(uint32_t);
    chunk_seq text_u32 = ExternalTransform<char, uint32_t>(text, tp,
        [](const char* in, size_t nl, size_t idx, const ChunkEmitter<uint32_t>& emit) {
            const size_t cap = emit.out_cap();
            size_t produced = 0, sub = 0;
            do {
                const size_t cnt = std::min(cap, nl - produced);
                uint32_t* out = emit.alloc();
                for (size_t i = 0; i < cnt; i++)
                    out[i] = (uint32_t)(unsigned char)in[produced + i] + 1u;
                memset((char*)out + cnt * sizeof(uint32_t), 0, CHUNK_SIZE - cnt * sizeof(uint32_t));
                emit.emit(out, cnt, idx * TFAN + sub);
                produced += cnt; sub++;
            } while (produced < nl);
        },
        /*max_out_per_input=*/TFAN);

    chunk_seq sa = d::dc3_rec(text_u32, n, out_prefix + "_dc3");
    d::sweep(tp);
    return sa;
}

}  // namespace ChunkSequenceOps

#endif  // CHUNK_DC3_H
