// Out-of-core suffix array construction (prefix doubling, sort-based).
//
// A port of parlaylib's problem-based-benchmark `suffix_array` (deps/parlaylib-
// examples/suffix_array.h) onto the chunk_seq data model. Upstream is a parallel
// prefix-doubling sort: sort suffixes by their first character, then repeatedly
// double the compared length by re-sorting on the pair (rank[p], rank[p+offset]).
// Upstream reads rank[p+offset] with a *random* gather into an in-DRAM array —
// exactly the access pattern the out-of-core library avoids. Here every such
// gather is turned into a **sort-join**, so the whole computation is a chain of
// streaming passes over the SSDs with no random access.
//
// Each doubling round (current sorted length = `offset`) is:
//
//   state PR = chunk_seq<(pos, rank)>            // rank of the suffix at pos
//   1. join-emit : for each (pos,rank) emit (pos, slot0, rank) and, if pos>=offset,
//                  (pos-offset, slot1, rank).  The slot1 record delivers a
//                  suffix's rank to the position `offset` to its left.
//   2. sort      : direct_sample_sort by (a_pos, slot)  -> each position's slot0
//                  is immediately followed by its slot1 (if any).
//   3. pair      : combine each position's two records into the doubling key
//                  SaKey(pos, k0=rank[pos], k1=rank[pos+offset] or 0).  The slot0/
//                  slot1 pair can straddle a chunk boundary, resolved with a
//                  one-element forward halo taken from a per-chunk DRAM array
//                  (the first record of every chunk).
//   4. sort      : direct_sample_sort SaKey by (k0,k1).  THIS is the duplicate-key-
//                  heavy sort (tied ranks); SaKey's ==/< compare (k0,k1) ONLY, with
//                  pos carried but uncompared, so direct_sample_sort's dedup
//                  assigner engages and the tied-key bucket run self-balances.  (A
//                  record that also compared pos would funnel a whole tied group
//                  into one bucket and overflow DRAM — verified separately.)
//   5. re-rank   : assign each element the sorted index of its (k0,k1) group head
//                  (a two-level max-scan; the group-boundary test needs the
//                  previous element's key, supplied per chunk from a DRAM array of
//                  last-keys).  Emits the next round's PR = (pos, newrank) and the
//                  number of distinct groups.  When #groups == n every suffix has a
//                  unique rank and the sort is complete; the sorted-by-key order is
//                  the suffix array (its pos column).
//
// Every pass is an ExternalTransform / RemoveWorker / direct_sample_sort — all
// density-independent — so the intermediate chunk_seqs may be dense-except-last
// (as sort output is) without any repack.  No delayed layer, no DensePack, no
// dense-input requirement.
//
// Constraints (as upstream): n < 2^32, characters treated as unsigned.  Records
// are 16 B (ranks and positions are 32-bit), which divides CHUNK_SIZE so every
// chunk stays O_DIRECT-aligned.

#ifndef CHUNK_SUFFIX_ARRAY_H
#define CHUNK_SUFFIX_ARRAY_H

#include <cstdint>
#include <cstring>
#include <filesystem>
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
namespace sa_detail {
// elem_prefix, sweep, PR, RankResult, and the generic assign_ranks live in
// chunk_sa_common.h (shared with chunk_dc3.h).

// Doubling key: sort key = (k0,k1); pos carried but EXCLUDED from comparison so
// direct_sample_sort's DeduplicatingAssigner engages on tied keys.
struct SaKey {
    uint32_t k0, k1, pos, pad;
    bool operator==(const SaKey& o) const { return k0 == o.k0 && k1 == o.k1; }
    bool operator<(const SaKey& o) const {
        return k0 < o.k0 || (k0 == o.k0 && k1 < o.k1);
    }
};
static_assert(CHUNK_SIZE % sizeof(SaKey) == 0, "SaKey must divide CHUNK_SIZE");

// Join record: sort key = (a_pos, slot); rank carried.
struct JRec {
    uint32_t a_pos, slot, rank, pad;
    bool operator==(const JRec& o) const { return a_pos == o.a_pos && slot == o.slot; }
    bool operator<(const JRec& o) const {
        return a_pos < o.a_pos || (a_pos == o.a_pos && slot < o.slot);
    }
};
static_assert(CHUNK_SIZE % sizeof(JRec) == 0, "JRec must divide CHUNK_SIZE");

static inline uint64_t key64(const SaKey& k) {
    return ((uint64_t)k.k0 << 32) | (uint64_t)k.k1;
}

// SaKey group membership for assign_ranks: equal (k0,k1).
static inline bool sakey_same(const SaKey& a, const SaKey& b) {
    return key64(a) == key64(b);
}

// ── bootstrap: one SaKey per character (k0 = char, k1 = 0, pos = index) ────────
inline chunk_seq initial_keys(const chunk_seq& text, const std::string& pfx) {
    auto pre = elem_prefix(text, 1);
    constexpr size_t FAN = sizeof(SaKey);   // 1-byte input -> 16-byte output
    return ExternalTransform<char, SaKey>(text, pfx,
        [pre](const char* in, size_t n, size_t idx, const ChunkEmitter<SaKey>& emit) {
            const size_t cap = emit.out_cap();
            size_t produced = 0, sub = 0;
            do {
                const size_t cnt = std::min(cap, n - produced);
                SaKey* out = emit.alloc();
                for (size_t i = 0; i < cnt; i++)
                    out[i] = SaKey{(uint32_t)(unsigned char)in[produced + i], 0,
                                   (uint32_t)(pre[idx] + produced + i), 0};
                memset((char*)out + cnt * sizeof(SaKey), 0, CHUNK_SIZE - cnt * sizeof(SaKey));
                emit.emit(out, cnt, idx * FAN + sub);
                produced += cnt;
                sub++;
            } while (produced < n);
        },
        /*max_out_per_input=*/FAN);
}

// ── round step 1: join-emit (pos,slot0,rank) + (pos-offset,slot1,rank) ────────
inline chunk_seq join_emit(const chunk_seq& prs, uint32_t offset, const std::string& pfx) {
    // Each PR emits up to 2 JRecs; in-cap (PR)=CHUNK/8, out-cap (JRec)=CHUNK/16,
    // so up to 2*(CHUNK/8)/(CHUNK/16) = 4 output blocks per input chunk.
    constexpr size_t FAN = 4;
    return ExternalTransform<PR, JRec>(prs, pfx,
        [offset](const PR* in, size_t n, size_t idx, const ChunkEmitter<JRec>& emit) {
            const size_t cap = emit.out_cap();
            JRec* out = emit.alloc();
            size_t fill = 0, sub = 0;
            auto put = [&](const JRec& r) {
                if (fill == cap) {
                    emit.emit(out, cap, idx * FAN + sub);   // full block, no tail
                    out = emit.alloc();
                    fill = 0;
                    sub++;
                }
                out[fill++] = r;
            };
            for (size_t i = 0; i < n; i++) {
                put(JRec{in[i].pos, 0, in[i].rank, 0});
                if (in[i].pos >= offset) put(JRec{in[i].pos - offset, 1, in[i].rank, 0});
            }
            if (fill > 0) {
                memset((char*)out + fill * sizeof(JRec), 0, CHUNK_SIZE - fill * sizeof(JRec));
                emit.emit(out, fill, idx * FAN + sub);
            } else {
                free(out);
            }
        },
        /*max_out_per_input=*/FAN);
}

// ── round step 3: pair each position's slot0/slot1 into a doubling key ────────
inline chunk_seq pair_keys(const chunk_seq& sorted_j, const std::string& pfx) {
    const size_t nc = sorted_j.chunks.size();

    // Forward halo of one element: the first JRec of every chunk (index order),
    // so a chunk whose last record is a slot0 can see the slot1 that opens the
    // next chunk.
    std::vector<JRec> first(nc);
    std::vector<char> has_first(nc, 0);
    RemoveWorker<JRec>(sorted_j, 10, [&](ChunkSequenceReader<JRec>& r) {
        while (true) {
            auto [p, cnt, idx] = r.Poll();
            if (p == nullptr) break;
            if (cnt > 0) { first[idx] = p[0]; has_first[idx] = 1; }
            r.allocator.Free(p);
        }
        return 0;
    });

    return ExternalTransform<JRec, SaKey>(sorted_j, pfx,
        [nc, &first, &has_first](const JRec* in, size_t n, size_t idx,
                                 const ChunkEmitter<SaKey>& emit) {
            SaKey* out = emit.alloc();
            size_t fill = 0;
            for (size_t i = 0; i < n; i++) {
                if (in[i].slot != 0) continue;   // slot1: consumed by its slot0
                JRec nxt{};
                bool has_nxt = false;
                if (i + 1 < n) { nxt = in[i + 1]; has_nxt = true; }
                else if (idx + 1 < nc && has_first[idx + 1]) { nxt = first[idx + 1]; has_nxt = true; }
                const uint32_t k1 =
                    (has_nxt && nxt.a_pos == in[i].a_pos && nxt.slot == 1) ? nxt.rank : 0;
                out[fill++] = SaKey{in[i].rank, k1, in[i].a_pos, 0};
            }
            memset((char*)out + fill * sizeof(SaKey), 0, CHUNK_SIZE - fill * sizeof(SaKey));
            emit.emit(out, fill, idx);
        },
        /*max_out_per_input=*/1);
}

// ── round step 5: assign group-head ranks over a key-sorted SaKey seq ─────────
// Thin wrapper over the shared assign_ranks: group by (k0,k1), carry pos.
inline RankResult assign_ranks(const chunk_seq& sorted_k, const std::string& pfx) {
    return sa_detail::assign_ranks<SaKey>(
        sorted_k, pfx, sakey_same, [](const SaKey& k) { return k.pos; });
}

// ── extract the pos column of a key-sorted (== suffix-array-ordered) SaKey seq ─
inline chunk_seq extract_sa(const chunk_seq& sorted_k, const std::string& pfx) {
    return ExternalTransform<SaKey, uint32_t>(sorted_k, pfx,
        [](const SaKey* in, size_t n, size_t idx, const ChunkEmitter<uint32_t>& emit) {
            uint32_t* out = emit.alloc();
            for (size_t i = 0; i < n; i++) out[i] = in[i].pos;
            memset((char*)out + n * sizeof(uint32_t), 0, CHUNK_SIZE - n * sizeof(uint32_t));
            emit.emit(out, n, idx);
        },
        /*max_out_per_input=*/1);
}

}  // namespace sa_detail

/**
 * Out-of-core suffix array of `text` (a chunk_seq<char>).  Returns an index-
 * ordered chunk_seq<uint32_t> of the n suffix start positions in lexicographic
 * order.  `out_prefix` names the result files; all intermediates are swept off
 * the drives before return.  n = text length must be < 2^32.
 */
inline chunk_seq ChunkSuffixArray(const chunk_seq& text, const std::string& out_prefix) {
    namespace d = sa_detail;

    size_t n = 0;
    for (const chunk& c : text.chunks) n += c.used;   // chars
    CHECK(n < (size_t{1} << 32)) << "suffix_array: n must be < 2^32";
    if (n == 0) return {};

    // Every intermediate prefix ends with '_' so a name-prefix sweep of e.g.
    // "..._k1_" never also matches "..._k10_".
    const std::string base = out_prefix + "_satmp_";
    auto sp = [&](const char* what, size_t stage) {
        return base + what + std::to_string(stage) + "_";
    };

    // Bootstrap: rank suffixes by their first character.
    const std::string ip = base + "init_";
    chunk_seq keys = d::initial_keys(text, ip);
    std::string kp = sp("k", 0);
    chunk_seq sorted_k = direct_sample_sort<d::SaKey>(keys, std::less<>{}, kp);
    d::sweep(ip);

    std::string pr_pfx = sp("pr", 0);
    d::RankResult rr = d::assign_ranks(sorted_k, pr_pfx);
    CHECK(rr.n == n) << "suffix_array: element count drift (" << rr.n << " vs " << n << ")";

    uint32_t offset = 1;
    const size_t max_rounds = 64;   // >= ceil(log2 n) + slack; guards infinite loop
    for (size_t stage = 1; rr.groups < n; stage++) {
        CHECK(stage <= max_rounds) << "suffix_array: doubling did not converge";
        chunk_seq pr = std::move(rr.pr);
        d::sweep(kp);   // previous round's sorted keys no longer needed

        const std::string jp = sp("j", stage), j2p = sp("js", stage), skp = sp("s", stage);
        const std::string nkp = sp("k", stage), npr = sp("pr", stage);

        chunk_seq j = d::join_emit(pr, offset, jp);
        d::sweep(pr_pfx);   // PR consumed
        chunk_seq sorted_j = direct_sample_sort<d::JRec>(j, std::less<>{}, j2p);
        d::sweep(jp);
        chunk_seq keys2 = d::pair_keys(sorted_j, skp);
        d::sweep(j2p);
        sorted_k = direct_sample_sort<d::SaKey>(keys2, std::less<>{}, nkp);
        d::sweep(skp);

        rr = d::assign_ranks(sorted_k, npr);
        CHECK(rr.n == n) << "suffix_array: element count drift in stage " << stage;

        kp = nkp;
        pr_pfx = npr;
        // Doubling: offset saturates rather than overflowing past n.
        offset = (offset > n) ? offset : offset * 2;
    }

    // sorted_k is now ordered by an all-distinct key == suffix array order.
    chunk_seq sa = d::extract_sa(sorted_k, out_prefix);
    d::sweep(kp);
    d::sweep(pr_pfx);
    return sa;
}

}  // namespace ChunkSequenceOps

#endif  // CHUNK_SUFFIX_ARRAY_H
