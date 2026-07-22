// Shared helpers for the out-of-core suffix-array examples.
//
// Both suffix-array examples in this directory are streaming, sort-based, and
// avoid random access by turning every "gather rank[q]" into a sort-join:
//   - chunk_suffix_array.h : prefix doubling (O(log n) rounds, ~2 sorts/round)
//   - chunk_dc3.h          : DC3 / skew (O(n); geometric 2/3 shrink per level)
// This header holds the pieces they share so the naming pass in particular (a
// two-level max-scan over a key-sorted sequence, with per-chunk DRAM boundary
// metadata) lives in one tested place.
//
// Everything here is density-independent (built on ExternalTransform /
// RemoveWorker), so it tolerates the dense-except-last chunk_seqs that
// direct_sample_sort produces without any repack.  Positions and ranks are
// 32-bit (n < 2^32).

#ifndef CHUNK_SA_COMMON_H
#define CHUNK_SA_COMMON_H

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "absl/log/check.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_seq_reader.h"
#include "ChunkSequence/external_engine.h"
#include "utils/file_utils.h"
#include "configs.h"

namespace ChunkSequenceOps {
namespace sa_detail {

// (pos, rank) — the naming pass's output element (rank is 1-based; see below).
struct PR { uint32_t pos, rank; };
static_assert(CHUNK_SIZE % sizeof(PR) == 0, "PR must divide CHUNK_SIZE");

// Elements before each chunk, in logical (index) order; pre[nc] = n.  Tolerates
// a non-dense (sort-output) chunk_seq: uses each chunk's `used` byte count.
inline std::vector<size_t> elem_prefix(const chunk_seq& seq, size_t esz) {
    const size_t nc = seq.chunks.size();
    std::vector<size_t> cnt(nc, 0);
    for (const chunk& c : seq.chunks) {
        CHECK(c.index < nc) << "suffix_array: chunk_seq not index-ordered";
        cnt[c.index] = c.used / esz;
    }
    std::vector<size_t> pre(nc + 1, 0);
    for (size_t i = 0; i < nc; i++) pre[i + 1] = pre[i] + cnt[i];
    return pre;
}

// Remove every file on every drive whose name begins with `prefix` (sort leaves
// tag-suffixed intermediates a GetFileName enumeration would miss).
inline void sweep(const std::string& prefix) {
    for (const std::string& dir : GetSSDList()) {
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
            const std::string name = e.path().filename().string();
            if (name.rfind(prefix, 0) == 0) std::filesystem::remove(e.path(), ec);
        }
    }
}

struct RankResult { chunk_seq pr; size_t groups; size_t n; };

// ── naming / re-rank over a key-sorted sequence ───────────────────────────────
// Given a chunk_seq of `T` already sorted so equal-key elements are adjacent,
// assign each element the sorted index of its group head + 1 (a two-level
// max-scan) and emit PR{ get_pos(elem), head_index + 1 }.  Ranks are **1-based**:
// rank 0 is reserved as the beyond-end sentinel, which must sort strictly below
// every real suffix so a suffix that runs off the end orders before a longer one
// sharing its prefix.  Returns {pr, #distinct groups, n}; #groups == n means all
// keys are unique.
//
//   same(a, b)   : are a and b in the same key group (equal sort key)?
//   get_pos(a)   : the position column to carry into PR.
//
// The group-boundary test at a chunk seam needs the previous chunk's last key,
// supplied from a per-chunk DRAM array of first/last elements (O(n_chunks)).
template <typename T, typename SameGroup, typename GetPos>
inline RankResult assign_ranks(const chunk_seq& sorted, const std::string& pfx,
                               SameGroup same, GetPos get_pos) {
    const size_t nc = sorted.chunks.size();
    auto pre = elem_prefix(sorted, sizeof(T));
    const size_t n = pre[nc];

    // pass 1: per-chunk first/last element, internal group-head count + max head
    // index (a head is an element whose key differs from its predecessor).
    std::vector<T> firstk(nc), lastk(nc);
    std::vector<long long> internal_max(nc, -1);
    std::vector<size_t> internal_cnt(nc, 0);
    RemoveWorker<T>(sorted, 10, [&](ChunkSequenceReader<T>& r) {
        while (true) {
            auto [p, cnt, idx] = r.Poll();
            if (p == nullptr) break;
            if (cnt > 0) {
                firstk[idx] = p[0];
                lastk[idx] = p[cnt - 1];
                long long mx = -1;
                size_t heads = 0;
                for (size_t i = 1; i < cnt; i++) {
                    if (!same(p[i - 1], p[i])) { mx = (long long)(pre[idx] + i); heads++; }
                }
                internal_max[idx] = mx;
                internal_cnt[idx] = heads;
            }
            r.allocator.Free(p);
        }
        return 0;
    });

    // seed[c] = max head index over chunks < c; first_head[c] = does chunk c open
    // a new group (its first key differs from chunk c-1's last key)?
    std::vector<long long> seed(nc, -1);
    std::vector<char> first_head(nc, 0);
    long long running = -1;
    size_t groups = 0;
    for (size_t c = 0; c < nc; c++) {
        if (pre[c + 1] == pre[c]) { seed[c] = running; continue; }  // empty chunk
        seed[c] = running;
        const bool fh = (c == 0) || !same(lastk[c - 1], firstk[c]);
        first_head[c] = fh;
        long long chunk_head = internal_max[c];
        if (fh) chunk_head = std::max(chunk_head, (long long)pre[c]);
        if (chunk_head >= 0) running = std::max(running, chunk_head);
        groups += internal_cnt[c] + (fh ? 1u : 0u);
    }

    // pass 2: inclusive max-scan of head indices; emit (pos, head_index + 1).
    chunk_seq pr = ExternalTransform<T, PR>(sorted, pfx,
        [nc, &pre, &seed, &first_head, same, get_pos](
            const T* in, size_t n2, size_t idx, const ChunkEmitter<PR>& emit) {
            PR* out = emit.alloc();
            long long run = seed[idx];
            for (size_t i = 0; i < n2; i++) {
                const bool flag = (i == 0) ? (bool)first_head[idx] : !same(in[i - 1], in[i]);
                if (flag) run = (long long)(pre[idx] + i);
                out[i] = PR{get_pos(in[i]), (uint32_t)(run + 1)};
            }
            memset((char*)out + n2 * sizeof(PR), 0, CHUNK_SIZE - n2 * sizeof(PR));
            emit.emit(out, n2, idx);
        },
        /*max_out_per_input=*/1);

    return {std::move(pr), groups, n};
}

}  // namespace sa_detail
}  // namespace ChunkSequenceOps

#endif  // CHUNK_SA_COMMON_H
