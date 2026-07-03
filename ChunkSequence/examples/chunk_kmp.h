#ifndef CHUNK_KMP_H
#define CHUNK_KMP_H

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

#include "absl/log/check.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_seq_reader.h"
#include "ChunkSequence/dense_pack.h"
#include "utils/file_utils.h"
#include "configs.h"

namespace ChunkSequenceOps {
namespace detail {

/**
 * A produced batch for ChunkKmp: owns the reader-pool text buffers (still
 * referenced as each chunk's overlap by its left neighbor, so they are freed
 * only at destruction) plus the per-chunk match-position sequences that
 * DensePack packs.  run(b) reads results[b] from the settled Batch, which is
 * move-stable (the outer vector's element storage is heap-allocated).
 */
template<typename CharT>
struct KmpBatch {
    std::unique_ptr<ChunkSequenceReader<CharT>> reader;  // keeps the pool alive
    std::vector<CharT*> bufs;  // one per chunk, index-sorted order
    std::vector<parlay::sequence<uint64_t>> results;

    KmpBatch() = default;
    KmpBatch(KmpBatch&&) = default;
    KmpBatch& operator=(KmpBatch&&) = default;
    KmpBatch(const KmpBatch&) = delete;
    KmpBatch& operator=(const KmpBatch&) = delete;
    ~KmpBatch() {
        if (reader)
            for (CharT* b : bufs) reader->allocator.Free(b);
    }

    size_t size() const { return results.size(); }
    DensePackRun<uint64_t> run(size_t b) const {
        return {results[b].data(), results[b].size()};
    }
};

/**
 * Sequential KMP over one chunk's text plus its overlap into the next chunk.
 * Ported from parlaylib's block loop (examples/knuth_morris_pratt.h): scan
 * text[0, n_b + ov) but only report matches that START in [0, n_b); the
 * stale-tail condition `(i - tail) <= n_b` stops as soon as no partial match
 * that started in range can still complete.  The overlap lives in a separate
 * buffer (the right neighbor's head), hence the two-segment `at()`.
 *
 * One fix over the parlay original: after reporting a match it left
 * tail == m-1, so the next comparison read pattern[m] (one past the end);
 * we instead continue from failure[tail], which still finds overlapping
 * matches (standard KMP).
 */
template<typename CharT>
inline void KmpScanChunk(const CharT* text, long n_b,
                         const CharT* overlap, long ov,
                         const CharT* pattern, long m,
                         const long* failure,
                         uint64_t global_pos,
                         parlay::sequence<uint64_t>& out) {
    auto at = [&](long i) { return i < n_b ? text[i] : overlap[i - n_b]; };
    long tail = -1;
    for (long i = 0; i < n_b + ov && (i - tail) <= n_b; i++) {
        const CharT c = at(i);
        while (tail != -1 && c != pattern[tail + 1])
            tail = failure[tail];
        if (c == pattern[tail + 1]) tail++;
        if (tail == m - 1) {
            out.push_back(global_pos + (uint64_t)(i - tail));
            tail = failure[tail];
        }
    }
}

} // namespace detail

/**
 * Knuth-Morris-Pratt search over an out-of-core text: find every occurrence of
 * `pattern` in the chunk_seq of CharT elements `seq`, returning the global
 * match-start positions as a tightly packed, index-ordered chunk_seq of
 * uint64_t (in text order).
 *
 * Each chunk is scanned sequentially by one worker (KMP is inherently
 * sequential within a block) and chunks are processed in parallel — the
 * out-of-core analogue of parlaylib's block-parallel KMP example.  A match
 * starting in chunk k can extend up to m-1 chars into chunk k+1; since a
 * DensePack batch holds 128 consecutive chunks in memory sorted by index, that
 * overlap is read straight from the right neighbor's buffer.  Only at batch
 * seams (1 chunk in 128) is the next chunk's head fetched with one small
 * synchronous O_DIRECT read.
 *
 * Requires m <= CHUNK_SIZE/sizeof(CharT) (the pattern fits in a chunk, so a
 * match spans at most 2 chunks) and a dense input (every chunk but the last
 * full — the library invariant), so only the final chunk can be short.
 *
 * @tparam CharT    Element type of the text (must match the chunk_seq).
 * @tparam Pattern  Random-access container of CharT with size().
 */
template<typename CharT = char, typename Pattern>
chunk_seq ChunkKmp(const chunk_seq& seq,
                   const std::string& result_prefix,
                   const Pattern& pattern) {
    const size_t n_in = seq.chunks.size();
    const long m = (long)pattern.size();
    const size_t epct = CHUNK_SIZE / sizeof(CharT);
    CHECK((size_t)m <= epct) << "ChunkKmp: pattern must fit within one chunk";
    if (m == 0 || n_in == 0) return {};

    // Local copies of the pattern and its failure function (built sequentially,
    // exactly as in the parlay original; both are at most one chunk in size).
    std::vector<CharT> pat(m);
    for (long i = 0; i < m; i++) pat[i] = pattern[i];
    std::vector<long> failure(m, -1);
    for (long r = 1, l = -1; r < m; r++) {
        while (l != -1 && pat[l + 1] != pat[r])
            l = failure[l];
        if (pat[l + 1] == pat[r])
            failure[r] = ++l;
    }

    // Global text position of each chunk's first element (exclusive prefix sum
    // of element counts — metadata only, no I/O).
    std::vector<uint64_t> pos_of(n_in + 1);
    pos_of[0] = 0;
    for (size_t i = 0; i < n_in; i++)
        pos_of[i + 1] = pos_of[i] + seq.chunks[i].used / sizeof(CharT);

    // Bytes of chunk `c` a left neighbor may need as overlap: min(m-1, count).
    auto head_bytes = [&](const chunk& c) {
        return std::min((size_t)(m - 1) * sizeof(CharT), c.used);
    };

    return DensePack<uint64_t>(n_in, result_prefix,
        [&](size_t base, size_t batch_n) {
            // Read this batch's contiguous slice [base, base+batch_n) with its
            // own reader, so completions can only belong to this batch.
            chunk_seq sub;
            sub.chunks.assign(seq.chunks.begin() + base,
                              seq.chunks.begin() + base + batch_n);
            auto reader = std::make_unique<ChunkSequenceReader<CharT>>();
            reader->PrepChunks(sub);
            reader->Start(5, 32, 16);

            struct BC { CharT* buf; size_t n; size_t idx; };
            std::vector<BC> bc(batch_n);
            for (size_t i = 0; i < batch_n; i++) {
                auto [ptr, n, cidx] = reader->Poll();
                bc[i] = {ptr, n, cidx};
            }
            // Restore logical order so bc[b+1] is bc[b]'s right neighbor.
            std::sort(bc.begin(), bc.end(),
                      [](const BC& a, const BC& b) { return a.idx < b.idx; });

            // Seam overlap: the last chunk of this batch needs the head of the
            // next batch's first chunk — one small synchronous O_DIRECT read
            // (begin_addr is CHUNK_SIZE-aligned).
            CharT* seam = nullptr;
            size_t seam_count = 0;
            if (base + batch_n < n_in && m > 1) {
                const chunk& next = seq.chunks[base + batch_n];
                const size_t bytes = head_bytes(next);
                seam = (CharT*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT,
                                             AlignUp(bytes));
                CHECK(seam != nullptr) << "ChunkKmp: seam allocation failed";
                int fd = open(next.filename.c_str(), O_DIRECT | O_RDONLY);
                SYSCALL(fd);
                SYSCALL(pread(fd, seam, AlignUp(bytes), (off_t)next.begin_addr));
                SYSCALL(close(fd));
                seam_count = bytes / sizeof(CharT);
            }

            detail::KmpBatch<CharT> batch;
            batch.reader = std::move(reader);
            batch.bufs.resize(batch_n);
            batch.results.resize(batch_n);
            parlay::parallel_for(0, batch_n, [&](size_t b) {
                batch.bufs[b] = bc[b].buf;
                const CharT* overlap;
                long ov;
                if (b + 1 < batch_n) {          // right neighbor is in-batch
                    overlap = bc[b + 1].buf;
                    ov = (long)std::min((size_t)(m - 1), bc[b + 1].n);
                } else {                        // batch seam (or last chunk)
                    overlap = seam;
                    ov = (long)seam_count;
                }
                detail::KmpScanChunk<CharT>(bc[b].buf, (long)bc[b].n,
                                            overlap, ov, pat.data(), m,
                                            failure.data(), pos_of[base + b],
                                            batch.results[b]);
            }, /*granularity=*/1);

            free(seam);
            return batch;
        });
}

} // namespace ChunkSequenceOps

#endif // CHUNK_KMP_H
