#ifndef CHUNK_FLAT_MAP_H
#define CHUNK_FLAT_MAP_H

#include <algorithm>
#include <cstdlib>
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
 * A produced batch for ChunkFlatMap: owns the reader-pool input buffers (still
 * referenced as each chunk's forward halo by its left neighbor, so they are
 * freed only at destruction) plus the per-chunk output sequences that DensePack
 * packs.  run(b) reads results[b] from the settled Batch, which is move-stable
 * (the outer vector's element storage is heap-allocated), so the pointer is
 * valid even when parlay::sequence uses its small-buffer form.
 */
template<typename T, typename R>
struct FlatMapBatch {
    std::unique_ptr<ChunkSequenceReader<T>> reader;  // keeps the pool alive
    std::vector<T*> bufs;  // one per chunk, index-sorted order
    std::vector<parlay::sequence<R>> results;

    FlatMapBatch() = default;
    FlatMapBatch(FlatMapBatch&&) = default;
    FlatMapBatch& operator=(FlatMapBatch&&) = default;
    FlatMapBatch(const FlatMapBatch&) = delete;
    FlatMapBatch& operator=(const FlatMapBatch&) = delete;
    ~FlatMapBatch() {
        if (reader)
            for (T* b : bufs) reader->allocator.Free(b);
    }

    size_t size() const { return results.size(); }
    DensePackRun<R> run(size_t b) const {
        return {results[b].data(), results[b].size()};
    }
};

} // namespace detail

/**
 * Out-of-core analogue of parlay::flatten(parlay::map(seq, f)) with an optional
 * forward halo: the stored-input sibling of ChunkFlatTabulate (which generates
 * its input from an index range).
 *
 * Reads `seq` chunk-by-chunk and calls f on each in parallel, packing the
 * returned per-chunk sequences densely (via DensePack) into an index-ordered
 * chunk_seq.  Each call receives, besides its own chunk, a read-only view of
 * the following chunk's first `halo` elements — the "forward halo" — so a body
 * can catch events that straddle a chunk boundary (e.g. a pattern match that
 * starts in this chunk and finishes in the next).  The engine sources the halo
 * from the in-batch right neighbor already resident in DRAM (127/128 chunks),
 * or, at a DensePack batch seam, from one small synchronous O_DIRECT read of
 * the next chunk's head.  `halo == 0` skips the seam read entirely, giving a
 * plain per-chunk flat-map with no neighbor access.
 *
 * The body must report only outputs "belonging to" its own chunk — i.e. events
 * whose logical start falls in [global_start, global_start + n) — so the halo
 * is used purely as lookahead and no output is double-counted.
 *
 * Requires halo < CHUNK_SIZE/sizeof(T) (the halo is satisfiable from the single
 * next chunk) and a dense input (every chunk but the last full — the library
 * invariant), so only the final chunk can be short.  At the very last chunk of
 * the sequence the halo is empty (halo_n == 0, halo may be nullptr).
 *
 * @tparam T  Input element type (must match the chunk_seq).
 * @tparam R  Output element type (sizeof(R) <= 8, the DensePack on-disk limit).
 * @tparam F  Callable: (const T* data, size_t n, uint64_t global_start,
 *             const T* halo, size_t halo_n) -> parlay::sequence<R>
 */
template<typename T, typename R, typename F>
chunk_seq ChunkFlatMap(const chunk_seq& seq, const std::string& result_prefix,
                       size_t halo, F f) {
    const size_t n_in = seq.chunks.size();
    if (n_in == 0) return {};
    CHECK(halo < CHUNK_SIZE / sizeof(T))
        << "ChunkFlatMap: halo must be smaller than one chunk";

    // Global element index of each chunk's first element (exclusive prefix sum
    // of element counts — metadata only, no I/O).
    std::vector<uint64_t> pos_of(n_in + 1);
    pos_of[0] = 0;
    for (size_t i = 0; i < n_in; i++)
        pos_of[i + 1] = pos_of[i] + seq.chunks[i].used / sizeof(T);

    // Bytes of chunk `c` a left neighbor may need as its halo: min(halo, count).
    auto head_bytes = [&](const chunk& c) {
        return std::min(halo * sizeof(T), c.used);
    };

    return DensePack<R>(n_in, result_prefix,
        [&](size_t base, size_t batch_n) {
            // Read this batch's contiguous slice [base, base+batch_n) with its
            // own reader, so completions can only belong to this batch.
            chunk_seq sub;
            sub.chunks.assign(seq.chunks.begin() + base,
                              seq.chunks.begin() + base + batch_n);
            auto reader = std::make_unique<ChunkSequenceReader<T>>();
            reader->PrepChunks(sub);
            reader->Start(5, 32, 16);

            struct BC { T* buf; size_t n; size_t idx; };
            std::vector<BC> bc(batch_n);
            for (size_t i = 0; i < batch_n; i++) {
                auto [ptr, n, cidx] = reader->Poll();
                bc[i] = {ptr, n, cidx};
            }
            // Restore logical order so bc[b+1] is bc[b]'s right neighbor.
            std::sort(bc.begin(), bc.end(),
                      [](const BC& a, const BC& b) { return a.idx < b.idx; });

            // Seam halo: the last chunk of this batch needs the head of the
            // next batch's first chunk — one small synchronous O_DIRECT read
            // (begin_addr is CHUNK_SIZE-aligned).
            T* seam = nullptr;
            size_t seam_count = 0;
            if (halo > 0 && base + batch_n < n_in) {
                const chunk& next = seq.chunks[base + batch_n];
                const size_t bytes = head_bytes(next);
                seam = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT,
                                         AlignUp(bytes));
                CHECK(seam != nullptr) << "ChunkFlatMap: seam allocation failed";
                int fd = open(next.filename.c_str(), O_DIRECT | O_RDONLY);
                SYSCALL(fd);
                SYSCALL(pread(fd, seam, AlignUp(bytes), (off_t)next.begin_addr));
                SYSCALL(close(fd));
                seam_count = bytes / sizeof(T);
            }

            detail::FlatMapBatch<T, R> batch;
            batch.reader = std::move(reader);
            batch.bufs.resize(batch_n);
            batch.results.resize(batch_n);
            parlay::parallel_for(0, batch_n, [&](size_t b) {
                batch.bufs[b] = bc[b].buf;
                const T* hbuf;
                size_t hn;
                if (halo == 0) {                // no halo requested
                    hbuf = nullptr;
                    hn = 0;
                } else if (b + 1 < batch_n) {   // right neighbor is in-batch
                    hbuf = bc[b + 1].buf;
                    hn = std::min(halo, bc[b + 1].n);
                } else {                        // batch seam (or last chunk)
                    hbuf = seam;
                    hn = seam_count;
                }
                batch.results[b] = f(bc[b].buf, bc[b].n, pos_of[base + b],
                                     hbuf, hn);
            }, /*granularity=*/1);

            free(seam);
            return batch;
        });
}

} // namespace ChunkSequenceOps

#endif // CHUNK_FLAT_MAP_H
