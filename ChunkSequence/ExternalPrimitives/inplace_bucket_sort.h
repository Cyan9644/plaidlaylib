#ifndef INPLACE_BUCKET_SORT_H
#define INPLACE_BUCKET_SORT_H

// sort_buckets_inplace — the phase-2 base sorter for external_samplesort.
//
// After the count-sort has scattered the input into per-bucket external
// sequences (each sized to fit in DRAM), every bucket still needs to be sorted
// internally.  This is the out-of-core analogue of Peter's WorkerOnlyPhase2: it
// does the minimum I/O and holds no long-lived thread pools.
//
// One parallel_for task per bucket (granularity 1): each task reads its bucket's
// chunks into one contiguous O_DIRECT buffer, sorts it in place, and writes the
// chunks straight back to the *same* files/offsets they came from.  Because the
// on-disk layout is reused verbatim, the buckets' chunk_seq headers are already
// correct on return — the caller just flattens them.
//
// Keeping read+sort+write together per bucket (rather than three separate global
// passes) is what overlaps I/O with compute: while one worker sorts bucket i,
// other workers are reading/writing buckets j,k — the same read/sort/write
// pipelining peter_samplesort gets from its per-worker io_uring double-buffer,
// here for free from the work-stealing scheduler across buckets.
//
// Why this beats the old per-bucket materialize()+to_chunk_seq() path: that path
// span up a fresh RemoveWorker reader pool (~10 threads + an io_uring ring) *and*
// a fresh UnorderedFileWriter (one thread per drive) for *every* bucket, and ran
// all buckets through an outer parallel_for at once — hundreds of threads and
// io_uring rings contending on a handful of cores (and churning RLIMIT_MEMLOCK).
// Here the only parallelism is two flat parallel_fors over chunks (I/O) and one
// over buckets (the in-DRAM sort), so nothing is oversubscribed.
//
// Density assumption: chunk_count_sort emits dense buckets (every chunk full at
// ELEMS_PER_CHUNK except the final partial one), so reading chunk ci into buffer
// slot ci*ept keeps both the O_DIRECT target alignment (each slot is
// CHUNK_SIZE-aligned) and the logical contiguity the in-place sort needs.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include <parlay/primitives.h>

#include "absl/log/check.h"

#include "ChunkSequence/chunk_seq.h"
#include "utils/file_utils.h"
#include "configs.h"

namespace ChunkSequenceOps {

// Sort each bucket in `buckets` in place on disk (see file comment).  The chunk
// headers are unchanged, so `buckets` remains a valid vector of sorted
// external sequences that the caller can flatten().
template <typename T = uint64_t, typename Less = std::less<>>
void sort_buckets_inplace(std::vector<chunk_seq>& buckets, Less less = {}) {
    static_assert(CHUNK_SIZE % sizeof(T) == 0,
        "sizeof(T) must divide CHUNK_SIZE for O_DIRECT alignment");
    const size_t ept = CHUNK_SIZE / sizeof(T);
    const size_t B   = buckets.size();

    parlay::parallel_for(0, B, [&](size_t b) {
        const size_t nc = buckets[b].chunks.size();
        if (nc == 0) return;
        T* buf = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, nc * CHUNK_SIZE);
        CHECK(buf != nullptr) << "sort_buckets_inplace: buffer alloc failed";

        // Read the bucket's chunks into slot ci*ept (dense: every chunk full at
        // ept except the last, so slots stay CHUNK_SIZE-aligned for O_DIRECT and
        // the live elements land contiguously in [0, nelem)).
        size_t nelem = 0;
        for (size_t ci = 0; ci < nc; ci++) {
            const chunk& c = buckets[b].chunks[ci];
            int fd = open(c.filename.c_str(), O_RDONLY | O_DIRECT);
            SYSCALL(fd);
            SYSCALL(pread(fd, buf + ci * ept, AlignUp(c.used), (off_t)c.begin_addr));
            close(fd);
            nelem += c.used / sizeof(T);
        }

        parlay::sort_inplace(parlay::make_slice(buf, buf + nelem), less);

        // Zero-pad the tail of the final (partial) chunk's block so its
        // write-back is a deterministic full O_DIRECT block.
        const size_t padded = nc * ept;
        if (nelem < padded)
            std::memset(buf + nelem, 0, (padded - nelem) * sizeof(T));

        for (size_t ci = 0; ci < nc; ci++) {
            const chunk& c = buckets[b].chunks[ci];
            int fd = open(c.filename.c_str(), O_WRONLY | O_DIRECT);
            SYSCALL(fd);
            SYSCALL(pwrite(fd, buf + ci * ept, CHUNK_SIZE, (off_t)c.begin_addr));
            close(fd);
        }
        free(buf);
    }, /*granularity=*/1);
}

}  // namespace ChunkSequenceOps

#endif  // INPLACE_BUCKET_SORT_H

