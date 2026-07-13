#ifndef INPLACE_BUCKET_SORT_H
#define INPLACE_BUCKET_SORT_H

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

// Rewrite each bucket in `buckets` in place on disk (see file comment): bucket b
// is read into one contiguous DRAM buffer and handed to
// `processor(b, buf, nelem)`, which permutes/overwrites those nelem elements;
// the buffer is then written back over the bucket's own chunks.  The chunk
// headers are unchanged, so `buckets` remains a valid vector of external
// sequences that the caller can flatten().
template <typename T = uint64_t, typename Processor>
void process_buckets_inplace(std::vector<chunk_seq>& buckets, Processor processor) {
    static_assert(CHUNK_SIZE % sizeof(T) == 0,
        "sizeof(T) must divide CHUNK_SIZE for O_DIRECT alignment");
    const size_t ept = CHUNK_SIZE / sizeof(T);
    const size_t B   = buckets.size();

    parlay::parallel_for(0, B, [&](size_t b) {
        const size_t nc = buckets[b].chunks.size();
        if (nc == 0) return;
        T* buf = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, nc * CHUNK_SIZE);
        CHECK(buf != nullptr) << "process_buckets_inplace: buffer alloc failed";

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

        processor(b, buf, nelem);

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

// Sort each bucket in place on disk: the phase-2 base sorter for
// external_samplesort.
template <typename T = uint64_t, typename Less = std::less<>>
void sort_buckets_inplace(std::vector<chunk_seq>& buckets, Less less = {}) {
    process_buckets_inplace<T>(buckets, [&](size_t, T* buf, size_t nelem) {
        parlay::sort_inplace(parlay::make_slice(buf, buf + nelem), less);
    });
}

}  // namespace ChunkSequenceOps

#endif  // INPLACE_BUCKET_SORT_H

