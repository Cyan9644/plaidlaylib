#ifndef EXTERNAL_ENGINE_H
#define EXTERNAL_ENGINE_H

#include <atomic>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

#include "parlay/primitives.h"
#include "parlay/utilities.h"   // parlay::hash64
#include "absl/log/check.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_seq_reader.h"
#include "utils/file_utils.h"
#include "utils/unordered_file_writer.h"
#include "configs.h"

namespace ChunkSequenceOps {

/**
 * The unified out-of-core transform engine shared by the eager streaming
 * primitives (ChunkMap, ChunkScan pass 2, …).
 *
 * Two building blocks live here:
 *   - ChunkEmitter<R>    : allocate output blocks and hand them to the writer,
 *                          recording a chunk descriptor for each.
 *   - ExternalTransform  : drive read -> body -> emit -> write for the
 *                          "one-or-more emit(s) per input chunk" family.
 *   - RemoveWorker     : drive read -> per-worker fold for the scalar family
 *                          (ChunkReduce, ChunkFindIf, ChunkScan pass 1).
 *
 * All three route through the single standardized reader (ChunkSequenceReader)
 * and writer (UnorderedFileWriter), so a chunk_seq produced by the engine is
 * indistinguishable from one produced by tabulate/dense-pack.
 *
 * Ownership rule for ExternalTransform: the engine owns each input buffer and
 * frees it back to the reader pool after `body` returns.  A body therefore
 * copies whatever it needs into freshly emitted output blocks (emit.alloc());
 * it must not retain or free the input pointer.  (This drops the old in-place
 * T==R buffer reuse; the extra in-DRAM copy is negligible next to the SSD
 * read+write it accompanies.)
 */
template<typename R>
class ChunkEmitter {
public:
    ChunkEmitter(const std::vector<std::string>& filenames,
                 std::vector<std::atomic<size_t>>& file_offsets,
                 std::atomic<size_t>& out_count,
                 std::vector<chunk>& out_chunks,
                 UnorderedFileWriter<R>& writer,
                 size_t num_drives)
        : filenames_(filenames), file_offsets_(file_offsets),
          out_count_(out_count), out_chunks_(out_chunks),
          writer_(writer), num_drives_(num_drives) {
        static_assert(CHUNK_SIZE % sizeof(R) == 0,
            "sizeof(R) must divide CHUNK_SIZE for O_DIRECT alignment");
    }

    // Number of R elements that fit in one output block.
    size_t out_cap() const { return CHUNK_SIZE / sizeof(R); }

    // Allocate a fresh, O_DIRECT-aligned CHUNK_SIZE output block.
    R* alloc() const {
        R* p = (R*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
        CHECK(p != nullptr) << "ChunkEmitter: allocation failed";
        return p;
    }

    /**
     * Emit one output block holding `count` valid R elements.  The caller must
     * have zero-padded the tail out to CHUNK_SIZE.  `logical_index` fixes this
     * block's position in the output order; ExternalTransform sorts by it and
     * (when compacting) renumbers to a dense 0..k-1 to restore the
     * index-ordered invariant.  Drive placement is deterministic balls-in-bins
     * via parlay::hash64 of the emission slot.
     */
    void emit(std::shared_ptr<R> buf, size_t count, size_t logical_index) const {
        const size_t slot = out_count_.fetch_add(1);
        const size_t d    = parlay::hash64(slot) % num_drives_;
        const size_t base = file_offsets_[d].fetch_add(CHUNK_SIZE);
        out_chunks_[slot] = chunk{filenames_[d], base, count * sizeof(R), logical_index};
        writer_.Push(std::move(buf), CHUNK_SIZE / sizeof(R), d, base);
    }

    // Convenience overload: adopt a raw block from alloc() (freed with free()).
    void emit(R* buf, size_t count, size_t logical_index) const {
        emit(std::shared_ptr<R>(buf, free), count, logical_index);
    }

private:
    const std::vector<std::string>& filenames_;
    std::vector<std::atomic<size_t>>& file_offsets_;
    std::atomic<size_t>& out_count_;
    std::vector<chunk>& out_chunks_;
    UnorderedFileWriter<R>& writer_;
    size_t num_drives_;
};

/**
 * Streaming transform: read every chunk of `seq`, hand each to `body`, write
 * whatever `body` emits, and return the resulting chunk_seq.
 *
 * body signature:
 *   void body(const T* in, size_t n, size_t index, const ChunkEmitter<R>& emit)
 * where `in`/`n` are the input chunk's data and element count and `index` is
 * its position in the input.  A body may emit any number of output blocks (up
 * to `max_out_per_input`); use `index * max_out_per_input + sub` as the
 * logical index of the sub-th emitted block to keep global order.
 *
 * @param max_out_per_input  Upper bound on emits per input chunk (sizes the
 *                           output descriptor table).
 * @param compact  Renumber output chunk indices to a dense 0..k-1 after
 *                  sorting (restores out.chunks[i].index == i).
 */
template<typename T, typename R = T, typename Body>
chunk_seq ExternalTransform(const chunk_seq& seq,
                            const std::string& result_prefix,
                            Body body,
                            size_t max_out_per_input = 1,
                            bool compact = true) {
    const size_t num_drives = GetSSDList().size();

    // Create/truncate one output file per drive.  The writer opens files with
    // O_CREAT but not O_TRUNC, so stale data from a prior run is cleared here.
    std::vector<std::string> filenames(num_drives);
    parlay::parallel_for(0, num_drives, [&](size_t d) {
        filenames[d] = GetFileName(result_prefix, d);
        int fd = open(filenames[d].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        SYSCALL(fd);
        SYSCALL(close(fd));
    }, /*granularity=*/1);

    ChunkSequenceReader<T> reader;
    reader.PrepChunks(seq);
    reader.Start(5, 32, 16);

    // One io_uring writer thread per drive; queue_size caps in-flight buffers
    // to 64 * 4 MB = 256 MB of DRAM at any moment.
    UnorderedWriterConfig wcfg;
    wcfg.num_threads   = num_drives;
    wcfg.io_uring_size = 32;
    wcfg.queue_size    = 64;
    wcfg.num_files     = num_drives;
    UnorderedFileWriter<R> writer;
    writer.Start(filenames, wcfg);

    // Output descriptors are filled by the emitter at slot = fetch_add order.
    std::vector<chunk> out_chunks(seq.chunks.size() * max_out_per_input);
    std::atomic<size_t> out_count{0};
    std::vector<std::atomic<size_t>> file_offsets(num_drives);
    for (auto& a : file_offsets) a.store(0, std::memory_order_relaxed);

    ChunkEmitter<R> emit(filenames, file_offsets, out_count, out_chunks, writer,
                         num_drives);

    parlay::parallel_for(0, parlay::num_workers(), [&](size_t) {
        while (true) {
            auto [ptr, n, idx] = reader.Poll();
            if (ptr == nullptr) break;
            body((const T*)ptr, n, idx, emit);
            reader.allocator.Free(ptr);
        }
    }, /*granularity=*/1);

    writer.Wait();

    // The reader delivers chunks out of order, so restore global order by the
    // logical index the body assigned, then densify if requested.
    out_chunks.resize(out_count.load());
    std::sort(out_chunks.begin(), out_chunks.end(),
              [](const chunk& a, const chunk& b) { return a.index < b.index; });
    if (compact)
        for (size_t k = 0; k < out_chunks.size(); k++) out_chunks[k].index = k;

    return {out_chunks};
}

/**
 * Scalar-fold driver: read every chunk of `seq` and return one accumulator per
 * parlay worker.  `worker(reader)` polls the shared reader to exhaustion
 * (Poll() returns nullptr when done) and returns its local accumulator; the
 * caller combines the per-worker results (e.g. parlay::reduce).  No writer is
 * started.  Workers may also just scatter into a shared array by chunk index
 * (as ChunkScan's pass 1 does) and return a placeholder.
 */
template<typename T, typename WorkerFn>
auto RemoveWorker(const chunk_seq& seq, size_t reader_threads, WorkerFn worker)
    -> parlay::sequence<std::invoke_result_t<WorkerFn, ChunkSequenceReader<T>&>> {
    ChunkSequenceReader<T> reader;
    reader.PrepChunks(seq);
    // More IO threads than default keep the SSDs saturated; CPU is not the limit.
    reader.Start(reader_threads, 32, 8);
    return parlay::tabulate(parlay::num_workers(),
                            [&](size_t) { return worker(reader); },
                            /*granularity=*/1);
}

} // namespace ChunkSequenceOps

#endif // EXTERNAL_ENGINE_H
