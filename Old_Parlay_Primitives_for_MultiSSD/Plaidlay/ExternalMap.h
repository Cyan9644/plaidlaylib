#ifndef EXTERNAL_MAP_H
#define EXTERNAL_MAP_H
#include "config_threads.h"
#include <pthread.h>
#include "plaidlay.h"
#include <cassert>
#include <math.h>
#include <iostream>
#include <fcntl.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include "chunk_header.h"
#include <liburing.h>
#include <cstring>
#include <atomic>
#include <array>
#include <algorithm>
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/utilities.h>
#include "utils/unordered_file_reader_modified.h"
#include "utils/unordered_file_writer_modified.h"


#ifndef NUM_SSDS
#define NUM_SSDS 30
#endif

// Map every element of an External_Sequence through `f`, producing a new
// External_Sequence on disk. This is the chunk-model analogue of
// sequence_algorithms/map.h; its read/process/write structure follows
// externalFilter.h.
//
// T is the input element type, R the output element type (defaults to T). Unlike
// filter, map is element-for-element, so an input chunk holding more output bytes
// than fit in a 4 MiB block (only possible when sizeof(R) > sizeof(T)) fans out
// into several output blocks. The fan-out factor is a compile-time constant.
//
// On-disk block convention is the same as the other primitives: every block
// occupies a full 4 MiB region and chunk_header.used records the valid byte
// count. That requires the 4 MiB block to be a whole number of T's and R's,
// which holds for all power-of-two element sizes (guarded below).
template<typename T, typename R = T, typename MapFn>
External_Sequence ExternalMap(External_Sequence &seq, MapFn f,
                              const std::vector<std::string> &new_filenames) {
    constexpr size_t B = CHUNK_SIZE;  // 4 MiB block, matches the reader/writer
    static_assert(B % sizeof(T) == 0,
                  "block size must be a whole number of input elements");
    static_assert(B % sizeof(R) == 0,
                  "block size must be a whole number of output elements");

    constexpr size_t out_cap = B / sizeof(R);  // output elements per full block
    // Max output blocks one input chunk can produce. ceil(in_cap / out_cap) ==
    // ceil(sizeof(R) / sizeof(T)); == 1 whenever sizeof(R) <= sizeof(T).
    constexpr size_t FANOUT = (sizeof(R) + sizeof(T) - 1) / sizeof(T);

    auto &chunk_headers = seq.ordered_underlying_sequence;

    UnorderedChunkReader<T, B> reader;
    reader.PrepFiles(chunk_headers);
    reader.Start();

    size_t expected_reads = (chunk_headers.size() + NUM_SSDS - 1) / NUM_SSDS;

    UnorderedChunkWriter<R> writer;
    UnorderedChunkWriterConfig wconfig;
    wconfig.num_threads = WRITER_THREADS;
    writer.Start(new_filenames, wconfig);

    // Upper bound on output chunk count; trimmed to out_count at the end.
    External_Sequence sequence(chunk_headers.size() * FANOUT);
    parlay::sequence<chunk_header> *out = &sequence.ordered_underlying_sequence;

    std::atomic<size_t> out_count{0};  // global slot allocator for output headers
    std::array<std::atomic<size_t>, NUM_SSDS> file_offsets{};

    size_t read_count = 0;
    while (read_count < expected_reads) {
        parlay::parallel_for(0, NUM_SSDS, [&](size_t i) {
            auto [ptr, size, _, index, which_chunk, filename] = reader.Poll();
            if (ptr == nullptr) {
                return;  // no chunk for this slot (partial final batch)
            }

            // One output block per up-to-out_cap run of elements. An empty input
            // chunk still emits a single empty output block so block structure is
            // preserved (mirrors filter keeping a header per input chunk).
            size_t num_sub = (size == 0) ? 1 : (size + out_cap - 1) / out_cap;

            for (size_t s = 0; s < num_sub; s++) {
                size_t lo = s * out_cap;
                size_t hi = std::min(size, (s + 1) * out_cap);
                size_t cnt = hi - lo;

                R *obuf = (R *) aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, B);
                for (size_t j = 0; j < cnt; j++) {
                    obuf[j] = f(ptr[lo + j]);
                }

                // file_offsets[] is atomic, out_count is atomic, and writer.Push
                // enqueues onto a thread-safe queue, so reserving the offset/slot
                // and pushing directly from the parallel loop is safe even with a
                // variable number of outputs per thread. mt19937 is not
                // thread-safe; choose the sink SSD with a deterministic hash.
                size_t ssd = parlay::hash64(read_count * NUM_SSDS * FANOUT +
                                            i * FANOUT + s) % NUM_SSDS;
                size_t base = file_offsets[ssd].fetch_add(B);
                size_t slot = out_count.fetch_add(1);

                chunk_header chunked;
                chunked.index = index * FANOUT + s;  // keeps sub-blocks ordered
                chunked.filename = new_filenames[ssd];
                chunked.used = cnt * sizeof(R);
                chunked.begin_address = base;
                (*out)[slot] = chunked;

                // Push a full B-byte block (out_cap elements) so the write is
                // O_DIRECT aligned; `used` records how much is valid.
                writer.Push(std::shared_ptr<R>(obuf, free), out_cap, ssd, base);
            }

            reader.allocator.Free(ptr);
        });
        read_count++;
    }

    writer.Wait();

    out->resize(out_count.load());
    std::sort(sequence.begin(), sequence.end(),
              [](const chunk_header &a, const chunk_header &b) {
                  return a.index < b.index;
              });
    // Fan-out leaves index gaps (index*FANOUT + s). Re-densify to 0..M-1 so
    // downstream code that assumes contiguous indices (e.g. LocalExternalScan in
    // externalScan.h) stays correct. Order is already established by the sort.
    for (size_t k = 0; k < out->size(); k++) {
        (*out)[k].index = k;
    }
    return sequence;
}

// Convenience overload: derive the NUM_SSDS output filenames from a prefix,
// mirroring the primes(n, prefix) wrapper in examples/external/primes.h.
template<typename T, typename R = T, typename MapFn>
External_Sequence ExternalMap(External_Sequence &seq, MapFn f,
                              const std::string &prefix) {
    std::vector<std::string> new_filenames;
    new_filenames.reserve(NUM_SSDS);
    for (int i = 0; i < NUM_SSDS; i++) {
        new_filenames.push_back(prefix + "_" + std::to_string(i));
    }
    return ExternalMap<T, R>(seq, f, new_filenames);
}

#endif
