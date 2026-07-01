//wrapper for common paradigm
//allows you to compute many single-pass operations on a streamed collection 
//cross-chunk states are not supported yet
#ifndef EXTERNAL_ENGINE_H
#define EXTERNAL_ENGINE_H

#include "config_threads.h"
#include "plaidlay.h"
#include "chunk_header.h"
#include <atomic>
#include <array>
#include <algorithm>
#include <memory>
#include <vector>
#include <string>
#include <cstdlib>
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/utilities.h>
#include "utils/unordered_file_reader_modified.h"
#include "utils/unordered_file_writer_modified.h"

#ifndef NUM_SSDS
#define NUM_SSDS 30
#endif


template <typename R>
class ChunkEmitter{
public:
    ChunkEmitter(const std::vector<std::string>& filenames,
                 std::array<std::atomic<size_t>, NUM_SSDS>& file_offsets,
                 std::atomic<size_t>& out_count,
                 parlay::sequence<chunk_header>& out_headers,
                 UnorderedChunkWriter<R>& writer,
                 size_t block_bytes,
                 size_t out_cap)
        :filenames_(filenames),
          file_offsets(file_offsets),
          out_count_(out_count),
          out_headers_(out_headers),
          writer_(writer),
          block_size_bytes(block_bytes),
          out_cap_(out_cap) {}

    size_t out_cap() const{return out_cap_;}
    R* alloc() const{
        return (R*) aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, block_size_bytes);
    }

    void emit(R* buf, size_t count, size_t logical_index) const{
        size_t slot = out_count_.fetch_add(1);
        size_t ssd = parlay::hash64(slot) % NUM_SSDS;
        size_t base = file_offsets[ssd].fetch_add(block_size_bytes);

        chunk_header h;
        h.index = logical_index;
        h.filename = filenames_[ssd];
        h.used = count * sizeof(R);
        h.begin_address = base;
        out_headers_[slot] = h;

        writer_.Push(std::shared_ptr<R>(buf, free), out_cap_, ssd, base);
    }

private:
    const std::vector<std::string>& filenames_;
    std::array<std::atomic<size_t>, NUM_SSDS>& file_offsets;
    std::atomic<size_t>& out_count_;
    parlay::sequence<chunk_header>& out_headers_;
    UnorderedChunkWriter<R>& writer_;
    size_t block_size_bytes;
    size_t out_cap_;
};
template <typename T, typename R = T, typename Body>
External_Sequence ExternalTransform(External_Sequence& seq,
                                    const std::vector<std::string>& filenames,
                                    Body body,
                                    size_t max_out_per_input = 1,
                                    bool compact = true) {
    constexpr size_t B = CHUNK_SIZE;
    static_assert(B % sizeof(R) == 0,
                  "block size must be a whole number of output elements");
    constexpr size_t out_cap = B / sizeof(R);

    auto& chunk_headers = seq.ordered_underlying_sequence;

    UnorderedChunkReader<T, B> reader;
    reader.PrepFiles(chunk_headers);
    reader.Start();

    size_t expected_reads = (chunk_headers.size() + NUM_SSDS - 1) / NUM_SSDS;

    UnorderedChunkWriter<R> writer;
    UnorderedChunkWriterConfig wconfig;
    wconfig.num_threads = WRITER_THREADS;
    writer.Start(filenames, wconfig);

    External_Sequence result(chunk_headers.size() * max_out_per_input);
    parlay::sequence<chunk_header>& out = result.ordered_underlying_sequence;

    std::atomic<size_t> out_count{0};
    std::array<std::atomic<size_t>, NUM_SSDS> file_offsets{};

    ChunkEmitter<R> emit(filenames, file_offsets, out_count, out, writer, B,
                         out_cap);

    size_t read_count = 0;
    while (read_count < expected_reads) {
        parlay::parallel_for(0, NUM_SSDS, [&](size_t ) {
            auto [ptr, size, _, index, which_chunk, filename] = reader.Poll();
            if (ptr == nullptr) {
                return;
            }
            body((const T*) ptr, size, index, emit);
            reader.allocator.Free(ptr);
        });
        read_count++;
    }

    writer.Wait();

    out.resize(out_count.load());
    std::sort(result.begin(), result.end(),
              [](const chunk_header& a, const chunk_header& b) {
                  return a.index < b.index;
              });
    if (compact){
        for (size_t k = 0; k < out.size(); k++) out[k].index = k;
    }
    return result;
}

#endif
