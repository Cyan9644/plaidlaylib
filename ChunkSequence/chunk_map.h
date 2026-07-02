#ifndef CHUNK_MAP_H
#define CHUNK_MAP_H

#include <algorithm>
#include <cstring>
#include <string>

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/external_engine.h"
#include "configs.h"

namespace ChunkSequenceOps {

/**
 * Apply f to every element across all chunks in seq, writing the results back
 * out as an index-ordered chunk_seq (out.chunks[i].index == i), so results are
 * directly chainable.
 *
 * Implemented as a thin body on ExternalTransform: each input chunk is mapped
 * into one or more output blocks.  When sizeof(R) > sizeof(T) an input chunk's
 * n elements may not fit in one output block, so up to FANOUT output blocks are
 * emitted per input; ExternalTransform sorts + densifies the emitted indices.
 *
 * Relies on the input's index-ordered invariant (seq.chunks[i].index == i).
 *
 * @tparam T  Input element type.
 * @tparam R  Output element type (defaults to T).
 */
template<typename T, typename R = T, typename F>
chunk_seq ChunkMap(const chunk_seq& seq, const std::string& result_prefix, F f) {
    // Worst-case output blocks per input chunk: an input chunk holds at most
    // CHUNK_SIZE/sizeof(T) elements, each becoming one R; an output block holds
    // CHUNK_SIZE/sizeof(R).  ceil(sizeof(R)/sizeof(T)) bounds the ratio.
    constexpr size_t FANOUT = (sizeof(R) + sizeof(T) - 1) / sizeof(T);

    return ExternalTransform<T, R>(seq, result_prefix,
        [f](const T* in, size_t n, size_t index, const ChunkEmitter<R>& emit) {
            const size_t cap = emit.out_cap();
            size_t produced = 0, sub = 0;
            // do/while so empty input chunks (n == 0) still emit one empty
            // output chunk, preserving the chunk-for-chunk structure.
            do {
                const size_t cnt = std::min(cap, n - produced);
                R* out = emit.alloc();
                for (size_t i = 0; i < cnt; i++) out[i] = f(in[produced + i]);
                memset((char*)out + cnt * sizeof(R), 0, CHUNK_SIZE - cnt * sizeof(R));
                emit.emit(out, cnt, index * FANOUT + sub);
                produced += cnt;
                sub++;
            } while (produced < n);
        },
        /*max_out_per_input=*/FANOUT);
}

} // namespace ChunkSequenceOps

#endif // CHUNK_MAP_H
