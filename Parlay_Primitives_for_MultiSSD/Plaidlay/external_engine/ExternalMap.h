#ifndef EXTERNAL_ENGINE_MAP_H
#define EXTERNAL_ENGINE_MAP_H

#include "external_engine.h"

// ExternalMap re-expressed on top of ExternalTransform. Compare with the
// hand-rolled ../ExternalMap.h: the batch loop, the per-output offset/slot
// atomics, the SSD hashing, and the final trim+sort+densify are all owned by
// the engine now. What's left is the element-for-element mapping and the
// fan-out arithmetic.
//
// map is element-for-element, so one input chunk fans out into
// FANOUT = ceil(sizeof(R)/sizeof(T)) output blocks (== 1 whenever
// sizeof(R) <= sizeof(T)). An empty input chunk still emits one empty block so
// block structure is preserved.
template <typename T, typename R = T, typename MapFn>
External_Sequence ExternalMap(External_Sequence& seq, MapFn f,
                              const std::vector<std::string>& new_filenames) {
    constexpr size_t B = CHUNK_SIZE;
    static_assert(B % sizeof(T) == 0,
                  "block size must be a whole number of input elements");
    static_assert(B % sizeof(R) == 0,
                  "block size must be a whole number of output elements");
    constexpr size_t FANOUT = (sizeof(R) + sizeof(T) - 1) / sizeof(T);

    return ExternalTransform<T, R>(
        seq, new_filenames,
        [&](const T* in, size_t size, size_t index, ChunkEmitter<R>& out) {
            const size_t out_cap = out.out_cap();
            // One output block per up-to-out_cap run; empty chunk -> one block.
            size_t num_sub = (size == 0) ? 1 : (size + out_cap - 1) / out_cap;
            for (size_t s = 0; s < num_sub; s++) {
                size_t lo = s * out_cap;
                size_t hi = std::min(size, (s + 1) * out_cap);
                R* obuf = out.alloc();
                for (size_t j = lo; j < hi; j++) obuf[j - lo] = f(in[j]);
                // index * FANOUT + s keeps a chunk's sub-blocks ordered; the
                // engine re-densifies the gaps after the sort.
                out.emit(obuf, hi - lo, index * FANOUT + s);
            }
        },
        /*max_out_per_input=*/FANOUT);
}

// Convenience overload: derive the NUM_SSDS output filenames from a prefix.
template <typename T, typename R = T, typename MapFn>
External_Sequence ExternalMap(External_Sequence& seq, MapFn f,
                              const std::string& prefix) {
    std::vector<std::string> new_filenames;
    new_filenames.reserve(NUM_SSDS);
    for (int i = 0; i < NUM_SSDS; i++) {
        new_filenames.push_back(prefix + "_" + std::to_string(i));
    }
    return ExternalMap<T, R>(seq, f, new_filenames);
}

#endif
