#ifndef EXTERNAL_ENGINE_FILTER_H
#define EXTERNAL_ENGINE_FILTER_H

#include "external_engine.h"

// ExternalFilter re-expressed on top of ExternalTransform. Compare with the
// hand-rolled ../externalFilter.h: the two-phase slot/bad_flags/file_offsets
// bookkeeping, the batch loop, and the reader/writer plumbing are all gone --
// what's left is the actual per-chunk work.
//
// Each input chunk produces exactly one output block (the survivors), so the
// fan-out bound is 1 and the emitted index is just the input chunk's index.
//
// The body is templated (not std::function) so the predicate inlines per
// element. The public predicate type stays std::function for a stable API.
template <typename T>
External_Sequence ExternalFilter(External_Sequence& seq,
                                 const std::function<bool(const T)>& predicate,
                                 const std::vector<std::string>& new_filenames) {
    return ExternalTransform<T, T>(
        seq, new_filenames,
        [&](const T* in, size_t size, size_t index, ChunkEmitter<T>& out) {
            T* buf = out.alloc();
            size_t k = 0;
            for (size_t j = 0; j < size; j++) {
                if (predicate(in[j])) buf[k++] = in[j];  // k <= size <= out_cap
            }
            out.emit(buf, k, index);  // one block per input chunk (may be empty)
        },
        /*max_out_per_input=*/1);
}

// Convenience overload: derive the NUM_SSDS output filenames from a prefix.
template <typename T>
External_Sequence ExternalFilter(External_Sequence& seq,
                                 const std::function<bool(const T)>& predicate,
                                 const std::string& prefix) {
    std::vector<std::string> new_filenames;
    new_filenames.reserve(NUM_SSDS);
    for (int i = 0; i < NUM_SSDS; i++) {
        new_filenames.push_back(prefix + "_" + std::to_string(i));
    }
    return ExternalFilter<T>(seq, predicate, new_filenames);
}

#endif
