#ifndef SORT_BUCKETS_H
#define SORT_BUCKETS_H

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include <unistd.h>

#include <parlay/primitives.h>

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_seq_reader.h"
#include "ChunkSequence/ExternalPrimitives/flatten.h"
#include "configs.h"

namespace ChunkSequenceOps {

template <typename T = uint64_t, typename Less = std::less<>>
chunk_seq sort_buckets(std::vector<chunk_seq>& buckets, Less less = {},
                       const std::string& prefix = "sorted") {
    static_assert(CHUNK_SIZE % sizeof(T) == 0,
        "sizeof(T) must divide CHUNK_SIZE for O_DIRECT alignment");
    const size_t B = buckets.size();

    std::vector<size_t> bn(B, 0);
    for (size_t b = 0; b < B; b++)
        for (const chunk& c : buckets[b].chunks) bn[b] += c.used / sizeof(T);

    size_t budget = ((size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE)) / 4;
    if (const char* e = getenv("SORT_BUCKETS_BUDGET_BYTES")) budget = std::stoull(e);
    const size_t budget_elems = std::max<size_t>(1, budget / sizeof(T));

    const size_t num_threads = std::max<size_t>(1, GetSSDList().size());

    std::vector<chunk_seq> wave_outputs;
    size_t wave_id = 0;

    size_t lo = 0;
    while (lo < B) {
        size_t hi = lo, wave_elems = 0;
        while (hi < B && (hi == lo || wave_elems + bn[hi] <= budget_elems)) {
            wave_elems += bn[hi];
            hi++;
        }

        std::vector<size_t> local_base(hi - lo);
        {
            size_t acc = 0;
            for (size_t b = lo; b < hi; b++) { local_base[b - lo] = acc; acc += bn[b]; }
        }

        parlay::sequence<T> buf(wave_elems);

        chunk_seq merged;
        std::vector<size_t> dest;
        for (size_t b = lo; b < hi; b++) {
            size_t off = local_base[b - lo];
            for (const chunk& c : buckets[b].chunks) {
                chunk nc = c;
                nc.index = merged.chunks.size();
                merged.chunks.push_back(nc);
                dest.push_back(off);
                off += c.used / sizeof(T);
            }
        }

        fprintf(stderr, "[sb] wave %zu buckets[%zu,%zu) wave_elems=%zu chunks=%zu budget_elems=%zu\n",
                wave_id, lo, hi, wave_elems, merged.chunks.size(), budget_elems);
        if (!merged.chunks.empty()) {
            ChunkSequenceReader<T> reader;
            reader.PrepChunks(merged);
            reader.Start(num_threads, 32, 16, 128);
            while (true) {
                auto [ptr, n, idx] = reader.Poll();
                if (ptr == nullptr) break;
                CHECK(idx < dest.size()) << "sort_buckets: idx " << idx << " >= " << dest.size();
                CHECK(dest[idx] + n <= wave_elems)
                    << "sort_buckets: overrun dest=" << dest[idx] << " n=" << n
                    << " wave_elems=" << wave_elems;
                std::memcpy(buf.data() + dest[idx], ptr, n * sizeof(T));
                reader.allocator.Free(ptr);
            }
        }
        fprintf(stderr, "[sb] wave %zu read done, sorting\n", wave_id);

        parlay::parallel_for(lo, hi, [&](size_t b) {
            T* base = buf.data() + local_base[b - lo];
            parlay::sort_inplace(parlay::make_slice(base, base + bn[b]), less);
        }, 1);

        wave_outputs.push_back(
            to_chunk_seq(buf, prefix + "_" + std::to_string(wave_id++)));
        lo = hi;
    }

    return flatten(wave_outputs);
}

}  // namespace ChunkSequenceOps

#endif  // SORT_BUCKETS_H
