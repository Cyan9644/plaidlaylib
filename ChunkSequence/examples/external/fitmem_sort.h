//partial external samplesort, assumes that buckets CAN fit in memory
#ifndef FITMEM_SORT_H
#define FITMEM_SORT_H
#include <algorithm>
#include <atomic>
#include <functional>
#include <random>

#include <parlay/primitives.h>
#include <parlay/random.h>
#include "ChunkSequence/ExternalPrimitives/materialize.h"
#include "ChunkSequence/ExternalPrimitives/scan_find.h"
#include "ChunkSequence/ExternalPrimitives/chunk_count_sort2.h"
#include "ChunkSequence/ExternalPrimitives/flatten.h"


namespace ChunkSequenceOps{
// Out-of-core sample sort, one bucketing level only.  Unlike the fully external
// sample_sort (external_samplesort.h), this variant assumes each bucket, after a
// single round of oversampled-pivot partitioning, is small enough to fit in
// DRAM, so it materializes and sorts each bucket directly rather than recursing.
template <typename T, typename Less = std::less<>>
chunk_seq fitmem_sort(chunk_seq& seq, Less less1 = {}) {
static std::atomic<size_t> ss_counter{0};
const std::string tag = std::to_string(ss_counter++);
size_t n = 0;
  for(size_t r = 0; r < seq.chunks.size(); r++){
n+= seq.chunks[r].used;
  }
  n/=sizeof(T);

  if (n < (2 << 12)){

    auto i = ChunkSequenceOps::materialize<T>(seq);

    parlay::sort_inplace(i);
    return ChunkSequenceOps::to_chunk_seq(i, "fs_base_" + tag);

  }

  int sample_size = 31;
  int over = 8;
  parlay::random_generator gen;
  std::uniform_int_distribution<long> dis(0, n-1);

    auto less2 = [&](std::pair<size_t, T> i, std::pair<size_t, T> j){
        return less1(i.second, j.second);
    };
    parlay::sequence<std::pair<size_t, T>> pivots(sample_size * over);
    parlay::parallel_for(0, sample_size * over, [&](long o){
        auto temp = gen[o];
        pivots[o].first = dis(temp);

    });


    parlay::sequence<size_t> scan_seq(seq.chunks.size());
    scan_size<T>(seq, scan_seq);

    parlay::parallel_for(0, sample_size * over, [&](size_t count){

        pivots[count].second = scan_find<T>(seq, scan_seq, pivots[count].first);
    });


    pivots = parlay::sort(pivots, less2);
    pivots = parlay::tabulate(sample_size,[&] (long i)
    {
        return pivots[i*over];
    });
    auto num_buckets = sample_size + 1;


auto seconds = parlay::map(pivots, [](const auto& p){ return p.second; });
  parlay::internal::heap_tree ss(seconds);

// Route every key into its bucket in a single streaming pass over seq, deriving
// the bucket from the value (ss.rank) on the fly.  The earlier version first
// materialized a full size_t bucket-id chunk_seq to disk with ChunkMap (an 8n
// write + 8n read) purely to feed chunk_count_sort2; folding the rank into the
// count sort drops that entire pass -- seq is now read once here, not twice.
std::vector<chunk_seq> externalSequenceVector(num_buckets);
ChunkSequenceOps::chunk_count_sort_by_key<T>(
    seq, num_buckets, externalSequenceVector,
    [&](T e){ return ss.rank(e, less1); },
    "fs_bucket_" + tag);

// Each bucket is assumed to fit in DRAM: pull it in, sort it in place, and write
// the sorted run back out.  No recursion (that is external_samplesort's job).
parlay::parallel_for(0, num_buckets, [&](long i){
    auto v = ChunkSequenceOps::materialize<T>(externalSequenceVector[i]);
    parlay::sort_inplace(v, less1);
    externalSequenceVector[i] = ChunkSequenceOps::to_chunk_seq(
        v, "fs_sorted_" + tag + "_" + std::to_string(i));
});

return ChunkSequenceOps::flatten(externalSequenceVector);
}

}

#endif
