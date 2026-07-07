#ifndef EXTERNAL_QUICK_SORT_H
#define EXTERNAL_QUICK_SORT_H
#include <algorithm>
#include <atomic>
#include <functional>
#include <random>

#include <parlay/primitives.h>
#include <parlay/random.h>
#include "ChunkSequence/chunk_map.h"
#include "ChunkSequence/chunk_histogram_by_index.h"
#include "ChunkSequence/ExternalPrimitives/materialize.h"
#include "ChunkSequence/chunk_pack.h"
#include "ChunkSequence/ExternalPrimitives/scan_find.h"
#include "ChunkSequence/ExternalPrimitives/chunk_count_sort2.h"
#include "ChunkSequence/ExternalPrimitives/flatten.h"


namespace ChunkSequenceOps{
template <typename T, typename Less = std::less<>>
chunk_seq quicksort(chunk_seq& seq, Less less1 = {}) {
static std::atomic<size_t> ss_counter{0};
const std::string tag = std::to_string(ss_counter++);
size_t n = 0;
  for(size_t r = 0; r < seq.chunks.size(); r++){
n+= seq.chunks[r].used;
  }
  n/=sizeof(T);
  
  if (n < 5096){

    auto i = ChunkSequenceOps::materialize<T>(seq);

    std::sort(i.begin(), i.end(), less1);
    return ChunkSequenceOps::to_chunk_seq(i, "ss_base_" + tag);

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
  
auto ids = ChunkMap<T, size_t>(seq, "ss_id_" + tag,[&](T e){
    return ss.rank(e, less1);
});

std::vector<chunk_seq> externalSequenceVector(num_buckets);
ChunkSequenceOps::chunk_count_sort2<T>(seq, ids, externalSequenceVector, "ss_bucket_" + tag);

parlay::parallel_for(0, num_buckets, [&](long i){

    size_t z = 0;
    for (const auto& c :externalSequenceVector[i].chunks){
        z += c.used / sizeof(T);
    }
    if (z ==n) {
        auto v = ChunkSequenceOps::materialize<T>(externalSequenceVector[i]);
        std::sort(v.begin(), v.end(), less1);
        externalSequenceVector[i] = ChunkSequenceOps::to_chunk_seq(
            v, "ss_deg_" + tag + "_" + std::to_string(i));
        return;
    }

    externalSequenceVector[i] = sample_sort<T>(externalSequenceVector[i], less1);


});












return ChunkSequenceOps::flatten(externalSequenceVector);


}

}

#endif
