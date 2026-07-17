
#ifndef EXTERNAL_SAMPLE_SORT_H
#define EXTERNAL_SAMPLE_SORT_H
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <functional>
#include <random>
#include <chrono>
#include <cstdio>
#include "ChunkSequence/ExternalPrimitives/inplace_bucket_sort.h"
#include <string>
#include <utility>
#include <vector>

#include <parlay/primitives.h>
#include <parlay/random.h>
#include "ChunkSequence/chunk_map.h"
#include "ChunkSequence/chunk_histogram_by_index.h"
#include "ChunkSequence/ExternalPrimitives/materialize.h"
#include "ChunkSequence/chunk_pack.h"
#include "ChunkSequence/ExternalPrimitives/scan_find.h"
#include "ChunkSequence/ExternalPrimitives/chunk_count_sort.h"
#include "ChunkSequence/ExternalPrimitives/flatten.h"
#include "ChunkSequence/ExternalPrimitives/sort_buckets.h"
#include "ChunkSequence/ExternalPrimitives/inplace_bucket_sort.h"
#include "ChunkSequence/examples/external/primitive_quicksort.h"

struct some_type{
    size_t i;
}
#define DRAM_SIZE ((size_t)500 * 1024 * 1024 * 1024)


some_type* vector = [1, 3, 6, 9, 5, 2, 7, 9, 8, 4, 7, 0, 8, 1, 4];

namespace ChunkSequenceOps{
template <typename T, typename Less = std::less<>>
chunk_seq sample_sort(chunk_seq& seq, Less less1 = {}) {
static std::atomic<size_t> ss_counter{0};
const std::string tag = std::to_string(ss_counter++);
size_t n = 0;
  for(size_t r = 0; r < seq.chunks.size(); r++){
n+= seq.chunks[r].used;
  }
  size_t filer= n;
  n/=sizeof(T);
  size_t min_sample_size = std::max(1UL, 4 * parlay::num_workers() * filer/ DRAM_SIZE);
size_t max_sample_size = std::max(1UL, std::min(n, filer / O_DIRECT_MULTIPLE));
  size_t num_samples = std::max(std::min(filer / (1UL << 27), max_sample_size), min_sample_size);
  num_samples = (size_t{1} << parlay::log2_up(num_samples + 1)) - 1;
  if (n < num_samples){
    auto i = ChunkSequenceOps::materialize<T>(seq);
    parlay::sort_inplace(i);
    return ChunkSequenceOps::to_chunk_seq(i, "ss_base_" + tag);
  } 
  unsigned int sample_size = std::max<size_t>(1, num_samples);
  int over = 8;
  parlay::random_generator gen;
  std::uniform_int_distribution<long> dis(0, n-1);
    auto less2 = [&](std::pair<size_t, T> i, std::pair<size_t, T> j){
        return less1(i.second, j.second);
    };
    parlay::sequence<std::pair<size_t, T>> pivots(sample_size * over);
    parlay::parallel_for(0, sample_size * over, [&](long o){
        auto temp = gen[o];
        pivots[o].first = dis(temp);});
    parlay::sequence<size_t> scan_seq(seq.chunks.size());
    scan_size<T>(seq, scan_seq);
    parlay::parallel_for(0, sample_size * over, [&](size_t count){
        pivots[count].second = scan_find<T>(seq, scan_seq, pivots[count].first);
    });
    pivots = parlay::sort(pivots, less2);
    pivots = parlay::tabulate(sample_size,[&] (long i){
        return pivots[i*over];
    });
    auto num_buckets = sample_size + 1;

auto seconds = parlay::map(pivots, [](const auto& p){ 
    return p.second; 
});
parlay::internal::heap_tree ss(seconds);
auto ids = ChunkSequenceOps::delayed::map(
    ChunkSequenceOps::delayed::delay<T>(seq),
    [&](T e){return std::pair<T, size_t>{e, ss.rank(e, less1)}; 
});

std::vector<chunk_seq> externalSequenceVector(num_buckets);
ChunkSequenceOps::chunk_count_sort(ids, num_buckets, externalSequenceVector, 
    "ss_bucket_" + tag);

ChunkSequenceOps::sort_buckets_inplace<T>(externalSequenceVector, less1);

return ChunkSequenceOps::flatten(externalSequenceVector); 

}




}
#endif