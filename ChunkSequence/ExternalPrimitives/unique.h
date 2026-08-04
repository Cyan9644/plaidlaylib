#ifndef CHUNK_UNIQUE_H
#define CHUNK_UNIQUE_H

#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "utils/logger.h"
#include "ChunkSequence/ExternalPrimitives/count_sort.h"
#include "ChunkSequence/ExternalPrimitives/flatten.h"
#include "ChunkSequence/ExternalPrimitives/materialize.h"
#include "ChunkSequence/chunk_delayed.h"
#include "ChunkSequence/chunk_seq.h"
#include "absl/log/check.h"
#include "configs.h"


//this is adapted straight from the filter method.
//the basic logic we want is to break into blocks
//specifically using a hash function, so that identical elements will go to the same block.
//from there, we just run deduplication in parallel on each block.
namespace ChunkSequenceOps{

template <typename T = uint64_t, typename Hash = std::hash<T>,
         typename Less = std::less<>>
chunk_seq unique(const chunk_seq& seq, const std::string& result_prefix,size_t num_buckets = 0, Hash hash = {}, Less less = {}){
  if (seq.chunks.empty()) return{};
  if (num_buckets == 0){
    size_t file_size = 0;
    for (const chunk& c : seq.chunks) file_size +=c.used;
    size_t min_sample_size =std::max(1UL, 4 * parlay::num_workers() * file_size / MAIN_MEMORY_SIZE);
    size_t max_sample_size = std::max(
        1UL, std::min(file_size / sizeof(T), file_size / O_DIRECT_MULTIPLE));
    num_buckets = std::max(std::min(file_size / (1UL << 27), max_sample_size),
                           min_sample_size) +1;}
  namespace d = ChunkSequenceOps::delayed;
  auto ids = d::map(d::delay<T>(seq), [hash, num_buckets](T v){
    return std::pair<T, size_t>{v, (size_t)(hash(v) % num_buckets)};
  });

  std::vector<chunk_seq> buckets(num_buckets);
  ChunkSequenceOps::count_sort(ids, num_buckets, buckets, result_prefix);

  parlay::parallel_for(0, num_buckets, [&](size_t b){
    if (buckets[b].chunks.empty()) return;
    auto vals = ChunkSequenceOps::sequential_materialize<T>(buckets[b]);
    parlay::sort_inplace(vals, less);
    vals.resize(std::unique(vals.begin(), vals.end()) - vals.begin());
    buckets[b] = ChunkSequenceOps::sequential_to_chunk_seq(
        vals, result_prefix + "_u" + std::to_string(b));
  });

  return ChunkSequenceOps::flatten(buckets);}

}  // namespace ChunkSequenceOps



#endif
