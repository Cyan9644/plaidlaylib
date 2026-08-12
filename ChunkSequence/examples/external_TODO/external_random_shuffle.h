#ifndef EXTERNAL_RANDOM_SHUFFLE_H
#define EXTERNAL_RANDOM_SHUFFLE_H
#include <parlay/primitives.h>
#include <parlay/random.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ChunkSequence/Primitives/operation.h"
#include "ChunkSequence/Primitives/count_sort.h"
#include "ChunkSequence/Primitives/flatten.h"
#include "ChunkSequence/Primitives/small_sequence_ops.h"
#include "ChunkSequence/Primitives/materialize.h"
#include "ChunkSequence/Primitives/scan_find.h"
#include "ChunkSequence/Primitives/sort_buckets.h"
#include "ChunkSequence/Primitives/histogram_by_index.h"
#include "ChunkSequence/Primitives/map.h"
#include "ChunkSequence/Primitives/pack.h"
#include "ChunkSequence/examples/external_TODO/primitive_quicksort.h"
#include "configs.h"
#include "utils/file_utils.h"


namespace plaid {

//this is an implemented random shuffle that uses basically the same idea as samplesort
//the difference is that our delayed map is computed with a function that returns a random pivot value
//and then within each "bucket," ideally actually distributed across drives, we shuffle the data
template <typename T, typename Less = std::less<>>
chunk_seq random_shuffle(chunk_seq& seq, Less less1 = {}, size_t disk_span = 1) {
  static std::atomic<size_t> shuffle_counter{0};
  const std::string tag = std::to_string(shuffle_counter++);

  size_t n = 0;
  for (size_t r = 0; r < seq.chunks.size(); r++) {
    n += seq.chunks[r].used;
  }
  size_t filer = n;
  n /= sizeof(T);

  size_t min_sample_size =
      std::max(1UL, 4 * parlay::num_workers() * filer / MAIN_MEMORY_SIZE);

  size_t max_sample_size =
      std::max(1UL, std::min(n, filer / O_DIRECT_MULTIPLE));

  size_t num_samples =
      std::max(std::min(filer / (1UL << 27), max_sample_size), min_sample_size);

  unsigned int sample_size = std::max<size_t>(1, num_samples);
  auto num_buckets = sample_size + 1;
  parlay::random_generator gen;
  namespace d = plaid::delayed;
  auto src = d::delay<T>(seq);
    //the mapping should return a random bucket index
  auto ids = d::map(
      d::zip(src, d::tabulate(src.length(), [](size_t i) { return i; })),
      [&, num_buckets](const std::pair<T, size_t>& e) {
        auto g = gen[e.second];
        std::uniform_int_distribution<size_t> dis(0, num_buckets - 1);
        return std::pair<T, size_t>{e.first, dis(g)};
      });

  std::vector<chunk_seq> externalSequenceVector(num_buckets);
  //bucket randomly
  plaid::count_sort(ids, num_buckets, externalSequenceVector,
                               "ss_bucket_" + tag, disk_span);
    //sort each bucket, which is assumed to be able to fit in DRAM. In theory this might not be the case,
    //since the bucket IDs are random, but this is true within reason.
  plaid::apply<ChunkOperation::Shuffle, T>(externalSequenceVector, less1);
  auto result = plaid::flatten(externalSequenceVector);

  return result;







}

}

#endif