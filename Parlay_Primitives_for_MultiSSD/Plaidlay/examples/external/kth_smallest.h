#include <algorithm>
#include <functional>
#include <random>

#include <parlay/primitives.h>
#include <parlay/random.h>

#include "externalSeq.h"
#include "helper/heap_tree.h"

// template <typename Range, typename Less = std::less<>>

auto kth_smallest(ExternalSequence &seq, long k, Less less = {}) {
//   long n = in.size();
//   if (n <= 1000) return parlay::sort(in,less)[k];

  // pick 31 pivots by randomly choosing 8 * 31 keys, sorting them,
  // and taking every 8th key (i.e. oversampling)


  int sample_size = 31;
  int over = 8;
  parlay::random_generator gen;
  std::uniform_int_distribution<long> dis(0, n-1);

  //this part is totally fine for DRAM, it is very unlikely that sample_size  * over exceeds oru 512 GB in any setting
  auto pivots = parlay::sort(parlay::tabulate(sample_size*over, [&] (long i) {
    auto r = gen[i];
    return in[dis(r)];}), less);

//find actual pivots, which are every over_sampleth selected pivot
  pivots = parlay::tabulate(sample_size,[&] (long i) {return pivots[i*over];});

  // Determine which of the 32 buckets each key belongs in

  //using a heap tree is not okay for this setting because all the data can't fit in DRAM, so we'll use the bucketing algorithm from samplesort



    //scatter gather config currently takes this config:
    // std::vector<FileInfo> Run(std::vector<FileInfo> &input_files,
    //                           const std::string &result_prefix,
    //                           const AssignerFunction assigner,
    //                           const ProcessorFunction processor,
    //                           const ScatterGatherConfig &config)

    ScatterGatherConfig config;
    config.bucketed_writer_config.num_buckets = pivots.size();
    auto results = scatter_gather.Run(input_files, result_prefix,
                                          assigner.GetAssigner(),
                                          simple_processor,
                                          config);
//   heap_tree ss(pivots);
//   auto ids = parlay::tabulate(n, [&] (long i) -> unsigned char {
//     return ss.find(in[i], less);});

  // Count how many in keys are each bucket

  //this needs to be refactored
  auto sums = parlay::histogram_by_index(ids, sample_size+1);

  // find which bucket k belongs in, and pack the keys in that bucket into next
  
    //this 
    // ExternalSequence seq;
    // size_t total;
   auto [seq, total] = externalSeqOps::scan(sums);

   //problem: calculating the upper bound directly is very difficult
  auto id = std::upper_bound(offsets.begin(), offsets.end(), k) - offsets.begin() - 1;
  auto next = parlay::pack(in, externalSeqOps::delayed_map(ids, [=] (auto b) {return b == id;}));

  // recurse on much smaller set, adjusting k as needed
  return kth_smallest(next, k - offsets[id], less);
}