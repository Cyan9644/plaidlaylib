//this kth_smallest assumes that a bucket CAN fit in memory
#ifndef FITMEM_KTH_SMALLEST_H
#define FITMEM_KTH_SMALLEST_H
#include <algorithm>
#include <functional>
#include <random>

#include <parlay/primitives.h>
#include <parlay/random.h>
#include "ChunkSequence/chunk_histogram_by_index.h"
#include "ChunkSequence/ExternalPrimitives/materialize.h"
#include "ChunkSequence/chunk_pack.h"
// parlay::internal::heap_tree comes in via <parlay/primitives.h> above; its
// header has no include guard, so do NOT include it a second time here.
#include "ChunkSequence/ExternalPrimitives/scan_find.h"



//we are currently assuming that not all elemsents go into 1 bucket, for obvious reasons.
namespace ChunkSequenceOps{
//randomized ~O(n) algorithm
template <typename T, typename Less = std::less<>>
T fitmem_kth_smallest(chunk_seq& seq, long k, Less less1 = {}) {
size_t n = 0;
  for(size_t r = 0; r < seq.chunks.size(); r++){
n+= seq.chunks[r].used; //add the used size of each chunk to the n; this tells us the number of total elements in the sequence
  }
  n/=sizeof(T);
//   long n = seq.chunks.size();
  
  if (n < 1536){ //1536 elements is the point at which we say we can materialize and sort directly

    auto i = ChunkSequenceOps::materialize<T>(seq); //materialize external sequence to parlay sequence (not yet cleanly implemented)
    //no reason to sort over select
    std::nth_element(i.begin(), i.begin() + k, i.end(), less1);
    return i[k];

  } 

  // pick 31 pivots by randomly choosing 8 * 31 keys, sorting them,
  // and taking every 8th key (i.e. oversampling)
  int sample_size = 31;
  int over = 8;
  parlay::random_generator gen;
  std::uniform_int_distribution<long> dis(0, n-1);
//   auto pivots = parlay::sort(ChunkSequence::tabulate(sample_size*over, [&] (long i) {
//     auto r = gen[i];
//     return seq[dis(r)];}), less); 
//     //need to find some way around this random access
//   pivots = parlay::tabulate(sample_size,[&] (long i) {return pivots[i*over];});
  
    auto less2 = [&](std::pair<size_t, T> i, std::pair<size_t, T> j){
        return less1(i.second, j.second);
    };
    //get a set of random indices
    parlay::sequence<std::pair<size_t, T>> pivots(sample_size * over);
    parlay::parallel_for(0, sample_size * over, [&](long o){
        auto temp = gen[o];
        pivots[o].first = dis(temp);

    });
    // auto pivots = parlay::tabulate(sample_size*over, [&] (long i) {
    // return = gen[i];
    // });

    //we now have the random indices, so we just need to figure out where these live in the actual chunk sequence
    // for(size_t count = 0; count < sample_size * over; count++){
        
    //     pivots[count].second = LinearFind<T>(seq, pivots[count].first); //this is intended to find the value in question
    // }
    parlay::sequence<size_t> scan_seq(seq.chunks.size());
    scan_size<T>(seq, scan_seq);

    parlay::parallel_for(0, sample_size * over, [&](size_t count){
  
        pivots[count].second = scan_find<T>(seq, scan_seq, pivots[count].first); //this is intended to find the value in question
    });

    //we now have the values of the pivots in memory

    //take the oversampleth pivots
    pivots = parlay::sort(pivots, less2);
    pivots = parlay::tabulate(sample_size,[&] (long i) 
    {
        return pivots[i*over];
    });


  // Determine which of the 32 buckets each key belongs in
//   auto seconds = pivots | std::views::elements<1>; //heap tree needs a random access container so we can't use views
//for this. also we're on C++ 17 instead of 20
auto seconds = parlay::map(pivots, [](const auto& p){ return p.second; });
  parlay::internal::heap_tree ss(seconds);

  // Bucket of a key = its rank among the pivots.  Cheap (a heap_tree walk, ~5
  // comparisons for 32 buckets), so we recompute it inline in both remaining
  // passes rather than materializing a bucket-id chunk_seq to disk with ChunkMap
  // (which cost a write of the ids plus a read-back in each of the two uses).
  auto key_of = [&](T e){ return (size_t)ss.rank(e, less1); };

  // Count how many keys fall in each bucket, straight from the values.
auto sums = ChunkSequenceOps::ChunkHistogramByKey<T>(seq, sample_size + 1, key_of);

  // find which bucket k belongs in, and pack the keys in that bucket into next
  auto [offsets, total] = parlay::scan(sums);
  auto id = std::upper_bound(offsets.begin(), offsets.end(), k) - offsets.begin() - 1;

  // Pack the winning bucket directly on the value predicate -- one read pass over
  // seq, no id chunk_seq to read alongside it.
auto next = ChunkSequenceOps::pack_value<T>(
    seq, "fk_next_" + std::to_string(n),
    [&, id](T e){ return key_of(e) == (size_t)id; });
  // The winning bucket is assumed to fit in DRAM (its probabilistic size bound is
  // O(n/sqrt(m)) for m buckets), so instead of recursing out-of-core we pull it
  // in and select directly, adjusting k to the bucket-local rank (k - offsets[id]).
  // parlay::kth_smallest returns an *iterator* to the selected element (into the
  // sequence passed in), so dereference it while the materialized bucket is still
  // alive (its lifetime extends to the end of this full return expression).
  return *parlay::kth_smallest(ChunkSequenceOps::materialize<T>(next),
                               (size_t)(k - offsets[id]), less1);
}

}

#endif
