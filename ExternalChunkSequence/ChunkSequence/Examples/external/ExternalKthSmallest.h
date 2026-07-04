#ifndef EXTERNAL_KTH_SMALLESTH_H
#define EXTERNAL_KTH_SMALLESTH_H
#include <algorithm>
#include <functional>
#include <random>

#include <parlay/primitives.h>
#include <parlay/random.h>
#include "ChunkSequence/chunk_map.h"
#include "ChunkSequence/ExternalPrimitives/external_histogram_by_index.h"
#include "ChunkSequence/ExternalPrimitives/materialize.h"
#include "ChunkSequence/chunk_pack.h"
// parlay::internal::heap_tree comes in via <parlay/primitives.h> above; its
// header has no include guard, so do NOT include it a second time here.
#include "ChunkSequence/ExternalPrimitives/LinearFind.h"



//we are currently assuming that not all elemsents go into 1 bucket, for obvious reasons.
namespace ChunkSequenceOps{
//randomized ~O(n) algorithm
template <typename T, typename Less = std::less<>>
T kth_smallest(chunk_seq& seq, long k, Less less1 = {}) {
size_t n = 0;
  for(size_t r = 0; r < seq.chunks.size(); r++){
n+= seq.chunks[r].used; //add the used size of each chunk to the n; this tells us the number of total elements in the sequence
  }
  n/=sizeof(T);
//   long n = seq.chunks.size();
  
  if (n < 1536){ //1536 elements is the point at which we say we can materialize and sort directly

    auto i = ChunkSequenceOps::materialize<T>(seq); //materialize external sequence to parlay sequence (not yet cleanly implemented)
    //ok but there's not really a reason to sort
    return parlay::sort(i,less1)[k];

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
    parlay::parallel_for(0, sample_size * over, [&](size_t count)){
  
        pivots[count].second = LinearFind<T>(seq, pivots[count].first); //this is intended to find the value in question
    }
    }
    
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
  
//   auto ids = ChunkSequenceOps::tabulate(seq, [&] (long i) -> unsigned char {
//     return ss.find(in[i], less);});
// auto ids = ChunkMap<T, unsigned char>(seq,"ids_prefix", [&](T e){ return ss.find(e, less1);});
auto ids = ChunkMap<T, unsigned char>(seq, "id_" + std::to_string(n),[&](T e){
    return ss.rank(e, less1);
});
  // Count how many in keys are each bucket
//   auto sums = ChunkSequenceOps::histogram_by_index(ids, sample_size+1);
auto sums = ChunkSequenceOps::histogram_by_index<unsigned char>(ids, sample_size+1);

  // find which bucket k belongs in, and pack the keys in that bucket into next
  auto [offsets, total] = parlay::scan(sums);
  auto id = std::upper_bound(offsets.begin(), offsets.end(), k) - offsets.begin() - 1;
//   auto next = ChunkSequenceOps::pack(seq, "next_prefix", parlay::delayed_map(ids, [=] (auto b) {return b == id;}));
auto flags=ChunkMap<unsigned char, bool>(ids, "flags_" + std::to_string(n),[=] (unsigned char b){ 
    return b == id;
}
);
auto next= ChunkSequenceOps::pack<T>(seq, "next_" +std::to_string(n), flags);
  // recurse on much smaller set, adjusting k as needed
    //note that currently this is on a parlay sequence:: we'll want to make this an external sequence 
    //as soon as we can implement the pack method
  return kth_smallest<T>(next, k - offsets[id], less1);
}

}

#endif