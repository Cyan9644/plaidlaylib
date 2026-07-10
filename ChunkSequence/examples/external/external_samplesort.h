//purely external samplesort, assumes that buckets CANNOT fit in memory

#ifndef EXTERNAL_SAMPLE_SORT_H
#define EXTERNAL_SAMPLE_SORT_H
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
#include "ChunkSequence/ExternalPrimitives/chunk_count_sort.h"
#include "ChunkSequence/ExternalPrimitives/flatten.h"
#include "ChunkSequence/ExternalPrimitives/sort_buckets.h"
#include "ChunkSequence/examples/external/primitive_quicksort.h"

#define DRAM_SIZE ((size_t)500 * 1024 * 1024 * 1024) //==500 GB


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
// size_t max_sample_size = std::max(1UL, std::min(n / sizeof(T), filer / O_DIRECT_MULTIPLE));
size_t max_sample_size = std::max(1UL, std::min(n, filer / O_DIRECT_MULTIPLE));
  auto num_samples = std::max(std::min(filer / (1UL << 27), max_sample_size), min_sample_size);
  
  if (n < num_samples){
    //we're likely going to want a fast quicksort method that takes better advantage of our 
    //memory representation than just external->materialize->sort->external
    //but this clearly should work so we'll go with it for now, also any external quicksort would need to materialize everything anyway
    //but could overlap the I/O with computation.

    auto i = ChunkSequenceOps::materialize<T>(seq); //it would be good to make this materialize into a parlay sequence (now done)

    // std::sort(i.begin(), i.end(), less1);
    // i = parlay::sort(i);
    parlay::sort_inplace(i);
    //this is going to be really slow if we just materialize and write back, probably the best thing to do is to pass a parlay sequence
    //at the recurring call based on the size, but this is kind of messy
    //the easiest way to fix the problem is to just make materialize faster by parallelizing it
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
  

// std::vector<chunk_seq> externalSequenceVector(num_buckets);
// ChunkSequenceOps::chunk_count_sort<T>(seq, ids, externalSequenceVector, "ss_bucket_" + tag);

// std::vector<chunk_seq> externalSequenceVector(num_buckets);
// ChunkSequenceOps::chunk_count_sort_by_key<T>(
//     seq, num_buckets, externalSequenceVector,
//     [&](T e){ return ss.rank(e, less1); },
//     "ss_bucket_" + tag);

// //one potential optimization here is that ChunkMap doesn't really need to write its data back to disk
// //we could just make this delayed
// //the issue then is that chunk_count_sort takes a materialized chunk sequence, so we'll need to edit it or make a new one
//this is exactly what the code does now
auto ids = ChunkSequenceOps::delayed::map(ChunkSequenceOps::delayed::delay<T>(seq),
    [&](T e){ return std::pair<T, size_t>{e, ss.rank(e, less1)}; });

std::vector<chunk_seq> externalSequenceVector(num_buckets);
ChunkSequenceOps::chunk_count_sort(ids, num_buckets, externalSequenceVector,
                                   "ss_bucket_" + tag);

//it should now be the case that externalSequenceVector is a full vector of the individual external sequences
//in this case we just need a simple flatten to put all the chunk headers into a single list
//because each individual sequence should be sorted and they're in order,
//we'll get a total sorted ordering
parlay::parallel_for(0, num_buckets, [&](long i){

    size_t z = 0;
    for (const auto& c :externalSequenceVector[i].chunks){
        z += c.used / sizeof(T);
    }
    if (z ==n) {
        //if we're not going to get anything from the partition, i.e. we have an empty partition
        auto v = ChunkSequenceOps::materialize<T>(externalSequenceVector[i]);
        std::sort(v.begin(), v.end(), less1);
        externalSequenceVector[i] = ChunkSequenceOps::to_chunk_seq(
            v, "ss_deg_" + tag + "_" + std::to_string(i));
        return;
    }

    //recurring on this samplesort chain is pretty expensive due to the buffer allocation
    //we might consider just an external quicksort for this level, as it would use much less memory.
    //probably multiple levels of parallelism won't help much regardless since we need to do reads
    //for the pivots = large overhead on recurring calls

    //so instead we're going to call the external quicksort method, which we actually have a method for now but it's 
    //implemented using primitives.
    //to get the best performance out of this example, we're going to want to implement it manually with a reader/writer
    externalSequenceVector[i] = primitive_quicksort<T>(externalSequenceVector[i], less1);


});

return ChunkSequenceOps::flatten(externalSequenceVector);


//this sort_buckets method doesn't actually help the performance
// return ChunkSequenceOps::sort_buckets<T>(externalSequenceVector, less1,
//                                          "ss_deg_" + tag);
// auto sums = ChunkSequenceOps::ChunkHistogramByIndex<unsigned char>(ids, sample_size+1);

//   auto [offsets, total] = parlay::scan(sums);
//   auto id = std::upper_bound(offsets.begin(), offsets.end(), k) - offsets.begin() - 1;

// auto next = ChunkSequenceOps::pack_if<T, unsigned char>(
//     seq, "next_" + std::to_string(n), ids,
//     [id](unsigned char b){ return b == id; });


//the basic paradigm here in standard samplesort is that you should have the offset count from the counting sort
//this gives you the logical indices for the first and last portions of the array you need to sort in parallel
//obviously this relies on you being able to index into the array efficiently, which implies we're working in DRAM

//Ok so it seems like this is going to take a very long time -- we need both the chunkmap of heap positions 
//and the the original sequence

//sadly the parlay::internal::count_sort is difficult here -- in the parlay example, they rely on it to reorder the sequence
//such that all the 0-bucketed elements stably are arranged 
//One thing that we should note here is that physically moving data is really going to be terrible
//whatever we do, it needs to move data virtually


//we have now packed the values by bucket into 32 smaller external seqeunces
//okay, so now we can actually just sort everything in parallel
//the issue is that this method returns a parlay::sequence
// return sample_sort<T>(next, k - offsets[id], less1, original_size);





}

}  // namespace ChunkSequenceOps

#endif