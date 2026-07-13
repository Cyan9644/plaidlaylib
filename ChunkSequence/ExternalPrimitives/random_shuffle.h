#ifndef CHUNK_RANDOM_SHUFFLE_H
#define CHUNK_RANDOM_SHUFFLE_H

#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/random.h>

#include "absl/log/check.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_seq_reader.h"
#include "ChunkSequence/chunk_delayed.h"
#include "ChunkSequence/n_reader.h"
#include "ChunkSequence/ExternalPrimitives/materialize.h"
#include "ChunkSequence/ExternalPrimitives/chunk_count_sort.h"
#include "ChunkSequence/ExternalPrimitives/flatten.h"
#include "ChunkSequence/ExternalPrimitives/inplace_bucket_sort.h"
#include "utils/file_utils.h"
#include "utils/unordered_file_writer.h"
#include "configs.h"





#ifndef DRAM_SIZE
#define DRAM_SIZE ((size_t)500 * 1024 * 1024 * 1024)
#endif
//this is a bucketing method to randomly shuffle data to SSD and return the new sequence
//this particular method uses the high-level abstractions so that we can compare the performance against a low-level reader/writer paradigm
template <typename T>
chunk_seq random_shuffle_method(chunk_seq& seq, const std::string& prefix = "rs") {


// parlay::sequence<chunk_seq>

namespace d = ChunkSequenceOps::delayed;
parlay::random_generator gen;

static std::atomic<size_t> ss_counter{0};
const std::string tag = std::to_string(ss_counter++);
// SsPhaseTimer _pt(tag.c_str());
size_t n = 0;
  for(size_t r = 0; r < seq.chunks.size(); r++){
n+= seq.chunks[r].used;
  }
  size_t filer= n;
  n/=sizeof(T);
  
  size_t min_sample_size = std::max(1UL, 4 * parlay::num_workers() * filer/ DRAM_SIZE);
// size_t max_sample_size = std::max(1UL, std::min(n / sizeof(T), filer / O_DIRECT_MULTIPLE));
size_t max_sample_size = std::max(1UL, std::min(n, filer / O_DIRECT_MULTIPLE));
  size_t num_samples = std::max(std::min(filer / (1UL << 27), max_sample_size), min_sample_size);
  //no rounding up to 2^k-1 here: that is a samplesort heap_tree (pivot tree) artifact, and a
  //shuffle has no pivot tree -- it just costs extra buckets, i.e. extra files and smaller writes
  if(n < num_samples){
    auto par = ChunkSequenceOps::materialize<T>(seq);
    par = parlay::random_shuffle(par);

    // return ChunkSequenceOps::to_chunk_seq(par, "random_base_" + tag "_" + std::to_string(i));
    return ChunkSequenceOps::to_chunk_seq(par, prefix + "_base_" + tag);
  }
  auto num_buckets = num_samples + 1;

std::vector<chunk_seq> externalSequenceVector(num_buckets);

   

    // auto ids = ChunkSequenceOps::delayed::map(ChunkSequenceOps::delayed::delay<T>(seq),[&](T o, size_t r){
    //    parlay::random_generator gen;
    //   std::uniform_int_distribution<long> dis(0, num_buckets-1);
    //     auto g = gen[r];
    //     return std::pair<T, size_t>{o, (size_t)dis(g)};
    // });
   
 auto src = ChunkSequenceOps::delayed::delay<T>(seq);
 auto ids =ChunkSequenceOps::delayed::map(d::zip(src,ChunkSequenceOps::delayed::tabulate(src.length(), [](size_t i){ return i; })),[&, num_buckets](const std::pair<T, size_t>& e) {
  auto g = gen[e.second];
  std::uniform_int_distribution<size_t> dis(0, num_buckets - 1);
  return std::pair<T, size_t>{e.first, dis(g)};
});


    // ChunkSequenceOps::inplace_bucket_sort(seq, ids, externalSequenceVector,"random_bucket_" + tag);

  
    ChunkSequenceOps::chunk_count_sort(ids, num_buckets, externalSequenceVector, prefix + "_bucket_" + tag);
    // Less less = //we want this less function to allow us to sort by chunk filename

    // parlay::sort_inplace(seq2.chunks, less);

    // std::vector<size_t> bucket_indices(num_buckets);
    // //bucket-wise shuffle
    // //issue: we need to actually find where each bucket begins and ends. This same logic could be used for samplesort
    // std::string store_filename = seq2.chunks[0].filename;
    // uint32_t index = 0;
    // for(size_t j = 1; j < seq2.chunks.size(); j++){

    //   if(seq2.chunks[j].filename != store_filename){
    //     bucket_indices[index++] = j-1;
    //     store_filename = seq2.chunks[j].filename;
    //   }
    // }
    // bucket_indices[index+1] = seq2.chunks.size(); //fill in the last index since it won't be filled in the loop because it had the last filename

    //we know here that each bucket has its own filename and therefore cannot have bled into another chunk,
    //so we don't need to cut by indices, and the count sort's chunks can be shuffled in place
      // auto seed = 42;
    // parlay::random rng(seed);
    // ChunkSequenceOps::process_buckets_inplace<T>(externalSequenceVector,[&](size_t b, T* buf, size_t nelem){
    //     auto shuffled = parlay::random_shuffle(parlay::make_slice(buf, buf + nelem),rng.fork(b));
    //     std::memcpy(buf, shuffled.data(), nelem * sizeof(T));
    // });


    //we know here that each bucket has its own filename and therefore cannot have bled into another chunk,
    //so we don't need to cut by indices
    // auto new_seq = ChunkSequenceOps::delayed::cut_by_chunk(seq, bucket_indices[i], bucket_indices[i+1]); 
    // auto parlay_seq = ChunkSequenceOps::materialize(new_seq);
    // parlay_seq = parlay::random_shuffle(parlay_seq);
    // externalSequenceVector[i] = ChunkSequenceOps::to_chunk_seq(parlay_seq);
    // auto parlay_seq = ChunkSequenceOps::sequential_materialize<T>(externalSequenceVector[i]);
    // parlay_seq = parlay::random_shuffle(parlay_seq);
    // externalSequenceVector[i]= ChunkSequenceOps::to_chunk_seq(parlay_seq, prefix + "_out_" + tag +  "_" + std::to_string(i));
    //   });
    auto seed = 42;
    parlay::random rng(seed);
    ChunkSequenceOps::process_buckets_inplace<T>(externalSequenceVector,[&](size_t b, T* buf, size_t nelem){
        auto shuffled = parlay::random_shuffle(parlay::make_slice(buf, buf + nelem),rng.fork(b));
        std::memcpy(buf, shuffled.data(), nelem * sizeof(T));
    });

//     //filenames are now contiguously ordered

// //     //chunks are now in sorted order, so we just need to read the entire bucket (everything up to our current filename) into memory and shuffle the contents

    return ChunkSequenceOps::flatten(externalSequenceVector);
//   std::uniform_int_distribution<long> dis(0, n-1);
    //   auto locals = RemoveWorker<T>(seq, /*reader_threads=*/10,
    //     [&, num_buckets, epct](ChunkSequenceReader<T>& reader) {
    //         while (true) {
    //             auto [ptr, m, idx] = reader.Poll();
    //             if (ptr == nullptr) break;
    //             for (size_t j = 0; j < m; j++) {
    //                 if (pred(ptr[j])) {
    //                     best = std::min(best, idx * epct + j);
    //                     break;  // first match in this chunk is its smallest index
    //                 }
    //             }
    //             reader.allocator.Free(ptr);
    //         }
    //         return best;
    //     });
    // parlay::parallel_for(0, num_buckets, [&]{


    // });

}



//
// Permutation — originally written by Peter Li, ported to the chunk_seq interface by claude

namespace ChunkSequenceOps {

template<typename T = uint64_t>
class Permutation {
private:

    /**
     * Compute a sensible bucket count from the sequence about to be permuted.
     * Peter's GetBucketSize, reading the byte size off the chunk headers instead
     * of a FileInfo list.
     */
    static size_t GetBucketCount(const chunk_seq &seq) {
        // FIXME: considerations for bucket count
        //   (1) each bucket should be small enough to fit in main memory; ideally they should be small enough that we
        //       can process buckets concurrently to overlap IO and computation
        size_t file_size = 0;
        for (const chunk &c: seq.chunks) {
            file_size += c.used;
        }
        // FIXME: assuming no bucket is skewed to the point where it is 3 times the average size
        size_t min_sample_size = std::max(1UL, 4 * parlay::num_workers() * file_size / MAIN_MEMORY_SIZE);
        // bucket count cannot exceed the number of elements; it should also not result in very tiny files
        size_t max_sample_size = std::max(1UL, std::min(file_size / sizeof(T), file_size / O_DIRECT_MULTIPLE));
        // FIXME: need more stuff here; ~128MB per bucket is temporary
        return std::max(std::min(file_size / (1UL << 27), max_sample_size), min_sample_size);
    }

public:

    /**
     * Scatter / process / gather, the chunk_seq form of ScatterGather::Run.
     *
     * @param assigner   (value, global index) -> bucket in [0, num_buckets)
     * @param processor  (bucket, buffer, n) -> rewrites the bucket's n elements
     *                   in DRAM, in place (it may not change how many there are)
     */
    template<typename Assigner, typename Processor>
    chunk_seq Run(const chunk_seq &seq, const std::string &result_prefix,
                  size_t num_buckets, Assigner assigner, Processor processor) {
        CHECK(num_buckets > 0) << "Permutation::Run: need at least one bucket";
        namespace d = ChunkSequenceOps::delayed;

        // Phase 1 (AssignToBucket).  The assigner wants the element's global
        // index, so zip the input against the identity — a generated leaf, so it
        // costs no I/O — and count-sort the resulting {value, bucket} pairs.
        auto src =ChunkSequenceOps::delayed::delay<T>(seq);
        auto ids =ChunkSequenceOps::delayed::map(d::zip(src,ChunkSequenceOps::delayed::tabulate(src.length(), [](size_t i) { return i; })),
                          [&](const std::pair<T, size_t> &e) {
                              return std::pair<T, size_t>{e.first, assigner(e.first, e.second)};
                          });

        std::vector<chunk_seq> buckets(num_buckets);
        chunk_count_sort(ids, num_buckets, buckets, result_prefix);

        // Phase 2 (ProcessBucket): every bucket is DRAM-sized by construction, so
        // each is read back, processed, and written over its own chunks.
        process_buckets_inplace<T>(buckets, processor);

        // Gather: the buckets, in bucket order, are the output sequence.
        return flatten(buckets);
    }

    /**
     * Randomly permute an out-of-core sequence.
     *
     * Deterministic for a given seed (parlay's convention: the default seed
     * reproduces the same permutation on every run).
     */
    chunk_seq Permute(const chunk_seq &seq,
                      const std::string &result_prefix = "perm",
                      size_t seed = 0) {
        const size_t num_buckets = GetBucketCount(seq) + 1;
        parlay::random_generator gen(seed);
        parlay::random rng(seed);

        const auto simple_assigner = [&, num_buckets](const T &, size_t index) {
            auto r = gen[index];
            // The distribution is stateless but not thread-safe to share, and the
            // assigner runs on every worker; a fresh one per element is two words.
            std::uniform_int_distribution<size_t> dist(0, num_buckets - 1);
            return dist(r);
        };
        const auto simple_processor = [&](size_t bucket, T *buffer, size_t n) {
            auto shuffled = parlay::random_shuffle(parlay::make_slice(buffer, buffer + n),
                                                   rng.fork(bucket));
            std::memcpy(buffer, shuffled.data(), n * sizeof(T));
        };
        return Run(seq, result_prefix, num_buckets, simple_assigner, simple_processor);
    }
};

} // namespace ChunkSequenceOps




// //this is the version of random shuffle that does not rely on primitives, and is therefore the official version for the library
// //one interesting comparison that we can draw later is how this performs relative to 
// template <typename T, typename Less = std::less<>>>
// chunk_seq random_shuffle(chunk_seq& seq) {
// // static std::atomic<size_t> ss_counter{0};
// // const std::string tag = std::to_string(ss_counter++);

// size_t n = 0;
//   for(size_t r = 0; r < seq.chunks.size(); r++){
// n+= seq.chunks[r].used;
//   }
//   size_t filer= n;
//   n/=sizeof(T);

//    size_t min_sample_size = std::max(1UL, 4 * parlay::num_workers() * filer/ DRAM_SIZE);

// size_t max_sample_size = std::max(1UL, std::min(n, filer / O_DIRECT_MULTIPLE));
//   size_t num_samples = std::max(std::min(filer / (1UL << 27), max_sample_size), min_sample_size);
// unsigned int sample_size = std::max<size_t>(1, num_samples);
// size_t num_buckets = sample_size + 1;
// parlay::random_generator gen;
// std::uniform_int_distribution<long> dis(0, num_buckets); //random generation for random bucketing

// parlay::internal::heap_tree ss(seconds);

// std::vector<std::vector<T>> buffers[NUM_SSDS]; //buffer list
// auto remove_from_queue = ChunkSequenceOps::RemoveWorker<T>(seq,  /*reader_threads=*/10, [&](ChunkSequenceReader<T>& reader, size_t i){

//     while(true){
    
//         //poll once; this thread will continue and keep polling until it blocks, which means there's nothing left in the queue
//         auto [ptr, size,index ] = reader.Poll();
//         buffers[i] = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);

//         if(ptr == nullptr) break; //the null should apply to all threads and the poll itself is threadsafe
//         // for(size_t k=0; k <size; k++){
//         //     buffers[i][k] = pt
            
//         // }

//         //now we want to directly assign a random bucket to each of the elements 
//         memcpy(buffers[i], ptr, size);

//         auto ids = parlay::map(buffers[i], [&](size_t i){

//           return dis(gen[i]); //return a random bucket ID
//         });

//         //now we have a sequence of values and their bucket IDs, so we need to add these to per-file buckets that will be written
//         //once full. This is a little bit tricky
//         //it will need to build an external (chunk) sequence and return it to the caller because we need to act on its data later on

//         reader.allocator.Free(ptr); //need to free ptr to allow more reads to be polled
//     }
//     //post-loop, we assume that all data is on SSDs in their proper buckets, which means we just need to read in and randomize these buckets internally
//     //we're also assuming that we have a new external sequence seq2 which has the new data
//     //we have no idea what the ordering of the chunks is on SSD right now in terms of files, but we know that each bucket is represented by a single file
//     //the good news is that we can sort the chunk headers by filename to get the data to read in and shuffle
//     //we're also making the assumption that each individual bucket can fit in main memory, which is not so crazy


//     Less less = //we want this less function to allow us to sort by chunk filename

//     parlay::sort_inplace(seq2.chunks, less);

//     //chunks are now in sorted order, so we just need to read the entire bucket (everything up to our current filename) into memory and shuffle the contents

//     //probably we can get the filenames returned from the bucketing step, but just in case we can't
//     int i = 0;
//     size_t counter = 0;
//     while(i < num_buckets){
//       std::string current_filename = seq.chunks[counter].filename;
//       while(counter < seq.chunks.size() && seq.chunks[counter].filename == current_filename){
//         counter++;
//       }
    

//     }
//     return remove;
// });

// }










#endif