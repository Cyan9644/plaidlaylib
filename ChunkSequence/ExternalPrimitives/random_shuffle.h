#ifndef CHUNK_RANDOM_SHUFFLE_H
#define CHUNK_RANDOM_SHUFFLE_H

#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <parlay/parallel.h>

#include "absl/log/check.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_seq_reader.h"
#include "ChunkSequence/chunk_delayed.h"
#include "ChunkSequence/n_reader.h"
#include "utils/file_utils.h"
#include "utils/unordered_file_writer.h"
#include "configs.h"



//this is a bucketing method to randomly shuffle data to SSD and return the new sequence
//this particular method uses the high-level abstractions so that we can compare the performance against a low-level reader/writer paradigm
template <typename T, typename Less = std::less<>>
chunk_seq& random_shuffle_method(chunk_seq& seq, Less less1 = {}) {


// parlay::sequence<chunk_seq>


static std::atomic<size_t> ss_counter{0};
const std::string tag = std::to_string(ss_counter++);
SsPhaseTimer _pt(tag.c_str());
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
  num_samples = (size_t{1} << parlay::log2_up(num_samples + 1)) - 1;
  if(n < num_samples){
    auto par = ChunkSequenceOps::materialize(seq);
    parlay::random_shuffle(par);
    return ChunkSequenceOps::to_chunk_seq(par);
  }
  auto num_buckets = sample_size + 1;
std::vector<std::vector<chunk_seq>> externalSequenceVector(num_buckets);

    parlay::random_generator gen;
    std::uniform_int_distribution<long> dis(0, num_buckets);

    auto ids = ChunkSequenceOps::delayed::map(ChunkSequenceOps::delayed::delay<T>(seq),[&](size_t o){ 
        return dis(gen[o]);
    });

    ChunkSequenceOps::inplace_bucket_sort(seq, ids, exteranlSequenceVector,"random_bucket_" + tag);
    return ChunkSeqeunceOps::flatten(externalSequenceVector);
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







#endif