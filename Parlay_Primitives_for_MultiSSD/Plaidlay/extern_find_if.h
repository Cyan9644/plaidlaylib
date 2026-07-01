#include <algorithm>
#include <cmath>
#include <array>
#include <atomic>
#include <random>
#include <string>
#include <vector>

#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include "ExternalBoolean.h"


#define PRACTICAL_SSD 8 //this is the greatest number of reads that could practically be
//done simultaneously -- any more and we saturate the bandwidth. Ideally you'd be doing like NUM_SSDs reads.
#define NUM_SSDS 30 //however, we have no idea what SSDs each chunk lives on. In this case, maybe it's better to do 30 reads at a time
#define over_val 2 << 10
// **************************************************************
// Finds location i of first element  that satisfies a predicate
// Importantly it runs with only O(i) work, while using O(log^2 i) span.
// Based on doubling search.
// This is available in parlaylib.
// **************************************************************


//goal: we would like to overlap the I/O
//however, a standard doubling search is not going to cut it here, as the chunk number will quickly grow too large
//to be reasonable to read simultaneously.


//the maximum parallelism we can achieve is over PRACTICAL_SSDs SSDs, so there's not really a reason to do doubling search here.
//we'll just do parallel over the chunks, but since we don't know which chunks live on which SSDs,
//this presents its own set of problems. For now we'll just read 30 at a time and do the normal stuff like filter does.


//this external method is intended to return the logical index of the first occurence where p of the element evals to true

#include <algorithm>
#include <cmath>
#include <array>
#include <atomic>
#include <random>
#include <string>
#include <vector>

#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include "ExternalBoolean.h"


#define PRACTICAL_SSD 8 //this is the greatest number of reads that could practically be 
//done simultaneously -- any more and we saturate the bandwidth. Ideally you'd be doing like NUM_SSDs reads.
#define NUM_SSDS 30 //however, we have no idea what SSDs each chunk lives on. In this case, maybe it's better to do 30 reads at a time
#define over_val 2 << 10
// **************************************************************
// Finds location i of first element  that satisfies a predicate
// Importantly it runs with only O(i) work, while using O(log^2 i) span.
// Based on doubling search.
// This is available in parlaylib.
// **************************************************************


//goal: we would like to overlap the I/O
//however, a standard doubling search is not going to cut it here, as the chunk number will quickly grow too large
//to be reasonable to read simultaneously.


//the maximum parallelism we can achieve is over PRACTICAL_SSDs SSDs, so there's not really a reason to do doubling search here.
//we'll just do parallel over the chunks, but since we don't know which chunks live on which SSDs,
//this presents its own set of problems. For now we'll just read 30 at a time and do the normal stuff like filter does.



// //this external method is intended to return the logical index of the first occurence where p of the element evals to true
// size_t find_if(External_Sequence& seq, UnaryPredicate&& p){


//     size_t n = seq.size();

//     size_t search = 1000;

//     size_t start = 0;

//     size_t i;

//     auto& chunk_headers = seq.ordered_underlying_sequence;
//     UnorderedChunkReader<T, 4 << 20> reader;
//     reader.PrepFiles(chunk_headers); 
//     reader.Start();

//     size_t expected_reads;
//     chunk_headers.size() % NUM_SSDS == 0 ? expected_reads = (chunk_headers.size() / NUM_SSDS) : expected_reads = chunk_headers.size() / NUM_SSDS + 1;

//     constexpr size_t buffer_size_bytes = 4 << 20, buffer_size = buffer_size_bytes / sizeof(T);

//     std::vector<T*> buffer(NUM_SSDS);

//     // std::array<std::atomic<size_t>, NUM_SSDS> file_offsets{};

//     size_t read_count = 0;
//     std::random_device rd;
//     std::mt19937 gen(rd()); 
//     while(read_count < expected_reads){

//     // for(int i = 0; i < NUM_SSDS; i++){
//     //     buffer[i] = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, buffer_size_bytes);
//     // }

//     std::atomic<int> counter(0);
//     std::atomic<int> counter_bad(0);

//     std::vector<std::pair<bool, size_t>> flags(NUM_SSDS, {false, over_val});


//     // std::uniform_int_distribution<int> distrib(0, NUM_SSDS-1);

//     // std::vector<unsigned int> random_holder(NUM_SSDS);
//     std::atomic<bool> bad_flags[NUM_SSDS];

//     // std::vector<int> slot_for(NUM_SSDS, -1);

//     // for(int k = 0; k < NUM_SSDS; k++){
//     //     random_holder[k] = distrib(gen);
//     //     bad_flags[k] = false;
//     // }

//     parlay::parallel_for(0, NUM_SSDS, [&](size_t i){

//     size_t buffer_index = 0;
//     size_t next_index = 0;
//     auto [ptr, size, _, index, which_chunk, filename] = reader.Poll(); 
//     if (ptr == nullptr) {
//         std::cout << "something went wrong or maybe not, null ptr";
//         counter_bad++;

//         bad_flags[i] = true;
//         }
//     else{
//         size_t j = 0;
//         size_t store = buffer_index;
//         while (j < size) {
//                 if (p(ptr[j])) {//if we have a hit on the unary predicate
//                     //what we actually want to return here is the index of the hit, which is tough
//                     //we don't want to return a local index, but a global one. 
//                 flags[i] = {true, j};
//                 break;
//                 }
//                 j++;
                
//             }

        

//         //option 2, this would be ideal if we had larger chunk sizes I guess
//         //but probably too much parallelism to be viable
//         // long loc = parlay::reduce(parlay::delayed_tabulate(ptr[size]-ptr[0]), [&](size_t k){
//         //     return p[k] ? k : size}), parlay::minimum<size_t>());

//         // if(loc < size){
//         //     return loc;
//         // }


        
//     }
// }
// auto item = std::find_if(flags.begin(), flags.end(), [search_id](const auto& flag) {
//         return item.first == true; 
//     });
//     if(item < flags.end()){
//         //find the global index offset, which is the local offset + the batch offset + the past read offset
//         if(item > 0){
//         return (item-1) * buffer_size_bytes + flags[item].second + read_count * NUM_SSDS * buffer_size_bytes;
//         }
//         else{
//             return flags[item].second + read_count * NUM_SSDS * buffer_size_bytes;
//         }
//     }

        
//         });
    

//         read_count++;



       

//         // for(int r= 0; r < NUM_SSDS; r++){
//         //     auto found
//         //     // if(flags[r].first == true)
//         //     // if(!(bad_flags[r])){
//         //     //     size_t base_offset = file_offsets[random_holder[r]].fetch_add(buffer_size_bytes);
//         //     //     (*chunk_header_arr)[write_count * NUM_SSDS + slot_for[r]].begin_address = base_offset;
//         //     //     writer.Push(std::shared_ptr<T>(buffer[r], free), buffer_size, random_holder[r], base_offset);
//         //     // }
//         //     // else{
//         //     //     free(buffer[r]);
//         //     //     }
//         //     // }

        
//         // }


// }




//goal: we would like to overlap the I/O
//however, a standard doubling search is not going to cut it here, as the chunk number will quickly grow too large
//to be reasonable to read simultaneously.


//the maximum parallelism we can achieve is over PRACTICAL_SSDs SSDs, so there's not really a reason to do doubling search here.
//we'll just do parallel over the chunks, but since we don't know which chunks live on which SSDs,
//this presents its own set of problems. For now we'll just read 30 at a time and do the normal stuff like filter does.



// //this external method is intended to return the logical index of the first occurence where p of the element evals to true
// size_t find_if(External_Sequence& seq, UnaryPredicate&& p){

template<typename T, typename UnaryPredicate>
size_t find_if(External_Sequence& seq, UnaryPredicate&& p){

    auto& chunk_headers = seq.ordered_underlying_sequence;
    UnorderedChunkReader<T, 4 << 20> reader;
    reader.PrepFiles(chunk_headers);
    reader.Start();

    size_t expected_reads;
    chunk_headers.size() % NUM_SSDS == 0 ? expected_reads = (chunk_headers.size() / NUM_SSDS) : expected_reads = chunk_headers.size() / NUM_SSDS + 1;

    constexpr size_t buffer_size_bytes = 4 << 20, buffer_size = buffer_size_bytes / sizeof(T);

    // Total number of logical elements across all chunks. This is the "not
    // found" sentinel (matching parlay::find_if, which returns the sequence
    // length when no element satisfies the predicate). seq.size() is the *chunk*
    // count, which is far smaller than any real element index, so it must not be
    // used here -- doing so makes std::min below always collapse to that tiny
    // value and the search returns the chunk count for every real match.
    size_t n = 0;
    for (const auto &h : chunk_headers) n += h.used / sizeof(T);

    size_t read_count = 0;
    size_t a = n;

    while(read_count <  expected_reads){
        std::vector<std::pair<bool, size_t>> flags(NUM_SSDS,{false, n});
        parlay::parallel_for(0, NUM_SSDS, [&](size_t i){
            auto [ptr,size, _, index, which_chunk, filename] = reader.Poll();
            if(ptr == nullptr){
                return;
            }
            for(size_t j=0; j < size; j++) {
                if (p(ptr[j])){ 
                    flags[i] = {true, index * buffer_size + j};
                    break;
    }
            }
            reader.allocator.Free(ptr); //needed? i guess
        });

   
        for(const auto& f:flags){
            if(f.first) a = std::min(a, f.second);
        }

        read_count++;
    }

    return a;  
}
