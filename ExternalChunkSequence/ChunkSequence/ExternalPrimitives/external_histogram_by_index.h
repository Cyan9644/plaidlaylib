#ifndef EXTERNAL_HISTOGRAM_H
#define EXTERNAL_HISTOGRAM_H
#include <pthread.h>
#include "ChunkSequence/external_engine.h"
#include <cassert>
#include <math.h>
#include <iostream>
#include <fcntl.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <liburing.h>
#include <cstring>
#include <random>
#include <array>
#include <algorithm>
#include <parlay/parallel.h>
#include <parlay/primitives.h>


#define NUM_SSDS 30
#ifndef PRACTICAL_SSDS 
#define PRACTICAL_SSDS 8
#endif

// template<typename T>
// parlay::sequence<size_t> ExternalHistogramByIndexExternalSeq(const chunk_seq& seq, const std::vector<std::string> &new_filenames, size_t num_unique){


//     //one assumption we make is that we have a dense case, otherwise this counting sort idea doesn't really work
//     //because it wastes so much memory
//     constexpr size_t buffer_size_bytes = 4 << 20, buffer_size = buffer_size_bytes / sizeof(T);
//     size_t num_chunks = (num_unique + buffer_size - 1) / buffer_size; //is this right? I think this is left over from the iota logic where num_unique = n
//     size_t read_count = 0;
//     std::vector<size_t>* store(num_unique) = calloc(num_unique * sizeof(size_t*));
//       auto& chunk_headers = seq.ordered_underlying_sequence;
//       size_t expected_reads;
//     // size_t expected_reads = (chunk_headers.size() + NUM_SSDS - 1) / NUM_SSDS;
//     chunk_headers.size() % PRACTICAL_SSDS == 0 ? expected_reads = (chunk_headers.size() / NUM_SSDS) : expected_reads = chunk_headers.size() / PRACTICAL_SSDS + 1;
//     UnorderedChunkReader<T, 4 << 20> reader;
//     reader.PrepFiles(chunk_headers); //prepfiles needs to be changed to accomodate chunk headers
//     reader.Start();
//     std::vector<T*> store_local(PRACTICAL_SSDS);
 
//     while(read_count < expected_read_count){
//         //instead of calculating the expected number of read batches, maybe the best way to do this is to check the reader.poll?
//         //but this seems much simpler
//         parlay::parallel_for(0, PRACTICAL_SSDS, [&](size_t i){
//             auto [ptr, size, _, index, which_chunk, filename] = reader.Poll();
//             //no reason to use aligned alloc since we don't need to write this
//             store_local[i] = (size_t*)calloc(num_unique);
//             size_t buffer_index = 0;
//             for(size_t k = 0; k < size; k++){
//                     store_local[i][ptr[k]]++;
//             }
//             });
//     //we now need to add the respective buffers back to the in-memory sequence 
//     //perhaps this can be done with a parlay tabulate or map or something clever but a sequential add isn't too bad

//         for(int i =0; i < PRACTICAL_SSDS; i++){
//             // for(int k = 0; k < num_unique; k++){

//             //     store[k]+=store_local[i][k];
            
//             // }
//             store = parlay::map(store, [&](size_t j){
//                 return store[j] + store_local[i][j]; //maybe wasteful if too many 0 elements or the sequence is not large enough 
//                 //to get much benefit from parallelism
//             });
//             // auto tab = parlay::tabulate(store, [&](size_t j)){
//             //     return store[j] + store_local[i][j];
//             // }
//             free(store_local[i]);
//         }
//         read_count++;
//     }

//     return store;

//     // std::sort(seq.begin(), seq.end(), [&](const chunk_header& i, const chunk_header& j){
//     //     return i.index < j.index;
//     // });
//     // return seq;
// }

namespace ChunkSequenceOps {

template<typename T>
parlay::sequence<size_t> histogram_by_index(const chunk_seq& seq, size_t num_unique){
    //as I understand it::
    //removeworker is a function template that starts the reader's io_uring producer threads then generates
    //one worker tas per hardware worker polling a single reader
    //an arbitrary worker takes the next chunk to enforce load balancing
    //poll blocks if the queue is empty but filling, will return nulptr once all readers have finished and the queue is empty
    //this stops the workers
    auto remove_from_queue = ChunkSequenceOps::RemoveWorker<T>(seq,  /*reader_threads=*/10, [&](ChunkSequenceReader<T>& reader){
        //create parlay seq init to 0 with num_unique values
        parlay::sequence<size_t> remove(num_unique, 0);
        while(true){
            //poll once; this thread will continue and keep polling until it blocks, which means there's nothing left in the queue
            auto [ptr, size,index ] = reader.Poll();
            
            if(ptr == nullptr) break; //the null should apply to all threads and the poll itself is threadsafe
            for(size_t k=0; k <size; k++){
                remove[ptr[k]]++; //logical increment
                
            }
            reader.allocator.Free(ptr); //need to free ptr to allow more reads to be polled
        }
        return remove;
    });
    parlay::sequence<size_t> total(num_unique, 0); //final counts sequence
    for(auto& remove : remove_from_queue){
        //inner loops is parallel so there are no race conditions, 
        // parlay::parallel_for(0, num_unique, [&](size_t j){
        //     total[j] += remove[j];
        // });

        //add the local buffer counts to the total
        total = parlay::tabulate(num_unique, [&](size_t j){
            return total[j] + remove[j];
        });
    }
    return total;

}

} // namespace ChunkSequenceOps

#endif