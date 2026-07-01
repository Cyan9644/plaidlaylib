#ifndef FLATTEN_H
#define FLATTEN_H

#include "parlay/primitives.h"
#include "plaidlay.h"
#include <cassert>
#include <math.h>
#include <iostream>
#include <fcntl.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <liburing.h>
#include <cstring>
#include <parlay/parallel.h>
#include <parlay/primitives.h>

size_t predecessor_search(const std::vector<size_t>& input, size_t find);
template <typename T>
long binary_search(const std::vector<size_t>& input, T find, long low, long high);

namespace plaidlayNaive{
    template <typename T>
    naiveSeq<T> block_flatten_in_dram(const naiveSeq<naiveSeq<T>>& input){
        size_t total_elements = 0;
        size_t i = 0;
        std::vector<size_t> over_seq(input.size());
        over_seq[0] = 0;
        over_seq[1] = input[0].size();
        for(i=2;i<input.size(); i++){
            over_seq[i] = input[i-1].size() + over_seq[i-1];
        }

        i=0;
        for (; i < input.size();i++) {
            total_elements += input[i].size();
        }

        bool has_remainder = total_elements % BLOCKSIZE;

        // naiveSeq<T> block_offsets(has_remainder ?  total_elements / BLOCKSIZE + 1 : total_elements / BLOCKSIZE);
        naiveSeq<T> finalseq(total_elements);

        //compute scan over input sizes
        // over_seq = plaidlayNaive::scan(over_seq);

        long num_blocks = total_elements/BLOCKSIZE;

        parlay::parallel_for(0, num_blocks, [&](long i) {

            auto final_block_start = i * BLOCKSIZE;
            auto final_block_end = std::min((size_t)(((i+1) * BLOCKSIZE)), total_elements);

            //find which sequence our block actually starts in in the input array 
            auto block_actual_start = predecessor_search(over_seq, final_block_start);
            size_t k = final_block_start;
            while(k < final_block_end){

                long offset = k-over_seq[block_actual_start];

                if((size_t) offset == input[block_actual_start].size()){
                    ++block_actual_start;
                }
                else{
                    finalseq[k] = input[block_actual_start][offset];
                    k++;
                }
        }
        });

        if(has_remainder){

            //pick up the rest sequentially
            long last_index = num_blocks * BLOCKSIZE;
            //all blocks have completed at this point, so the sequential portion is responsible for elements of num_blocks * BLOCKSIZE and up
            auto final_start = predecessor_search(over_seq, last_index);
            size_t a = last_index;
            while(a < total_elements){

            long offset = a-over_seq[final_start];
            //copy into new sequence
            // bool k_plus = false;
            // if(offset != input[final_start].size()){
            //     k_plus = true;
            // }
            // offset == input[final_start].size() ? block=input[++final_start] : finalseq[k] = input[final_start][offset];
            // if(k_plus) k++;
            if((size_t)offset == input[final_start].size()){
                ++final_start;
            }
            else{
                finalseq[a] = input[final_start][offset];
                a++;
            }
            }
        }
        return finalseq;
    }
}


// //don't use I guess
// template <typename T>
// long binary_search(const std::vector<size_t>& input, T find, long low, long high){

// if(low == high || low > high){
//     return high;
// }

// T med = input[low + ((high-low)/2)];

// if(med == find){
//     return (low + ((high-low)/2));
// }

// if(med < find){

//     return binary_search(input, find, low + ((high-low)/2)+ 1, high);

// }
// else{
//     return binary_search(input, find, low, low + ((high-low)/2)-1);
// }

// }


//finds the index of the greatest element smaller or equal to the desired value
inline size_t predecessor_search(const std::vector<size_t>& input, size_t find){
    size_t high = input.size()-1; size_t low = 0; size_t answer;

    while(low <= high){
        size_t mid = low + ((high-low)/2);
        if(input[mid] <= find){
            answer = mid;
            low = mid+1;
        }
        else high = mid-1;
    }
    return answer;
}




template<typename T>
FileInfo FlattenFileSequential(const FileInfo &in_file, const std::string &out_file){

    struct QueueData {
        T *ptr;
        size_t size;
        size_t index;

        QueueData(T *ptr, size_t size, size_t index) : ptr(ptr), size(size), index(index) {}
    };
    const auto cmp = [](QueueData a, QueueData b) {
        return a.index > b.index;
    };
    UnorderedFileReader<T> reader;
    reader.PrepFiles({in_file});
    reader.Start();
    UnorderedFileWriter<T> writer(out_file);
    std::priority_queue<QueueData, std::vector<QueueData>, decltype(cmp)> queue(cmp);
    constexpr size_t buffer_size_bytes = 4 << 20, buffer_size = buffer_size_bytes / sizeof(T);



}




#endif