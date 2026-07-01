#ifndef EXTERNAL_IOTA_H
#define EXTERNAL_IOTA_H
#include "config_threads.h"
#include <pthread.h>
#include "plaidlay.h"
#include <cassert>
#include <math.h>
#include <iostream>
#include <fcntl.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include "chunk_header.h"
#include <liburing.h>
#include <cstring>
#include <random>
#include <array>
#include <algorithm>
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include "utils/unordered_file_reader_modified.h"
#include "utils/unordered_file_writer_modified.h"


#define NUM_SSDS 30
#ifndef PRACTICAL_SSDS 
#define PRACTICAL_SSDS 8
#endif




template<typename T>
External_Sequence ExternalHistogramByIndex(External_Sequence& seq, const std::vector<std::string> &new_filenames, size_t num_unique) {


    //one assumption we make is that we have a dense case, otherwise this counting sort idea doesn't really work
    //because it wastes so much memory

    constexpr size_t buffer_size_bytes = 4 << 20, buffer_size = buffer_size_bytes / sizeof(T);

    size_t num_chunks = (n + buffer_size - 1) / buffer_size;
    size_t expected_write_count = (num_chunks + NUM_SSDS - 1) / NUM_SSDS;
    
    UnorderedChunkWriter<T> writer;
    UnorderedChunkWriterConfig wconfig;
    wconfig.num_threads = WRITER_THREADS; 

    writer.Start(new_filenames, wconfig);

    std::vector<T*> buffer(NUM_SSDS);
    
    size_t write_count = 0;
    std::array<std::atomic<size_t>, NUM_SSDS> file_offsets{};


    std::random_device rd;
    std::mt19937 gen(rd()); 

    std::array<size_t> store(num_unique);


    while(write_count < expected_write_count){

    //we may eventually want to change this: there is no intrinsic ordering to the histogram by index, we just need to make 
    // for(int i = 0; i < PRACTICAL_SSDS; i++){
    //     buffer[i] = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, buffer_size_bytes);
    // }
    // std::atomic<int> counter(0);
    // std::atomic<int> counter_bad(0);
    

    // std::uniform_int_distribution<int> distrib(0, NUM_SSDS-1);

    // std::vector<unsigned int> random_holder(NUM_SSDS);
    // std::atomic<bool> bad_flags[NUM_SSDS];

    // std::vector<int> slot_for(NUM_SSDS, -1);

    // for(int k = 0; k < NUM_SSDS; k++){
    //     random_holder[k] = distrib(gen);
    //     bad_flags[k] = false;
    // }

    auto& chunk_headers = seq.ordered_underlying_sequence;
    UnorderedChunkReader<T, 4 << 20> reader;
    reader.PrepFiles(chunk_headers); //prepfiles needs to be changed to accomodate chunk headers
    reader.Start();

    //instead of calculating the expected number of read batches, maybe the best way to do this is to check the reader.poll?
    //but this seems much simpler
    size_t expected_reads;
    // size_t expected_reads = (chunk_headers.size() + NUM_SSDS - 1) / NUM_SSDS;
    chunk_headers.size() % NUM_SSDS == 0 ? expected_reads = (chunk_headers.size() / NUM_SSDS) : expected_reads = chunk_headers.size() / NUM_SSDS + 1;

    parlay::parallel_for(0, PRACTICAL_SSDS, [&](size_t i){
        
    //no reason to use aligned alloc since we don't need to write this
    size_t* buffer = (size_t*)malloc(num_unique);
    

    size_t buffer_index = 0;


    // size_t global_chunk = write_count * NUM_SSDS + i;


        //this was for an incomplete case, but we don't care about that here
        // if(global_chunk >= num_chunks){
        //     bad_flags[i] = true;
        //     return;
        // }

        // size_t begin_val = global_chunk * buffer_size;
    // size_t valid = (begin_val < n) ? std::min(buffer_size, n - begin_val) : 0;

     for(size_t k = 0; k < buffer_size; k++){
            size_t val = 
            buffer[k] = begin_val + k;
    }

        chunk_header chunked;
        chunked.index = global_chunk;
        chunked.filename = new_filenames[random_holder[i]];
        chunked.used = valid * sizeof(T);
        chunked.begin_address = 0;

        int slot = counter.fetch_add(1);
        slot_for[i] = slot;
        
        seq.ordered_underlying_sequence[write_count * NUM_SSDS + slot] = chunked;

            
        });

        for(int r= 0; r < NUM_SSDS; r++){
            if(!(bad_flags[r])){
                size_t base_offset = file_offsets[random_holder[r]].fetch_add(buffer_size_bytes);
                seq.ordered_underlying_sequence[write_count * NUM_SSDS + slot_for[r]].begin_address = base_offset;
                writer.Push(std::shared_ptr<T>(buffer[r], free), buffer_size, random_holder[r], base_offset);
            }
            else{
                free(buffer[r]);
            }
        }

        write_count++;
        }

    writer.Wait();


    std::sort(seq.begin(), seq.end(), [&](const chunk_header& i, const chunk_header& j){
        return i.index < j.index;
    });
    return seq;
}



#endif