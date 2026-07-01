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
// #define NUM_CHUNKS_PER_BATCH 30

//this is surprisingly not that easy to do by hand, maybe should use claude code for it
//essentially what we want to do is create an external sequence on disk that we can logically access 
//as if it were in DRAM naturally
//it's a weekend, so I'm doing this manually. Sorry if there are bugs, claude can clean it up



//seq is assumed to be of size n%chunksize == 0 ? n/chunksize : n/chunksize + 1
template<typename T>
External_Sequence ExternalIota(size_t n, External_Sequence& seq, const std::vector<std::string> &new_filenames) {
    //probably we have been passed an empty external sequence and it's our job to build it from scratch and 
    //then reorder it at the end

    // UnorderedChunkReader<T, 4 << 20> reader;
    // reader.PrepFiles(chunk_headers); 
    // reader.Start();

    // size_t expected_reads;
    // chunk_headers.size() % NUM_SSDS == 0 ? expected_reads = (chunk_headers.size() / NUM_SSDS) : expected_reads = chunk_headers.size() / NUM_SSDS + 1;


    constexpr size_t buffer_size_bytes = 4 << 20, buffer_size = buffer_size_bytes / sizeof(T);

    // each chunk holds buffer_size elements; a batch writes NUM_SSDS chunks (one per SSD).
    size_t num_chunks = (n + buffer_size - 1) / buffer_size;             // total chunks needed (ceil)
    size_t expected_write_count = (num_chunks + NUM_SSDS - 1) / NUM_SSDS; // total batches (ceil)
    
    UnorderedChunkWriter<T> writer;
    UnorderedChunkWriterConfig wconfig;
    wconfig.num_threads = WRITER_THREADS; 

    writer.Start(new_filenames, wconfig);

    std::vector<T*> buffer(NUM_SSDS);
    
    size_t write_count = 0;
    std::array<std::atomic<size_t>, NUM_SSDS> file_offsets{};


    // size_t read_count = 0;
    std::random_device rd;
    std::mt19937 gen(rd()); 
    while(write_count < expected_write_count){

    for(int i = 0; i < NUM_SSDS; i++){
        buffer[i] = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, buffer_size_bytes);
    }

    
    std::atomic<int> counter(0);
    std::atomic<int> counter_bad(0);
    

    std::uniform_int_distribution<int> distrib(0, NUM_SSDS-1);

    std::vector<unsigned int> random_holder(NUM_SSDS);
    std::atomic<bool> bad_flags[NUM_SSDS];

    std::vector<int> slot_for(NUM_SSDS, -1);

    for(int k = 0; k < NUM_SSDS; k++){
        random_holder[k] = distrib(gen);
        bad_flags[k] = false;
    }

  
    parlay::parallel_for(0, NUM_SSDS, [&](size_t i){

    size_t buffer_index = 0;
    // size_t next_index = 0;
    // auto [ptr, size, _, index, which_chunk, filename] = reader.Poll(); 
    // if (ptr == nullptr) {
    //     std::cout << "something went wrong or maybe not, null ptr";
    //     counter_bad++;
    //     //bad flags means that the reader did not go through on this phase
    //     bad_flags[i] = true;
    //     }

        //we are currently in the parallel for loop
        //the way we use buffer_index, etc. here are a little bit different from filter

        //the global index of the chunk this thread produces in this batch
        size_t global_chunk = write_count * NUM_SSDS + i;

        //in the final batch some SSD slots don't correspond to a real chunk
        if(global_chunk >= num_chunks){
            bad_flags[i] = true; //no chunk for this slot; it is skipped in the push loop below
            return;
        }

        size_t begin_val = global_chunk * buffer_size; //first iota value belonging to this chunk
        //number of in-range elements; the last real chunk may be partial
        size_t valid = (begin_val < n) ? std::min(buffer_size, n - begin_val) : 0;

        for(size_t k = 0; k < buffer_size; k++){
            buffer[i][k] = begin_val + k; //values past `valid` are padding; `used` records the real count
        }

        chunk_header chunked;
        chunked.index = global_chunk;
        chunked.filename = new_filenames[random_holder[i]];
        chunked.used = valid * sizeof(T);
        chunked.begin_address = 0; //we'll change this later

        //this is a reasonable approach because we need to find where this header is in the list later --
        //maybe it is better to modify the writer to return the base offset directly but currently this is necessary
        //because we don't want to calculate the base offset in the parallel for 
        //we can't immediately push though since we need to increment write_count, so this is the best we can do for now.
        int slot = counter.fetch_add(1);
        slot_for[i] = slot;
        
        //the difference between this method and filter is that all we're doing is writing -- we don't do any reads 
        //so there's no need for a slot. We're just going to reorder the chunks headers at the end anyway since we know what they contain.
        // seq.add_header(chunked);
        seq.ordered_underlying_sequence[write_count * NUM_SSDS + slot] = chunked;

        // (*chunk_header_arr)[write_count * NUM_SSDS + slot] = chunked;
            
        });
        // read_count++;

        //we are doing this to avoid the use of another atomic (even though we need one for slot)
        //the idea is that the actual writing can't really be done in the parallel loop because we can't increment the shared
        //buffer counter without a lock, so we can't find which SSD we need to go to.
        for(int r= 0; r < NUM_SSDS; r++){
            if(!(bad_flags[r])){
                size_t base_offset = file_offsets[random_holder[r]].fetch_add(buffer_size_bytes);
                seq.ordered_underlying_sequence[write_count * NUM_SSDS + slot_for[r]].begin_address = base_offset;
                writer.Push(std::shared_ptr<T>(buffer[r], free), buffer_size, random_holder[r], base_offset);
            }
            else{
                free(buffer[r]); //slot had no real chunk; release the buffer we allocated for it
            }
        }

        write_count++;
        }

    writer.Wait();

    //we actually know exactly how many chunks we'll have
    // size_t total_valid = (expected_write_count - 1) * NUM_SSDS + (n % NUM_SSDS == 0 ? NUM_SSDS : n % NUM_SSDS);
    // chunk_header_arr->resize(total_valid);

    std::sort(seq.begin(), seq.end(), [&](const chunk_header& i, const chunk_header& j){
        return i.index < j.index;
    });
    return seq;
}



#endif