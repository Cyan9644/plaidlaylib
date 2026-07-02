#ifndef EXTERNAL_FILTER_H
#define EXTERNAL_FILTER_H
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




//the job of this function is to compute the last element of the prefix sum for each block and store this in DRAM
template<typename T>
std::vector<T> LocalExternalScan(External_Sequence &seq, const std::vector<std::string> &new_filenames) {
    auto& chunk_headers = seq.ordered_underlying_sequence;
    UnorderedChunkReader<T, 4 << 20> reader;
    reader.PrepFiles(chunk_headers);
    reader.Start();

    size_t expected_reads;
    chunk_headers.size() % NUM_SSDS == 0 ? expected_reads = (chunk_headers.size() / NUM_SSDS) : expected_reads = chunk_headers.size() / NUM_SSDS + 1;

    // External_Sequence sequence = External_Sequence(seq.size());
    // std::vector<chunk_header>* chunk_header_arr = &sequence.ordered_underlying_sequence;
    // constexpr size_t buffer_size_bytes = 4 << 20, buffer_size = buffer_size_bytes / sizeof(T);
    // UnorderedChunkWriter<T> writer;
    // UnorderedChunkWriterConfig wconfig;
    // wconfig.num_threads = 2; 

    // writer.Start(new_filenames, wconfig);

    // std::vector<T*> buffer(NUM_SSDS);
    
    // size_t write_count = 0;
    // std::array<std::atomic<size_t>, NUM_SSDS> file_offsets{};

    std::vector<T> local_offset_holder(chunk_headers.size()+1); //we are allocating + 1 size here because we eventually want to return the total
    //sum of all the elements, which means we need this last one as well.
    size_t read_count = 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    while(read_count < expected_reads){

    // for(int i = 0; i < NUM_SSDS; i++){
    //     buffer[i] = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, buffer_size_bytes);
    // }

    
    // std::atomic<int> counter(0);
    // std::atomic<int> counter_bad(0);
    

    // std::uniform_int_distribution<int> distrib(0, NUM_SSDS-1);

    std::vector<unsigned int> random_holder(NUM_SSDS);
    std::atomic<bool> bad_flags[NUM_SSDS];

    std::vector<int> slot_for(NUM_SSDS, -1);

    for(int k = 0; k < NUM_SSDS; k++){
        random_holder[k] = distrib(gen);
        bad_flags[k] = false;
    }

  
    parlay::parallel_for(0, NUM_SSDS, [&](size_t i){

    // size_t buffer_index = 0;
    // size_t next_index = 0;
    auto [ptr, size, _, index, which_chunk, filename] = reader.Poll(); //one thing that needs to be changed: these scan offsets need to be in the absolute correct order
    //because we need to compute another scan on them in the future

    //maybe we can think up a new algorithm to get around this, but I doubt it because the DRAM space is limited
    
    if (ptr == nullptr) {
        std::cout << "something went wrong or maybe not, null ptr\n";
        counter_bad++;

        bad_flags[i] = true;
        }
    else{
        
   
        size_t j = 0;
        size_t store = buffer_index;
        size_t local_sum = 0;
        while (j < size) {
                local_sum += ptr[j];
                j++;

            }
             if(index == chunk_headers.size()-1){

                //if this is the final chunk, we want to store the total in local_offset_holder[last] so that we can easily get the full size later
                local_offset_holder[index+1] = local_sum + ptr[j-1];
                //this value should now be the sum of all elements in the last block, which we can later add to the cumulative block offsets
                //to get the size
            };
            local_offset_holder[index] = local_sum; // index is given back by the reader we suppose is ordered
    }
        });
        read_count++;
       

        // for(int r= 0; r < NUM_SSDS; r++){
         
        //     if(!(bad_flags[r])){
        //         size_t base_offset = file_offsets[random_holder[r]].fetch_add(buffer_size_bytes);
        //         (*chunk_header_arr)[write_count * NUM_SSDS + slot_for[r]].begin_address = base_offset;
        //         writer.Push(std::shared_ptr<T>(buffer[r], free), buffer_size, random_holder[r], base_offset);
        //     }
        //     else{
        //         free(buffer[r]);
        //         }
        //     }

     
        }
        local_offset_holder[local_offset_holder.size()-1] = 

    // writer.Wait();

    
    // size_t total_valid = (expected_reads - 1) * NUM_SSDS + (chunk_headers.size() % NUM_SSDS == 0 ? NUM_SSDS : chunk_headers.size() % NUM_SSDS);
    // chunk_header_arr->resize(total_valid);

    // std::sort(sequence.begin(), sequence.end(), [&](const chunk_header& i, const chunk_header& j){
    //     return i.index < j.index;
    // });
    return local_offset_holder;
}


template<typename T>
std::pair<External_Sequence, size_t> ExternalScanFile(External_Sequence &seq,const std::vector<std::string> &new_filenames, std::vector<size_t>& offsets) {

    auto& chunk_headers = seq.ordered_underlying_sequence;
    UnorderedChunkReader<T, 4 << 20> reader;
    reader.PrepFiles(chunk_headers); 
    reader.Start();

    size_t expected_reads;

    chunk_headers.size() % NUM_SSDS == 0 ? expected_reads = (chunk_headers.size() / NUM_SSDS) : expected_reads = chunk_headers.size() / NUM_SSDS + 1;

    External_Sequence sequence = External_Sequence(seq.size());

    parlay::sequence<chunk_header>* chunk_header_arr = &sequence.ordered_underlying_sequence;
    constexpr size_t buffer_size_bytes = 4 << 20, buffer_size = buffer_size_bytes / sizeof(T);
    UnorderedChunkWriter<T> writer;
    UnorderedChunkWriterConfig wconfig;
    wconfig.num_threads = 2; 

    writer.Start(new_filenames, wconfig);

    std::vector<T*> buffer(NUM_SSDS);

    size_t write_count = 0;
    std::array<std::atomic<size_t>, NUM_SSDS> file_offsets{};

    size_t read_count = 0;

   // size_t running_count = 0; //it will be inefficient to keep a running count as a shared variable because the work is parallelized here.
    //probably it's best to just find the very last chunk of the scan and add its corresponding chunk offset to the last element found there + the value of that element in the original sequence


    std::random_device rd;
    std::mt19937 gen(rd()); 
    while(read_count < expected_reads){

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
    size_t next_index = 0;

    auto [ptr, size, _, index, which_chunk, filename] = reader.Poll(); //we are again assuming for scan that this speciic reader will return items in order of their index,

    //which also means we don't need to sort them in the end
    //however, maybe for this step it is okay to use an unordered reader and sort if that results in some speedup.

    if (ptr == nullptr) {
        std::cout << "something went wrong or maybe not, null ptr";
        counter_bad++;

        bad_flags[i] = true;
        }
    else{

        size_t j = 0;
        size_t store = buffer_index;

        while (j < size) {
               
                buffer[i][buffer_index] = ptr[buffer_index++]; 
                buffer[i][buffer_index] += offsets[j];
                // buffer_index++;
                

                j++;
            }
        chunk_header chunked; 
        chunked.index = index;
        chunked.filename =new_filenames[random_holder[i]]; 
        chunked.used = (buffer_index-store) * sizeof(T);

        chunked.begin_address = 0;
        int slot = counter.fetch_add(1);
        slot_for[i] = slot;
        (*chunk_header_arr)[write_count * NUM_SSDS + slot] = chunked;

    }
        });
        read_count++;

        for(int r= 0; r < NUM_SSDS; r++){

            if(!(bad_flags[r])){
                size_t base_offset = file_offsets[random_holder[r]].fetch_add(buffer_size_bytes);
                (*chunk_header_arr)[write_count * NUM_SSDS + slot_for[r]].begin_address = base_offset;
                writer.Push(std::shared_ptr<T>(buffer[r], free), buffer_size, random_holder[r], base_offset);
            }
            else{
                free(buffer[r]);
                }
            }

        write_count++;
        }

    //the way we 
    writer.Wait();

    size_t total_valid = (expected_reads - 1) * NUM_SSDS + (chunk_headers.size() % NUM_SSDS == 0 ? NUM_SSDS : chunk_headers.size() % NUM_SSDS);
    chunk_header_arr->resize(total_valid);

    std::sort(sequence.begin(), sequence.end(), [&](const chunk_header& i, const chunk_header& j){
        return i.index < j.index;
    });
    return sequence;
}


template<typename T>
std::pair<ExternalSequence, size_t> ExternalScan(const ExternalSequence &seq, const std::string &prefix) {

    // std::vector<size_t> store(files.size());
    std::vector<size_t> store = LocalExternalScan<T>(seq, prefix);


    
    //after this finishes, store should contain the total lengths.
    //this means we need to calculate a scan over them to get the file offsets
    auto file_offsets = std::vector<size_t>(store.size());

    //we now have the absolute offsets for each chunk, so we need to go back through and add those values to the chunk scan values
    //while recomputing the scan


    //problem: we also need the total sum, so maybe this exclusive scan should in fact have a size one greater than the required 
    //for storing the scan values
    //maybe it's okay to not output the sum along with it
    std::exclusive_scan(store.begin(), store.end(), file_offsets.begin(), size_t(0));


    std::pair<ExternalSequence, size_t> paired = ExternalScanFile<T>(files, GetFileName(prefix), file_offsets);
  
}

#endif
