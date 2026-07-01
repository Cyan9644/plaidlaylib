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
// #define NUM_CHUNKS_PER_BATCH 30


template<typename T>
External_Sequence ExternalFilter(External_Sequence &seq, const std::function<bool(const T)>& predicate, const std::vector<std::string> &new_filenames) {
    // std::vector<chunk_header> chunk_headers = seq.ordered_underlying_sequence;
    // std::atomic<bool> lock = true;
    auto& chunk_headers = seq.ordered_underlying_sequence;
    UnorderedChunkReader<T, 4 << 20> reader;
    reader.PrepFiles(chunk_headers); //prepfiles needs to be changed to accomodate chunk headers
    reader.Start();

    //instead of calculating the expected number of read batches, maybe the best way to do this is to check the reader.poll?
    //but this seems much simpler
    size_t expected_reads;
    // size_t expected_reads = (chunk_headers.size() + NUM_SSDS - 1) / NUM_SSDS;
    chunk_headers.size() % NUM_SSDS == 0 ? expected_reads = (chunk_headers.size() / NUM_SSDS) : expected_reads = chunk_headers.size() / NUM_SSDS + 1;


    //THIS EVENTUALLY NEEDS TO BE CHANGED
    //we are currently cheating a bit by not merging non-full blocks; this allows us to allocate the exact amount of space needed

    External_Sequence sequence = External_Sequence(seq.size());
    // std::vector<chunk_header>* chunk_header_arr = &((std::vector<chunk_header>*)sequence.ordered_underlying_sequence);
    parlay::sequence<chunk_header>* chunk_header_arr = &sequence.ordered_underlying_sequence;
    constexpr size_t buffer_size_bytes = 4 << 20, buffer_size = buffer_size_bytes / sizeof(T);
    UnorderedChunkWriter<T> writer;
    UnorderedChunkWriterConfig wconfig;
    wconfig.num_threads = 2; 

    //new_filenames is a list of NUM_SSDS length that contains filenames, one for each SSD.
    //we want to sample from this list when we push
    // new_filenames is already a vector<string>; pass it straight through
    // (wrapping it in braces makes the std::string/vector overloads ambiguous).
    writer.Start(new_filenames, wconfig);

    std::vector<T*> buffer(NUM_SSDS);
    
    // unsigned long block_index = 0;
    size_t write_count = 0;
    std::array<std::atomic<size_t>, NUM_SSDS> file_offsets{};


    size_t read_count = 0;
    //problem: there's no .top method
    // while(reader.top() != nullptr){
    std::random_device rd;
    std::mt19937 gen(rd()); //?
    while(read_count < expected_reads){

    for(int i = 0; i < NUM_SSDS; i++){
        buffer[i] = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, buffer_size_bytes);
    }

    // std::vector<UnorderedFileReader<T>> readers;
    // parlay::parallel_for(0, readers.size(), [&](long i){
    //     readers[i].prepFiles()
    // });
    // parlay::parallel_for(block_index *NUM_SSDS, min((1 +block_index) * NUM_SSDS, chunk_headers.size()), [&]{
    
    // std::priority_queue<QueueData, std::vector<QueueData>, decltype(cmp)> queue(cmp);
    std::atomic<int> counter(0);
    std::atomic<int> counter_bad(0);
    

    std::uniform_int_distribution<int> distrib(0, NUM_SSDS-1);

    //no parlay::tabulate here because the # of chunks is so small
    std::vector<unsigned int> random_holder(NUM_SSDS);
    std::atomic<bool> bad_flags[NUM_SSDS];

    std::vector<int> slot_for(NUM_SSDS, -1);

    for(int k = 0; k < NUM_SSDS; k++){
        random_holder[k] = distrib(gen);
        bad_flags[k] = false;
    }
    // std::vector<size_t> slist();

  
    // int random_number = distrib(gen);
    // size_t base_offset = file_offsets[random_number].fetch_add(buffer_size_bytes * NUM_SSDS);
    parlay::parallel_for(0, NUM_SSDS, [&](size_t i){

    size_t buffer_index = 0;
    size_t next_index = 0;
    //poll is threadsafe
    auto [ptr, size, _, index, which_chunk, filename] = reader.Poll(); //this poll can return an arbitrary chunk, so we're going to need to 
        //check the metadata to figure out what chunk it is
        //but the metadata isn't embedded in the data, so we can't read it.
        //the existing call uses things that aren't actually returned by reader.poll() like which_chunk 
        //because we expect that this will be implemented later.
        //if we knew what file we were reading from, we could check where we are to determine which chunk we must be in
        //but as far as I know the unordered 
    if (ptr == nullptr) {
        std::cout << "something went wrong or maybe not, null ptr";
        counter_bad++;

        bad_flags[i] = true;
        }
    else{
    
        size_t j = 0;
        size_t store = buffer_index;
        //this is safe to do because no thread can write outside of its bounds -- it is limited to the elements returned by reader.poll
        //and its buffer index is outside the range of any other thread.
        //every individual size is less than or equal to buffer_size
        while (j < size) {
                if (predicate(ptr[j])) {
                    buffer[i][buffer_index] = ptr[j];
                    buffer_index++;
                }
                // if (buffer_size == buffer_index) {
                //     // writer.Push(std::shared_ptr<T>(buffer, free), buffer_size);
                //     // write_count++;
                //     buffer = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, buffer_size_bytes);
                //     buffer_index = 0;
                // }
                j++;
            }
        chunk_header chunked; //this is intended to be the chunk header that will go into the final vector of chunk headers
        chunked.index = index;
        chunked.filename =new_filenames[random_holder[i]]; //we need to figure out what filename to write back to -- this should be given by the poll 
        chunked.used = (buffer_index-store) * sizeof(T);
        // chunked.begin_address = chunk_headers[write_count * NUM_SSDS + i].get_begin_address(); //begin_address should be the offset into the file, which we can take from


        // size_t base_offset = file_offsets[random_holder[i]].fetch_add(buffer_size_bytes * NUM_SSDS);
        // chunked.begin_address = base_offset + i * buffer_size_bytes; //(write_count*NUM_SSDS + i) * buffer_size_bytes;

        // size_t base_offset = file_offsets[random_holder[i]].fetch_add(buffer_size_bytes);
        // chunked.begin_address = base_offset;

        //the previous chunk

        // chunk_header_arr[write_count * NUM_SSDS + j]  = chunked;//get next position in the chunk header array
        //  bool check_yes = 1;

        // bool check_no = 0;

        // while(!(lock.compare_exchange_strong(check_yes, check_no))){

        // check_yes = true;
        // check_no = false;

        // }

        // size_t base_offset = file_offsets[random_holder[i]].fetch_add(buffer_size_bytes);
        // chunked.begin_address = base_offset;

        chunked.begin_address = 0;
        int slot = counter.fetch_add(1);
        slot_for[i] = slot;
        (*chunk_header_arr)[write_count * NUM_SSDS + slot] = chunked;

        // counter++;
            // reader.allocator.Free(top.ptr);
        // lock = 1;
            
        //now we assume that everything here has been processed into the final buffer
    }
        });
        read_count++;

        //at this point, all threads have completed the parallel work and are ready to be written
        //we need to ensure a stable ordering, though, so we should sort them

        //if you don't include a file offset, it round robins to SSDS

       
        //we're pushing and this should be written to a random SSD to spread writes across them
        //writer is threadsafe

        //problem: if we push all of the chunks onto the random_numberth SSD, we may end up overloading some SSDs
        //and not using others hardly at all. one solution is to just issue these writes in a for loop with a higher granularity.
        for(int r= 0; r < NUM_SSDS; r++){
            // writer.Push(std::shared_ptr<T>(buffer, free), buffer_size * NUM_SSDS, random_number);
            // if(counter_bad == 0){
            // writer.Push((buffer + r * buffer_size), buffer_size, random_holder[r]);
            // }
            //this else statement should run and check if the index we intend to push was actually used or not, if not then we can't push
            // else{
                // auto iter = bad_flags.end();
                // if(!(iter == std::find(bad_flags.start(), bad_flags.end(), r))){
                //     writer.Push((buffer + r * buffer_size), buffer_size, random_holder[r]);
                // }
         
            if(!(bad_flags[r])){
                size_t base_offset = file_offsets[random_holder[r]].fetch_add(buffer_size_bytes);
                (*chunk_header_arr)[write_count * NUM_SSDS + slot_for[r]].begin_address = base_offset;
                writer.Push(std::shared_ptr<T>(buffer[r], free), buffer_size, random_holder[r], base_offset);
            }
            else{
                free(buffer[r]);
                }
            }

        //write_count tracks the batch index; must increment here so subsequent batches use the correct slot range
        write_count++;
        }

        //somehow we need to figure out a way to free the buffer

        // free(buffer);


        //the buffer will be reallocated at the top of the while loop
        // buffer = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, buffer_size_bytes);
        // buffer_index = 0;
    // }
    
    

    //we do NOT need to make a file end marker -- this is a relic of the old reader which didn't have chunk metadata

    // size_t end_size = AlignUp(buffer_size * NUM_SSDS * sizeof(T) + METADATA_SIZE);
    // if (end_size > buffer_size_bytes * NUM_SSDS) {
    //     // rare situation where the size of the metadata exceeds sizeof(T), resulting
    //     // in insufficient buffer size
    //     buffer = (T*)realloc(buffer, end_size);
    // }

    // buffer = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, buffer_size_bytes * NUM_SSDS);

    // MakeFileEndMarker((unsigned char *) buffer,
    //                   end_size,
    //                   buffer_size * NUM_SSDS * sizeof(T));
    // writer.Push(std::shared_ptr<T>(buffer, free), end_size / sizeof(T));
    writer.Wait();
    // size_t file_size = (write_count + 1) * buffer_size_bytes * NUM_SSDS;
    // size_t true_size = write_count * buffer_size_bytes + buffer_index * NUM_SSDS * sizeof(T);

    
    size_t total_valid = (expected_reads - 1) * NUM_SSDS + (chunk_headers.size() % NUM_SSDS == 0 ? NUM_SSDS : chunk_headers.size() % NUM_SSDS);
    chunk_header_arr->resize(total_valid);

    std::sort(sequence.begin(), sequence.end(), [&](const chunk_header& i, const chunk_header& j){
        return i.index < j.index;
    });
    return sequence;
}



#endif
