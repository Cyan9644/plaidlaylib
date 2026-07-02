#ifndef SCAN_H
#define SCAN_H

// #include "utils/file_info.h"
// #include "utils/unordered_file_reader.h"
// #include "utils/unordered_file_writer.h"
#include "filter.h"
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
#include "utils/file_info.h"
#include "utils/file_utils.h"
#include "utils/unordered_file_writer.h"
#include "utils/unordered_file_reader.h"
#include <parlay/primitives.h>
#include <queue>


namespace plaidlayNaive{
    //simple scan for the flatten method
    template <typename T>
    naiveSeq<T> scan_inclusive(naiveSeq<T> input){
        naiveSeq<T> ret(input.size());
        ret[0] = input[0];
        if(input.size() == 1){
            return ret;
        }
        size_t i = 1;
        for(; i < input.size(); i++){
            ret[i] = input[i] + ret[i-1];
        }
        return ret;
    }


    template <typename T>
    naiveSeq<T> scan(naiveSeq<T> input){
        naiveSeq<T> ret(input.size());
        if(input.size() == 1){
            ret[0]=0;
            return ret;
        }
        else if(input.size()==2){
            ret[0]=0;
            ret[1]=input[0];
            return ret;
        }
        size_t i = 2;
    
        ret[0]= 0;
        ret[1] = input[0];
        for(; i < input.size(); i++){

            ret[i] = input[i-1] + ret[i-1];
        }
        return ret;
    }

    //parallel scan method based on the specifications from Parallel Block Delayed Sequences
    //do a scan on each block, do a scan on the offsets, and recalculate the block scans
    template <typename T>
    naiveSeq<T> block_scan(naiveSeq<T>& input){

        naiveSeq<T> output(input.size());

        bool remainder = input.size() % BLOCKSIZE;

        long num_blocks = input.size()/BLOCKSIZE;

        long size;
        
        if(remainder){
            size = num_blocks+1;
        }
        else{
            size = num_blocks;
        }
        std::vector<long> offsets(size);
        parlay::parallel_for(0, size, [&](long i){

            long block_start = i * BLOCKSIZE;

            long block_end = std::min((i+1) * BLOCKSIZE,(long)input.size()) ;

            output[block_start]=0;

            long r = block_start+1; //don't want to take information from beyond the block boundaries

            for(; r <block_end; r++){

            output[r] = output[r-1] + input[r-1];
            }

            offsets[i] = output[block_end-1] + input[block_end-1];

        });

        //get offsets scan
        std::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin(), 0L);

        parlay::parallel_for(0, size, [&](long i){

            long block_start = i * BLOCKSIZE;

            long block_end = std::min((i+1) * BLOCKSIZE,(long)input.size());

            long r = block_start;

            for(; r< block_end; r++){

                output[r]+=offsets[i];

            }


        });

        return output;


    }

}

//parallel scan method based on the specifications from Parallel Block Delayed Sequences
//do a scan on each block, do a scan on the offsets, and recalculate the block scans
template <typename T>
FileInfo NaiveScanFile(const FileInfo &in_file, const std::string& out_file, std::vector<size_t> &file_offsets, size_t i){

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
    UnorderedFileWriter<T> writer;
    UnorderedWriterConfig wconfig;
    wconfig.num_threads = 1;
    writer.Start(std::vector<std::string>{out_file}, wconfig);
    // UnorderedFileWriter<T> writer(out_file);
    std::priority_queue<QueueData, std::vector<QueueData>, decltype(cmp)> queue(cmp);

    constexpr size_t buffer_size_bytes = 4 << 20, buffer_size = buffer_size_bytes / sizeof(T);
    size_t buffer_index = 0;
    auto buffer = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, buffer_size_bytes);
    size_t write_count = 0;
    size_t next_index = 0;
    // size_t buffer_reset = 1;
    // bool very_first_time = 1;
    T running_total = file_offsets[i];
    while (true) {
       
        auto [ptr, size, _, index] = reader.Poll();
        if (ptr == nullptr) {
            CHECK(queue.empty());
            break;
        }
        queue.emplace(ptr, size, index);
        while (!queue.empty()) {
            auto top = queue.top();
            if (top.index != next_index) {
                break;
            }
            queue.pop();
            next_index += top.size;
            // process buffer
            size_t i = 0;

            while(i < top.size){

                buffer[buffer_index++] = running_total;
                running_total += top.ptr[i];

                if(buffer_index == buffer_size){
                    writer.Push(std::shared_ptr<T>(buffer, free), buffer_size);
                    write_count++;
                    buffer = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, buffer_size_bytes);
                    buffer_index = 0;

                }
                ++i;
            }
            // buffer_reset = true; //this reset means that we have reached the end of the current chunk
        }
    }
    
    size_t end_size = AlignUp(buffer_index * sizeof(T) + METADATA_SIZE);
    if (end_size > buffer_size_bytes) {
        // rare situation where the size of the metadata exceeds sizeof(T), resulting
        // in insufficient buffer size
        //required to use realloc here and not align for pipeline
        // buffer = (T*)realloc(buffer, end_size);
        // aligned_free

        T* new_buffer = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, end_size);
        memcpy(new_buffer, buffer, buffer_index * sizeof(T));
        free(buffer);
        buffer=  new_buffer;
    }
    MakeFileEndMarker((unsigned char *) buffer,
                      end_size,
                      buffer_index * sizeof(T));
    writer.Push(std::shared_ptr<T>(buffer, free), end_size / sizeof(T));
    writer.Wait();
    // size_t file_size = (write_count + 1) * buffer_size_bytes;
    size_t file_size = write_count * buffer_size_bytes + end_size;
    size_t true_size = write_count * buffer_size_bytes + buffer_index * sizeof(T);
    return {out_file, in_file.file_index, true_size, file_size};


}


template<typename T>
void put_in_mem(const std::vector<FileInfo> &files, const std::string &prefix, std::vector<size_t> &store, size_t i){
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
    reader.PrepFiles({files[i]});
    reader.Start();
    // UnorderedFileWriter<T> writer;
    // UnorderedWriterConfig wconfig;
    // wconfig.num_threads = 1;
    // writer.Start(std::vector<std::string>{out_file}, wconfig);
    // UnorderedFileWriter<T> writer(out_file);
    std::priority_queue<QueueData, std::vector<QueueData>, decltype(cmp)> queue(cmp);

    constexpr size_t buffer_size_bytes = 4 << 20, buffer_size = buffer_size_bytes / sizeof(T);
    size_t buffer_index = 0;
    // auto buffer = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, buffer_size_bytes);
    size_t write_count = 0;
    size_t next_index = 0;
    // size_t buffer_reset = 1;
    // bool very_first_time = 1;
    T running_total = 0;
    while (true) {
       
        auto [ptr, size, _, index] = reader.Poll();
        if (ptr == nullptr) {
            CHECK(queue.empty());
            break;
        }
        queue.emplace(ptr, size, index);
        while (!queue.empty()) {
            auto top = queue.top();
            if (top.index != next_index) {
                break;
            }
            queue.pop();
            next_index += top.size;
            // process buffer
            size_t i = 0;

            while(i < top.size){

                // buffer[buffer_index++] = running_total;
                running_total += top.ptr[i];

                if(buffer_index == buffer_size){
                    // writer.Push(std::shared_ptr<T>(buffer, free), buffer_size);
                    // write_count++;
                    // buffer = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, buffer_size_bytes);
                    buffer_index = 0;

                }
                ++i;
            }
            // buffer_reset = true; //this reset means that we have reached the end of the current chunk
        }
    }
    
    // size_t end_size = AlignUp(buffer_index * sizeof(T) + METADATA_SIZE);
    // if (end_size > buffer_size_bytes) {
    //     // rare situation where the size of the metadata exceeds sizeof(T), resulting
    //     // in insufficient buffer size
    //     //required to use realloc here and not align for pipeline
    //     // buffer = (T*)realloc(buffer, end_size);
    //     // aligned_free

    //     T* new_buffer = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, end_size);
    //     memcpy(new_buffer, buffer, buffer_index * sizeof(T));
    //     free(buffer);
    //     buffer=  new_buffer;
    // }
    // MakeFileEndMarker((unsigned char *) buffer,
    //                   end_size,
    //                   buffer_index * sizeof(T));

    //this will be the end marker. We'll need to calculate a simple scan over this to find the offsets for each file's running total
    store[i] = running_total;


} 

template<typename T>
ExternalSequence Scan(const std::vector<FileInfo> &files, const std::string &prefix) {

    std::vector<size_t> store(files.size());
    parlay::parallel_for(0, store.size(), [&](size_t i){

        put_in_mem<T>(files, prefix, store, i);


    });
    //after this finishes, store should contain the total lengths.
    //this means we need to calculate a scan over them to get the file offsets
    auto file_offsets = std::vector<size_t>(store.size());
    std::exclusive_scan(store.begin(), store.end(), file_offsets.begin(), size_t(0));



    auto result = parlay::map(parlay::iota(files.size()), [&](size_t i) {
        return NaiveScanFile<T>(files[i], GetFileName(prefix, i), file_offsets, i);
    }, 1);

    return {result.begin(), result.end()};
}


#endif
