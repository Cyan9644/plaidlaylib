#ifndef FILTER_H
#define FILTER_H
#include <pthread.h>
#include "utils/file_info.h"
#include "utils/file_utils.h"
#include "utils/unordered_file_writer.h"
#include "utils/unordered_file_reader.h"
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
#include <queue>

#define NUMBLOCKS 1
#define BLOCKSIZE 512
//copied from plaidlay.h


namespace plaidlayNaive {
    template <typename T>
    naiveSeq<T> block_flatten_in_dram(const naiveSeq<naiveSeq<T>>& input);
}
pthread_mutex_t locking = PTHREAD_MUTEX_INITIALIZER; 
template <typename T, typename Func>
naiveSeq<T> filter_range(const naiveSeq<T>& seq, Func f, unsigned long start, unsigned long end){

        auto seq2 = plaidlayNaive::cut(seq, start, end);
        std::vector<T> out;

        
        for (const T& elem: seq2) {
            if (f(elem)) {
                out.push_back(elem);
            }
        }
        return naiveSeq<T>(out);


    }


//don't use this
template <typename T>

naiveSeq<T> flatten_range(const naiveSeq<naiveSeq<T>>& seq, unsigned long start, unsigned long end) {
        // naiveSeq<T> res;
        std::vector<T> res;
        for(; start < end; start++){
            for(const auto& in: seq[start]){
                
                res.push_back(in);
            }
        }
        return naiveSeq<T>(res);


    }




//filters in parallel (must be in DRAM)
//this can be improved on by removing the lock -- use out.resize to write in parallel without synchronization overhead
//this also doesn't preserve ordering, whoops
template <typename T, typename Func>
naiveSeq<T> naive_parallel_dram_filter_lock(naiveSeq<T> sequence, Func f){

    std::vector<naiveSeq<T>> out;

    std::atomic<bool> lock = true;

    auto num_blocks = (int)(sequence.size() / BLOCKSIZE);

    auto remainder = (int)(sequence.size() % BLOCKSIZE);

    parlay::parallel_for(0, num_blocks, [&](long i){

    naiveSeq<T> filtered;

    i != num_blocks-1 ? filtered = plaidlayNaive::filter(plaidlayNaive::cut(sequence, BLOCKSIZE * i, BLOCKSIZE * (i+1)), f) : filtered = plaidlayNaive::filter(plaidlayNaive::cut(sequence, BLOCKSIZE * i, BLOCKSIZE * (i+1) + remainder));

    bool check_yes = 1;

    bool check_no = 0;

    while(!(lock.compare_exchange_strong(check_yes, check_no))){

        check_yes = true;
        check_no = false;

    }

    out.push_back(filtered);
    lock = 1;


    });



    auto outer = naiveSeq<naiveSeq<T>>(out);
    return plaidlayNaive::flatten(outer);;


}

template <typename T, typename Func>
naiveSeq<T> naive_parallel_dram_filter(naiveSeq<T> sequence, Func f){

    std::vector<naiveSeq<T>> out;

    auto num_blocks = (int)(sequence.size() / BLOCKSIZE);

    auto remainder = (int)(sequence.size() % BLOCKSIZE);

    out.resize(num_blocks);
    parlay::parallel_for(0, num_blocks, [&](long i){


        out[i] = plaidlayNaive::filter(plaidlayNaive::cut(sequence, BLOCKSIZE * i, BLOCKSIZE * (i+1)), f);


    });
    //we can just treat remainder as another block instead of doing this sequentially

    auto remain = plaidlayNaive::filter(plaidlayNaive::cut(sequence, BLOCKSIZE * num_blocks, BLOCKSIZE * num_blocks + remainder), f);

    out.push_back(remain);

    auto outer = naiveSeq<naiveSeq<T>>(out);

    return plaidlayNaive::flatten(outer);


}

//directly from graduate Peter's code
template<typename T>
FileInfo FilterFileSequential(const FileInfo &in_file, const std::string &out_file, const std::function<bool(const T)> predicate) {
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

    //to set up the unordered file reader (multithreaded, does not guarantee ordering on thread finishes), we need to call prepfiles and start
    reader.PrepFiles({in_file});
    reader.Start();
    UnorderedFileWriter<T> writer(out_file);

    std::priority_queue<QueueData, std::vector<QueueData>, decltype(cmp)> queue(cmp);
    //4 << 20 == 4 * 2^20
    constexpr size_t buffer_size_bytes = 4 << 20, buffer_size = buffer_size_bytes / sizeof(T);
    size_t buffer_index = 0;
    //we expect O_DIRECT to be on to avoid the OS's page cache, which means we need to align our memory directly
    auto buffer = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, buffer_size_bytes);
    size_t write_count = 0;
    size_t next_index = 0;
    while (true) {
        //.poll will block until a block read is returned or a sentinel 0 value is given back (which would mean that the reader is closed)
        auto [ptr, size, _, index] = reader.Poll();

        if (ptr == nullptr) {
            //if there is nothing left to read
            CHECK(queue.empty());
            break;
        }
        //put on pq for writing (in-place)
        //the queue here is a min-heap ordered by smallest index at the top, which means that the element at the top of the queue should be the next writable position
        //although it may not match what we expect yet since the reader is multithreaded. This means we need to wait for the correct index to show up at the top of the queue
        //so that we can maintain stability (initial ordering of elements) in the filtering
        queue.emplace(ptr, size, index);
        while (!queue.empty()) {
            auto top = queue.top();
            if (top.index != next_index) {
                //this is intended to enforce an ordering on writes -- if the queue is not empty but the top index is not the desired one, we need to go back and read more blocks
                break;
            }
            //take the top element which is the next to write
            queue.pop();
            //increment next_index, which indicates how many elements have been consumed

            //FIXME

            // I think this should actually be top.size
            next_index += top.size;
            // process buffer
            size_t i = 0;
            while (i < top.size) {//should be top.size, not size from the block read earlier. that would be fine if we only had one pass but this 
                //is designed to handle the case where reads come in out of order and need to be reordered
                if (predicate(top.ptr[i])) {
                    //include in filter
                    buffer[buffer_index] = top.ptr[i];
                    buffer_index++;
                }
                if (buffer_size == buffer_index) {
                    //easier to think about this as checking buffer_index == buffer_size, so when the buffer is full
                    writer.Push(std::shared_ptr<T>(buffer, free), buffer_size);
                    write_count++;
                    //reallocate buffer since we pushed the last one onto the write queue
                    buffer = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, buffer_size_bytes);
                    buffer_index = 0;
                }
                i++;
            }
        }
    }

    //now one of the break conditions has been achieved, which means that we are done reading

    //end size is the next multiple up I assume of the O_DIRECT_MEMORY_ALIGNMENT
    size_t end_size = AlignUp(buffer_index * sizeof(T) + METADATA_SIZE);
    if (end_size > buffer_size_bytes) {
        // rare situation where the size of the metadata exceeds sizeof(T), resulting
        // in insufficient buffer size
        buffer = (T*)realloc(buffer, end_size);//this buffer is not aligned to O_DIRECT_MEMORY_ALIGNMENT
    }
    MakeFileEndMarker((unsigned char *) buffer,
                      end_size,
                      buffer_index * sizeof(T));
    writer.Push(std::shared_ptr<T>(buffer, free), end_size);
    //wait for all writes to complete? Seems that we could be overlapping io more by writing as we read
    writer.Wait();
    size_t file_size = (write_count + 1) * buffer_size_bytes;
    size_t true_size = write_count * buffer_size_bytes + buffer_index * sizeof(T);
    return {out_file, in_file.file_index, true_size, file_size};
}


template<typename T>
FileInfo FilterFileParallel(const FileInfo &in_file, const std::string &out_file, const std::function<bool(const T)> predicate) {
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

    //to set up the unordered file reader (multithreaded, does not guarantee ordering on thread finishes), we need to call prepfiles and start
    reader.PrepFiles({in_file});
    reader.Start();
    UnorderedFileWriter<T> writer(out_file);

    std::priority_queue<QueueData, std::vector<QueueData>, decltype(cmp)> queue(cmp);
    //4 << 20 == 4 * 2^20
    constexpr size_t buffer_size_bytes = 4 << 20, buffer_size = buffer_size_bytes / sizeof(T);
  
    //we expect O_DIRECT to be on to avoid the OS's page cache, which means we need to align our memory directly
    
    size_t write_count = 0;
    size_t next_index = 0;
    UnorderedWriterConfig config;
    config.io_uring_size = 8;
    config.num_threads = 5;
    size_t buffering = 0;
    std::atomic<bool> lock = true;
    std::atomic<bool> lock2 = true;
    std::atomic<bool> lock_index = true;
    auto buffer = std::vector<T*>(config.num_threads);
    
    for(; buffering < config.num_threads; buffering++){
        buffer[buffering] = (T*) aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, buffer_size_bytes);
    }


    while (true) {
        // if(!queue.empty()){
        // auto top = queue.top();
        bool shared_flag = 0;
        size_t overall_buffer_index;
        auto threads_complete = 0;
        parlay::parallel_for(0, config.num_threads, [&](long p){
        size_t buffer_index = 0;
        // buffer_index = 0;
        //.poll will block until a block read is returned or a sentinel 0 value is given back (which would mean that the reader is closed)
        //reader.poll is threadsafe
        auto [ptr, size, _, index] = reader.Poll();
        pthread_mutex_lock(&locking);
        if (ptr == nullptr) {
            //if there is nothing left to read
            //pthread_mutex_lock(&locking);
            CHECK(queue.empty());
            
            shared_flag= 1;
            //pthread_mutex_unlock(&locking);

        }
        if(!shared_flag){
        //put on pq for writing (in-place)
        //the queue here is a min-heap ordered by smallest index at the top, which means that the element at the top of the queue should be the next writable position
        //although it may not match what we expect yet since the reader is multithreaded. This means we need to wait for the correct index to show up at the top of the queue
        //so that we can maintain stability (initial ordering of elements) in the filtering

        //need LOCK here -- after everything is working we'll replace these atomic variable mutexes with regular locks
        
        // bool check_yes2 = 1;

        // bool check_no2 = 0;

        // while(!(lock2.compare_exchange_strong(check_yes2, check_no2))){

        // check_yes2 = true;
        // check_no2 = false;

        // }
        //pthread_mutex_lock(&locking);
        queue.emplace(ptr, size, index);
      //don't unlock here because we would need to do it agian immediately
    
        while (!queue.empty()) {//maybe need a lock here? sad for performance
            auto top = queue.top();
            //pthread_mutex_unlock(&locking);
            if (top.index != next_index) {
                //this is intended to enforce an ordering on writes -- if the queue is not empty but the top index is not the desired one, we need to go back and read more blocks
                break;
            }
            //take the top element which is the next to write
                   
        bool check_yes = 1;

        bool check_no = 0;
        //pthread_mutex_lock(&locking);
            queue.pop();
        //pthread_mutex_unlock(&locking);
            
            //increment next_index, which indicates how many elements have been consumed

            next_index += top.size;
            // process buffer
            size_t i = 0;
            pthread_mutex_unlock(&locking);
            while (i < top.size) {//should be top.size, not size from the block read earlier. that would be fine if we only had one pass but this 
                //is designed to handle the case where reads come in out of order and need to be reordered
                if (predicate(top.ptr[i])) {
                    //include in filter
                    buffer[p][buffer_index] = top.ptr[i];
                    buffer_index++;
                }
                    if (buffer_size == buffer_index) {
                    //easier to think about this as checking buffer_index == buffer_size, so when the buffer is full
                    //writer.push is threadsafe
                    writer.Push(std::shared_ptr<T>(buffer[p], free), buffer_size);
                    write_count++;
                
                    buffer[p] = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, buffer_size_bytes);
                    bool check_lock1 = 1;
                    bool check_lock2 = 0;
                    //because we're guarding single updates here, we can use atomic variables
                     //pthread_mutex_lock(&locking);
          overall_buffer_index+=buffer_index;

                //pthread_mutex_unlock(&locking);


                    // while(!(lock_index.compare_exchange_strong(check_lock1, check_lock2))){

                    //     check_lock1 = true;
                    //     check_lock2 = false;
                    //     }
                    // overall_buffer_index+=buffer_index;
                 
                    buffer_index = 0;
                }
    
                i++;
            }
            pthread_mutex_lock(&locking);
            //need this additional lock to ensure that the lock is held when we check queue.empty() and top() the next time
            //pthread_mutex_lock(&locking);

        }
        pthread_mutex_unlock(&locking); 
    
    }
    else{
        return;
    }
    });
    if(shared_flag){
        break;
    }
  
    }


    //now one of the break conditions has been achieved, which means that we are done reading

    //end size is the next multiple up I assume of the O_DIRECT_MEMORY_ALIGNMENT
    size_t end_size = AlignUp(buffer_size * sizeof(T) + METADATA_SIZE);

    //we're not going to worry about this for right now
    // if (end_size > buffer_size_bytes) {
    //     // rare situation where the size of the metadata exceeds sizeof(T), resulting
    //     // in insufficient buffer size
    //     buffer = (T*)realloc(buffer, end_size);//this buffer is not aligned to O_DIRECT_MEMORY_ALIGNMENT
    // }
    MakeFileEndMarker((unsigned char *) buffer[0],
                      end_size,
                    buffer_size * sizeof(T));
    writer.Push(std::shared_ptr<T>(buffer[0], free), end_size);
    //TODO: we need to eventually free all of the buffers to prevent memory leaks
    //wait for all writes to complete?
    writer.Wait();
    size_t file_size = (write_count + 1) * buffer_size_bytes;
    size_t true_size = write_count * buffer_size_bytes + buffer_size * sizeof(T);
    return {out_file, in_file.file_index, true_size, file_size};
}



template<typename T>
FileInfo FilterFileParallel_in_block(const FileInfo &in_file, const std::string &out_file, const std::function<bool(const T)> predicate) {
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
    size_t buffer_index = 0;
    auto buffer = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, buffer_size_bytes);
    size_t write_count = 0;
    size_t next_index = 0;
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
            long num_blocks = top.size / BLOCKSIZE;
            bool has_remainder = top.size % BLOCKSIZE;
            long size;
            has_remainder ? size = ++num_blocks : size = num_blocks;

            //instead of allocating the full blocksize and then resizing afterwards, we'll spend some CPU time to make a second pass 
            //and avoid this intermediate allocation
            naiveSeq<naiveSeq<T>> myseqseq(size);
            
            
            //instead of parallel_for, we could call a regular DRAM parallel filter method
            parlay::parallel_for(0, size, [&](long o){
                auto counter =  0; //this is the counter that will tell us how much we need to allocate.
                //we're assuming that memory is the bottleneck here, so we can't make an additional vector to hold the indices of the results to avoid reevaluating the predicates
              
                long block_start = o * BLOCKSIZE;
                long copy = block_start;
                long block_end = std::min<size_t>((o+1) * BLOCKSIZE, (long) top.size);
                for(; copy< block_end; copy++){
                    auto cate = top.ptr[copy];
                    if(predicate(cate)){
                        counter++;
                    }

                }

                myseqseq[o] = naiveSeq<T>(counter);
                counter = 0;
                for(; block_start < block_end; block_start++){
                    auto cate = top.ptr[block_start];
                    if(predicate(cate)){
                        myseqseq[o][counter++] = top.ptr[block_start];
                    }

                }
                auto new_seq = plaidlayNaive::block_flatten_in_dram(myseqseq);
                auto new_ct = 0;
                for(; new_ct < counter; new_ct++){
                    buffer[block_start+new_ct] = new_seq[new_ct];
                }


                //NEED ATOMIC VAR HERE TO SHOW WHEN FULL
            });
            
            // buffer = reinterpret_cast<T*>(flatten(myseqseq));
            writer.Push(std::shared_ptr<T>(buffer, free), buffer_size);
            buffer = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, buffer_size_bytes);
 
        }
    }
    size_t end_size = AlignUp(buffer_index * sizeof(T) + METADATA_SIZE);
    if (end_size > buffer_size_bytes) {
        // rare situation where the size of the metadata exceeds sizeof(T), resulting
        // in insufficient buffer size
        buffer = (T*)realloc(buffer, end_size); //this buffer is not aligned to O_DIRECT_MEMORY_ALIGNMENT
    }
    MakeFileEndMarker((unsigned char *) buffer,
                      end_size,
                      buffer_index * sizeof(T));
    writer.Push(std::shared_ptr<T>(buffer, free), end_size);
    writer.Wait();
    size_t file_size = (write_count + 1) * buffer_size_bytes;
    size_t true_size = write_count * buffer_size_bytes + buffer_index * sizeof(T);
    return {out_file, in_file.file_index, true_size, file_size};

}




#endif