// #ifndef FILTER_H
// #define FILTER_H
// #include <pthread.h>
// #include "plaidlay.h"
// #include <cassert>
// #include <math.h>
// #include <iostream>
// #include <fcntl.h>
// #include <sys/time.h>
// #include <unistd.h>
// #include <stdlib.h>
// #include <liburing.h>
// #include <cstring>
// #include <parlay/parallel.h>
// #include <parlay/primitives.h>

// // template<typename T>
// // FileInfo FilterFileExternalSequential(const FileInfo &in_file, const std::string &out_file, const std::function<bool(const T)> predicate) {
// //     struct QueueData {
// //         T *ptr;
// //         size_t size;
// //         size_t index;

// //         QueueData(T *ptr, size_t size, size_t index) : ptr(ptr), size(size), index(index) {}
// //     };
// //     const auto cmp = [](QueueData a, QueueData b) {
// //         return a.index > b.index;
// //     };
    


// //     UnorderedFileReader<T> reader;

// //     //to set up the unordered file reader (multithreaded, does not guarantee ordering on thread finishes), we need to call prepfiles and start
// //     reader.PrepFiles({in_file});
// //     reader.Start();
// //     UnorderedFileWriter<T> writer(out_file);

// //     std::priority_queue<QueueData, std::vector<QueueData>, decltype(cmp)> queue(cmp);
// //     //4 << 20 == 4 * 2^20
// //     constexpr size_t buffer_size_bytes = 4 << 20, buffer_size = buffer_size_bytes / sizeof(T);
// //     size_t buffer_index = 0;
// //     //we expect O_DIRECT to be on to avoid the OS's page cache, which means we need to align our memory directly
// //     auto buffer = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, buffer_size_bytes);
// //     size_t write_count = 0;
// //     size_t next_index = 0;
// //     while (true) {
// //         //.poll will block until a block read is returned or a sentinel 0 value is given back (which would mean that the reader is closed)
// //         auto [ptr, size, _, index] = reader.Poll();

// //         if (ptr == nullptr) {
// //             //if there is nothing left to read
// //             CHECK(queue.empty());
// //             break;
// //         }
// //         //put on pq for writing (in-place)
// //         //the queue here is a min-heap ordered by smallest index at the top, which means that the element at the top of the queue should be the next writable position
// //         //although it may not match what we expect yet since the reader is multithreaded. This means we need to wait for the correct index to show up at the top of the queue
// //         //so that we can maintain stability (initial ordering of elements) in the filtering
// //         queue.emplace(ptr, size, index);
// //         while (!queue.empty()) {
// //             auto top = queue.top();
// //             if (top.index != next_index) {
// //                 //this is intended to enforce an ordering on writes -- if the queue is not empty but the top index is not the desired one, we need to go back and read more blocks
// //                 break;
// //             }
// //             //take the top element which is the next to write
// //             queue.pop();
// //             //increment next_index, which indicates how many elements have been consumed

// //             //FIXME

// //             // I think this should actually be top.size
// //             next_index += top.size;
// //             // process buffer
// //             size_t i = 0;
// //             while (i < top.size) {//should be top.size, not size from the block read earlier. that would be fine if we only had one pass but this 
// //                 //is designed to handle the case where reads come in out of order and need to be reordered
// //                 if (predicate(top.ptr[i])) {
// //                     //include in filter
// //                     buffer[buffer_index] = top.ptr[i];
// //                     buffer_index++;
// //                 }
// //                 if (buffer_size == buffer_index) {
// //                     //easier to think about this as checking buffer_index == buffer_size, so when the buffer is full
// //                     writer.Push(std::shared_ptr<T>(buffer, free), buffer_size);
// //                     write_count++;
// //                     //reallocate buffer since we pushed the last one onto the write queue
// //                     buffer = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, buffer_size_bytes);
// //                     buffer_index = 0;
// //                 }
// //                 i++;
// //             }
// //         }
// //     }

// //     //now one of the break conditions has been achieved, which means that we are done reading

// //     //end size is the next multiple up I assume of the O_DIRECT_MEMORY_ALIGNMENT
// //     size_t end_size = AlignUp(buffer_index * sizeof(T) + METADATA_SIZE);
// //     if (end_size > buffer_size_bytes) {
// //         // rare situation where the size of the metadata exceeds sizeof(T), resulting
// //         // in insufficient buffer size
// //         buffer = (T*)realloc(buffer, end_size);//this buffer is not aligned to O_DIRECT_MEMORY_ALIGNMENT
// //     }
// //     MakeFileEndMarker((unsigned char *) buffer,
// //                       end_size,
// //                       buffer_index * sizeof(T));
// //     writer.Push(std::shared_ptr<T>(buffer, free), end_size);
// //     //wait for all writes to complete? Seems that we could be overlapping io more by writing as we read
// //     writer.Wait();
// //     size_t file_size = (write_count + 1) * buffer_size_bytes;
// //     size_t true_size = write_count * buffer_size_bytes + buffer_index * sizeof(T);
// //     return {out_file, in_file.file_index, true_size, file_size};
// // }