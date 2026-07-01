//
// Created by peter on 8/29/24.
//

#ifndef SORTING_FILTER_H
#define SORTING_FILTER_H

#include "utils/file_info.h"
#include "utils/unordered_file_reader.h"
#include "utils/unordered_file_writer.h"

#include "parlay/primitives.h"

template<typename T>
FileInfo FilterFile(const FileInfo &in_file, const std::string &out_file, const std::function<bool(const T)> predicate) {
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
            while (i < top.size) {
                if (predicate(top.ptr[i])) {
                    buffer[buffer_index] = top.ptr[i];
                    buffer_index++;
                }
                if (buffer_size == buffer_index) {
                    writer.Push(std::shared_ptr<T>(buffer, free), buffer_size);
                    write_count++;
                    buffer = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, buffer_size_bytes);
                    buffer_index = 0;
                }
                i++;
            }
            reader.allocator.Free(top.ptr);
        }
    }
    size_t end_size = AlignUp(buffer_index * sizeof(T) + METADATA_SIZE);
    if (end_size > buffer_size_bytes) {
        // rare situation where the size of the metadata exceeds sizeof(T), resulting
        // in insufficient buffer size
        buffer = (T*)realloc(buffer, end_size);
    }
    MakeFileEndMarker((unsigned char *) buffer,
                      end_size,
                      buffer_index * sizeof(T));
    writer.Push(std::shared_ptr<T>(buffer, free), end_size / sizeof(T));
    writer.Wait();
    size_t file_size = (write_count + 1) * buffer_size_bytes;
    size_t true_size = write_count * buffer_size_bytes + buffer_index * sizeof(T);
    return {out_file, in_file.file_index, true_size, file_size};
}

template<typename T>
std::vector<FileInfo> Filter(const std::vector<FileInfo> &files,
                             const std::string &prefix,
                             const std::function<bool(const T)> predicate) {
    auto result = parlay::map(parlay::iota(files.size()), [&](size_t i) {
        return FilterFile(files[i], GetFileName(prefix, i), predicate);
    }, 1);
    return {result.begin(), result.end()};
}

#endif //SORTING_FILTER_H
