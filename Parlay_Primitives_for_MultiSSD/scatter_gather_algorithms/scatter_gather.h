//
// Created by peter on 7/31/24.
//

#ifndef SORTING_SCATTER_GATHER_H
#define SORTING_SCATTER_GATHER_H

#include <utility>
#include <vector>
#include <string>
#include <functional>

#include "parlay/primitives.h"
#include "parlay/internal/get_time.h"
#include "parlay/alloc.h"

#include "configs.h"
#include "utils/file_utils.h"
#include "utils/unordered_file_reader.h"
#include "utils/ordered_file_writer.h"
#include "utils/type_allocator.h"

struct BucketedWriterConfig {
    size_t num_threads = 2;
    // Need to be explicitly specified.
    size_t num_buckets = -1;
};

struct ScatterGatherConfig {
    UnorderedReaderConfig reader_config;
    BucketedWriterConfig bucketed_writer_config;
    // Prints detailed performance statistics
    bool benchmark_mode = false;
};

/**
 * Perform external memory sample sort
 *
 * @tparam T The type to be sorted
 */
template<typename T>
class ScatterGather {
public:
    typedef std::function<size_t(const T &, size_t)> AssignerFunction;
    typedef std::function<void(T **, size_t)> ProcessorFunction;

private:

    using BucketData = AllocatorData<SAMPLE_SORT_BUCKET_SIZE>;
    using bucket_allocator = AlignedTypeAllocator<BucketData, O_DIRECT_MULTIPLE>;

    /**
     * Keep polling from the reader for data and then assign items into buckets according to how they compare
     * with samples. Buckets are flushed to the writer whenever they're full.
     *
     * The function is meant to be run as a thread and exits when the reader returns nullptr (no more input available)
     *
     * @tparam Comparator
     * @param samples
     * @param flush_threshold The number of bytes in each bucket. This is the threshold at which a bucket gets sent
     * to the writer. It may or may not be written to disk at the writer's discretion.
     * @param comp
     */
    void AssignToBucket(size_t num_buckets, const AssignerFunction assigner, const std::vector<FileInfo> &files) {
        // reads from the reader and put result into a thread-local buffer; send to intermediate_writer when buffer is full
        size_t buffer_size = SAMPLE_SORT_BUCKET_SIZE / sizeof(T);
        // each bucket stores a pointer to an array, which will hold temporary values in that bucket
        T *buckets[num_buckets];
        unsigned int buffer_index[num_buckets];
        for (size_t i = 0; i < num_buckets; i++) {
            buckets[i] = (T *) bucket_allocator::alloc();
            buffer_index[i] = 0;
        }
        while (true) {
            auto [data, size, file_index, data_index] = reader.Poll();
            if (data == nullptr) {
                break;
            }
            const size_t index_start = files[file_index].before_size + data_index;
            for (size_t i = 0; i < size; i++) {
                // use binary search to find which bucket it belongs to
                size_t bucket_index = assigner(data[i], index_start + i);
                // move data to bucket
                buckets[bucket_index][buffer_index[bucket_index]++] = data[i];
                // flush if bucket is full
                if (buffer_index[bucket_index] == buffer_size) {
                    buffer_index[bucket_index] = 0;
                    intermediate_writer.Write(bucket_index, buckets[bucket_index], buffer_size);
                    buckets[bucket_index] = (T *) bucket_allocator::alloc();
                }
            }
            reader.allocator.Free(data);
        }
        // cleanup partially full buckets
        for (size_t i = 0; i < num_buckets; i++) {
            // if bucket is empty, free and do nothing else; otherwise send to writer
            if (buffer_index[i] == 0) {
                bucket_allocator::free((BucketData *) buckets[i]);
            } else {
                intermediate_writer.Write(i, buckets[i], buffer_index[i]);
            }
        }
    }

    /**
     * Run a file's content in internal memory
     * @tparam Comparator
     * @param file_info
     * @param target_file File to which the result of the sorting algorithm will be written
     * @param comparator
     * @return Information of the resulting file
     */
    FileInfo ProcessBucket(const FileInfo &file_info, const std::string &target_file,
                           const ProcessorFunction processor) {
        // use parlay's sorting utility to sort this bucket
        T *buffer = (T *) ReadEntireFile(file_info.file_name, file_info.file_size);
        size_t n = file_info.true_size / sizeof(T);
        processor(&buffer, n);
        int fd = open(target_file.c_str(), O_WRONLY | O_DIRECT | O_CREAT, 0644);
        SYSCALL(fd);
        SYSCALL(write(fd, buffer, file_info.file_size));
        close(fd);
        // FIXME: this is very expensive because munmap is called every time
        free(buffer);
        return {target_file, file_info};
    }

    // reader for the input files
    UnorderedFileReader<T> reader;
    // writer to handle all the buckets created in phase 1 of sample sort
    OrderedFileWriter<T, SAMPLE_SORT_BUCKET_SIZE> intermediate_writer;

    parlay::sequence<FileInfo>
    QueuePhase2(const std::string &result_prefix, const ProcessorFunction &processor,
                const std::vector<FileInfo> &bucket_list) {
        parlay::internal::timer timer("phase 2 internal");
        const size_t num_files = bucket_list.size();
        parlay::sequence<FileInfo> results(num_files, FileInfo("", 0, 0, 0));
        std::atomic<size_t> current_file = 0, files_read = 0;
        SimpleQueue<std::pair<size_t, T *>> read_queue, write_queue;
        std::vector<std::thread> read_workers, write_workers;
        read_queue.SetSizeLimit(256);
        write_queue.SetSizeLimit(256);
        for (size_t worker_thread = 0; worker_thread < 256; worker_thread++) {
            read_workers.emplace_back([&]() {
                while (true) {
                    size_t index = current_file++;
                    if (index >= num_files) {
                        return;
                    }
                    const FileInfo &file = bucket_list[index];
                    auto pointer = (T *) ReadEntireFile(file.file_name, file.file_size);
                    read_queue.Push({index, pointer});
                    index = ++files_read;
                    if (index == num_files) {
                        read_queue.Close();
                    }
                }
            });
            write_workers.emplace_back([&]() {
                while (true) {
                    auto [res, code] = write_queue.Poll({0, nullptr});
                    if (code == QueueCode::FINISH) {
                        return;
                    }
                    auto [index, pointer] = res;
                    FileInfo file = bucket_list[index];
                    auto file_name = GetFileName(result_prefix, index);
                    int fd = open(file_name.c_str(), O_WRONLY | O_DIRECT | O_CREAT, 0644);
                    SYSCALL(fd);
                    SYSCALL(write(fd, pointer, file.file_size));
                    close(fd);
                    free(pointer);
                    results[index] = {file_name, bucket_list[index]};
                }
            });
        }
        timer.next("Worker threads created");
        parlay::parallel_for(0, parlay::num_workers() - 4, [&](size_t _) {
            while (true) {
                auto [res, code] = read_queue.Poll({0, nullptr});
                if (code == QueueCode::FINISH) {
                    return;
                }
                auto [index, pointer] = res;
                size_t n = bucket_list[index].true_size / sizeof(T);
                processor(&pointer, n);
                write_queue.Push(std::pair(index, pointer));
            }
        }, 1);
        write_queue.Close();
        timer.next("All processor functions completed");
        for (auto &worker: read_workers) {
            worker.join();
        }
        for (auto &worker: write_workers) {
            worker.join();
        }
        timer.next("Workers joined");
        timer.stop();
        return results;
    }

    parlay::sequence<FileInfo>
    WorkerOnlyPhase2(const std::string &result_prefix, const ProcessorFunction &processor,
                     const std::vector<FileInfo> &bucket_list) {
        struct LocalFile {
            int fd;
            T *buffer;
            FileInfo info;
        };
        std::atomic<size_t> current_file = 0;
        size_t num_files = bucket_list.size();
        parlay::sequence<FileInfo> results(num_files, FileInfo("", 0, 0, 0));
        parlay::parallel_for(0, parlay::num_workers(), [&](size_t worker_id) {
            auto sleep_time = 5000 * parlay::worker_id();
            usleep(sleep_time);
            struct io_uring read_ring, write_ring;
            io_uring_queue_init(4, &read_ring, IORING_SETUP_SINGLE_ISSUER);
            io_uring_queue_init(4, &write_ring, IORING_SETUP_SINGLE_ISSUER);
            LocalFile previous, current, next;
            bool need_reap_read = false,
                need_submit_read = true,
                need_process = false,
                need_reap_write = false,
                need_submit_write = false;
            while (need_submit_read || need_reap_write) {
                previous = current;
                current = next;
                // reap read
                if (need_reap_read) {
                    struct io_uring_cqe *cqe;
                    SYSCALL(io_uring_wait_cqe(&read_ring, &cqe));
                    SYSCALL(cqe->res);
                    io_uring_cqe_seen(&read_ring, cqe);
                    close(current.fd);
                    need_process = true;
                } else {
                    need_process = false;
                }
                // submit read
                size_t index = current_file++;
                if (index >= num_files) {
                    need_submit_read = false;
                }
                if (need_submit_read) {
                    const auto &file_info = bucket_list[index];
                    next.fd = open(file_info.file_name.c_str(), O_RDONLY | O_DIRECT);
                    SYSCALL(next.fd);
                    next.buffer = (T *) std::aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, file_info.file_size);
                    next.info = file_info;
                    next.info.file_index = index;
                    auto sqe = io_uring_get_sqe(&read_ring);
                    io_uring_prep_read(sqe, next.fd, next.buffer, file_info.file_size, 0);
                    io_uring_submit(&read_ring);
                    need_reap_read = true;
                } else {
                    need_reap_read = false;
                }

                // process
                if (need_process) {
                    processor(&current.buffer, current.info.true_size / sizeof(T));
                    current.info.file_name = GetFileName(result_prefix, current.info.file_index);
                    int fd = open(current.info.file_name.c_str(),
                                  O_DIRECT | O_WRONLY | O_CREAT,
                                  0644);
                    SYSCALL(fd);
                    current.fd = fd;
                    need_submit_write = true;
                } else {
                    need_submit_write = false;
                }

                // reap write
                if (need_reap_write) {
                    struct io_uring_cqe *cqe;
                    SYSCALL(io_uring_wait_cqe(&write_ring, &cqe));
                    SYSCALL(cqe->res);
                    io_uring_cqe_seen(&write_ring, cqe);
                    SYSCALL(close(previous.fd));
                    free(previous.buffer);
                    results[previous.info.file_index] = previous.info;
                }

                // submit write
                if (need_submit_write) {
                    auto sqe = io_uring_get_sqe(&write_ring);
                    io_uring_prep_write(sqe, current.fd, current.buffer, current.info.file_size, 0);
                    io_uring_submit(&write_ring);
                    need_reap_write = true;
                } else {
                    need_reap_write = false;
                }
            }
        }, 1);
        return results;
    }

    parlay::sequence<FileInfo>
    SimplePhase2(const std::string &result_prefix, const ProcessorFunction &processor,
                 const std::vector<FileInfo> &bucket_list) {
        return parlay::tabulate(bucket_list.size(), [&](size_t i) {
            const auto &file_info = bucket_list[i];
            auto result_name = GetFileName(result_prefix, i);
            ProcessBucket(file_info, result_name, processor);
            return FileInfo(result_name, file_info);
        }, 1);
    }

public:

    std::vector<FileInfo> Run(std::vector<FileInfo> &input_files,
                              const std::string &result_prefix,
                              const AssignerFunction assigner,
                              const ProcessorFunction processor,
                              const ScatterGatherConfig &config) {
        parlay::internal::timer timer("Scatter gather internal", true);
        reader.PrepFiles(input_files);
        reader.Start(config.reader_config);
        // FIXME: change this file name to a different one (possibly randomized?)
        size_t num_buckets = config.bucketed_writer_config.num_buckets;
        intermediate_writer.Initialize("spfx_", num_buckets, 1 << 20);
        timer.next("Start phase 1 (assign to buckets)");
        std::vector<FileInfo> bucket_list;
        auto intermedia_io_threads = config.bucketed_writer_config.num_threads;
        CHECK(intermedia_io_threads < parlay::num_workers());
        parlay::par_do([&]() {
            parlay::parallel_for(0, intermedia_io_threads, [&](size_t i) {
                intermediate_writer.RunIOThread(&intermediate_writer);
            }, 1);
        }, [&]() {
            parlay::parallel_for(0, parlay::num_workers() - intermedia_io_threads, [&](int i) {
                AssignToBucket(num_buckets, assigner, input_files);
            }, 1);
            // retrieve buckets from intermediate_writer
            bucket_list = intermediate_writer.ReapResult();
        });
        bucket_allocator::finish();
        // Print detailed statistics under benchmark mode, otherwise just print the time
        if (config.benchmark_mode) {
            double throughput = GetThroughput(input_files, timer.next_time());
            std::cout << "Throughput1: " << throughput << "GB\n";
        } else {
            timer.next("After assign to bucket and before phase 2");
        }
        parlay::sequence<FileInfo> results = WorkerOnlyPhase2(result_prefix, processor, bucket_list);
        if (config.benchmark_mode) {
            double throughput = GetThroughput(input_files, timer.next_time());
            std::cout << "Throughput2: " << throughput << "GB\n";
        } else {
            timer.next("After assign to bucket and before phase 2");
        }
        timer.stop();
        return {results.begin(), results.end()};
    }
};

#endif //SORTING_SCATTER_GATHER_H
