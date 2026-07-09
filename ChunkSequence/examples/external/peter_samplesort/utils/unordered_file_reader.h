//
// Created by peter on 3/2/24.
//

#ifndef SORTING_UNORDERED_FILE_READER_H
#define SORTING_UNORDERED_FILE_READER_H

#include "utils/logger.h"
#include "utils/file_utils.h"
#include "configs.h"
#include "utils/simple_queue.h"
#include "utils/type_allocator.h"
#include <thread>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <deque>
#include <liburing.h>
#include <utility>
#include <iostream>
#include <map>
#include <fcntl.h>

struct UnorderedReaderConfig {
    size_t num_threads = 2;
    size_t max_requests = 64;
    size_t queue_depth = 32;
    size_t buffer_queue_size = 1024;

    // Optional: per-file active-chunk bitmap. If non-null, indexed by the
    // FileInfo::file_index; each inner vector is a packed 64-bit bitset where
    // bit i indicates whether chunk i of that file should be read. When set,
    // workers skip inactive chunks (no read submitted, no buffer enqueued).
    const std::vector<std::vector<uint64_t>>* active_chunks_per_file = nullptr;

    UnorderedReaderConfig() = default;

    UnorderedReaderConfig(size_t num_threads,
                          size_t max_requests,
                          size_t queue_depth) : num_threads(num_threads),
                                                max_requests(max_requests),
                                                queue_depth(queue_depth) {}
};

/**
 * Read a list of files in no particular order. File sizes are assumed to be multiples of O_DIRECT_MULTIPLE and
 * all bytes in the file are plain data (no end-of-file size markers are used and there are not padding bytes).
 *
 * @tparam T The data type to be read from the file.
 */
template<typename T, size_t READ_SIZE = READER_READ_SIZE>
class UnorderedFileReader {
public:
    typedef std::tuple<T *, size_t, size_t, size_t> BufferData;

    struct ReaderAllocator {
        static constexpr size_t INITIAL_BUFFER_COUNT = 100;
        static constexpr size_t ALLOCATION_BUFFERS = 100;
        static constexpr size_t ALLOCATION_THRESHOLD = 20;

        std::vector<T *> free_list;
        // Only one thread can touch the free list at a time
        std::mutex free_list_lock;
        // Only one thread can perform memory allocation at a time
        std::mutex allocation_lock;
        // Memory allocations. This should be much smaller
        std::vector<T *> allocations;

        ReaderAllocator(size_t num_buffers = INITIAL_BUFFER_COUNT) {
            AllocateMoreMemory(INITIAL_BUFFER_COUNT);
        }

        ~ReaderAllocator() {
            for (T *ptr: allocations) {
                free(ptr);
            }
        }

        void AllocateMoreMemory(size_t num_pointers) {
            std::lock_guard<std::mutex> alloc_lock(allocation_lock);
            if (free_list.size() > ALLOCATION_THRESHOLD) {
                // Some other thread has already done the allocation
                // FIXME: this size call leads to a data race
                return;
            }
            T *ptr = (T *) std::aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, READ_SIZE * num_pointers);
            {
                std::lock_guard<std::mutex> lock(free_list_lock);
                for (size_t i = 0; i < num_pointers; i++) {
                    free_list.push_back((T *) ((intptr_t) ptr + i * READ_SIZE));
                }
            }
            allocations.push_back(ptr);
        }

        T *Alloc() {
            while (true) {
                std::unique_lock<std::mutex> lock(free_list_lock);
                if (!free_list.empty()) {
                    auto ret = free_list.back();
                    free_list.pop_back();
                    return ret;
                }
                // Free list is empty
                lock.unlock();
                AllocateMoreMemory(ALLOCATION_BUFFERS);
            }
        }

        void Free(T *ptr) {
            std::lock_guard<std::mutex> lock(free_list_lock);
            free_list.push_back(ptr);
        }
    };

    ReaderAllocator allocator;

    /**
     * Creates the object. Note that you should call PrepFiles and Start to really begin the file reader.
     *
     * @param buffer_queue_size Size of the buffer queue.
     * Larger values may improve performance at the cost of increased memory usage
     */
    explicit UnorderedFileReader() = default;

    ~UnorderedFileReader() {
        is_open = false;
        Wait();
    }

    void PrepFiles(const std::string &prefix) {
        files = FindFiles(prefix);
    }

    void PrepFiles(const std::vector<FileInfo> &file_list) {
        this->files = file_list;
    }

    void Start(const UnorderedReaderConfig &config = UnorderedReaderConfig()) {
        CHECK(config.num_threads > 0) << "Need at least 1 thread";
        buffer_queue.SetSizeLimit(config.buffer_queue_size);
        active_threads = (int) config.num_threads;
        for (size_t i = 0; i < config.num_threads; i++) {
            std::vector<FileInfo> file_list;
            for (size_t j = i; j < files.size(); j += config.num_threads) {
                file_list.push_back(files[j]);
            }
            worker_threads.push_back(
                    std::make_unique<std::thread>(RunFileReaderWorker, this, std::move(file_list),
                                                  config.queue_depth, config.max_requests,
                                                  config.active_chunks_per_file));
        }
    }

    /**
     * Add new piece of data to the buffer
     *
     * @param data
     * @param size
     */
    void Push(T *data, size_t size, size_t file_index, size_t data_index) {
        CHECK((size_t) data % O_DIRECT_MULTIPLE == 0) << "Buffers used by the UnorderedFileReader must be aligned.";
        buffer_queue.Push({data, size, file_index, data_index});
    }

    /**
     * Get a piece of data from the buffer. This call will hang until
     *   (1) a piece of data is available
     *   (2) the reader is closed (because some other thread closed it or because it ran out of data to read)
     * This function is thread-safe.
     *
     * @param timeout_microseconds The interval for checking whether the reader is still open.
     * @return pointer to a piece of data and its size if one is available; nullptr and 0 if the reader is closed
     *   and no longer has any data
     */
    BufferData Poll() {
        static BufferData default_result(nullptr, 0, 0, 0);
        return buffer_queue.Poll(default_result).first;
    }

    void Close() {
        buffer_queue.Close();
    }

    /**
     * Block until file reader finishes
     */
    void Wait() {
        for (auto &t: worker_threads) {
            if (t->joinable()) {
                t->join();
            }
        }
    }

    /**
     * Reset the reader for reuse after the previous Start() has finished
     * draining all files. Joins workers, reopens the buffer queue, and
     * leaves the allocator memory pool and `files` list intact so the next
     * Start() doesn't reallocate or rediscover files.
     */
    void Reset() {
        Wait();
        worker_threads.clear();
        active_threads = 0;
        is_open = true;
        buffer_queue.Reopen();
    }

private:
    // whether the file reader is actively running
    bool is_open = true;
    std::atomic<int> active_threads = 0;
    // list of files to read
    std::vector<FileInfo> files;
    // files that are available to tbe reused
    std::vector<FileInfo> available_files;
    // a single worker thread for managing file reading
    std::vector<std::unique_ptr<std::thread>> worker_threads;
    // a buffer queue containing data read from disk
    SimpleQueue<BufferData> buffer_queue;

    /**
     * A file that is currently being read
     */
    struct OpenedFile {
        int fd;
        size_t bytes_issued = 0;        // next file offset to consider for issue
        size_t chunks_active = 0;       // total chunks we plan to actually read
        size_t chunks_completed = 0;    // active chunks whose CQE has been reaped
        size_t file_size;
        const size_t file_index;
        // Optional per-file bitmap: bit i set => chunk i should be read.
        // If null, every chunk is active.
        const uint64_t* active_bits = nullptr;
        size_t active_bits_words = 0;

        OpenedFile(const std::string &name, size_t file_size, size_t file_index)
                : file_size(file_size), file_index(file_index) {
            fd = open(name.c_str(), O_DIRECT | O_RDONLY);
            SYSCALL(fd);
        }

        explicit OpenedFile(const FileInfo &info)
                : OpenedFile(info.file_name, info.file_size, info.file_index) {
        }

        ~OpenedFile() {
            SYSCALL(close(fd));
        }

        // Total number of chunks the file would have (rounded up).
        inline size_t num_chunks() const {
            return (file_size + READ_SIZE - 1) / READ_SIZE;
        }

        inline bool chunk_active(size_t idx) const {
            if (active_bits == nullptr) return true;
            size_t w = idx >> 6;
            if (w >= active_bits_words) return false;
            return (active_bits[w] >> (idx & 63)) & 1ULL;
        }

        // Advance bytes_issued past inactive chunks. Returns true iff the
        // chunk at the (new) bytes_issued is active and within file bounds.
        inline bool advance_to_active() {
            while (bytes_issued < file_size) {
                size_t idx = bytes_issued / READ_SIZE;
                if (chunk_active(idx)) return true;
                bytes_issued += std::min((size_t) READ_SIZE, file_size - bytes_issued);
            }
            return false;
        }
    };

    struct ReadRequest {
        size_t offset, read_size;
        OpenedFile *file;
        T *data;
    };

    static void RunFileReaderWorker(UnorderedFileReader *reader,
                                    std::vector<FileInfo> &&all_files,
                                    const size_t io_uring_size,
                                    const size_t max_outstanding_requests,
                                    const std::vector<std::vector<uint64_t>>* active_chunks_per_file) {
        struct io_uring ring;
        SYSCALL(io_uring_queue_init(io_uring_size, &ring, IORING_SETUP_SINGLE_ISSUER));

        std::deque<OpenedFile *> available_files;
        std::vector<OpenedFile *> completed_files;
        for (auto &file: all_files) {
            auto *f = new OpenedFile(file);
            if (active_chunks_per_file != nullptr &&
                file.file_index < active_chunks_per_file->size()) {
                const auto &bits = (*active_chunks_per_file)[file.file_index];
                f->active_bits = bits.data();
                f->active_bits_words = bits.size();
            }
            // Count active chunks to know the completion target. With no
            // bitmap supplied this is just the chunk count of the file.
            size_t total = f->num_chunks();
            if (f->active_bits == nullptr) {
                f->chunks_active = total;
            } else {
                size_t cnt = 0;
                for (size_t c = 0; c < total; ++c) {
                    if (f->chunk_active(c)) cnt++;
                }
                f->chunks_active = cnt;
            }
            if (f->chunks_active == 0) {
                completed_files.push_back(f);
            } else {
                available_files.push_back(f);
            }
        }

        size_t outstanding_requests = 0;
        auto *request_pool = (ReadRequest *) malloc(max_outstanding_requests * sizeof(ReadRequest));
        std::vector<ReadRequest *> available_requests;
        available_requests.reserve(max_outstanding_requests);
        for (size_t i = 0; i < max_outstanding_requests; i++) {
            available_requests.push_back(request_pool + i);
        }

        // if we haven't finished all the files and the reader is still open, keep looping
        while (completed_files.size() < all_files.size() && reader->is_open) {
            // reap io_uring result until there's nothing to reap
            while (outstanding_requests > 0) {
                // wait for 0 seconds for the cqe
                struct io_uring_cqe *cqe;
                int wait_result = io_uring_peek_cqe(&ring, &cqe);
                if (wait_result == 0) {
                    // process this cqe
                    SYSCALL(cqe->res);
                    auto *request = (ReadRequest *) io_uring_cqe_get_data(cqe);
                    // add data to buffer queue
                    reader->Push(request->data, request->read_size / sizeof(T),
                                 request->file->file_index,
                                 request->offset / sizeof(T));
                    auto *file = request->file;
                    file->chunks_completed++;
                    if (file->chunks_completed == file->chunks_active) {
                        completed_files.push_back(file);
                    }
                    available_requests.push_back(request);
                    outstanding_requests--;
                    io_uring_cqe_seen(&ring, cqe);
                } else {
                    // nothing to reap, break
                    break;
                }
            }
            // keep submitting new read requests to the io_uring until we are about to exceed to max size of the buffer
            while (!available_requests.empty() && !available_files.empty()) {
                auto file = available_files.front();
                available_files.pop_front();

                if (!file->advance_to_active()) {
                    // No more active chunks to issue. Outstanding completions
                    // will mark this file complete via the reap path. Don't
                    // push it back to available_files.
                    continue;
                }

                auto request = available_requests.back();
                available_requests.pop_back();
                request->file = file;
                request->offset = file->bytes_issued;
                auto read_size = std::min(READ_SIZE, file->file_size - file->bytes_issued);
                request->read_size = read_size;
                request->data = reader->allocator.Alloc();

                // issue a read on an opened file
                struct io_uring_sqe *sqe;
                sqe = io_uring_get_sqe(&ring);
                if (sqe == nullptr) {
                    LOG(ERROR) << "Request obtained but is unable to be fulfilled because the io_uring buffer is full.";
                    break;
                }

                // perform write
                io_uring_prep_read(sqe, file->fd, request->data, read_size, file->bytes_issued);
                io_uring_sqe_set_data(sqe, request);
                io_uring_submit(&ring);

                file->bytes_issued += read_size;
                // keep reusing this file if it may have more active chunks left
                if (file->bytes_issued < file->file_size) {
                    available_files.push_back(file);
                }
                outstanding_requests++;
            }
        }
        // cleanup
        io_uring_queue_exit(&ring);

        free(request_pool);
        CHECK(completed_files.size() == all_files.size())
                        << "Expected all files to be read when reader thread terminates: "
                        << all_files.size() << " files total, yet "
                        << completed_files.size() << " files are completed.";
        for (auto &file: completed_files) {
            delete file;
        }

        reader->active_threads--;
        if (reader->active_threads == 0) {
            reader->Close();
        }
    }
};

#endif //SORTING_UNORDERED_FILE_READER_H
