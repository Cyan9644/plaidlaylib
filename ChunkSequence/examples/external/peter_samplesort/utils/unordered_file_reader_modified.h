//
// Created by peter on 3/2/24.
//
// Modified variant of UnorderedChunkReader that reads a list of *chunks*
// (described by chunk_header-like records) instead of whole files. This is the
// reader used by Plaidlay/externalFilter.h.
//
// Differences from the original utils/unordered_file_reader.h:
//   * PrepFiles accepts a vector of chunk headers (duck-typed: each element must
//     expose .filename, .begin_address, .used, and .index). Each chunk is one
//     independent read of `used` bytes at `begin_address` within `filename`.
//   * Poll() returns a 6-tuple
//         (data_ptr, valid_element_count, file_index, index, which_chunk, filename)
//     where `index` is the chunk_header's global ordering key carried through
//     unchanged (so a downstream filter can restore a stable order), and
//     `filename`/`which_chunk` identify the source chunk.
//
// The on-disk convention (set by externalFilter.h) is that every block occupies
// a full READ_SIZE region on disk and `used` records how many bytes of it are
// valid. We therefore read only AlignUp(used) bytes and report `used/sizeof(T)`
// valid elements.

#ifndef SORTING_UNORDERED_FILE_READER_MODIFIED_H
#define SORTING_UNORDERED_FILE_READER_MODIFIED_H

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
#include <string>
#include <fcntl.h>

struct UnorderedChunkReaderConfig {
    size_t num_threads = 2;
    size_t max_requests = 64;
    size_t queue_depth = 32;
    size_t buffer_queue_size = 1024;

    UnorderedChunkReaderConfig() = default;

    UnorderedChunkReaderConfig(size_t num_threads,
                          size_t max_requests,
                          size_t queue_depth) : num_threads(num_threads),
                                                max_requests(max_requests),
                                                queue_depth(queue_depth) {}
};

/**
 * Read a list of chunks in no particular order. Each chunk is described by a
 * chunk_header (filename, begin_address, used, index) and is read independently.
 *
 * @tparam T The data type to be read from the file.
 * @tparam READ_SIZE The size (bytes) of a single on-disk block. Must match the
 *   block size used when the chunks were written (4 MiB for externalFilter).
 */
template<typename T, size_t READ_SIZE = READER_READ_SIZE>
class UnorderedChunkReader {
public:
    // (data, valid_element_count, file_index, index, which_chunk, filename)
    typedef std::tuple<T *, size_t, size_t, size_t, size_t, std::string> BufferData;

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

    explicit UnorderedChunkReader() = default;

    ~UnorderedChunkReader() {
        is_open = false;
        Wait();
    }

    /**
     * Prepare the reader with a list of chunk headers. Duck-typed: ChunkHeader
     * must expose .filename, .begin_address, .used, and .index members. Each
     * header becomes one independent read.
     */
    template<typename Headers>
    void PrepFiles(const Headers &headers) {
        chunks.clear();
        chunks.reserve(headers.size());
        for (const auto &h: headers) {
            chunks.push_back(ChunkRequest{h.filename, h.begin_address, h.used, h.index});
        }
    }

    void Start(const UnorderedChunkReaderConfig &config = UnorderedChunkReaderConfig()) {
        CHECK(config.num_threads > 0) << "Need at least 1 thread";
        buffer_queue.SetSizeLimit(config.buffer_queue_size);
        active_threads = (int) config.num_threads;
        for (size_t i = 0; i < config.num_threads; i++) {
            std::vector<ChunkRequest> chunk_list;
            for (size_t j = i; j < chunks.size(); j += config.num_threads) {
                chunk_list.push_back(chunks[j]);
            }
            worker_threads.push_back(
                    std::make_unique<std::thread>(RunFileReaderWorker, this, std::move(chunk_list),
                                                  config.queue_depth, config.max_requests));
        }
    }

    /**
     * Enqueue a piece of data for consumers. `size` is the number of *valid*
     * elements; `index` is the chunk's global ordering key.
     */
    void Push(T *data, size_t size, size_t file_index, size_t index,
              size_t which_chunk, std::string filename) {
        CHECK((size_t) data % O_DIRECT_MULTIPLE == 0) << "Buffers used by the UnorderedChunkReader must be aligned.";
        buffer_queue.Push({data, size, file_index, index, which_chunk, std::move(filename)});
    }

    /**
     * Get a piece of data from the buffer. Blocks until data is available or the
     * reader is closed and drained, in which case it returns
     * (nullptr, 0, 0, 0, 0, "").
     */
    BufferData Poll() {
        static BufferData default_result(nullptr, 0, 0, 0, 0, std::string());
        return buffer_queue.Poll(default_result).first;
    }

    void Close() {
        buffer_queue.Close();
    }

    void Wait() {
        for (auto &t: worker_threads) {
            if (t->joinable()) {
                t->join();
            }
        }
    }

    /**
     * Reset the reader for reuse after the previous Start() has finished
     * draining all chunks.
     */
    void Reset() {
        Wait();
        worker_threads.clear();
        active_threads = 0;
        is_open = true;
        buffer_queue.Reopen();
    }

private:
    // One unit of work: read `used` bytes at `begin_address` of `filename`.
    struct ChunkRequest {
        std::string filename;
        size_t begin_address;
        size_t used;   // valid bytes
        size_t index;  // global ordering key
    };

    // POD request handed to io_uring. Kept trivially constructible because the
    // request pool is malloc'd (no constructors are run). All per-chunk metadata
    // is recovered from `my_chunks`[chunk_idx] at completion time.
    struct ReadRequest {
        size_t offset;
        size_t read_size;
        size_t valid_elems;
        size_t chunk_idx;
        int fd;
        T *data;
    };

    // whether the file reader is actively running
    bool is_open = true;
    std::atomic<int> active_threads = 0;
    // chunks to read
    std::vector<ChunkRequest> chunks;
    // worker threads managing file reading
    std::vector<std::unique_ptr<std::thread>> worker_threads;
    // a buffer queue containing data read from disk
    SimpleQueue<BufferData> buffer_queue;

    static void RunFileReaderWorker(UnorderedChunkReader *reader,
                                    std::vector<ChunkRequest> &&my_chunks,
                                    const size_t io_uring_size,
                                    const size_t max_outstanding_requests) {
        struct io_uring ring;
        SYSCALL(io_uring_queue_init(io_uring_size, &ring, IORING_SETUP_SINGLE_ISSUER));

        // Open each distinct source file once; reuse the fd across its chunks.
        std::map<std::string, int> open_fds;
        auto get_fd = [&](const std::string &name) -> int {
            auto it = open_fds.find(name);
            if (it != open_fds.end()) {
                return it->second;
            }
            int fd = open(name.c_str(), O_DIRECT | O_RDONLY);
            SYSCALL(fd);
            open_fds.emplace(name, fd);
            return fd;
        };

        size_t outstanding_requests = 0;
        size_t next_chunk = 0;
        size_t completed = 0;
        auto *request_pool = (ReadRequest *) malloc(max_outstanding_requests * sizeof(ReadRequest));
        std::vector<ReadRequest *> available_requests;
        available_requests.reserve(max_outstanding_requests);
        for (size_t i = 0; i < max_outstanding_requests; i++) {
            available_requests.push_back(request_pool + i);
        }

        while (completed < my_chunks.size() && reader->is_open) {
            // reap completed reads
            while (outstanding_requests > 0) {
                struct io_uring_cqe *cqe;
                int wait_result = io_uring_peek_cqe(&ring, &cqe);
                if (wait_result == 0) {
                    SYSCALL(cqe->res);
                    auto *request = (ReadRequest *) io_uring_cqe_get_data(cqe);
                    const auto &c = my_chunks[request->chunk_idx];
                    reader->Push(request->data, request->valid_elems,
                                 request->chunk_idx, c.index,
                                 c.begin_address / READ_SIZE, c.filename);
                    available_requests.push_back(request);
                    outstanding_requests--;
                    completed++;
                    io_uring_cqe_seen(&ring, cqe);
                } else {
                    break;
                }
            }
            // issue new reads
            while (!available_requests.empty() && next_chunk < my_chunks.size()) {
                const auto &c = my_chunks[next_chunk];
                size_t valid_elems = c.used / sizeof(T);

                // An empty chunk (nothing survived a previous pass) needs no read;
                // hand back a non-null, zero-length buffer so the consumer still
                // sees it as a real (empty) chunk and preserves its slot/order.
                if (c.used == 0) {
                    T *data = reader->allocator.Alloc();
                    reader->Push(data, 0, next_chunk, c.index,
                                 c.begin_address / READ_SIZE, c.filename);
                    completed++;
                    next_chunk++;
                    continue;
                }

                // Read only the valid portion, rounded up for O_DIRECT. This is
                // <= READ_SIZE (the allocator buffer size) because used <= READ_SIZE.
                size_t read_size = AlignUp(c.used);

                auto *request = available_requests.back();
                available_requests.pop_back();
                request->fd = get_fd(c.filename);
                request->offset = c.begin_address;
                request->read_size = read_size;
                request->valid_elems = valid_elems;
                request->chunk_idx = next_chunk;
                request->data = reader->allocator.Alloc();

                struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
                if (sqe == nullptr) {
                    LOG(ERROR) << "Request obtained but is unable to be fulfilled because the io_uring buffer is full.";
                    reader->allocator.Free(request->data);
                    available_requests.push_back(request);
                    break;
                }

                io_uring_prep_read(sqe, request->fd, request->data, read_size, c.begin_address);
                io_uring_sqe_set_data(sqe, request);
                io_uring_submit(&ring);

                outstanding_requests++;
                next_chunk++;
            }
        }
        // cleanup
        io_uring_queue_exit(&ring);
        free(request_pool);
        for (auto &kv: open_fds) {
            SYSCALL(close(kv.second));
        }

        CHECK(completed == my_chunks.size())
                        << "Expected all chunks to be read when reader thread terminates: "
                        << my_chunks.size() << " chunks total, yet "
                        << completed << " chunks are completed.";

        reader->active_threads--;
        if (reader->active_threads == 0) {
            reader->Close();
        }
    }
};

#endif //SORTING_UNORDERED_FILE_READER_MODIFIED_H
