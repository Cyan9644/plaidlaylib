//
// Created by peter on 3/18/24.
//

#ifndef SORTING_ORDERED_FILE_WRITER_H
#define SORTING_ORDERED_FILE_WRITER_H

#include <fcntl.h>
#include <sys/uio.h>
#include <memory>
#include <vector>
#include <string>
#include <mutex>
#include <liburing.h>

#include "absl/log/log.h"
#include "parlay/parallel.h"
#include "parlay/alloc.h"

#include "utils/logger.h"
#include "configs.h"
#include "utils/file_info.h"
#include "utils/file_utils.h"
#include "utils/simple_queue.h"
#include "utils/type_allocator.h"

/**
 * Not really an ordered file writer. This class creates many buckets. Each bucket corresponds to a file.
 * Data sent to the same file are not written in order, but the order of the buckets are respected.
 *
 * @tparam T
 */
template<typename T, size_t BucketSize>
class OrderedFileWriter {
    struct Bucket;
    struct IOVectorRequest;
public:
    /**
     * If the default constructor is used, need to call Initialize later
     */
    OrderedFileWriter() = default;

    OrderedFileWriter(const std::string &prefix, size_t num_buckets, size_t flush_threshold) {
        Initialize(prefix, num_buckets, flush_threshold);
    }

    ~OrderedFileWriter() {
        CleanUp();
        for (size_t i = 0; i < num_buckets; i++) {
            buckets[i].~Bucket();
        }
        free(buckets);
        free(requests);
    }

    // For debugging: setting it to true will skip the write to disk step and deallocate right before
    bool skip_write = false;

    static void RunIOThread(OrderedFileWriter<T, SAMPLE_SORT_BUCKET_SIZE> *writer) {
        auto completions = &writer->free_requests;
        auto pending_requests = &writer->pending_requests;
        unsigned int requests_in_ring = 0;
        struct io_uring ring;
        const unsigned RING_DEPTH = 128;
        io_uring_queue_init(RING_DEPTH, &ring, IORING_SETUP_SINGLE_ISSUER);
        bool has_more_requests = true;
        while (has_more_requests || requests_in_ring > 0) {
            bool reap_required = requests_in_ring >= RING_DEPTH || !has_more_requests || completions->IsEmptyUnsafe();
            while (requests_in_ring > 0) {
                io_uring_cqe *cqe;
                if (reap_required) {
                    SYSCALL(io_uring_wait_cqe(&ring, &cqe));
                } else {
                    int res = io_uring_peek_cqe(&ring, &cqe);
                    if (res != 0) {
                        break;
                    }
                }
                io_uring_cqe_seen(&ring, cqe);
                SYSCALL(cqe->res);
                auto completed_request = (IOVectorRequest *) io_uring_cqe_get_data(cqe);
                requests_in_ring--;
                completed_request->Reset();
                completions->Push(completed_request);
                // at least one reap is done at this point; further reaps are optional
                reap_required = false;
            }
            bool need_submit = false;
            while(requests_in_ring < RING_DEPTH) {
                auto [request, code] = pending_requests->Poll(nullptr, 0);
                if (request == nullptr) {
                    if (code == QueueCode::FINISH) {
                        has_more_requests = false;
                    }
                    break;
                }
                if (writer->skip_write) {
                    IOVectorRequest *r = request;
                    r->Reset();
                    completions->Push(r);
                    continue;
                }
                io_uring_sqe *sqe = io_uring_get_sqe(&ring);
                if (sqe == nullptr) {
                    [[unlikely]]
                    LOG(ERROR) << "io_uring does not have enough sqe";
                    return;
                }
                io_uring_sqe_set_data(sqe, request);
                io_uring_prep_writev(sqe,
                                     request->fd,
                                     &request->io_vectors[0],
                                     request->iovec_count,
                                     request->offset);
                requests_in_ring++;
                need_submit = true;
            }
            if (need_submit) {
                SYSCALL(io_uring_submit(&ring));
            }
        }
        io_uring_queue_exit(&ring);
    }

    inline IOVectorRequest* NewRequest(int fd, size_t offset) {
        auto *r = free_requests.Poll().first;
        r->fd = fd;
        r->offset = offset;
        return r;
    }

    inline void SubmitRequest(IOVectorRequest* r) {
        pending_requests.Push(r);
    }

    /**
     * Initialize the file writer
     *
     * @param prefix Prefix of output file names
     * @param bucket_count Number of buckets to create (they're numbered starting from 0)
     * @param file_flush_threshold The threshold (in bytes) at which a chunk of data is sent to disk
     */
    void Initialize(const std::string &prefix, size_t bucket_count,
                    size_t file_flush_threshold, size_t request_pool_size = -1) {
        this->num_buckets = bucket_count;
        this->io_threshold = file_flush_threshold;
        // FIXME: adjust this
        request_pool_size = bucket_count * 10;

        requests = (IOVectorRequest*)malloc(request_pool_size * sizeof(IOVectorRequest));
        for (size_t i = 0;i < request_pool_size; i++) {
            IOVectorRequest *r = requests + i;
            new(r) IOVectorRequest();
            free_requests.Push(r);
        }

        // allocate memory for all buckets
        buckets = (Bucket*)malloc(bucket_count * sizeof(Bucket));
        for (size_t i = 0; i < bucket_count; i++) {
            std::string f_name = GetFileName(prefix, i);
            // do a placement new since the std::mutex in a BucketData can't be copied or moved
            new(&buckets[i]) Bucket(f_name);
            buckets[i].request = NewRequest(buckets[i].current_file, 0);
            // construct the result file; file sizes will be filled in later
            result_files.emplace_back(f_name, i, 0, 0);
        }
    }

    /**
     * Clean up half-full buckets and misaligned writes in each bucket. Must be called before writer exits to ensure
     * that everything is written to disk. This should not be called concurrently.
     */
    void CleanUp() {
        if (cleanup_started) {
            return;
        }
        cleanup_started = true;
        parlay::parallel_for(0, num_buckets, [&](size_t i) {
            auto [buffer, buffer_size, true_size] = buckets[i].GatherMisalignedPointers();
            Bucket *bucket = &buckets[i];
            IOVectorRequest *request = bucket->request;
            request->last_request = true;
            request->AddPointer((T*)buffer, buffer_size);
            bucket->file_size += request->current_size;
            SubmitRequest(request);
            result_files[i].file_size = bucket->file_size;
            result_files[i].true_size = bucket->file_size - buffer_size + true_size;
        }, 1);
        // no more requests sent to the writer
        pending_requests.Close();
    }

    /**
     * Write data to a particular bucket in the writer.
     *
     * Do not submit arrays whose size is not a multiple of O_DIRECT_MULTIPLE unless necessary.
     * @param bucket_number
     * @param pointer Ownership of pointer will now be assumed by the writer, which will deallocate it using Free.
     * @param count
     */
    void Write(size_t bucket_number, T *pointer, size_t count) {
        CHECK(bucket_number < num_buckets) << "Invalid bucket number";
        CHECK((size_t)pointer % O_DIRECT_MULTIPLE == 0) << "Write buffers must be aligned";
        Bucket *bucket = &buckets[bucket_number];
        auto size = count * sizeof(T);
        std::unique_lock<std::mutex> bucket_lock(bucket->lock);
        // misaligned pointer need special treatment since they won't fit in a iovec
        if (size % O_DIRECT_MULTIPLE != 0) {
            bucket->misaligned_pointers.emplace_back(pointer, count);
            return;
        }
        IOVectorRequest *request = bucket->request;
        request->AddPointer(pointer, size);
        if (request->current_size >= io_threshold || request->iovec_count >= IO_VECTOR_SIZE) {
            bucket->file_size += request->current_size;
            bucket->request = NewRequest(bucket->current_file, bucket->file_size);
            bucket_lock.unlock();
            SubmitRequest(request);
        }
    }

    [[nodiscard]] std::vector<FileInfo> ReapResult() {
        CleanUp();
        return std::move(result_files);
    }

private:
    bool cleanup_started = false;
    size_t num_buckets = 0;
    std::vector<FileInfo> result_files;
    Bucket *buckets = nullptr;
    IOVectorRequest *requests = nullptr;

    // should contain requests that are reset and ready to be reused
    SimpleQueue<IOVectorRequest*> free_requests;
    SimpleQueue<IOVectorRequest*> pending_requests;

    size_t io_threshold = 4 << 20;

    struct BucketData {
        char data[BucketSize];
    };
    using BucketAllocator = AlignedTypeAllocator<BucketData, O_DIRECT_MULTIPLE>;

    struct IOVectorRequest {
        bool last_request = false;
        int fd = -1;
        size_t offset = -1;
        size_t current_size = 0;
        // FIXME: add doc on why using iovec
        iovec io_vectors[IO_VECTOR_SIZE];
        uint32_t iovec_count = 0;

        IOVectorRequest() = default;

        void AddPointer(T* pointer, size_t size) {
            io_vectors[iovec_count].iov_base = pointer;
            io_vectors[iovec_count].iov_len = size;
            iovec_count += 1;
            current_size += size;
        }

        void Reset() {
            for (size_t i = 0; i < iovec_count - (last_request ? 1 : 0); i++) {
                BucketAllocator::free(reinterpret_cast<BucketData*>(io_vectors[i].iov_base));
            }
            if (last_request) {
                free(io_vectors[iovec_count - 1].iov_base);
            }
            current_size = 0;
            iovec_count = 0;
        }
    };

    struct Bucket {
        IOVectorRequest *request;
        int current_file;
        size_t true_file_size, file_size = 0;
        std::mutex lock;
        std::vector<std::pair<T *, size_t>> misaligned_pointers;

        Bucket() = delete;

        explicit Bucket(std::string &file_name) {
            current_file = open(file_name.c_str(), O_WRONLY | O_DIRECT | O_CREAT, 0644);
            SYSCALL(current_file);
        }

        ~Bucket() {
            SYSCALL(close(current_file));
        }

        /**
         * Write misaligned pointers (those whose size cannot be divided by O_DIRECT_MULTIPLE) to the file.
         *
         * Copies the content in each pointer into a buffer and then writes the buffer to disk using
         * synchronous IO.
         */
        std::tuple<unsigned char*, size_t, size_t> GatherMisalignedPointers() {
            size_t write_size = 0;
            for (size_t i = 0; i < misaligned_pointers.size(); i++) {
                auto [pointer, count] = misaligned_pointers[i];
                size_t pointer_size = count * sizeof(T);
                write_size += pointer_size;
            }
            // Divide and always round up because we need 2 more bytes than write_size
            size_t target_write_size = (write_size / O_DIRECT_MULTIPLE + 1) * O_DIRECT_MULTIPLE;
            if (target_write_size - write_size < METADATA_SIZE) {
                target_write_size += O_DIRECT_MULTIPLE;
            }
            auto *write_buffer = (unsigned char *)std::aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, target_write_size);
            size_t buffer_position = 0;
            for (size_t i = 0; i < misaligned_pointers.size(); i++) {
                auto [pointer, count] = misaligned_pointers[i];
                size_t pointer_size = count * sizeof(T);
                memcpy(write_buffer + buffer_position, pointer, pointer_size);
                BucketAllocator::free((BucketData*)pointer);
                buffer_position += pointer_size;
            }
            MakeFileEndMarker(write_buffer, target_write_size, write_size);
            // compute true file size (excluding garbage bytes and marker at the end) and file size on disk
            return {write_buffer, target_write_size, write_size};
        }
    };
};

#endif //SORTING_ORDERED_FILE_WRITER_H
