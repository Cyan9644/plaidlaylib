//
// Created by peter on 5/23/24.
//

#ifndef SORTING_RANDOM_READ_H
#define SORTING_RANDOM_READ_H

#include <vector>

#include "liburing.h"
#include "absl/log/log.h"
#include "parlay/sequence.h"

#include "configs.h"
#include "utils/file_utils.h"
#include "utils/logger.h"

constexpr size_t GetRandomBatchReadBufferSize(size_t size) {
    return AlignUp(size + O_DIRECT_MULTIPLE - 1);
}

template<typename T>
parlay::sequence<T> RandomBatchRead(const std::vector<FileInfo> &files,
                                    const parlay::sequence<size_t> &requests) {
    const size_t num_files = files.size();
    size_t size_prefix_sum[num_files];
    for (size_t i = 0; i < num_files; i++) {
        auto prev = i == 0 ? 0 : size_prefix_sum[i - 1];
        size_prefix_sum[i] = files[i].true_size + prev;
        if (files[i].true_size == 0) {
            [[unlikely]];
            LOG(ERROR) << "File's true size is not determined or an empty file is passed: " << files[i].file_name;
        }
    }

    struct ReadRequest {
        size_t offset;
        unsigned char buffer[GetRandomBatchReadBufferSize(sizeof(T))];
    };

    // open all files in parallel; could be sped up using async open calls, but performance gain is probably negligible
    int fds[num_files];
    parlay::parallel_for(0, num_files, [&](size_t i) {
        fds[i] = open(files[i].file_name.c_str(), O_RDONLY | O_DIRECT);
        SYSCALL(fds[i]);
    }, 1);

    const size_t num_threads = parlay::num_workers();
    const size_t segment_size = (requests.size() + num_threads - 1) / num_threads;

    return parlay::flatten(parlay::map(parlay::iota(num_threads), [&](size_t segment) {
        const size_t segment_start = segment_size * segment;
        const size_t segment_end = std::min(segment_size * (segment + 1), requests.size());
        if (segment_end <= segment_start) {
            return parlay::sequence<T>();
        }

        const size_t NUM_BUFFERS = 4096 * 4;
        auto *buffers = (ReadRequest *) malloc(sizeof(ReadRequest) * NUM_BUFFERS);
        std::vector<size_t> free_buffers;
        for (size_t i = 0; i < NUM_BUFFERS; i++) {
            free_buffers.push_back(i);
        }

        parlay::sequence<T> results;
        results.reserve(segment_end - segment_start);

        // If there are too many threads, io_uring_queue_init will fail due to insufficient locked memory
        const unsigned IO_URING_ENTRIES = 512;
        struct io_uring ring;
        SYSCALL(io_uring_queue_init(IO_URING_ENTRIES, &ring, IORING_SETUP_SINGLE_ISSUER));
        size_t i = segment_start, pending_requests = 0, requests_in_ring = 0;
        while (i < segment_end || requests_in_ring > 0) {
            // there are available buffers and remaining requests; keep submitting
            while (i < segment_end && !free_buffers.empty()) {
                auto byte_offset = requests[i] * sizeof(T);
                // Start and end of aligned read (both multiples of O_DIRECT_MULTIPLE)
                size_t start = AlignDown(byte_offset), end = AlignUp(byte_offset + sizeof(T));
                size_t buffer_index = free_buffers.back();
                free_buffers.pop_back();
                buffers[buffer_index].offset = byte_offset - start;
                auto file_num = std::upper_bound(size_prefix_sum, size_prefix_sum + num_files, start) - size_prefix_sum;
                CHECK((size_t)file_num < num_files);
                struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
                io_uring_prep_read(sqe, fds[file_num], buffers[buffer_index].buffer, end - start,
                                   file_num == 0 ? start : start - size_prefix_sum[file_num - 1]);
                io_uring_sqe_set_data(sqe, reinterpret_cast<void *>(buffer_index));
                i++;
                pending_requests++;
                if (pending_requests == IO_URING_ENTRIES || i == segment_end) {
                    int submitted = io_uring_submit(&ring);
                    SYSCALL(submitted);
                    requests_in_ring += submitted;
                    pending_requests = 0;
                }
            }
            // at this point, we have submitted as much as we can; now is time to reap
            bool wait = true;
            while (requests_in_ring > 0) {
                struct io_uring_cqe *cqe;
                if (wait) {
                    SYSCALL(io_uring_wait_cqe(&ring, &cqe));
                    wait = false;
                } else {
                    int result = io_uring_peek_cqe(&ring, &cqe);
                    if (result != 0) {
                        break;
                    }
                }
                SYSCALL(cqe->res);
                io_uring_cqe_seen(&ring, cqe);
                requests_in_ring--;
                auto buffer_index = reinterpret_cast<size_t>(io_uring_cqe_get_data(cqe));
                free_buffers.push_back(buffer_index);
                ReadRequest *buffer = &buffers[buffer_index];
                results.push_back(*reinterpret_cast<T *>(buffer->buffer + buffer->offset));
            }
        }
        io_uring_queue_exit(&ring);
        free(buffers);

        return results;
    }, 1));
}

#endif //SORTING_RANDOM_READ_H
