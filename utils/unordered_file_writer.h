//
// Created by peter on 3/2/24.
//

#ifndef SORTING_UNORDERED_FILE_WRITER_H
#define SORTING_UNORDERED_FILE_WRITER_H

#include "utils/logger.h"
#include "configs.h"
#include "utils/file_utils.h"
#include "utils/simple_queue.h"
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

struct UnorderedWriterConfig {
    size_t io_uring_size = IO_URING_BUFFER_SIZE;
    size_t queue_size = 1000;
    size_t num_threads = 1;
    // Needed if only a file prefix is provided.
    // If the list of file names is supplied, this is ignored.
    size_t num_files = SSD_COUNT;
    bool allow_expand = false;

    UnorderedWriterConfig() = default;

    UnorderedWriterConfig(size_t io_uring_size,
                          size_t queue_size,
                          size_t num_threads,
                          size_t num_files) : io_uring_size(io_uring_size),
                                              queue_size(queue_size),
                                              num_threads(num_threads),
                                              num_files(num_files) {}
};

template<typename T>
class UnorderedFileWriter {
private:
    struct WriteRequest;
    struct OpenedFile;
public:

    UnorderedFileWriter() = default;

    explicit UnorderedFileWriter(const std::string &prefix,
                                 const UnorderedWriterConfig &config = UnorderedWriterConfig()) {
        Start(prefix, config);
    }

    void Start(const std::string &prefix,
               const UnorderedWriterConfig &config = UnorderedWriterConfig()) {
        file_name_prefix = prefix;
        std::vector<std::string> file_names;
        for (size_t i = 0; i < config.num_files; i++) {
            file_names.push_back(GetFileName(prefix, i));
        }
        Start(file_names, config);
    }

    void Start(const std::vector<std::string> &file_names,
               const UnorderedWriterConfig &config) {
        wait_queue.SetSizeLimit(config.queue_size);
        num_files = file_names.size();
        allow_expand = config.allow_expand;
        CHECK(num_files > 0);
        if (num_files < config.num_threads) {
            [[unlikely]]
            LOG(WARNING)
                << "Writing to " << num_files << " files with " << config.num_threads << " threads. "
                << "Some threads will not get a file.";
        }
        for (size_t i = 0; i < num_files; i++) {
            auto file = new OpenedFile(file_names[i]);
            global_files.push_back(file);
        }
        CHECK(config.num_threads > 0);
        for (size_t t = 0; t < config.num_threads; t++) {
            std::vector<OpenedFile *> file_list;
            for (size_t file_index = t; file_index < num_files; file_index += config.num_threads) {
                file_list.push_back(global_files[file_index]);
            }
            if (file_list.empty()) {
                continue;
            }
            worker_threads.push_back(std::make_unique<std::thread>(
                    RunFileWriterWorker, this, file_list, config.io_uring_size));
        }
    }

    ~UnorderedFileWriter() {
        Wait();
        DLOG(INFO) << "FileWriter worker thread exited. " << num_files << " files created.";
    }

    void Push(std::shared_ptr<T> data, size_t size, size_t file_index = -1, size_t file_offset = -1) {
        // FIXME: need to align writes to 512 byte blocks, otherwise we won't be able to use O_DIRECT
        //   short term solution is to force multiples of 512 and throw an error otherwise;
        //   alternatively, use ftruncate to change the size of the file
        //   long term solution is to store the size of the last section in the end of the file (i.e. last 8 bytes)
        CHECK(size * sizeof(T) % O_DIRECT_MULTIPLE == 0)
                        << "Size (in bytes) must be aligned to the size of a page in O_DIRECT mode. "
                        << "Actual size: " << size * sizeof(T);
        CHECK((size_t) data.get() % O_DIRECT_MULTIPLE == 0)
                        << "Buffers used by the UnorderedFileWriter must be aligned.";
        auto request = new WriteRequest(std::move(data), size, file_index, file_offset);
        wait_queue.Push(request);
    }

    WriteRequest *Poll() {
        static WriteRequest empty_request({nullptr}, 0);
        auto [result, _] = wait_queue.Poll(&empty_request);
        return result;
    }

    void Close() {
        wait_queue.Close();
    }

    bool cleanup_done = false;

    /**
     * Block until file intermediate_writer finishes and return the number of files
     * @return
     */
    size_t Wait() {
        if (cleanup_done) {
            return num_files;
        }
        cleanup_done = true;
        Close();
        for (auto &thread: worker_threads) {
            if (thread->joinable()) {
                thread->join();
            }
        }
        for (auto &f: global_files) {
            delete f;
        }
        return num_files;
    }

    bool AllowExpand() {
        return allow_expand;
    }

    void ExpandFiles(size_t new_num_files) {
        CHECK(this->num_files < new_num_files);
        while (this->global_files.size() < new_num_files) {
            global_files.push_back(new OpenedFile(GetFileName(file_name_prefix, global_files.size())));
        }
        num_files = global_files.size();
    }

private:
    bool allow_expand = false;
    size_t num_files = 0;
    std::vector<std::unique_ptr<std::thread>> worker_threads;
    SimpleQueue<WriteRequest *> wait_queue;
    std::vector<OpenedFile *> global_files;
    std::string file_name_prefix;

    struct WriteRequest {
        std::shared_ptr<T> data;
        size_t size;
        size_t file_index = -1;
        size_t file_offset = -1;
        OpenedFile *file = nullptr;

        WriteRequest() : data(nullptr), size(0), file(nullptr), file_index(-1), file_offset(-1) {}

        WriteRequest(std::shared_ptr<T> data, size_t size) : data(std::move(data)), size(size) {}

        WriteRequest(std::shared_ptr<T> data, size_t size, size_t file_index, size_t file_offset)
                : data(std::move(data)), size(size), file_index(file_index), file_offset(file_offset) {}

    };

    struct OpenedFile {
        int fd;
        size_t bytes_issued = 0;
        size_t bytes_written = 0;

        explicit OpenedFile(const std::string &name) {
            fd = open(name.c_str(), O_DIRECT | O_WRONLY | O_CREAT, 0644);
            SYSCALL(fd);
        }

        ~OpenedFile() {
            SYSCALL(close(fd));
        }
    };

    enum Phase {
        NORMAL = 0,
        WAITING_FOR_COMPLETION = 1,
        ALL_DONE = 2
    };

    static void RunFileWriterWorker(UnorderedFileWriter *writer,
                                    const std::vector<OpenedFile *> files,
                                    const size_t io_uring_size) {
        CHECK(!files.empty());
        struct io_uring ring;
        SYSCALL(io_uring_queue_init(io_uring_size, &ring, IORING_SETUP_SINGLE_ISSUER));

        size_t current_file = 0;
        size_t outstanding_request = 0;
        size_t sqe_unavailable_count = 0;
        size_t max_outstanding_requests = io_uring_size * 2;

        // FIXME: do we need to acquire a mutex for the second check?
        Phase phase = NORMAL;
        while (phase != ALL_DONE) {
            // reap io_uring result
            while (outstanding_request > 0) {
                struct io_uring_cqe *cqe;
                int wait_result = io_uring_peek_cqe(&ring, &cqe);
                if (wait_result != 0) {
                    if (outstanding_request >= max_outstanding_requests || phase == WAITING_FOR_COMPLETION) {
                        wait_result = io_uring_wait_cqe(&ring, &cqe);
                    }
                }
                if (wait_result == 0) {
                    SYSCALL(cqe->res);
                    auto *request = (WriteRequest *) io_uring_cqe_get_data(cqe);
                    auto *file = request->file;
                    file->bytes_written += request->size * sizeof(T);

                    io_uring_cqe_seen(&ring, cqe);
                    outstanding_request--;
                    delete request;
                } else {
                    break;
                }
            }
            if (phase >= WAITING_FOR_COMPLETION) {
                if (outstanding_request == 0) {
                    phase = ALL_DONE;
                }
                continue;
            }
            bool submit_write = false;
            size_t requests_in_queue = 0;
            while (outstanding_request < max_outstanding_requests && requests_in_queue < io_uring_size) {
                auto *request = writer->Poll();
                if (request->size == 0) {
                    phase = WAITING_FOR_COMPLETION;
                    break;
                }
                // submit this IO request to ring
                struct io_uring_sqe *sqe;
                while (true) {
                    sqe = io_uring_get_sqe(&ring);
                    if (sqe == nullptr) {
                        [[unlikely]];
                        sqe_unavailable_count++;
                    } else {
                        break;
                    }
                }
                OpenedFile *file;
                if (request->file_index != (size_t) -1) {
                    if (request->file_index >= writer->num_files) {
                        [[unlikely]]
                        if (writer->allow_expand) {
                            writer->ExpandFiles(request->file_index + 1);
                        } else {
                            CHECK(request->file_index < writer->num_files);
                        }
                    }
                    file = writer->global_files[request->file_index];
                } else {
                    file = files[current_file];
                    current_file = (current_file + 1) % files.size();
                }
                request->file = file;

                size_t num_bytes = request->size * sizeof(T);

                size_t offset = file->bytes_issued;
                if (request->file_offset != (size_t) -1) {
                    offset = request->file_offset;
                }
                io_uring_prep_write(sqe, file->fd, request->data.get(), num_bytes, offset);
                file->bytes_issued += num_bytes;
                io_uring_sqe_set_data(sqe, request);
                submit_write = true;
                outstanding_request++;
                requests_in_queue++;
            }
            if (submit_write) {
                SYSCALL(io_uring_submit(&ring));
            }
            requests_in_queue = 0;
        }
        if (sqe_unavailable_count > 0) {
            [[unlikely]];
            LOG(WARNING) << "io_uring sqe entires were unavailable " << sqe_unavailable_count << " times. " <<
                         "Consider expanding the ring buffer.";
        }
        io_uring_queue_exit(&ring);
    }
};

#endif //SORTING_UNORDERED_FILE_WRITER_H
