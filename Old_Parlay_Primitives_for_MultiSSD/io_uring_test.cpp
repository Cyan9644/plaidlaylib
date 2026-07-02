#include <cstdlib>
#include <fcntl.h>
#include <liburing.h>
#include <cstring>
#include <sys/utsname.h>
#include <set>

#include "parlay/internal/get_time.h"

#include "utils/logger.h"
#include "utils/file_utils.h"

bool simple_write_test(const char *file_name, const void *data, int data_size) {
    bool success = true;
    LOG(INFO) << "Testing write to " << file_name;
    int fd = open(file_name, O_WRONLY | O_CREAT | O_DIRECT, 0644);
    SYSCALL(fd);
    auto res = write(fd, data, data_size);
    SYSCALL(res);
    if (res != data_size) {
        LOG(ERROR) << "Expected write of size " << data_size << ", got write of size " << res;
        success = false;
    }
    SYSCALL(close(fd));
    // now make sure the content written to the file can be read
    if (success) {
        fd = open(file_name, O_RDONLY | O_DIRECT);
        void *buffer = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, data_size);
        res = read(fd, buffer, data_size);
        SYSCALL(res);
        if (res != data_size) {
            LOG(ERROR) << "Expected " << data_size << " bytes read. Got " << res << " bytes.";
            success = false;
        }
        if (memcmp(buffer, data, data_size) != 0) {
            LOG(ERROR) << "File content does not match.";
            success = false;
        }
        free(buffer);
        SYSCALL(close(fd));
    }
    SYSCALL(unlink(file_name));
    return success;
}

bool io_uring_usable() {
    struct utsname u;
    uname(&u);
    LOG(INFO) << "Running on kernel version " << u.release;
    struct io_uring ring;
    int res = io_uring_queue_init(5, &ring, 0);
    if (res == 0) {
        io_uring_queue_exit(&ring);
        LOG(INFO) << "io_uring available";
        return true;
    }
    LOG(ERROR) << std::strerror(-res);
    LOG(ERROR) << "io_uring unavailable";
    return false;
}

bool test_io_uring(const char *file_name,
                   int *data,
                   size_t array_size,
                   size_t file_size,
                   size_t alignment = O_DIRECT_MEMORY_ALIGNMENT) {
    int fd = open(file_name, O_WRONLY | O_CREAT | O_DIRECT, 0644);
    SYSCALL(fd);

    struct io_uring ring;
    int init = io_uring_queue_init(3600, &ring, IORING_SETUP_SINGLE_ISSUER);
    SYSCALL(init);
    if (fd == -1 || init != 0) {
        cleanup:
        close(fd);
        unlink(file_name);
        return false;
    }

    std::set<size_t> expected_data;
    for (size_t i = 0; i < file_size / array_size; i++) {
        struct io_uring_sqe *sqe;
        do {
            sqe = io_uring_get_sqe(&ring);
        } while (sqe == nullptr);
        int *write_data = (int *) aligned_alloc(alignment, array_size);
        memcpy(write_data, data, array_size);
        write_data[0] = (int) i + 1;
        io_uring_prep_write(sqe, fd, write_data, array_size, i * array_size);
        io_uring_sqe_set_data(sqe, (void *) (i + 1));
        expected_data.insert(i + 1);
        SYSCALL(io_uring_submit(&ring));
    }
    for (size_t i = 0; i < file_size / array_size; i++) {
        struct io_uring_cqe *cqe;
        SYSCALL(io_uring_wait_cqe(&ring, &cqe));
        SYSCALL(cqe->res);
        auto index = (size_t) io_uring_cqe_get_data(cqe);
        if (expected_data.erase(index) != 1) {
            LOG(ERROR) << "Expected " << index << " in the set of ring data.";
            goto cleanup;
        }
        io_uring_cqe_seen(&ring, cqe);
    }
    if (!expected_data.empty()) {
        LOG(ERROR) << "Not all requests are reaped";
        goto cleanup;
    }
    SYSCALL(close(fd));
    fd = open(file_name, O_RDONLY | O_DIRECT);
    SYSCALL(fd);
    int **buffers = (int **) malloc(file_size / array_size * sizeof(int *));
    for (size_t i = 0; i < file_size / array_size; i++) {
        struct io_uring_sqe *sqe;
        do {
            sqe = io_uring_get_sqe(&ring);
        } while (sqe == nullptr);
        int *read_buffer = (int *) aligned_alloc(alignment, array_size);
        buffers[i] = read_buffer;
        io_uring_prep_read(sqe, fd, read_buffer, array_size, i * array_size);
        io_uring_sqe_set_data(sqe, (void *) i);
        SYSCALL(io_uring_submit(&ring));
    }
    for (size_t i = 0; i < file_size / array_size; i++) {
        struct io_uring_cqe *cqe;
        SYSCALL(io_uring_wait_cqe(&ring, &cqe));
        io_uring_cqe_seen(&ring, cqe);
        SYSCALL(cqe->res);
        auto index = (size_t) io_uring_cqe_get_data(cqe);
        data[0] = (int) index + 1;
        if (memcmp(data, buffers[index], array_size) != 0) {
            LOG(ERROR) << "Data mismatch";
            free(buffers);
            goto cleanup;
        }
    }
    free(buffers);
    io_uring_queue_exit(&ring);
    SYSCALL(close(fd));
    SYSCALL(unlink(file_name));
    return true;
}

bool test_io_uring_writev(const char *file_name) {
    struct io_uring ring;
    SYSCALL(io_uring_queue_init(4, &ring, IORING_SETUP_SINGLE_ISSUER));
    constexpr size_t IO_VECTOR_BYTES = 4096;

    auto make_io_vectors = [](size_t num_vectors, bool fill) {
        auto *io_vectors = static_cast<iovec *>(malloc(sizeof(struct iovec) * num_vectors));
        for (size_t vec_index = 0; vec_index < num_vectors; vec_index++) {
            io_vectors[vec_index].iov_base = std::aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, IO_VECTOR_BYTES);
            io_vectors[vec_index].iov_len = IO_VECTOR_BYTES;
            if (!fill) {
                continue;
            }
            // Fill pointers with data
            for (size_t i = 0; i < IO_VECTOR_BYTES; i++) {
                auto datum = (unsigned char) ((vec_index << 4) ^ i);
                reinterpret_cast<unsigned char *>(io_vectors[vec_index].iov_base)[i] = datum;
            }
        }
        return io_vectors;
    };

    for (size_t num_vectors = 1; num_vectors <= IO_VECTOR_SIZE; num_vectors *= 2) {
        LOG(INFO) << "Using " << num_vectors << " IO vectors";
        int fd = open(file_name, O_WRONLY | O_CREAT | O_DIRECT, 0644);
        SYSCALL(fd);
        struct iovec *io_vectors = make_io_vectors(num_vectors, true);

        // Submit and reap write
        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
        SYSCALL((intptr_t) sqe);
        io_uring_prep_writev(sqe, fd, io_vectors, num_vectors, 0);
        io_uring_submit(&ring);
        struct io_uring_cqe *cqe;
        SYSCALL(io_uring_wait_cqe(&ring, &cqe));
        SYSCALL(cqe->res);
        io_uring_cqe_seen(&ring, cqe);
        close(fd);

        // Submit and reap read
        fd = open(file_name, O_RDONLY | O_DIRECT);
        struct iovec *result = make_io_vectors(num_vectors, false);
        sqe = io_uring_get_sqe(&ring);
        SYSCALL((intptr_t) sqe);
        io_uring_prep_readv(sqe, fd, result, num_vectors, 0);
        io_uring_submit(&ring);
        SYSCALL(io_uring_wait_cqe(&ring, &cqe));
        SYSCALL(cqe->res);
        io_uring_cqe_seen(&ring, cqe);

        // Compare results
        for (size_t i = 0; i < num_vectors; i++) {
            auto iov1 = io_vectors[i], iov2 = result[i];
            if (std::memcmp(iov1.iov_base, iov2.iov_base, IO_VECTOR_BYTES) != 0) {
                LOG(ERROR) << "Error while comparing iovec:\n"
                           << "File: " << file_name << "\n"
                           << "Num of io vectors: " << num_vectors << "\n"
                           << "Mismatch fisrt occured at io vector number " << i;
                return false;
            }
        }
        SYSCALL(close(fd));
        SYSCALL(unlink(file_name));
        // Ignore memory leaks in allocated pointer here
        for (size_t i = 0; i < num_vectors; i++) {
            free(io_vectors[i].iov_base);
            free(result[i].iov_base);
        }
        free(io_vectors);
        free(result);
    }
    io_uring_queue_exit(&ring);
    return true;
}

void run_main_test() {
    LOG(INFO) << "Test setup";
    const size_t FILE_SIZE = 1UL << 28;
    const size_t N = 1UL << 20;
    const size_t ARRAY_SIZE = N * sizeof(int);
    int *data = (int *) aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, ARRAY_SIZE);
    for (size_t j = 0; j < N; j++) {
        data[j] = (int) (j * j) - 3;
    }
    LOG(INFO) << "Testing whether a file can be written to the current directory using O_DIRECT.";
    CHECK(simple_write_test("./io_test", data, ARRAY_SIZE));
    LOG(INFO) << "Testing whether a file can be written to all SSDs.";
    PopulateSSDList();
    LOG(INFO) << "Promised " << SSD_COUNT << " SSDs in config file. Testing each of them.";
    for (const auto &ssd_name: GetSSDList()) {
        CHECK(simple_write_test((ssd_name + "/io_test").c_str(), data, ARRAY_SIZE));
    }
    LOG(INFO) << "Testing whether io_uring can be used";
    CHECK(io_uring_usable());
    LOG(INFO) << "Testing whether io_uring can be used to write to all SSDs (small write)";
    for (const auto &ssd_name: GetSSDList()) {
        CHECK(test_io_uring((ssd_name + "/io_uring_test").c_str(), data, ARRAY_SIZE, ARRAY_SIZE));
    }
    LOG(INFO) << "Testing whether io_uring can be used to write to a SSD (big write)";
    CHECK(test_io_uring((GetSSDList()[0] + "/io_uring_test").c_str(), data, ARRAY_SIZE, FILE_SIZE));
    for (const auto &ssd_name: GetSSDList()) {
        LOG(INFO) << "Testing io_uring together with writev on " << ssd_name;
        CHECK(test_io_uring_writev((ssd_name + "/io_uring_test").c_str()));
    }
    LOG(INFO) << "All tests passed.";

    std::cout << "Check SSD O_DIRECT memory alignment requirements? Note that this may render the SSD unusable. "
              << "1: YES. 0: NO" << std::endl;
    int response;
    std::cin >> response;
    if (response) {
        for (const auto &ssd_name: GetSSDList()) {
            constexpr size_t MAX_ALIGN = 4096, MIN_ALIGN = 8;
            for (size_t alignment = MAX_ALIGN; alignment >= MIN_ALIGN; alignment /= 2) {
                auto file_name = ssd_name + "/io_uring_test";
                bool result = test_io_uring(
                        file_name.c_str(),
                        data,
                        ARRAY_SIZE,
                        // Try 10 times in case some of the malloc calls happened to have better alignment
                        ARRAY_SIZE * 10,
                        alignment);
                if (!result) {
                    LOG(ERROR) << "Test failed on " << ssd_name << " at alignment " << alignment;
                    break;
                }
                if (alignment == MIN_ALIGN) {
                    LOG(INFO) << ssd_name << " does not seem to care about memory alignment";
                }
            }
        }
    }
}

/**
 * Test a combination of number of io_uring entries and number of threads.
 * This number seems to be dependent on the kernel version.
 */
void run_io_uring_init_entries_test() {
    constexpr size_t entries = 20000;
    constexpr size_t threads = 2;
    struct io_uring rings[threads];
    for (size_t i = 0; i < threads; i++) {
        SYSCALL(io_uring_queue_init(entries, &rings[i], IORING_SETUP_SINGLE_ISSUER));
    }
    exit(0);
}

int main() {
    run_main_test();
}
