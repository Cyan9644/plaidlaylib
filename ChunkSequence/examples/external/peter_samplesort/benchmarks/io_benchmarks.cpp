//
// Created by peter on 2/18/25.
//

#include "benchmarks/io_benchmarks.h"

#include <map>
#include "utils/unordered_file_writer.h"
#include "utils/unordered_file_reader.h"
#include "utils/ordered_file_writer.h"
#include "utils/command_line.h"
#include "parlay/random.h"
#include "parlay/primitives.h"
#include "absl/log/log.h"
#include "utils/random_read.h"

const size_t SINGLE_IO_SIZE = 4 * (1UL << 20);

void OrderedFileWriterTest(int argc, char **argv) {
    using WriterData = AllocatorData<SAMPLE_SORT_BUCKET_SIZE>;
    using Allocator = AlignedTypeAllocator<WriterData, O_DIRECT_MULTIPLE>;

    CHECK(argc > 3) << "Expected an argument on total write size and number of buckets";
    using Type = long long;
    const std::string prefix = "test_files";
    const size_t TOTAL_WRITE_SIZE = 1UL << strtol(argv[2], nullptr, 10);
    const size_t SINGLE_WRITE_SIZE = SAMPLE_SORT_BUCKET_SIZE;
    const size_t NUM_BUCKETS = strtol(argv[3], nullptr, 10);
    parlay::internal::timer timer;
    OrderedFileWriter<Type, SAMPLE_SORT_BUCKET_SIZE> writer(prefix, NUM_BUCKETS,
                                                            4 * (1 << 20));
    parlay::par_do(
            [&]() {
                parlay::parallel_for(0, 2, [&](size_t i) {
                    writer.RunIOThread(&writer);
                }, 1);
            },
            [&]() {
                const size_t n = SINGLE_WRITE_SIZE / sizeof(Type);
                LOG(INFO) << "Starting writer loop";
                parlay::random_generator gen;
                parlay::parallel_for(
                        0, TOTAL_WRITE_SIZE / SINGLE_WRITE_SIZE, [&](size_t i) {
                            std::uniform_int_distribution<size_t> dis(0, NUM_BUCKETS - 1);
                            auto array = reinterpret_cast<Type *>(Allocator::alloc());
                            for (size_t index = 0; index < n; index++) {
                                array[index] = (Type) (index * index - 5 * index - 1);
                            }
                            auto r = gen[i];
                            writer.Write(dis(r), array, n);
                        }, 1);
                writer.CleanUp();
                LOG(INFO) << "Waiting for completion";
            });
    auto results = writer.ReapResult();
    double time = timer.next_time();
    std::cout << "Throughput: " << GetThroughput(results, time) << "GB\n";
}

void UnorderedReadTest(int argc, char **argv) {
    CHECK(argc > 2) << "Usage: " << argv[0] << " " << argv[1] << " <file prefix>";
    using Type = long long;
    parlay::internal::timer timer("Unordered read");
    auto files = FindFiles(std::string(argv[2]));
    size_t expected_size = 0;
    for (auto &file: files) {
        expected_size += file.file_size;
    }
    LOG(INFO) << "Starting reading " << files.size() << " files " << expected_size
              << " bytes " << (expected_size >> 30) << " GiB";
    timer.next("start benchmark");
    using Reader = UnorderedFileReader<Type, READER_READ_SIZE>;
    Reader reader;
    reader.PrepFiles(files);
    // FIXME: allocator is the bottleneck.
    //  jemalloc performs worse than glibc (1/3) while mimalloc is much faster.
    size_t io_uring_size = ParseLong(argv[3]);
    size_t num_io_threads = ParseLong(argv[4]);
    reader.Start(
            UnorderedReaderConfig(
                    num_io_threads,
                    io_uring_size * 2,
                    io_uring_size));
    size_t remaining_size = expected_size;
    const std::time_t time_limit = 30;
    const std::time_t start_time = std::time(nullptr);
    bool timeout = false;
    while (true) {
        auto [ptr, size, _, _2] = reader.Poll();
        if (ptr == nullptr || size == 0) {
            break;
        }
        reader.allocator.Free(ptr);
        remaining_size -= size * sizeof(Type);
        if (std::time(nullptr) - start_time >= time_limit) {
            timeout = true;
            break;
        }
    }
    if (!timeout) {
        CHECK(remaining_size == 0)
                        << "Still " << remaining_size << " bytes remaining unread";
    }
    double time_elapsed = timer.next_time();
    double throughput = (double) (expected_size - remaining_size) / 1e9 / time_elapsed;
    LOG(INFO) << "Throughput: " << throughput;
    if (timeout) {
        // Skip sanity checker in the reader
        LOG(INFO) << "Test timeout";
        exit(0);
    }
}

void UnorderedWriteTest(int argc, char **argv) {
    CHECK(argc > 3) << "Usage: " << argv[0] << " " << argv[1] << " <file prefix> <size (pow of 2)>";
    using Type = long long;
    const std::string prefix(argv[2]);
    const size_t TOTAL_WRITE_SIZE = 1UL << ParseLong(argv[3]);
    parlay::internal::timer timer("Unordered write");
    timer.next("experiment start");
    size_t io_uring_size = ParseLong(argv[4]);
    size_t num_io_threads = ParseLong(argv[5]);
    UnorderedWriterConfig config;
    config.io_uring_size = io_uring_size;
    config.num_threads = num_io_threads;
    UnorderedFileWriter<Type> writer(prefix, config);
    timer.next("preparing writes");
    size_t totalWriteSize = TOTAL_WRITE_SIZE;
    auto array = std::shared_ptr<Type>(
            (Type *) aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, SINGLE_IO_SIZE), free);
    for (size_t i = 0; i < SINGLE_IO_SIZE / sizeof(Type); i++) {
        array.get()[i] = (Type) (i * i - 5 * i - 1);
    }
    timer.next("starting to write");
    for (size_t i = 0; i < totalWriteSize / SINGLE_IO_SIZE; i++) {
        writer.Push(array, SINGLE_IO_SIZE / sizeof(Type));
    }
    writer.Wait();
    double time_elapsed = timer.next_time();
    double throughput = (double) TOTAL_WRITE_SIZE / 1e9 / time_elapsed;
    LOG(INFO) << "Throughput: " << throughput;
}

void RandomReadTest(int argc, char **argv) {
    using Type = uint64_t;
    if (argc < 4) {
        std::cout << argv[0] << " " << argv[1] << " <file prefix> <num reads>";
        return;
    }
    std::string prefix(argv[2]);
    size_t n = ParseLong(argv[3]);
    parlay::internal::timer timer("random read");

    std::random_device rd;
    auto files = FindFiles(prefix);
    GetFileInfo(files);
    CHECK(!files.empty()) << "No file with prefix " << prefix << " found";
    size_t total_size = 0;
    for (auto &file: files) {
        total_size += file.true_size;
    }

    parlay::random_generator gen;
    std::uniform_int_distribution<size_t> dis_file(0, total_size - 1);
    auto queries =
            parlay::tabulate(n, [&](size_t i) -> size_t {
                auto g = gen[i];
                auto index = dis_file(g) % (total_size / sizeof(Type));
                return index;
            });
    timer.next("Begin random read");
    auto result = RandomBatchRead<Type>(files, queries);
    double time = timer.next_time();
    LOG(INFO) << "Spent " << time << " seconds to read " << n << " elements of size " << sizeof(Type);
    double throughput = (double) (n * O_DIRECT_MULTIPLE / 1e9) / time;
    double iops = (double) n / time;
    std::cout.precision(10);
    std::cout << "Throughput (inc. wasted bandwidth): " << throughput << " GB/s. IOPS: " << iops;
}

void LargeReadTest(int argc, char **argv) {
    using T = size_t;
    constexpr auto NUM_IO_VECTORS = 16;
    struct Buffer {
        T *buffer;
        iovec io_vectors[NUM_IO_VECTORS];

        explicit Buffer(size_t size) {
            buffer = (T *) aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, size);
            for (size_t i = 0; i < NUM_IO_VECTORS; i++) {
                char *ptr = (char *) buffer;
                io_vectors[i].iov_base = ptr + (size / NUM_IO_VECTORS) * i;
                io_vectors[i].iov_len = size / NUM_IO_VECTORS;
            }
        }

        ~Buffer() { free(buffer); }
    };

    CHECK(argc >= 5);
    std::string prefix(argv[2]);
    auto files = FindFiles(prefix);
    size_t read_size = 1UL << ParseLong(argv[3]);
    LOG(INFO) << "Read size: " << read_size / (1 << 20) << " MiB";
    // WriteFiles<size_t>(prefix, total_size);
    CHECK(!files.empty());
    GetFileInfo(files);
    std::vector<int> fds;
    for (auto &file: files) {
        fds.push_back(open(file.file_name.c_str(), O_RDONLY | O_DIRECT));
    }

    size_t max_requests_in_ring = ParseLong(argv[4]);
    size_t buffer_size = read_size * max_requests_in_ring;
    std::queue<Buffer *> buffers;
    for (size_t i = 0; i < buffer_size; i += read_size) {
        auto buffer = new Buffer(read_size);
        buffers.push(buffer);
    }

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<size_t> file_distribution(0, fds.size() - 1);

    struct io_uring ring;
    io_uring_queue_init(max_requests_in_ring, &ring, IORING_SETUP_SINGLE_ISSUER);
    size_t requests_in_ring = 0;
    while (true) {
        while (requests_in_ring < max_requests_in_ring) {
            auto sqe = io_uring_get_sqe(&ring);
            SYSCALL((ptrdiff_t) sqe);
            size_t file_index = file_distribution(rng);
            FileInfo file = files[file_index];
            auto buffer = buffers.front();
            buffers.pop();
            std::uniform_int_distribution<size_t> offset_distribution(
                    0, file.file_size / read_size - 1);
            io_uring_prep_readv(sqe, fds[file_index], buffer->io_vectors,
                                NUM_IO_VECTORS, offset_distribution(rng) * read_size);
            io_uring_sqe_set_data(sqe, buffer);
            requests_in_ring++;
        }
        io_uring_submit(&ring);
        while (requests_in_ring > 0) {
            struct io_uring_cqe *cqe;
            if (requests_in_ring == max_requests_in_ring) {
                SYSCALL(io_uring_wait_cqe(&ring, &cqe));
            } else {
                int res = io_uring_peek_cqe(&ring, &cqe);
                if (res != 0) {
                    break;
                }
            }
            SYSCALL(cqe->res);
            requests_in_ring--;
            io_uring_cqe_seen(&ring, cqe);
            buffers.push((Buffer *) io_uring_cqe_get_data(cqe));
        }
    }
}

