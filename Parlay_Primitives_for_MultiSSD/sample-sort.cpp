//
// Created by peter on 3/21/24.
//

#include <random>

#include "scatter_gather_algorithms/sample_sort.h"
#include "utils/random_number_generator.h"
#include "utils/command_line.h"
#include "utils/unordered_file_writer.h"

template<typename NumberType>
struct DummyIterator {
    NumberType i, n;

    explicit DummyIterator(NumberType limit) : i(0), n(limit) {}

    DummyIterator(NumberType limit, NumberType start) : i(start), n(limit) {}

    DummyIterator operator+=(NumberType inc) {
        i += inc;
        return *this;
    }

    DummyIterator operator+(NumberType inc) {
        return DummyIterator(n, i + inc);
    }

    DummyIterator &operator++() {
        i++;
        return *this;
    }

    DummyIterator operator++(int) {
        DummyIterator<NumberType> old = *this;
        operator++();
        return old;
    }

    NumberType operator*() {
        return i;
    }

    bool operator==(const DummyIterator<NumberType> &o) {
        return this->i == o.i && this->n == o.n;
    }

    NumberType operator-(const DummyIterator<NumberType> &o) {
        return this->i - o.i;
    }

    NumberType operator[](NumberType index) {
        return index;
    }
};

template<typename T, typename Iterator>
parlay::sequence<std::tuple<size_t, T>>
CompareSortingResultFile(Iterator start, size_t start_index, const FileInfo &f) {
    start += start_index;
    auto current_file = (T *) ReadEntireFile(f.file_name, f.true_size);
    size_t n = f.true_size / sizeof(T);
    parlay::sequence<std::tuple<size_t, T>> mismatch_indices;
    for (size_t i = 0; i < n; i++) {
        auto expected = *start;
        auto actual = current_file[i];
        if (expected != actual) {
            mismatch_indices.push_back({start_index, actual});
            if (mismatch_indices.size() > 10) {
                return mismatch_indices;
            }
        }
        start++;
        start_index++;
    }
    free(current_file);
    return mismatch_indices;
}

/**
 * Compare the result of an in-memory sorting algorithm to an external memory sorting algorithm.
 *
 * @tparam T The type of data being sorted.
 * @tparam Iterator An iterator for the result of the in-memory sorting algorithm.
 * @param start Starting iterator
 * @param end Ending iterator
 * @param file_list A list of files resulting from the sorting algorithm
 */
template<typename T, typename Iterator>
void CompareSortingResult(Iterator start, Iterator end, const std::vector<FileInfo> &file_list) {
    // read each file and then compare the content of the file to the result of the in-memory sorting algorithm
    size_t total_size = end - start;
    size_t n_files = file_list.size();
    std::vector<size_t> prefix_sum(n_files + 1, 0);
    prefix_sum[0] = 0;
    for (size_t i = 1; i < n_files; i++) {
        prefix_sum[i] = prefix_sum[i - 1] + file_list[i - 1].true_size / sizeof(T);
    }
    size_t file_size = prefix_sum[n_files - 1] + file_list[n_files - 1].true_size / sizeof(T);
    if (file_size != total_size) {
        LOG(ERROR) << "Expected " << total_size << " numbers, got " << file_size;
        return;
    }
    auto mismatches = parlay::flatten(parlay::map(parlay::iota(n_files), [&](size_t i) {
        return CompareSortingResultFile<T>(start, prefix_sum[i], file_list[i]);
    }, 1));
    if (mismatches.empty()) {
        LOG(INFO) << "No mismatch found after comparing " << total_size << " elements. " <<
                  "All numbers in the permutation exist and are in order.";
        return;
    }
    for (size_t i = 0; i < std::min(10UL, mismatches.size()); i++) {
        auto [index, value] = mismatches[i];
        LOG(ERROR) << "Mismatch at index " << i << ": expected " << start[index] << ", got " << value << " instead.";
    }
}

template<typename T, typename Comparator>
void VerifySortingResult(const std::vector<FileInfo> &file_list, size_t expected_size, Comparator comp) {
    T prev = std::numeric_limits<T>().min();
    for (const auto &file: file_list) {
        auto arr = (T *) ReadEntireFile(file.file_name, file.file_size);
        size_t n = file.true_size / sizeof(T);
        expected_size -= n;
        for (size_t i = 0; i < n; i++) {
            // the current number should be no less than the previous number
            if (comp(arr[i], prev)) {
                LOG(ERROR) << "Error at file " << file.file_name << " index " << i << ": "
                           << prev << " is greater than " << arr[i];
                return;
            }
            prev = arr[i];
        }
        free(arr);
    }
    if (expected_size != 0) {
        LOG(ERROR) << "Size mismatch: " << expected_size << " more bytes expected. "
                   << "Due to wraparound of unsigned integers, this number might be spuriously large.";
    } else {
        LOG(INFO) << "Tests passed. All numbers are in ascending order.";
    }
}

void TestSampleSort(size_t count, bool regenerate) {
    using Type = uint64_t;
    std::string input_prefix = "numbers";
    std::string output_prefix = "sorted_numbers";
    if (regenerate) {
        LOG(INFO) << "Generating random numbers and writing them to disk";
        GenerateUniformRandomNumbers<Type>(input_prefix, count);
    } else {
        LOG(INFO) << "Skipping RNG";
    }
    // external memory sorting
    LOG(INFO) << "Performing external memory sorting";
    SampleSort<Type> sorter;
    auto input_files = FindFiles(input_prefix);
    auto result_files = sorter.Sort(input_files, output_prefix, std::less<>());

    LOG(INFO) << "Comparing result";
    VerifySortingResult<Type>(result_files, count, std::less<>());
}

void nop(void *ptr) {}

/**
 * Write some small numbers to disk for testing purposes.
 *
 * @param prefix The prefix of the names of the resulting files
 * @param n Number of items to be generated
 */
void GenerateSmallSample(const std::string &prefix, size_t n) {
    auto nums = (size_t *) aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, n * sizeof(size_t));
    {
        auto perm = parlay::random_permutation(n, parlay::random(std::random_device()()));
        parlay::parallel_for(0, n, [&](size_t i) {
            nums[i] = perm[i];
        });
    }
    UnorderedWriterConfig config;
    config.num_threads = 2;
    config.io_uring_size = 64;
    UnorderedFileWriter<size_t> writer(prefix, config);
    size_t step = std::min(1UL << 20, n);
    for (size_t i = 0; i < n; i += step) {
        writer.Push(std::shared_ptr<size_t>(nums + i, nop), std::min(step, n - i));
    }
    writer.Close();
    writer.Wait();
    free(nums);
}

DummyIterator<size_t> GetDummyArray(size_t n) {
    return DummyIterator<size_t>(n);
}

/**
 * Test the sorter with a small data sample
 * @param n Number of items in the test
 */
void TestSampleSortSmall(size_t n, bool generate) {
    const std::string prefix = "numbers", result_prefix = "result";
    DummyIterator<size_t> nums(n);
    if (generate) {
        GenerateSmallSample(prefix, n);
    }
    SampleSort<size_t> sorter;
    // Note that we don't know the number of files in the list. We just know the prefix.
    // We need to look for the files generated by GenerateSmallSample.
    auto file_list = FindFiles(prefix);
    LOG(INFO) << "Performing external memory sort.";
    auto result_files = sorter.Sort(file_list, result_prefix, std::less<>());
    LOG(INFO) << "Comparing sorting result.";
    CompareSortingResult<size_t>(nums, nums + n, result_files);
}

void generate(int argc, char **argv) {
    if (argc < 5) {
        LOG(ERROR) << "Usage: " << argv[0] << " gen <data size (power of 2)> <prefix> <type> <param>\n"
                   << "  types:\n"
                   << "  0: uniform random (param: limit)\n"
                   << "  1: permutation (in-memory)\n"
                   << "  2: zipfian (in-memory, param: s)\n"
                   << "  3: exponential (in-memory, param: s)";
        return;
    }
    size_t n = 1UL << ParseLong(argv[2]);
    std::string prefix(argv[3]);
    size_t sample_type = ParseLong(argv[4]);
    LOG(INFO) << "Generating " << n << " numbers with file prefix " << prefix << ". Sample type: " << sample_type
              << ".";
    switch (sample_type) {
        case 0:
            GenerateUniformRandomNumbers<size_t>(prefix, n, ParseLong(argv[5]));
            break;
        case 1:
            GenerateSmallSample(prefix, n);
            break;
        case 2:
            GenerateZipfianRandomNumbers<size_t>(prefix, n, ParseDouble(argv[5]));
            break;
        case 3:
            GenerateExponentialRandomNumbers<size_t>(prefix, n, ParseDouble(argv[5]));
            break;
        default:
            LOG(ERROR) << "Unknown sample type " << sample_type;
    }
}

void RunTest(int argc, char **argv) {
    if (argc < 4) {
        LOG(ERROR) << "Usage: " << argv[0] << " run <input prefix> <output prefix>";
        return;
    }
    std::string input_prefix(argv[2]), output_prefix(argv[3]);
    auto input_files = FindFiles(input_prefix);
    SampleSort<size_t> sorter;
    parlay::internal::timer timer("Sample sort");
    auto result_files = sorter.Sort(input_files, output_prefix, std::less<>());
    timer.next("DONE");
}

void verify_result(int argc, char **argv) {
    if (argc < 5) {
        LOG(ERROR) << "Usage: " << argv[0] << " verify <file prefix> <large data: 1|0> <data size>";
        return;
    }
    std::string prefix(argv[2]);
    bool large_data = (bool) ParseLong(argv[3]);
    size_t n = 1UL << ParseLong(argv[4]);
    auto files = FindFiles(prefix);
    CHECK(!files.empty()) << "No file with prefix " << prefix << " found.";
    GetFileInfo(files, true);
    if (large_data) {
        VerifySortingResult<size_t>(files, n, std::less<>());
    } else {
        auto dummy_array = GetDummyArray(n);
        CompareSortingResult<size_t>(dummy_array, dummy_array + n, files);
    }
}

int main(int argc, char **argv) {
    ParseGlobalArguments(argc, argv);
    if (argc < 2) {
        show_usage:
        LOG(ERROR) << "Usage: " << argv[0] << " <gen|run|verify> <command-specific options>";
        return 0;
    }
    std::map<std::string, std::function<void(int, char **)>> commands(
            {
                    {"gen",    generate},
                    {"run",    RunTest},
                    {"verify", verify_result}
            }
    );
    if (commands.count(argv[1])) {
        commands[argv[1]](argc, argv);
    } else {
        goto show_usage;
    }
}
