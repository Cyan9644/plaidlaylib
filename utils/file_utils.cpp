//
// Created by peter on 3/2/24.
//

#include <filesystem>
#include <sys/stat.h>
#include <fcntl.h>
#include <set>
#include <random>

#include "parlay/parallel.h"

#include "file_utils.h"
#include "utils/logger.h"

using std::filesystem::directory_iterator;
using std::filesystem::path;

/**
 * Find files whose name begin with the specified prefix.
 *
 * @param prefix All files that begin with prefix will be included in the result
 * @param parallel Whether to do this in parallel. Currently not used.
 * @return A list of file information.
 *   Note that we also get the file size as a by-product of iterating through the directory.
 */
[[nodiscard]] std::vector<FileInfo> FindFiles(const std::string &prefix, bool parallel) {
    // TODO: if this ever becomes the bottleneck, iterate through all SSDs in parallel
    std::vector<FileInfo> result;
    const auto ssd_list = GetSSDList();
    // go through every SSD
    for (const std::string &ssd_name: ssd_list) {
        path p(ssd_name);
        for (auto const &dir_entry: directory_iterator{p}) {
            auto path_str = dir_entry.path().string();
            size_t index = path_str.find("/" + prefix);
            if (index != std::string::npos) {
                size_t file_index = 0;
                // Usually the part after the prefix is the index, but if it's empty, we just ignore it.
                auto index_substring = path_str.substr(index + 1 + prefix.size());
                if (!index_substring.empty()) {
                    file_index = std::stol(index_substring);
                }
                result.emplace_back(path_str, file_index, 0, dir_entry.file_size());
            }
        }
    }
    std::sort(result.begin(), result.end(), [](const FileInfo &a, const FileInfo &b) {
        return a.file_index < b.file_index;
    });
    return result;
}

/**
 * @brief Retrieve information about the file if not available.
 *   Specifically, the size of the file on the file system and the true size of the file
 *   (excluding the garbage data at the end, which is used to pad spaces
 *   so that the file size is a multiple of O_DIRECT_MULTIPLE)
 *
 * @param info A list of files
 */
void GetFileInfo(std::vector<FileInfo> &info, bool eof_marker, bool compute_before_size) {
    parlay::parallel_for(0, info.size(), [&](size_t i) {
        if (info[i].file_size == 0) {
            info[i].file_size = GetFileSize(info[i].file_name);
        }
        if (eof_marker) {
            if (info[i].true_size == 0 && info[i].file_size > 0) {
                unsigned char buffer[O_DIRECT_MULTIPLE];
                ReadFileOnce(info[i].file_name, buffer, info[i].file_size - O_DIRECT_MULTIPLE);
                info[i].true_size = info[i].file_size - *(uint16_t *) (buffer + O_DIRECT_MULTIPLE - METADATA_SIZE);
            }
        } else {
            info[i].true_size = info[i].file_size;
        }
    }, 1);
    if (compute_before_size) {
        ComputeBeforeSize(info);
    }
}

std::vector<std::string> ssd_list;

void PopulateSSDList() {
    PopulateSSDList(SSD_COUNT, false, false);
}

void PopulateSSDList(size_t count, bool random, bool verbose) {
    CHECK(count <= SSD_COUNT);
    CHECK(ssd_list.empty());
    std::set<size_t> chosen;
    if (random) {
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<size_t> distribution(0, SSD_COUNT - 1);
        while (chosen.size() < count) {
            chosen.insert(distribution(rng));
        }
    } else {
        for (size_t i = 0; i < count; i++) {
            chosen.insert(i);
        }
    }
    std::vector<int> ssd_numbers(chosen.begin(), chosen.end());
    PopulateSSDList(ssd_numbers, verbose);
}

void PopulateSSDList(const std::vector<int> &ssd_numbers, bool verbose) {
    CHECK(!ssd_numbers.empty());
    std::string num_string;
    for (int num: ssd_numbers) {
        num_string += std::to_string(num) + ",";
    }
    if (verbose) {
        LOG(INFO) << "SSD numbers: " << num_string;
    }
    char buffer[1024];
    for (size_t i: ssd_numbers) {
        sprintf(buffer, SSD_ROOT.c_str(), i);
        ssd_list.emplace_back(buffer);
    }
    std::ostringstream imploded;
    std::copy(ssd_list.begin(), ssd_list.end(),
              std::ostream_iterator<std::string>(imploded, " "));
    if (verbose) {
        LOG(INFO) << "SSDs used: " << imploded.str();
    }
}

void CheckSSDList() {
    if (ssd_list.empty()) {
        [[unlikely]]
        LOG(WARNING) << "Number of SSDs to use is not specified. Defaulting to all of them.";
        PopulateSSDList();
    }
}

/**
 * Generate a file name and, in doing so, assign it to a SSD in a round-robin fashion
 *
 * @param prefix
 * @param file_number
 * @return
 */
std::string GetFileName(const std::string &prefix, size_t file_number) {
    CheckSSDList();
    size_t ssd_number = file_number % ssd_list.size();
    return ssd_list[ssd_number] + "/" + prefix + std::to_string(file_number);
}

/**
 * Read O_DIRECT_MULTIPLE bytes starting from the specified offset of the file
 *
 * @param file_name
 * @param buffer Buffer to which the result will be written. Its size must be at least O_DIRECT_MULTIPLE
 * @param offset The byte at which we want to start reading
 */
void ReadFileOnce(const std::string &file_name, void *buffer, size_t offset) {
    int fd = open(file_name.c_str(), O_RDONLY | O_DIRECT);
    SYSCALL(fd);
    CHECK(offset % O_DIRECT_MULTIPLE == 0)
                << "File read offset is " << offset << ", which is not a multiple of " << O_DIRECT_MULTIPLE;
    auto res = lseek64(fd, (long) offset, SEEK_SET);
    CHECK(res != off64_t(-1)) << std::strerror(errno) << " " << file_name << " at offset " << offset;
    SYSCALL(read(fd, buffer, O_DIRECT_MULTIPLE));
    SYSCALL(close(fd));
}

/**
 * Read read_size bytes from a file and store the result in a buffer. The read starts at <code>start</code> bytes.
 * This version is slightly slower since it copies memory twice. However, it is easier to use since no alignment
 * assumptions are made about start and read_size.
 *
 * @param file_name
 * @param buffer
 * @param start
 * @param read_size Must be much smaller than stack size limit (i.e. 8MB by default)
 * since the read buffer for this function is allocated on the stack.
 */
void ReadFileOnce(const std::string &file_name, void *buffer, size_t start, size_t read_size) {
    int fd = open(file_name.c_str(), O_RDONLY | O_DIRECT);
    SYSCALL(fd);
    size_t start_aligned = AlignDown(start);
    size_t end = start + read_size;
    // compute the nearest aligned byte
    size_t end_aligned = AlignUp(end);
    size_t aligned_read_size = end_aligned - start_aligned;
    auto res = lseek64(fd, (long) start_aligned, SEEK_SET);
    SYSCALL(res);
    // read everything into a temporary buffer and then copy the data to the original buffer
    unsigned char temp_buffer[aligned_read_size];
    SYSCALL(read(fd, temp_buffer, O_DIRECT_MULTIPLE));
    memcpy(buffer, temp_buffer + (start - start_aligned), read_size);
    SYSCALL(close(fd));
}

/**
 * Read an entire file into memory and store the result in a buffer
 *
 * @param file_name
 * @param read_size Number of bytes to be read
 * @return A pointer to the content of the file (needs to be freed by the caller using Free)
 */
void *ReadEntireFile(const std::string &file_name, size_t read_size) {
    // align the read for O_DIRECT
    read_size = AlignUp(read_size);
    void *buffer = std::aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, read_size);
    int fd = open(file_name.c_str(), O_RDONLY | O_DIRECT);
    SYSCALL(fd);
    // reads cannot exceed 2147479552 bytes on Linux, so we need this loop to perform multiple reads
    size_t result_size = 0;
    while (result_size < read_size) {
        size_t cur_size = read(fd, (char*)buffer + result_size, read_size - result_size);
        SYSCALL(cur_size);
        result_size += cur_size;
    }
    return buffer;
}

std::vector<std::string> GetSSDList() {
    CheckSSDList();
    return ssd_list;
}

void ComputeBeforeSize(std::vector<FileInfo> &files) {
    size_t total_size = 0;
    for (auto &f: files) {
        f.before_size = total_size;
        total_size += f.true_size;
    }
}

void MakeFileEndMarker(unsigned char *buffer, size_t size, size_t real_size) {
    constexpr size_t capacity = 1 << (8 * METADATA_SIZE);
    size_t byte_diff = size - real_size;
    CHECK(byte_diff < capacity);
    *(uint16_t*)(&buffer[size - METADATA_SIZE]) = (uint16_t)byte_diff;
}

double GetThroughput(size_t size, double time) {
    return ((double) size / 1e9) / time;
}

double GetThroughput(const std::vector<FileInfo> &files, double time) {
    size_t size = 0;
    for (const auto &f : files) {
        size += f.true_size == 0 ? f.file_size : f.true_size;
    }
    return GetThroughput(size, time);
}

inline size_t GetFileSize(const std::string &file_name) {
    struct stat stat_buf;
    SYSCALL(stat(file_name.c_str(), &stat_buf));
    return stat_buf.st_size;
}

void Read(int fd, void *buffer, size_t read_size) {
    // Loop for large buffers for both read/write.
    auto *p = static_cast<unsigned char*>(buffer);
    while (read_size > 0) {
        auto n = read(fd, p, read_size);
        CHECK(n > 0);
        p += n;
        read_size -= static_cast<size_t>(n);
    }
}

void Write(int fd, const void *buffer, size_t write_size) {
    auto *p = static_cast<const unsigned char*>(buffer);
    while (write_size > 0) {
        auto n = write(fd, p, write_size);
        CHECK(n > 0);
        p += n;
        write_size -= static_cast<size_t>(n);
    }
}
