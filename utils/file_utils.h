//
// Created by peter on 3/2/24.
//

#ifndef SORTING_FILE_UTILS_H
#define SORTING_FILE_UTILS_H

#include <vector>
#include <string>
#include <cerrno>
#include <chrono>
#include <thread>
#include <sys/resource.h>
#include <liburing.h>
#include "utils/file_info.h"
#include "configs.h"

// Raise the process's soft open-file limit (RLIMIT_NOFILE) up to its hard limit.
//
// This library opens one file per drive AND one io_uring instance (each of which
// costs a file descriptor) per worker thread, for every reader and writer.
// Primitives that fan out concurrent readers/writers -- notably the parallel
// recursive sorts (sample_sort's per-bucket parallel_for, primitive_quicksort's
// par_do) -- can therefore need thousands of descriptors at once, far past the
// common 1024 default soft limit, at which point io_uring_queue_init fails with
// EMFILE.  Bumping the soft limit to the hard limit needs no privilege; call it
// once at program start.  Returns the new soft limit (0 on failure).
inline size_t RaiseFdLimit() {
    struct rlimit rl;
    if (getrlimit(RLIMIT_NOFILE, &rl) != 0) return 0;
    rl.rlim_cur = rl.rlim_max;
    if (setrlimit(RLIMIT_NOFILE, &rl) != 0) return 0;
    return (size_t)rl.rlim_cur;
}

// Create an io_uring instance, retrying while the kernel transiently reports
// -ENOMEM.
//
// io_uring charges its ring pages against the process's RLIMIT_MEMLOCK (the
// locked-memory limit, distinct from the fd limit RaiseFdLimit() lifts, and on
// many boxes only ~64 MiB and not raisable without privilege).  The catch is
// that on some kernels -- notably the WSL2 "microsoft" kernel used on dev
// boxes -- the memory freed by io_uring_queue_exit() is reclaimed
// *asynchronously*: the exit call returns before a kernel workqueue actually
// hands the memlock accounting back.  A primitive that churns many short-lived
// rings -- the parallel recursive sorts spin up one ring per reader/writer
// worker for every subproblem across the bucket fanout -- can therefore momentarily
// exceed the limit even though its true steady-state footprint is tiny, and
// io_uring_queue_init() then fails with -ENOMEM.  The pending reclaim catches
// up within about a second, so a short bounded retry rides out the spike; a
// genuinely exhausted limit still falls through to the caller (which SYSCALL-
// checks the return).  Signature mirrors io_uring_queue_init.
inline int InitIoUringWithRetry(unsigned entries, struct io_uring *ring, unsigned flags) {
    // ~4 s of headroom at 20 ms spacing -- far more than the observed ~1 s
    // reclaim latency, while still bounded so a real exhaustion surfaces.
    constexpr int kMaxAttempts = 200;
    for (int attempt = 0;; attempt++) {
        int r = io_uring_queue_init(entries, ring, flags);
        if (r != -ENOMEM || attempt >= kMaxAttempts) return r;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

std::vector<FileInfo> FindFiles(const std::string &prefix, bool parallel = false);

inline size_t GetFileSize(const std::string &file_name);

void GetFileInfo(std::vector<FileInfo> &info, bool eof_marker = false, bool compute_before_size = true);

void ComputeBeforeSize(std::vector<FileInfo> &files);

inline size_t AlignDown(size_t original, size_t alignment) {
    return original / alignment * alignment;
}

inline size_t AlignUp(size_t original, size_t alignment) {
    return (original + alignment - 1) / alignment * alignment;
}

/**
 * Ensure that a byte offset conforms to disk alignment requirements by rounding down.
 * For example, 4098 would become 4096 if O_DIRECT_MULTIPLE is set to 4096.
 *
 * @param original
 * @return
 */
constexpr size_t AlignDown(size_t original) {
    return original / O_DIRECT_MULTIPLE * O_DIRECT_MULTIPLE;
}
/**
 * Ensure that a byte offset conforms to disk alignment requirements by rounding up.
 * For example, 4098 would become 8192 if O_DIRECT_MULTIPLE is set to 4096.
 *
 * @param original
 * @return
 */
constexpr size_t AlignUp(size_t original) {
    return (original + O_DIRECT_MULTIPLE - 1) / O_DIRECT_MULTIPLE * O_DIRECT_MULTIPLE;
}

void Read(int fd, void* buffer, size_t read_size);
void Write(int fd, const void *buffer, size_t write_size);

void ReadFileOnce(const std::string &file_name, void* buffer, size_t offset);
void ReadFileOnce(const std::string &file_name, void *buffer, size_t start, size_t read_size);

void* ReadEntireFile(const std::string &file_name, size_t read_size);

void PopulateSSDList();
void PopulateSSDList(size_t count, bool random, bool verbose);
void PopulateSSDList(const std::vector<int> &ssd_numbers, bool verbose);
std::string GetFileName(const std::string &prefix, size_t file_number);
std::vector<std::string> GetSSDList();

void MakeFileEndMarker(unsigned char *buffer, size_t size, size_t real_size);

double GetThroughput(size_t size, double time);
double GetThroughput(const std::vector<FileInfo> &files, double time);

#endif //SORTING_FILE_UTILS_H
