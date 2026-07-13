// Drive housekeeping shared by the head-to-head sort benchmarks
// (external_samplesort_vs_peter, direct_samplesort_vs_peter, samplesort_three_way).
//
// Those drivers time several out-of-core sorts in one process, on one set of
// drives.  Everything here runs strictly *between* the timed regions: it decides
// what state the drives are in when a sort starts, and never touches the sort
// itself.
//
// The subtle one is settle_drives().  A sort that runs after another was measured
// 15-25% slower on the dev box purely because of what the previous one left
// behind, and a bare sync()+sleep does not fix it:
//
//   - unlink() returns as soon as the directory entry is gone.  ext4 releases the
//     file's blocks when the journal transaction commits, in the background — and
//     a sort's files are gigabytes.  A sort started on top of that freeing queues
//     behind it.
//   - an input build leaves dirty pages that are still being written back.
//
// syncfs() on each mount forces both to completion for that file system; the
// settle sleep afterwards gives the devices themselves a moment to go idle
// (consumer SSDs keep doing GC/TRIM work after the kernel is done).  With this in
// place a sort runs at the same speed wherever it sits in the order, so each sort
// can be timed ONCE per data point — repeating a point and keeping the best would
// hide the bias at the price of multiplying the SSD write endurance it costs.
//
// Override the settle with BENCH_SETTLE_MS (milliseconds, default 2000).

#ifndef BENCH_DRIVES_H
#define BENCH_DRIVES_H

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "utils/file_utils.h"

namespace bench_drives {

inline size_t settle_ms() {
    static const size_t ms = [] {
        const char* e = getenv("BENCH_SETTLE_MS");
        return e ? std::stoull(e) : size_t{2000};
    }();
    return ms;
}

// Wait until the drives are actually done with whatever the last phase did to
// them: the file systems are synced (which completes both writeback and the block
// frees queued by removing a previous sort's files) and then left to settle.
// Always call this outside a timed region.
inline void settle_drives() {
    for (const std::string& dir : GetSSDList()) {
        int fd = open(dir.c_str(), O_RDONLY | O_DIRECTORY);
        if (fd < 0) continue;              // best effort: a mount we can't open
        syncfs(fd);
        close(fd);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms()));
}

// Remove every file on every drive whose name begins with one of `prefixes`, then
// settle.  A generic scan (not GetFileName(prefix, d)) because the sorts leave
// tag- and bucket-suffixed intermediates that a fixed 0..num_drives enumeration
// would miss.  This is how each sort is guaranteed to run on drives holding
// nothing but its own input — and, thanks to the settle, none of the previous
// sort's unfinished teardown.
inline void clear_drives(const std::vector<std::string>& prefixes) {
    for (const std::string& dir : GetSSDList()) {
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
            const std::string name = e.path().filename().string();
            for (const std::string& p : prefixes) {
                if (name.rfind(p, 0) == 0) {  // name starts with p
                    std::filesystem::remove(e.path(), ec);
                    break;
                }
            }
        }
    }
    settle_drives();
}

}  // namespace bench_drives

#endif  // BENCH_DRIVES_H
