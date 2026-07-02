// Bandwidth comparison: ChunkSequence vs sequence_algorithms
//
// Runs the same Map and Reduce workloads through both implementations on
// equivalent data ([0, n-1] identity permutation) and reports read throughput
// in GB/s so the two I/O paths can be compared directly.
//
// Map output files (bw_chunk_map_*, bw_seq_map_*) and the file-based input
// (bw_seq_*) are left on disk after the run and must be cleaned up manually.

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <fcntl.h>
#include <unistd.h>

#include "parlay/primitives.h"
#include "absl/log/check.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "utils/file_info.h"
#include "utils/logger.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_map.h"
#include "ChunkSequence/chunk_reduce.h"
#include "sequence_algorithms/map.h"
#include "sequence_algorithms/reduce.h"

// ── shared monoid ─────────────────────────────────────────────────────────────

struct SumMonoid {
    uint64_t identity = 0;
    uint64_t operator()(uint64_t a, uint64_t b) const { return a + b; }
};

// ── timing ────────────────────────────────────────────────────────────────────

using Clock = std::chrono::steady_clock;

static double elapsed(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ── data generation ───────────────────────────────────────────────────────────

// Write one file per SSD containing sequential uint64_t values [start, start+count).
// Returns FileInfo for each non-empty file with all fields populated.
static std::vector<FileInfo> make_seq_files(size_t n, const std::string& prefix) {
    const auto& ssds = GetSSDList();
    const size_t num_drives = ssds.size();
    const size_t elems_per_file = (n + num_drives - 1) / num_drives;

    std::vector<FileInfo> files(num_drives);
    parlay::parallel_for(0, num_drives, [&](size_t d) {
        const size_t start = d * elems_per_file;
        const size_t count = (start >= n) ? 0 : std::min(elems_per_file, n - start);
        if (count == 0) return;

        const size_t data_bytes = count * sizeof(uint64_t);
        const size_t file_bytes = AlignUp(data_bytes);
        const std::string fname = GetFileName(prefix, d);

        int fd = open(fname.c_str(), O_DIRECT | O_WRONLY | O_CREAT | O_TRUNC, 0644);
        SYSCALL(fd);

        uint64_t* buf = (uint64_t*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, file_bytes);
        CHECK(buf != nullptr);
        for (size_t i = 0; i < count; i++) buf[i] = (uint64_t)(start + i);
        if (file_bytes > data_bytes)
            memset((char*)buf + data_bytes, 0, file_bytes - data_bytes);
        SYSCALL(pwrite(fd, buf, file_bytes, 0));
        SYSCALL(close(fd));
        free(buf);

        // file_size == file_bytes because UnorderedFileReader reads the full
        // on-disk size; true_size carries the actual data byte count.
        files[d] = FileInfo(fname, d, data_bytes, file_bytes);
    }, /*granularity=*/1);

    // Drop drives that got no elements (only happens when n < num_drives).
    files.erase(
        std::remove_if(files.begin(), files.end(),
                       [](const FileInfo& f) { return f.true_size == 0; }),
        files.end());
    for (size_t i = 0; i < files.size(); i++) files[i].file_index = i;
    ComputeBeforeSize(files);
    return files;
}

// ── helpers ───────────────────────────────────────────────────────────────────

static size_t chunk_seq_bytes(const chunk_seq& seq) {
    size_t total = 0;
    for (const auto& c : seq.chunks) total += c.used;
    return total;
}

static double to_gb(size_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static void print_row(const std::string& label, size_t bytes, double secs) {
    std::cout << "  " << std::left  << std::setw(28) << label
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(7) << secs << "s"
              << "  " << std::setw(7) << std::setprecision(2)
              << to_gb(bytes) / secs << " GB/s\n";
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);

    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 5'000'000ULL;

    // ── generate data ─────────────────────────────────────────────────────────

    std::cout << "Generating chunk_seq perm(" << n << ")...\n" << std::flush;
    const chunk_seq cseq = ChunkSequenceOps::perm(n);
    const size_t cseq_total = chunk_seq_bytes(cseq);
    std::cout << "  " << cseq.chunks.size() << " chunks, "
              << std::fixed << std::setprecision(3) << to_gb(cseq_total) << " GB\n";

    std::cout << "Generating file-based seq perm(" << n << ")...\n" << std::flush;
    const std::vector<FileInfo> seq_files = make_seq_files(n, "bw_seq");
    size_t seq_total = 0;
    for (const auto& f : seq_files) seq_total += f.true_size;
    std::cout << "  " << seq_files.size() << " files, "
              << to_gb(seq_total) << " GB\n\n";

    // ── reduce ────────────────────────────────────────────────────────────────

    double chunk_reduce_s, reduce_s, chunk_map_s, map_s;

    std::cout << "--- Reduce (sum) ---\n";
    {
        auto t0 = Clock::now();
        volatile uint64_t r = ChunkSequenceOps::ChunkReduce<uint64_t>(cseq, SumMonoid{});
        chunk_reduce_s = elapsed(t0);
        (void)r;
        print_row("ChunkReduce", cseq_total, chunk_reduce_s);
    }
    {
        auto seq_copy = seq_files;   // Map/Reduce take by value
        auto t0 = Clock::now();
        volatile uint64_t r = Reduce<uint64_t>(seq_copy, SumMonoid{});
        reduce_s = elapsed(t0);
        (void)r;
        print_row("Reduce", seq_total, reduce_s);
    }

    // ── map ───────────────────────────────────────────────────────────────────

    std::cout << "\n--- Map (x -> x+1, read + write) ---\n";
    {
        auto t0 = Clock::now();
        ChunkSequenceOps::ChunkMap<uint64_t>(cseq, "bw_chunk_map",
                           [](uint64_t x) { return x + 1; });
        chunk_map_s = elapsed(t0);
        print_row("ChunkMap", cseq_total, chunk_map_s);
    }
    {
        auto seq_copy = seq_files;
        auto t0 = Clock::now();
        // NOTE: you can make this <uint64_t, uint64_t, false> to force the out of place branch
        Map<uint64_t>(seq_copy, "bw_seq_map",
                      std::function<uint64_t(uint64_t)>([](uint64_t x) { return x + 1; }));
        map_s = elapsed(t0);
        print_row("Map", seq_total, map_s);
    }

    std::cout << "\n(throughput is input bytes / wall time; "
                 "Map also writes the same volume to disk)\n";

    // Machine-readable line for the scaling driver (operation times only, in
    // seconds).  Columns: n,map_s,chunkmap_s,reduce_s,chunkreduce_s
    std::cout << "CSV," << n << ','
              << std::setprecision(9)
              << map_s << ',' << chunk_map_s << ','
              << reduce_s << ',' << chunk_reduce_s << '\n';
    return 0;
}
