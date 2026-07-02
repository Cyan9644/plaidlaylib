// find_if_benchmark.cpp
//
// Compares the external find_if (Plaidlay/extern_find_if.h, which streams an
// External_Sequence off the SSDs in batches of NUM_SSDS chunks) against the
// in-memory find_if (Plaidlay/examples/in_memory/find_if.h, the parlaylib
// doubling search). Structured after examples/external/primes_benchmark.cpp.
//
// Data set: an iota sequence (element value == its global index), written in
// *full* 4 MiB chunks by WriteIotaSequence below. This choice is deliberate:
// extern_find_if reports a match's global position as
//     chunk.index * buffer_size + j
// i.e. it assumes every chunk holds a full buffer_size elements. That holds for
// iota (only the final chunk may be short, and it sits last), so the external
// and in-memory searches return the *same* index and can be checked against
// each other. A variable-length sequence such as the primes sieve output does
// not satisfy that contract, so it would not give a meaningful index match.
//
// The predicate is "value >= threshold". Because value == index, the first
// match is exactly at index == threshold. By default threshold = N-1, so the
// match is the very last element and both implementations must traverse the
// whole sequence (extern_find_if has no early-out; it scans every chunk). Pass
// a position fraction in [0,1] to move the match earlier.
//
// Only the find_if call is timed; building the iota data on disk is setup.
//
// Build (from the Plaidlay/ directory):
//   g++ -std=c++17 -O2 -I . -I .. -I ../deps/abseil-cpp -I ./parlaylib/include \
//       examples/external/find_if_benchmark.cpp -o find_if_benchmark \
//       -lpthread -luring <abseil log libs>
//
// Run (defaults sweep 2^30 .. 2^36, one timed repeat, match at the last
// element, scratch files written flat in the cwd). The upper bound is 2^36
// because the in-memory side cannot hold a larger sequence in DRAM:
//   ./find_if_benchmark [min_exp] [max_exp] [repeats] [pos_frac] [ssd_base]
// ssd_base defaults to "" (flat files in the cwd, like primes_scaling_benchmark).
// Pass a mount prefix such as /mnt/ssd to spread one file per physical SSD across
// /mnt/ssd0../mnt/ssd29 instead (those mount dirs must exist and be writable).

#include "extern_find_if.h"                  // external find_if + reader/writer/External_Sequence
#include "examples/in_memory/find_if.h"      // in-memory find_if (overload on Range)

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using Elem = long;  // element type stored on disk and searched in memory

int g_failures = 0;

void Check(bool cond, const std::string &name) {
    std::cout << (cond ? "    [PASS] " : "    [FAIL] ") << name << std::endl;
    if (!cond) g_failures++;
}

double SecondsSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// One file per SSD. With a non-empty ssd_base the files land on the per-SSD
// mounts, e.g. /mnt/ssd0/find_if_bench_e30 .. /mnt/ssd29/find_if_bench_e30
// (requires those mount directories to exist and be writable). With an empty
// ssd_base the files are flat names created in the current working directory --
// e.g. find_if_bench_e30_ssd0 .. find_if_bench_e30_ssd29 -- which mirrors how
// primes_scaling_benchmark writes its scratch files and works without root-owned
// SSD mounts.
std::vector<std::string> SsdFiles(const std::string &ssd_base, int exp) {
    std::vector<std::string> names;
    names.reserve(NUM_SSDS);
    for (int i = 0; i < NUM_SSDS; i++) {
        if (ssd_base.empty()) {
            names.push_back("find_if_bench_e" + std::to_string(exp) + "_ssd" + std::to_string(i));
        } else {
            names.push_back(ssd_base + std::to_string(i) + "/find_if_bench_e" + std::to_string(exp));
        }
    }
    return names;
}

void RemoveFiles(const std::vector<std::string> &files) {
    for (const auto &f : files) unlink(f.c_str());
}

// Write the sequence [0, 1, ..., n-1] of T across NUM_SSDS files as full 4 MiB
// chunks, returning the index-ordered External_Sequence. Mirrors ExternalIota.h
// but is inlined here so this file only includes extern_find_if.h (ExternalIota.h
// re-includes the unguarded reader/writer headers and would clash).
template <typename T>
External_Sequence WriteIotaSequence(size_t n, const std::vector<std::string> &filenames) {
    constexpr size_t buffer_size_bytes = 4 << 20;
    constexpr size_t buffer_size = buffer_size_bytes / sizeof(T);

    const size_t num_chunks = (n + buffer_size - 1) / buffer_size;             // ceil
    const size_t expected_write_count = (num_chunks + NUM_SSDS - 1) / NUM_SSDS; // batches (ceil)

    External_Sequence seq(num_chunks);

    UnorderedChunkWriter<T> writer;
    UnorderedChunkWriterConfig wconfig;
    wconfig.num_threads = WRITER_THREADS;
    writer.Start(filenames, wconfig);

    std::vector<T *> buffer(NUM_SSDS);
    std::array<std::atomic<size_t>, NUM_SSDS> file_offsets{};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distrib(0, NUM_SSDS - 1);

    size_t write_count = 0;
    while (write_count < expected_write_count) {
        for (int i = 0; i < NUM_SSDS; i++) {
            buffer[i] = (T *) aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, buffer_size_bytes);
        }

        std::atomic<int> counter(0);
        std::vector<unsigned int> random_holder(NUM_SSDS);
        std::vector<char> bad(NUM_SSDS, 0);   // each slot written by exactly one thread
        std::vector<int> slot_for(NUM_SSDS, -1);
        for (int k = 0; k < NUM_SSDS; k++) random_holder[k] = distrib(gen);

        parlay::parallel_for(0, NUM_SSDS, [&](size_t i) {
            const size_t global_chunk = write_count * NUM_SSDS + i;
            if (global_chunk >= num_chunks) {       // final batch may have empty slots
                bad[i] = 1;
                return;
            }
            const size_t begin_val = global_chunk * buffer_size;  // first value in this chunk
            const size_t valid = (begin_val < n) ? std::min(buffer_size, n - begin_val) : 0;

            for (size_t k = 0; k < buffer_size; k++) {
                buffer[i][k] = (T)(begin_val + k);  // values past `valid` are padding (see `used`)
            }

            chunk_header ch;
            ch.index = global_chunk;
            ch.filename = filenames[random_holder[i]];
            ch.used = valid * sizeof(T);
            ch.begin_address = 0;                   // filled in after we know the offset
            const int slot = counter.fetch_add(1);
            slot_for[i] = slot;
            seq.ordered_underlying_sequence[write_count * NUM_SSDS + slot] = ch;
        });

        for (int r = 0; r < NUM_SSDS; r++) {
            if (!bad[r]) {
                const size_t base_offset = file_offsets[random_holder[r]].fetch_add(buffer_size_bytes);
                seq.ordered_underlying_sequence[write_count * NUM_SSDS + slot_for[r]].begin_address = base_offset;
                writer.Push(std::shared_ptr<T>(buffer[r], free), buffer_size, random_holder[r], base_offset);
            } else {
                free(buffer[r]);
            }
        }
        write_count++;
    }

    writer.Wait();
    std::sort(seq.begin(), seq.end(),
              [](const chunk_header &a, const chunk_header &b) { return a.index < b.index; });
    return seq;
}

struct Result {
    double mem_seconds;
    double ext_seconds;
    size_t mem_index;
    size_t ext_index;
    bool   mem_materialized;
};

// Above this many bytes the in-memory side searches a lazy (delayed) iota range
// instead of a materialized DRAM sequence, so the benchmark can still run past
// what DRAM holds. 512 GiB == 2^36 longs, the largest the in-memory side can
// hold in memory and the default upper bound of the sweep. Lower this if your
// machine has less DRAM (sizes above it fall back to a lazy range).
constexpr size_t kMaterializeMaxBytes = 512ull << 30;

Result RunOne(int exp, int repeats, double pos_frac, const std::string &ssd_base) {
    const size_t n = (size_t) 1 << exp;
    const Elem threshold = (Elem)(pos_frac * (double)(n - 1));  // first value >= threshold is at index == threshold
    auto pred = [threshold](Elem x) { return x >= threshold; };

    std::cout << "exp = " << exp << "  (N = " << n << " elements, "
              << (n * sizeof(Elem)) / (1ull << 30) << " GiB on disk, "
              << repeats << " timed repeat(s), match at index " << threshold << ")" << std::endl;

    // --- in-memory find_if ------------------------------------------------
    const bool materialize = n * sizeof(Elem) <= kMaterializeMaxBytes;
    double mem_best = 1e300;
    size_t mem_index = 0;
    if (materialize) {
        auto data = parlay::tabulate(n, [](size_t i) { return (Elem) i; });
        for (int r = 0; r < repeats; r++) {
            auto t0 = std::chrono::steady_clock::now();
            mem_index = (size_t) ::find_if(data, pred);  // ::-qualified: disable ADL, pick in-memory overload
            mem_best = std::min(mem_best, SecondsSince(t0));
        }
    } else {
        // Lazy range: O(1) memory, pure compute. (Does not model DRAM bandwidth.)
        auto data = parlay::delayed_tabulate(n, [](size_t i) { return (Elem) i; });
        for (int r = 0; r < repeats; r++) {
            auto t0 = std::chrono::steady_clock::now();
            mem_index = (size_t) ::find_if(data, pred);  // ::-qualified: disable ADL, pick in-memory overload
            mem_best = std::min(mem_best, SecondsSince(t0));
        }
    }

    // --- external find_if -------------------------------------------------
    std::vector<std::string> files = SsdFiles(ssd_base, exp);
    RemoveFiles(files);  // start clean so stale blocks can't be read back
    External_Sequence seq = WriteIotaSequence<Elem>(n, files);  // setup, not timed

    double ext_best = 1e300;
    size_t ext_index = 0;
    for (int r = 0; r < repeats; r++) {
        auto t0 = std::chrono::steady_clock::now();
        ext_index = ::find_if<Elem>(seq, pred);
        ext_best = std::min(ext_best, SecondsSince(t0));
    }
    RemoveFiles(files);

    // --- correctness ------------------------------------------------------
    Check(mem_index == (size_t) threshold, "in-memory found the expected index");
    Check(ext_index == (size_t) threshold, "external found the expected index");
    Check(mem_index == ext_index, "external and in-memory indices agree");

    const double gib = (double)(n * sizeof(Elem)) / (double)(1ull << 30);
    std::cout << "    in-memory: " << std::fixed << std::setprecision(4) << mem_best
              << " s" << (materialize ? "" : " (lazy)")
              << "   external: " << ext_best << " s"
              << "   (external read " << std::setprecision(2) << (gib / ext_best) << " GiB/s)"
              << std::endl;

    return Result{mem_best, ext_best, mem_index, ext_index, materialize};
}

}  // namespace

int main(int argc, char **argv) {
    int min_exp = 30;
    int max_exp = 36;
    int repeats = 1;
    double pos_frac = 1.0;          // match at the last element (worst case) by default
    std::string ssd_base = "";      // empty => flat scratch files in the cwd (primes-style)

    if (argc >= 2) min_exp = std::atoi(argv[1]);
    if (argc >= 3) max_exp = std::atoi(argv[2]);
    if (argc >= 4) repeats = std::max(1, std::atoi(argv[3]));
    if (argc >= 5) pos_frac = std::min(1.0, std::max(0.0, std::atof(argv[4])));
    if (argc >= 6) ssd_base = argv[5];

    std::cout << "=== find_if benchmark: external vs in-memory ===" << std::endl;
    std::cout << "NUM_SSDS = " << NUM_SSDS << ", chunk = 4 MiB, element = " << sizeof(Elem)
              << " B, position fraction = " << pos_frac << std::endl;
    std::cout << "in-memory materialized when data <= " << (kMaterializeMaxBytes >> 30)
              << " GiB, else lazy" << std::endl;

    std::vector<std::pair<int, Result>> table;
    for (int exp = min_exp; exp <= max_exp; exp++) {
        table.emplace_back(exp, RunOne(exp, repeats, pos_frac, ssd_base));
        std::cout << std::endl;
    }

    std::cout << "=== Summary (best of " << repeats << ") ===" << std::endl;
    std::cout << std::setw(6) << "exp" << std::setw(16) << "N"
              << std::setw(16) << "in-mem (s)" << std::setw(16) << "external (s)"
              << std::setw(12) << "speedup" << std::setw(10) << "match" << std::endl;
    for (const auto &[exp, res] : table) {
        const double speedup = res.ext_seconds > 0 ? res.mem_seconds / res.ext_seconds : 0.0;
        std::cout << std::setw(6) << exp
                  << std::setw(16) << ((size_t) 1 << exp)
                  << std::setw(16) << std::fixed << std::setprecision(4) << res.mem_seconds
                  << (res.mem_materialized ? " " : "L")
                  << std::setw(15) << res.ext_seconds
                  << std::setw(11) << std::setprecision(2) << speedup << "x"
                  << std::setw(10) << (res.mem_index == res.ext_index ? "yes" : "NO")
                  << std::endl;
    }
    std::cout << "(L = in-memory ran on a lazy range because the data exceeds the materialize cap)" << std::endl;

    std::cout << "================================================" << std::endl;
    if (g_failures == 0) {
        std::cout << "All correctness checks passed." << std::endl;
        return 0;
    }
    std::cout << g_failures << " check(s) failed." << std::endl;
    return 1;
}
