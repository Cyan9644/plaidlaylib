// Example: out-of-core prime sieve via ChunkFlatTabulate.
//
// Computes every prime in [0, n) with data stored across the SSDs.  Small primes
// (up to sqrt(n)) are sieved in memory, then each 4 MB virtual chunk [start, end)
// is sieved independently in parallel and its survivors collected; ChunkFlatTabulate
// packs the variable-length per-chunk survivor lists into a dense output chunk_seq.
//
// Dual-purpose, like the benchmarks: prints human-readable results AND a
// machine-readable "CSV," line that benchmarks/run_benches.py greps.  The examples
// sweep (make bench-examples) times chunk_primes() across a sweep of n.
//
// When it fits in RAM the driver also times parlaylib's own in-memory sieve
// (deps/parlaylib-examples/primes.h, the upstream original of in_mem_primes
// below) over the full range as a DRAM baseline, and cross-checks the prime
// counts (exits non-zero on a mismatch).  Its footprint is ~10n bytes (n+1
// flags + a materialized iota + the output).  Budget: half of physical RAM,
// override via EXAMPLE_INMEM_BUDGET_BYTES; when skipped the CSV field is left
// blank so the plotted in-mem line stops at the RAM cliff (as in
// delayed_compare).
//
//   usage: primesExample [global --flags] [n] [consolidate_out_path]
//     n                    sieve range (default 1e6)
//     consolidate_out_path if given, write the full prime list as packed
//                          uint64_t to this local file (skipped at bench scale)
//
// CSV line: CSV,<n>,<time_s>,<inmem_time_s>,<count>,<throughput_gb_s>
//
// Complexity: O(n log log n) work, O(c log n) span, c = n / ELEMS_PER_CHUNK.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "absl/log/check.h"

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

// Upstream parlaylib example (fetched by `make deps`), used only as the
// in-memory comparison baseline; the out-of-core sieve below stays
// self-contained on its own in_mem_primes.  Defines global primes(long) —
// no clash with anything here.
#include "parlaylib-examples/primes.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "utils/logger.h"
#include "ChunkSequence/chunk_flat_tabulate.h"
#include "ChunkSequence/chunk_seq.h"

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}
static double to_gb(size_t bytes) { return (double)bytes / (1024.0 * 1024.0 * 1024.0); }

static void cleanup_prefix(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++) unlink(GetFileName(prefix, d).c_str());
}

// **************************************************************
// In-memory sieve for small primes up to sqrt(n).
// At n = 2^40, sqrt(n) ~ 2^20: the small-primes list is a few MB — fits in RAM.
// **************************************************************
parlay::sequence<long> in_mem_primes(long n) {
    if (n < 2) return {};
    long sqrt_n = (long)std::sqrt((double)n);
    auto sqrt_primes = in_mem_primes(sqrt_n);
    parlay::sequence<bool> flags(n + 1, true);
    parlay::parallel_for(0, n / sqrt_n + 1, [&](long i) {
        long start = sqrt_n * i;
        long end   = (std::min)(start + sqrt_n, n + 1);
        for (long j = 0; j < (long)sqrt_primes.size(); j++) {
            long p     = sqrt_primes[j];
            long first = (std::max)(2 * p, (((start - 1) / p) + 1) * p);
            for (long k = first; k < end; k += p)
                flags[k] = false;
        }
    }, 1);
    flags[0] = flags[1] = false;
    return parlay::filter(parlay::iota<long>(n + 1),
                          [&](long i) { return flags[i]; });
}

// **************************************************************
// Out-of-core primes sieve via ChunkFlatTabulate.  Each 4 MB virtual chunk
// [start, end) is sieved independently against the small primes, and its
// surviving indices are collected as uint64_t; ChunkFlatTabulate packs the
// variable-length per-chunk lists into a dense, index-ordered chunk_seq.
// **************************************************************
chunk_seq chunk_primes(size_t n, const std::string& result_prefix) {
    long sqrt_n = (long)std::sqrt((double)n);
    // Correct for floating-point rounding: ensure sqrt_n^2 <= n < (sqrt_n+1)^2.
    while ((long long)(sqrt_n + 1) * (sqrt_n + 1) <= (long long)n) sqrt_n++;

    parlay::sequence<long> small = in_mem_primes(sqrt_n);

    return ChunkSequenceOps::ChunkFlatTabulate<uint64_t>(n + 1, result_prefix,
        [&](size_t start, size_t end) {
            std::vector<bool> flags(end - start, true);
            for (long p : small) {
                size_t first = std::max((size_t)(2 * p),
                                        (((start - 1) / p) + 1) * p);
                for (size_t k = first; k < end; k += (size_t)p)
                    flags[k - start] = false;
            }
            parlay::sequence<uint64_t> out;
            size_t lo = (start < 2) ? 2 : start;
            for (size_t i = lo; i < end; i++)
                if (flags[i - start]) out.push_back((uint64_t)i);
            return out;
        });
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
    const std::string out_path = (argc > 2) ? argv[2] : "";  // opt-in consolidate

    // RAM budget for the in-memory parlaylib baseline (as in delayed_compare):
    // resident set ≈ n+1 bool flags + the materialized iota(n+1) it filters
    // (8 B each) + the output ≈ 10n bytes.
    const size_t phys = (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
    size_t budget = phys / 2;
    if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES")) budget = std::stoull(e);
    const bool inmem_ok = n * 10 <= budget;

    std::cout << "Sieving primes in [0, " << n << ")..." << std::flush;
    const std::string prefix = "primes";
    auto t0 = Clock::now();
    chunk_seq primes_seq = chunk_primes(n, prefix);
    const double secs = elapsed(t0);
    std::cout << " done\n";

    size_t count = 0, out_bytes = 0;
    for (const auto& c : primes_seq.chunks) {
        count += c.used / sizeof(uint64_t);
        out_bytes += c.used;
    }

    std::cout << "pi(" << n << ") = " << count << "   "
              << std::fixed << std::setprecision(4) << secs << "s   "
              << std::setprecision(2) << to_gb(out_bytes) / secs
              << " GB/s (eff. output)\n";

    // In-memory baseline: parlaylib's primes(n) over the full range (returns
    // primes <= n, matching chunk_primes's inclusive range), cross-checked by
    // prime count.
    bool agree = true;
    double inmem_secs = 0;
    if (inmem_ok) {
        auto t1 = Clock::now();
        auto primes_mem = primes((long)n);
        inmem_secs = elapsed(t1);
        std::cout << "in-mem parlaylib sieve: pi(" << n << ") = " << primes_mem.size()
                  << "   " << std::setprecision(4) << inmem_secs << "s\n";
        if (primes_mem.size() != count) {
            std::cout << "*** MISMATCH: in-mem count " << primes_mem.size()
                      << " != out-of-core count " << count << " ***\n";
            agree = false;
        }
    } else {
        std::cout << "in-mem parlaylib sieve: skipped (~10n footprint exceeds "
                  << "RAM budget " << std::setprecision(2) << to_gb(budget) << " GB)\n";
    }

    // Read the last output chunk and print the final few primes (cheap: one chunk).
    if (count > 0) {
        const chunk& last = primes_seq.chunks.back();
        const size_t last_n = last.used / sizeof(uint64_t);
        void* buf = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
        CHECK(buf != nullptr);
        int fd = open(last.filename.c_str(), O_DIRECT | O_RDONLY);
        SYSCALL(fd);
        SYSCALL(pread(fd, buf, AlignUp(last.used), (off_t)last.begin_addr));
        close(fd);

        const uint64_t* elems = reinterpret_cast<const uint64_t*>(buf);
        const size_t show = std::min(last_n, size_t(10));
        std::cout << "last " << show << " prime(s):";
        for (size_t i = last_n - show; i < last_n; i++)
            std::cout << " " << elems[i];
        std::cout << "\n";
        free(buf);
    }

    // Opt-in: write the full prime sequence as packed uint64_t to a local file.
    // Skipped by default (and by the benchmark sweep) — at scale this is a huge
    // single-file write that is not part of the out-of-core algorithm.
    if (!out_path.empty()) {
        primes_seq.consolidate(out_path);
        std::cout << "written to " << out_path << "\n";
    }

    // Machine-readable line for benchmarks/run_benches.py (examples sweep).
    // Columns: n,time_s,inmem_time_s,count,throughput_gb_s
    // (inmem_time_s blank when the footprint exceeds the RAM budget).
    auto f9 = [](double v) { std::ostringstream o; o << std::setprecision(9) << v; return o.str(); };
    std::cout << "CSV," << n << ',' << f9(secs) << ','
              << (inmem_ok ? f9(inmem_secs) : std::string()) << ',' << count
              << ',' << f9(to_gb(out_bytes) / secs) << '\n';

    cleanup_prefix(prefix);  // don't leave the output on the drives across sweep points
    return agree ? 0 : 1;
}
