

#include "examples/external/primes.h"
#include "examples/in_memory/in_memory_primes.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool cond, const std::string &name) {
    if (cond) {
        std::cout << "    [PASS] " << name << std::endl;
    } else {
        std::cout << "    [FAIL] " << name << std::endl;
        g_failures++;
    }
}

double SecondsSince(std::chrono::steady_clock::time_point start) {
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

// The deterministic flag/output filenames the external sieve uses for a prefix
// (mirrors primes(n, prefix) in primes.h), so we can delete them afterwards.
std::vector<std::string> ScratchFiles(const std::string &prefix) {
    std::vector<std::string> names;
    for (int i = 0; i < NUM_SSDS; i++) {
        names.push_back(prefix + "_flag_" + std::to_string(i));
        names.push_back(prefix + "_prime_" + std::to_string(i));
    }
    return names;
}

void RemoveFiles(const std::vector<std::string> &files) {
    for (const auto &f : files) unlink(f.c_str());
}

// Read an external prime result back into a flat, globally-ordered vector. Each
// chunk_header points at `used` bytes of size_t living at begin_address in its
// file; sorting headers by index reproduces the ascending global order.
std::vector<size_t> ReadExternalPrimes(const External_Sequence &out) {
    parlay::sequence<chunk_header> headers = out.ordered_underlying_sequence;
    std::sort(headers.begin(), headers.end(),
              [](const chunk_header &a, const chunk_header &b) { return a.index < b.index; });

    std::vector<size_t> result;
    for (const auto &h : headers) {
        const size_t n = h.used / sizeof(size_t);
        if (n == 0) continue;
        std::vector<size_t> buf(n);
        int fd = open(h.filename.c_str(), O_RDONLY);
        if (fd < 0) {
            std::cerr << "could not open output file " << h.filename << std::endl;
            g_failures++;
            continue;
        }
        ssize_t r = pread(fd, buf.data(), h.used, (off_t) h.begin_address);
        if (r != (ssize_t) h.used) {
            std::cerr << "short read of output " << h.filename << std::endl;
            g_failures++;
        }
        close(fd);
        result.insert(result.end(), buf.begin(), buf.end());
    }
    return result;
}

// Report the first place two prime lists diverge, to make failures debuggable.
void ReportFirstDiff(const std::vector<size_t> &ext, const std::vector<long> &mem) {
    const size_t n = std::min(ext.size(), mem.size());
    for (size_t i = 0; i < n; i++) {
        if (ext[i] != (size_t) mem[i]) {
            std::cout << "      first mismatch at position " << i
                      << ": external=" << ext[i] << " in-memory=" << mem[i] << std::endl;
            return;
        }
    }
    std::cout << "      lists agree on first " << n << " entries; sizes differ ("
              << "external=" << ext.size() << ", in-memory=" << mem.size() << ")" << std::endl;
}

struct Result {
    double mem_seconds;
    double ext_seconds;
    size_t mem_count;
    size_t ext_count;
};

Result RunOne(size_t n, int repeats) {
    std::cout << "n = " << n << "  (" << repeats << " timed repeat(s) each)" << std::endl;

    // --- in-memory sieve --------------------------------------------------
    parlay::sequence<long> mem_primes;
    double mem_best = 1e300;
    for (int r = 0; r < repeats; r++) {
        auto t0 = std::chrono::steady_clock::now();
        mem_primes = primes((long) n);  // in_memory_primes.h overload: primes(long)
        mem_best = std::min(mem_best, SecondsSince(t0));
    }

    // --- external sieve ---------------------------------------------------
    std::vector<size_t> ext_primes;
    double ext_best = 1e300;
    std::vector<std::string> last_files;
    for (int r = 0; r < repeats; r++) {
        const std::string prefix = "primes_bench_" + std::to_string(n) + "_" + std::to_string(r);
        std::vector<std::string> files = ScratchFiles(prefix);
        RemoveFiles(files);  // start clean so stale blocks can't be read back

        auto t0 = std::chrono::steady_clock::now();
        External_Sequence out = primes(n, prefix);  // primes.h overload: primes(size_t, prefix)
        ext_best = std::min(ext_best, SecondsSince(t0));

        ext_primes = ReadExternalPrimes(out);  // read-back not counted in the timing
        RemoveFiles(files);
        last_files = files;
    }

    // --- correctness ------------------------------------------------------
    bool counts_match = ext_primes.size() == mem_primes.size();
    Check(counts_match, "prime counts match (external vs in-memory)");

    bool values_match = counts_match;
    if (counts_match) {
        for (size_t i = 0; i < ext_primes.size(); i++) {
            if (ext_primes[i] != (size_t) mem_primes[i]) { values_match = false; break; }
        }
    }
    Check(values_match, "prime values match in ascending order");
    if (!values_match) {
        std::vector<long> mem_vec(mem_primes.begin(), mem_primes.end());
        ReportFirstDiff(ext_primes, mem_vec);
    }

    std::cout << "    in-memory: " << std::fixed << std::setprecision(4) << mem_best
              << " s   external: " << ext_best << " s   ("
              << "primes found: " << mem_primes.size() << ")" << std::endl;

    return Result{mem_best, ext_best, mem_primes.size(), ext_primes.size()};
}

}  // namespace

int main(int argc, char **argv) {
    std::vector<size_t> sizes;
    int repeats = 1;

    if (argc >= 2) {
        sizes.push_back((size_t) std::stoull(argv[1]));
        if (argc >= 3) repeats = std::max(1, std::atoi(argv[2]));
    } else {
        // Default sweep: small (cheap, exercises batch boundaries) up to a size
        // big enough for the external pipeline's IO cost to dominate.
        sizes = {1000, 1000000, 10000000, 100000000};
    }

    std::cout << "=== Prime sieve benchmark: external vs in-memory ===" << std::endl;
    std::cout << "NUM_SSDS = " << NUM_SSDS << ", chunk = 4 MiB" << std::endl;

    std::vector<std::pair<size_t, Result>> table;
    for (size_t n : sizes) {
        table.emplace_back(n, RunOne(n, repeats));
        std::cout << std::endl;
    }

    std::cout << "=== Summary (best of " << repeats << ") ===" << std::endl;
    std::cout << std::setw(14) << "n" << std::setw(14) << "in-mem (s)"
              << std::setw(14) << "external (s)" << std::setw(12) << "speedup"
              << std::setw(14) << "#primes" << std::endl;
    for (const auto &[n, res] : table) {
        double speedup = res.ext_seconds > 0 ? res.mem_seconds / res.ext_seconds : 0.0;
        std::cout << std::setw(14) << n
                  << std::setw(14) << std::fixed << std::setprecision(4) << res.mem_seconds
                  << std::setw(14) << res.ext_seconds
                  << std::setw(11) << std::setprecision(2) << speedup << "x"
                  << std::setw(14) << res.mem_count << std::endl;
    }

    std::cout << "====================================================" << std::endl;
    if (g_failures == 0) {
        std::cout << "All correctness checks passed." << std::endl;
        return 0;
    }
    std::cout << g_failures << " check(s) failed." << std::endl;
    return 1;
}
