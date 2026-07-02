// Scaling harness for the external prime sieve (primes.h) vs the in-memory
// sieve, used to produce the time-vs-input-size plot.
//
// It runs exactly ONE variant for ONE input size per process invocation. That
// isolation is deliberate: the in-memory sieve is expected to exhaust RAM on the
// larger sizes, and when the OOM killer fires it takes down the whole process.
// By keeping each measurement in its own process, the Python driver
// (plot_primes_benchmark.py) loses only the single OOM'd data point and can keep
// sweeping the remaining sizes.
//
// On success it prints one CSV-friendly line to stdout and exits 0:
//     <variant>,<n>,<seconds>,<peak_rss_kb>,<count>
// On bad arguments it exits 2. An OOM kill surfaces to the parent as a SIGKILL
// (non-zero wait status), which the driver records as out-of-memory.

#include "examples/external/primes.h"
#include "examples/in_memory/in_memory_primes.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <sys/resource.h>
#include <unistd.h>

namespace {

double SecondsSince(std::chrono::steady_clock::time_point start) {
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

// Peak resident set size of this process. On Linux ru_maxrss is in kilobytes.
long PeakRssKb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;
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

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <mem|ext> <n>" << std::endl;
        return 2;
    }
    const std::string variant = argv[1];
    const size_t n = std::stoull(argv[2]);

    double seconds = 0.0;
    size_t count = 0;

    if (variant == "mem") {
        // In-memory sieve from in_memory_primes.h: primes(long).
        auto t0 = std::chrono::steady_clock::now();
        parlay::sequence<long> p = primes((long) n);
        seconds = SecondsSince(t0);
        count = p.size();
    } else if (variant == "ext") {
        // External sieve from primes.h: primes(size_t, prefix). Start from a
        // clean slate so stale on-disk blocks can't be read back, and clean up
        // afterwards so we don't leak ~n bytes of flag/output files per run.
        const std::string prefix = "primes_scaling_" + std::to_string(n);
        std::vector<std::string> files = ScratchFiles(prefix);
        RemoveFiles(files);

        auto t0 = std::chrono::steady_clock::now();
        External_Sequence out = primes(n, prefix);
        seconds = SecondsSince(t0);

        for (const auto &h : out.ordered_underlying_sequence)
            count += h.used / sizeof(size_t);
        RemoveFiles(files);
    } else {
        std::cerr << "unknown variant: " << variant << " (expected mem|ext)" << std::endl;
        return 2;
    }

    std::printf("%s,%zu,%.6f,%ld,%zu\n",
                variant.c_str(), n, seconds, PeakRssKb(), count);
    return 0;
}
