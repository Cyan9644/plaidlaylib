// Zip-depth scaling benchmark: how a many-way delayed zip chain scales as the
// number of zipped sequences (== the chain's nesting depth) grows.
//
// Builds a left-nested zip chain over K distinct source chunk_seqs,
// collapsing each zip back to a scalar immediately (the same
// zip-then-map-to-scalar fold idiom used by
// ChunkSequence/examples/chunk_bigint_add.h's carry-lookahead add), so the
// chain's *value type* stays uint64_t at every depth while its *node type*
// (map_node<zip_node<...>>) still deepens by one level per K. Each depth is
// therefore a distinct C++ type, built via compile-time template recursion
// (build_chain<K>) — consistent with chunk_delayed.h's no-std::function
// design, and the only way to build a runtime-chosen zip depth without type
// erasure.
//
// A single run self-sweeps K = 1,2,4,8,... (doubling) up to max_k, timing
// cd::reduce(sum) over the K-way chain at each depth and printing one CSV line
// per depth (unlike the other benchmarks, which sweep by re-invoking the
// binary with a different n — depth here is a compile-time recursion, not a
// runtime parameter, so one process covers the whole sweep).
//
// All K sources are independently generated (distinct files/prefixes) so
// zip's Planner never dedups them by chunk_seq* — this measures genuine
// K-way read scaling, not the read-dedup fast path the "faster zipping"
// commit added for shared sources.
//
// argv[1] = n (elements per source, default 1<<20)
// argv[2] = max_k (default 64; hard compile-time ceiling MAX_K = 128)
//
// CSV line per depth: CSV,<k>,<n>,<reduce_s>,<throughput_gb_s>,<agree>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_delayed.h"

namespace cd = ChunkSequenceOps::delayed;

constexpr size_t MAX_K = 128;

struct SumMonoid {
    uint64_t identity = 0;
    uint64_t operator()(uint64_t a, uint64_t b) const { return a + b; }
};

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}
static double to_gb(size_t bytes) { return (double)bytes / (1024.0 * 1024.0 * 1024.0); }

static void cleanup_prefix(const std::string& prefix) {
    for (size_t d = 0; d < GetSSDList().size(); d++)
        unlink(GetFileName(prefix, d).c_str());
}

// Left-nested zip chain over srcs[0..K), each zip immediately collapsed to a
// scalar so every fold step's map lambda is pair<uint64_t,uint64_t> -> uint64_t
// regardless of depth.
template<size_t K>
auto build_chain(const std::vector<chunk_seq>& srcs) {
    if constexpr (K == 1) {
        return cd::map(cd::delay<uint64_t>(srcs[0]), [](uint64_t x) { return x; });
    } else {
        auto prev = build_chain<K - 1>(srcs);
        return cd::map(cd::zip(prev, cd::delay<uint64_t>(srcs[K - 1])),
                       [](std::pair<uint64_t, uint64_t> p) { return p.first + p.second; });
    }
}

static bool g_agree = true;

template<size_t K>
void run_depth(const std::vector<chunk_seq>& srcs, size_t n, size_t max_k) {
    auto chain = build_chain<K>(srcs);

    auto t0 = Clock::now();
    uint64_t r = cd::reduce(chain, SumMonoid{});
    double secs = elapsed(t0);

    const uint64_t per_source = n * (n - 1) / 2;
    const uint64_t expected = (uint64_t)K * per_source;
    const bool ok = (r == expected);
    if (!ok) {
        std::cout << "  *** MISMATCH at k=" << K << ": got " << r
                  << " expected " << expected << " ***\n";
        g_agree = false;
    }

    const size_t in_bytes = K * n * sizeof(uint64_t);
    std::cout << "  k=" << std::setw(4) << std::left << K << std::right
              << std::fixed << std::setprecision(4) << std::setw(9) << secs << "s   "
              << std::setprecision(2) << to_gb(in_bytes) / secs << " GB/s"
              << (ok ? "" : "   MISMATCH") << "\n";

    auto f9 = [](double v) { std::ostringstream o; o << std::setprecision(9) << v; return o.str(); };
    std::cout << "CSV," << K << ',' << n << ',' << f9(secs) << ','
              << f9(to_gb(in_bytes) / secs) << ',' << (ok ? 1 : 0) << '\n';

    if constexpr (K * 2 <= MAX_K) {
        if (K * 2 <= max_k) run_depth<K * 2>(srcs, n, max_k);
    }
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    const size_t n     = (argc > 1) ? std::stoull(argv[1]) : (size_t(1) << 20);
    size_t max_k        = (argc > 2) ? std::stoull(argv[2]) : 64;
    if (max_k > MAX_K) {
        std::cout << "max_k " << max_k << " exceeds compiled ceiling " << MAX_K
                  << "; clamping.\n";
        max_k = MAX_K;
    }

    std::cout << "Generating " << max_k << " source chunk_seqs of n=" << n << " each...\n";
    std::vector<chunk_seq> srcs(max_k);
    for (size_t i = 0; i < max_k; i++)
        srcs[i] = ChunkSequenceOps::tabulate<uint64_t>(
            n, "zsrc" + std::to_string(i), [](size_t j) { return (uint64_t)j; });
    std::cout << "\n--- zip chain depth sweep: reduce(sum) ---\n";

    run_depth<1>(srcs, n, max_k);

    for (size_t i = 0; i < max_k; i++) cleanup_prefix("zsrc" + std::to_string(i));

    std::cout << "\n" << (g_agree ? "agree=1 (all depths match)"
                                  : "agree=0 (MISMATCH — see above)") << "\n";
    return g_agree ? 0 : 1;
}
