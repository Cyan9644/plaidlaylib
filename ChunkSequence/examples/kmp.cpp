// Example: out-of-core Knuth-Morris-Pratt string search via ChunkKmp.
//
// Builds an n-char text across the SSDs (4-letter alphabet, deterministic from
// parlay::hash64), then finds every occurrence of an m-char pattern with
// ChunkKmp: each 4 MB chunk is scanned sequentially by one worker, chunks in
// parallel, with cross-chunk matches caught via batch-local overlap.  The
// pattern is the text's own first m characters, so at least one match always
// exists (more occur by chance for small m).
//
// Dual-purpose, like the benchmarks: prints human-readable results AND a
// machine-readable "CSV," line that benchmarks/run_benches.py greps.  The
// examples sweep (make bench-examples) times the search across a sweep of n
// with m held constant.
//
//   usage: kmpExample [global --flags] [n] [m]
//     n   text length in chars (default 1e6)
//     m   pattern length in chars (default 16)
//
// CSV line: CSV,<n>,<m>,<build_s>,<search_s>,<count>,<throughput_gb_s>
//   throughput = text bytes read / search_s (search is one streaming read pass).
//
// Complexity: O(n) work, O(n / ELEMS_PER_CHUNK) span.

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

#include "parlay/primitives.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/examples/chunk_kmp.h"
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

// Deterministic 4-letter text: char i of the text, computable anywhere.
static char text_at(size_t i) { return (char)('a' + parlay::hash64(i) % 4); }

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
    const size_t m = (argc > 2) ? std::stoull(argv[2]) : 16;

    // Pattern = the text's own first m chars, so a match at 0 is guaranteed.
    std::string pattern(m, '\0');
    for (size_t i = 0; i < m; i++) pattern[i] = text_at(i);

    const std::string text_prefix = "kmp_text";
    const std::string out_prefix  = "kmp_out";

    std::cout << "Building " << n << "-char text..." << std::flush;
    auto t0 = Clock::now();
    chunk_seq text = ChunkSequenceOps::tabulate<char>(n, text_prefix, text_at);
    const double build_s = elapsed(t0);
    std::cout << " done (" << std::fixed << std::setprecision(4) << build_s
              << "s)\n";

    std::cout << "Searching for " << m << "-char pattern \"" << pattern
              << "\"..." << std::flush;
    t0 = Clock::now();
    chunk_seq matches = ChunkSequenceOps::ChunkKmp<char>(text, out_prefix, pattern);
    const double search_s = elapsed(t0);
    std::cout << " done\n";

    size_t count = 0;
    for (const auto& c : matches.chunks) count += c.used / sizeof(uint64_t);
    const double gb_s = to_gb(n) / search_s;

    std::cout << count << " match(es)   "
              << std::setprecision(4) << search_s << "s   "
              << std::setprecision(2) << gb_s << " GB/s (text read)\n";

    // Machine-readable line for benchmarks/run_benches.py (examples sweep).
    // Columns: n,m,build_s,search_s,count,throughput_gb_s
    auto f9 = [](double v) { std::ostringstream o; o << std::setprecision(9) << v; return o.str(); };
    std::cout << "CSV," << n << ',' << m << ',' << f9(build_s) << ','
              << f9(search_s) << ',' << count << ',' << f9(gb_s) << '\n';

    // Don't leave text/output on the drives across sweep points.
    cleanup_prefix(text_prefix);
    cleanup_prefix(out_prefix);
    return 0;
}
