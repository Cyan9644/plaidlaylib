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
// When the text fits in RAM the driver also times parlaylib's own in-memory
// KMP (deps/parlaylib-examples/knuth_morris_pratt.h) on the same text as a
// DRAM baseline, and cross-checks the match count and the exact positions
// (read back element-wise; exits non-zero on a mismatch).  Budget: half of
// physical RAM, override via
// EXAMPLE_INMEM_BUDGET_BYTES; when skipped the CSV field is left blank so the
// plotted in-mem line stops at the RAM cliff (as in delayed_compare).
//
//   usage: kmpExample [global --flags] [n] [m]
//     n   text length in chars (default 1e6)
//     m   pattern length in chars (default 16)
//
// CSV line: CSV,<n>,<m>,<build_s>,<search_s>,<inmem_search_s>,<count>,<throughput_gb_s>
//   throughput = text bytes read / search_s (search is one streaming read pass).
//
// Complexity: O(n) work, O(n / ELEMS_PER_CHUNK) span.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

#include "absl/log/check.h"

#include "parlay/primitives.h"

// Upstream parlaylib example (fetched by `make deps`): the in-memory baseline.
#include "parlaylib-examples/knuth_morris_pratt.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "utils/trace_marker.h"
#include "utils/logger.h"
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

// Element-wise check of an out-of-core uint64_t result against the in-mem
// baseline's sequence: read each output chunk back off the drives in index
// order and compare every value.  Only called when the baseline ran, so
// `expected` fits in RAM by construction.
template <typename Seq>
static bool contents_equal(const chunk_seq& cs, const Seq& expected) {
    void* buf = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
    CHECK(buf != nullptr);
    bool ok = true;
    size_t j = 0;
    for (const chunk& c : cs.chunks) {
        if (!ok || c.used == 0) continue;
        int fd = open(c.filename.c_str(), O_DIRECT | O_RDONLY);
        SYSCALL(fd);
        SYSCALL(pread(fd, buf, AlignUp(c.used), (off_t)c.begin_addr));
        close(fd);
        const uint64_t* elems = reinterpret_cast<const uint64_t*>(buf);
        const size_t cnt = c.used / sizeof(uint64_t);
        for (size_t i = 0; i < cnt && ok; i++, j++)
            ok = j < expected.size() && elems[i] == (uint64_t)expected[j];
    }
    free(buf);
    return ok && j == expected.size();
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

    // RAM budget for the in-memory parlaylib baseline (as in delayed_compare):
    // its resident set is ≈ the n-char text itself.
    const size_t phys = (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
    size_t budget = phys / 2;
    if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES")) budget = std::stoull(e);
    const bool inmem_ok = n <= budget;

    const std::string text_prefix = "kmp_text";
    const std::string out_prefix  = "kmp_out";

    std::cout << "Building " << n << "-char text..." << std::flush;
    trace_mark("build_start");
    auto t0 = Clock::now();
    chunk_seq text = ChunkSequenceOps::tabulate<char>(n, text_prefix, text_at);
    const double build_s = elapsed(t0);
    trace_mark("build_end");
    std::cout << " done (" << std::fixed << std::setprecision(4) << build_s
              << "s)\n";

    std::cout << "Searching for " << m << "-char pattern \"" << pattern
              << "\"..." << std::flush;
    trace_mark("op_start");
    t0 = Clock::now();
    chunk_seq matches = ChunkSequenceOps::ChunkKmp<char>(text, out_prefix, pattern);
    const double search_s = elapsed(t0);
    trace_mark("op_end");
    std::cout << " done\n";

    size_t count = 0;
    for (const auto& c : matches.chunks) count += c.used / sizeof(uint64_t);
    const double gb_s = to_gb(n) / search_s;

    std::cout << count << " match(es)   "
              << std::setprecision(4) << search_s << "s   "
              << std::setprecision(2) << gb_s << " GB/s (text read)\n";

    // In-memory baseline: parlaylib's knuth_morris_pratt on the same text
    // (built in DRAM outside the timed region), cross-checked by match count.
    bool agree = true;
    double inmem_search_s = 0;
    if (inmem_ok) {
        auto text_mem    = parlay::tabulate(n, text_at);   // parlay::sequence<char>
        auto pattern_mem = parlay::tabulate(m, text_at);
        t0 = Clock::now();
        auto matches_mem = knuth_morris_pratt(text_mem, pattern_mem);
        inmem_search_s = elapsed(t0);
        std::cout << "in-mem parlaylib KMP: " << matches_mem.size() << " match(es)   "
                  << std::setprecision(4) << inmem_search_s << "s\n";
        if (matches_mem.size() != count) {
            std::cout << "*** MISMATCH: in-mem count " << matches_mem.size()
                      << " != out-of-core count " << count << " ***\n";
            agree = false;
        } else if (!contents_equal(matches, matches_mem)) {
            std::cout << "*** MISMATCH: in-mem match positions differ from "
                      << "out-of-core output ***\n";
            agree = false;
        }
    } else {
        std::cout << "in-mem parlaylib KMP: skipped (text exceeds RAM budget "
                  << std::setprecision(2) << to_gb(budget) << " GB)\n";
    }

    // Machine-readable line for benchmarks/run_benches.py (examples sweep).
    // Columns: n,m,build_s,search_s,inmem_search_s,count,throughput_gb_s
    // (inmem_search_s blank when the text exceeds the RAM budget).
    auto f9 = [](double v) { std::ostringstream o; o << std::setprecision(9) << v; return o.str(); };
    std::cout << "CSV," << n << ',' << m << ',' << f9(build_s) << ','
              << f9(search_s) << ',' << (inmem_ok ? f9(inmem_search_s) : std::string())
              << ',' << count << ',' << f9(gb_s) << '\n';

    // Don't leave text/output on the drives across sweep points.
    cleanup_prefix(text_prefix);
    cleanup_prefix(out_prefix);
    return agree ? 0 : 1;
}
