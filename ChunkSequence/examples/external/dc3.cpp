// Example: out-of-core suffix array construction via ChunkDC3 (DC3 / skew).
//
// Builds an n-char text across the SSDs (4-letter alphabet, deterministic from
// parlay::hash64), then computes the suffix array with ChunkDC3
// (examples/external/chunk_dc3.h): the Kärkkäinen–Sanders skew algorithm made
// streaming — every random rank gather is a sort-join, the final merge is one
// comparator sort, and the recursion drops to an in-memory base case once a
// level's text fits in DRAM.  Unlike the prefix-doubling sibling
// (suffix_array.cpp), DC3 moves only a constant multiple of the input (the
// recursion shrinks 2/3 per level) rather than an O(log n) multiple, so it is
// the natural out-of-core suffix array — this example is the head-to-head.
//
// Dual-purpose, like the other examples: prints human-readable results AND a
// machine-readable "CSV," line that benchmarks/run_benches.py greps.  The
// examples sweep (make bench-examples) times the construction across a sweep of n.
//
// When the text fits in RAM the driver also times parlaylib's own in-memory
// suffix_array (deps/parlaylib-examples/suffix_array.h) on the same text as a
// DRAM baseline, and cross-checks the full array element-wise (read back off the
// drives; exits non-zero on a mismatch) — the differential test in the spirit of
// the benchmarks.  Budget: half of physical RAM, override via
// EXAMPLE_INMEM_BUDGET_BYTES; additionally capped at n < 2^31 (upstream indexes
// with unsigned int).  When skipped the CSV field is left blank so the plotted
// in-mem line stops at the RAM cliff.
//
//   usage: dc3Example [global --flags] [n]
//     n   text length in chars (default 1e6)
//
// CSV line: CSV,<n>,<build_s>,<sa_s>,<inmem_sa_s>,<count>,<throughput_gb_s>
//   throughput = text bytes / sa_s.

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
// Its `suffix_array` is defined at global scope with no include guard, so this
// is the only upstream example header this translation unit may include.
#include "parlaylib-examples/suffix_array.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/examples/external/chunk_dc3.h"

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}
static double to_gb(size_t bytes) { return (double)bytes / (1024.0 * 1024.0 * 1024.0); }

static void cleanup_prefix(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++) unlink(GetFileName(prefix, d).c_str());
}

// Deterministic 4-letter text: char i, computable anywhere (non-zero chars).
static char text_at(size_t i) { return (char)('a' + parlay::hash64(i) % 4); }

// Element-wise check of the out-of-core uint32 suffix array against the in-mem
// reference: read each output chunk back off the drives in index order and
// compare every value.
static bool contents_equal(const chunk_seq& seq, const parlay::sequence<unsigned int>& expected) {
    std::vector<const chunk*> ordered;
    for (const auto& c : seq.chunks) ordered.push_back(&c);
    std::sort(ordered.begin(), ordered.end(),
              [](const chunk* a, const chunk* b) { return a->index < b->index; });

    void* buf = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
    CHECK(buf != nullptr) << "dc3: readback buffer alloc failed";
    size_t j = 0;
    bool ok = true;
    for (const chunk* c : ordered) {
        if (c->used == 0) continue;
        int fd = open(c->filename.c_str(), O_RDONLY | O_DIRECT);
        SYSCALL(fd);
        SYSCALL(pread(fd, buf, AlignUp(c->used), (off_t)c->begin_addr));
        close(fd);
        const auto* elems = (const uint32_t*)buf;
        const size_t cnt = c->used / sizeof(uint32_t);
        for (size_t i = 0; i < cnt && ok; i++, j++)
            ok = j < expected.size() && elems[i] == (uint32_t)expected[j];
    }
    free(buf);
    return ok && j == expected.size();
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    RaiseFdLimit();
    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
    CHECK(n > 0 && n < (size_t{1} << 32)) << "dc3: need 0 < n < 2^32";

    const size_t phys = (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
    size_t budget = phys / 2;
    if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES")) budget = std::stoull(e);
    const bool inmem_ok = (n < (size_t{1} << 31)) && (n <= budget / 32);

    const std::string text_prefix = "dc3_text";
    const std::string out_prefix  = "dc3_out";

    std::cout << std::fixed;
    std::cout << "Building " << n << "-char text..." << std::flush;
    auto t0 = Clock::now();
    chunk_seq text = ChunkSequenceOps::tabulate<char>(n, text_prefix, text_at);
    const double build_s = elapsed(t0);
    std::cout << " done (" << std::setprecision(4) << build_s << "s)\n";

    std::cout << "Building suffix array (DC3)..." << std::flush;
    t0 = Clock::now();
    chunk_seq sa = ChunkSequenceOps::ChunkDC3(text, out_prefix);
    const double sa_s = elapsed(t0);
    std::cout << " done\n";

    size_t count = 0;
    for (const auto& c : sa.chunks) count += c.used / sizeof(uint32_t);
    const double gb_s = to_gb(n) / sa_s;

    std::cout << "|SA| = " << count << "   " << std::setprecision(4) << sa_s << "s   "
              << std::setprecision(2) << gb_s << " GB/s (text size / build time)\n";

    // In-memory baseline: parlaylib's suffix_array on the same text (built in
    // DRAM outside the timed region), cross-checked element-wise.
    bool agree = true;
    double inmem_sa_s = 0;
    if (inmem_ok) {
        auto text_mem = parlay::tabulate(n, [](size_t i) { return (unsigned char)text_at(i); });
        t0 = Clock::now();
        auto sa_mem = suffix_array(text_mem);   // parlay::sequence<unsigned int>
        inmem_sa_s = elapsed(t0);
        std::cout << "in-mem parlaylib suffix_array: |SA| = " << sa_mem.size() << "   "
                  << std::setprecision(4) << inmem_sa_s << "s\n";
        if (sa_mem.size() != count) {
            std::cout << "*** MISMATCH: in-mem |SA| " << sa_mem.size()
                      << " != out-of-core |SA| " << count << " ***\n";
            agree = false;
        } else if (!contents_equal(sa, sa_mem)) {
            std::cout << "*** MISMATCH: in-mem suffix array differs from out-of-core output ***\n";
            agree = false;
        } else {
            std::cout << "cross-check: out-of-core suffix array matches parlaylib\n";
        }
    } else {
        std::cout << "in-mem parlaylib suffix_array: skipped (n exceeds RAM/index budget)\n";
    }

    // Machine-readable line for benchmarks/run_benches.py (examples sweep).
    // Columns: n,build_s,sa_s,inmem_sa_s,count,throughput_gb_s
    // (inmem_sa_s blank when the baseline was skipped).
    auto f9 = [](double v) { std::ostringstream o; o << std::setprecision(9) << v; return o.str(); };
    std::cout << "CSV," << n << ',' << f9(build_s) << ',' << f9(sa_s) << ','
              << (inmem_ok ? f9(inmem_sa_s) : std::string()) << ',' << count << ','
              << f9(gb_s) << '\n';

    // Don't leave text/output on the drives across sweep points.
    cleanup_prefix(text_prefix);
    cleanup_prefix(out_prefix);
    return agree ? 0 : 1;
}
