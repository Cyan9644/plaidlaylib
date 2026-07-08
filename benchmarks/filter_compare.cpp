// Old vs new delayed-filter windowing benchmark.
//
// ChunkSequenceOps::delayed::filter is a windowed terminal built directly on
// for_each_window, so it inherited the node-tree rewrite (407b97d "alt zip
// impl with nodes, will bench") and the read-dedup Planner/Resolver machinery
// added on top of it (840ff7e "faster zipping") even though filter itself
// never zips anything: every chunk's single read now goes through the generic
// Planner::need / Resolver::next path instead of the old design's direct
// per-batch ChunkSequenceReader. This benchmark quantifies that windowing
// overhead as n grows, by timing three variants of the same filter on the
// same input:
//
//   eager        ChunkSequenceOps::ChunkFilter        — current eager
//                (DensePack-based); a device-throughput reference point, the
//                same role "raw read" plays in delayed_compare.cpp.
//   old-delayed  ChunkSequenceOps::old_delayed::filter — pre-node-tree design
//                (benchmarks/old_filter/old_chunk_delayed.h, frozen at c5c3406).
//   new-delayed  ChunkSequenceOps::delayed::filter      — current node/Planner
//                windowing.
//
// Cross-substrate correctness check on every run (element count + sum of
// survivors, all read back via the current eager ChunkReduce); exits non-zero
// on mismatch.  Intermediates are cleaned up after each pipeline step.
//
// CSV line: CSV,<n>,<in_bytes>,<eager_s>,<old_delayed_s>,<new_delayed_s>,<agree>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <string>
#include <unistd.h>

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_reduce.h"
#include "ChunkSequence/chunk_filter.h"
#include "ChunkSequence/chunk_delayed.h"
#include "benchmarks/old_filter/old_chunk_delayed.h"

namespace cd  = ChunkSequenceOps::delayed;
namespace ocd = ChunkSequenceOps::old_delayed;

struct SumMonoid {
    uint64_t identity = 0;
    uint64_t operator()(uint64_t a, uint64_t b) const { return a + b; }
};

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

static size_t chunk_seq_bytes(const chunk_seq& seq) {
    size_t total = 0;
    for (const auto& c : seq.chunks) total += c.used;
    return total;
}
static double to_gb(size_t bytes) { return (double)bytes / (1024.0 * 1024.0 * 1024.0); }

static void cleanup_prefix(const std::string& prefix) {
    for (size_t d = 0; d < GetSSDList().size(); d++)
        unlink(GetFileName(prefix, d).c_str());
}

static void print_row(const std::string& label, size_t in_bytes, double secs) {
    std::cout << "  " << std::left << std::setw(16) << label << std::right
              << std::fixed << std::setprecision(4) << std::setw(9) << secs << "s   "
              << std::setprecision(2) << to_gb(in_bytes) / secs << " GB/s (eff. input)\n";
}

// Read back a filtered chunk_seq's element count and sum for cross-checking.
static std::pair<size_t, uint64_t> count_and_sum(const chunk_seq& out) {
    size_t count = 0;
    for (const auto& c : out.chunks) count += c.used / sizeof(uint64_t);
    uint64_t sum = ChunkSequenceOps::ChunkReduce<uint64_t>(out, SumMonoid{});
    return {count, sum};
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 5'000'000ULL;

    auto keep_even = [](uint64_t x) { return x % 2 == 0; };

    std::cout << "Generating chunk_seq perm(" << n << ")..." << std::flush;
    const chunk_seq cseq = ChunkSequenceOps::perm(n);
    const size_t in_bytes = chunk_seq_bytes(cseq);
    std::cout << " " << cseq.chunks.size() << " chunks, "
              << std::fixed << std::setprecision(3) << to_gb(in_bytes) << " GB\n\n";

    double eager_s = 0, old_s = 0, new_s = 0;
    bool agree = true;
    auto check = [&](const char* what, bool ok) {
        if (!ok) { std::cout << "  *** MISMATCH: " << what << " ***\n"; agree = false; }
    };

    std::cout << "--- filter(keep even) ---\n";

    auto t0 = Clock::now();
    chunk_seq out_eager = ChunkSequenceOps::ChunkFilter<uint64_t>(cseq, "bw_fc_eager", keep_even);
    eager_s = elapsed(t0);
    auto [cnt_eager, sum_eager] = count_and_sum(out_eager);
    cleanup_prefix("bw_fc_eager");
    print_row("eager", in_bytes, eager_s);

    auto t1 = Clock::now();
    chunk_seq out_old = ocd::filter(ocd::delay(cseq), "bw_fc_old", keep_even);
    old_s = elapsed(t1);
    auto [cnt_old, sum_old] = count_and_sum(out_old);
    cleanup_prefix("bw_fc_old");
    print_row("old-delayed", in_bytes, old_s);
    check("count  old == eager", cnt_old == cnt_eager);
    check("sum    old == eager", sum_old == sum_eager);

    auto t2 = Clock::now();
    chunk_seq out_new = cd::filter(cd::delay(cseq), "bw_fc_new", keep_even);
    new_s = elapsed(t2);
    auto [cnt_new, sum_new] = count_and_sum(out_new);
    cleanup_prefix("bw_fc_new");
    print_row("new-delayed", in_bytes, new_s);
    check("count  new == eager", cnt_new == cnt_eager);
    check("sum    new == eager", sum_new == sum_eager);

    cleanup_prefix("perm");

    std::cout << "\n" << (agree ? "agree=1 (all substrates match)"
                                : "agree=0 (MISMATCH — see above)") << "\n";

    auto f9 = [](double v) { std::ostringstream o; o << std::setprecision(9) << v; return o.str(); };
    std::cout << "CSV," << n << ',' << in_bytes
              << ',' << f9(eager_s) << ',' << f9(old_s) << ',' << f9(new_s)
              << ',' << (agree ? 1 : 0) << '\n';
    return agree ? 0 : 1;
}
