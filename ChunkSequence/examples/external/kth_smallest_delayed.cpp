// Benchmark: kth-smallest selection, head-to-head — ChunkPartition
// (kth_smallest_fast, ExternalKthSmallest.h) vs. delayed::lazy_filter
// (kth_smallest_delayed, kth_smallest_delayed.h) vs. in-memory parlaylib
// (examples/in_memory/kth_smallest.h) — on the same keys.
//
// Both out-of-core substrates use the identical oversampled-pivot /
// heap_tree bucketing scheme; the only difference is how the winning bucket
// (of 32) reaches the next recursion level:
//   - kth_smallest_fast:    ChunkPartition writes ALL 32 buckets in one
//                           combined pass, then recurses into the 1 it needs
//                           and discards the other 31's data it just wrote.
//   - kth_smallest_delayed: delayed::lazy_filter represents "just bucket
//                           id" as a fused node with no writes; if it
//                           already fits the DRAM budget it goes straight
//                           into a parlay::sequence via materialize (zero
//                           disk writes for that level), otherwise it is
//                           force()'d — writing ONLY that one bucket.
//
// See kth_smallest_delayed.h's file banner for why this (a 32-way split
// where 31 branches are thrown away every level) is the positive case for
// lazy_filter, in contrast to chunk_convex_hull_lazy_filter.h's quickhull
// rewrite (a 2-way split where both branches are kept, so lazy_filter is
// pure overhead there).
//
// Fairness — same discipline as samplesort_three_way.cpp: each contestant
// gets its own freshly built input, the drives are settled after a build and
// after removing a contestant's files (bench_drives.h), and a *_FIRST knob
// lets you confirm the measured times don't depend on run order.
//
//   usage: kth_smallest_delayedExample [global --flags] [n] [k]
//     n   number of keys (default 1e6)
//     k   rank to select, 0-based (default n/2, the median)
//     KTHD_FIRST                which disk contestant goes first, 0..1
//                                (default 0; a check knob, not a measurement
//                                one — see bench_drives.h)
//     KTH_SMALLEST_DRAM_BUDGET_BYTES  DRAM base-case budget shared by both
//                                out-of-core algorithms
//     EXAMPLE_INMEM_BUDGET_BYTES      RAM budget for the in-mem baseline
//
// CSV line:
//   CSV,<n>,<k>,<fast_build_s>,<fast_select_s>,<delayed_build_s>,
//       <delayed_select_s>,<inmem_select_s>,<result>,<eager_write_bytes>,
//       <delayed_write_bytes>,<throughput_gb_s>
//   throughput = input bytes / delayed_select_s. <inmem_select_s> is left
//   BLANK past the RAM budget, so the plotted DRAM line stops at the cliff.

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/examples/external/ExternalKthSmallest.h"
#include "ChunkSequence/examples/external/bench_drives.h"
#include "ChunkSequence/examples/external/kth_smallest_delayed.h"
#include "ChunkSequence/examples/in_memory/kth_smallest.h"
#include "absl/log/check.h"
#include "parlay/primitives.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "utils/trace_marker.h"

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

using bench_drives::clear_drives;
using bench_drives::settle_drives;

static double to_gb(size_t bytes) {
  return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

// Same deterministic, duplicate-free key as kth_smallest.cpp, so both
// examples' results are directly comparable.
static uint64_t key_at(size_t i) { return parlay::hash64(i); }

static const std::vector<std::string> kFastPrefixes = {"kthd_fast_in",
                                                        "kth_next_"};
static const std::vector<std::string> kDelayedPrefixes = {"kthd_delayed_in",
                                                           "kdl_next_"};

namespace {
struct Contestant {
  std::string name;
  std::string label;  // short slug for trace_mark, e.g. build_start_<label>
  std::vector<std::string> prefixes;
  std::function<double()> build;   // -> build seconds
  std::function<double()> select;  // -> select seconds
  double build_s = 0, select_s = 0;
  uint64_t result = 0;
};
}  // namespace

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
  const size_t k = (argc > 2) ? std::stoull(argv[2]) : n / 2;
  CHECK(n > 0 && k < n) << "need n > 0 and 0 <= k < n (n=" << n << ", k=" << k
                        << ")";

  size_t first = 0;
  if (const char* e = getenv("KTHD_FIRST")) first = std::stoull(e) % 2;

  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys;
  if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  const bool inmem_ok = n <= budget / 16;

  chunk_seq fast_seq, delayed_seq;
  size_t eager_write_bytes = 0, delayed_write_bytes = 0;

  std::vector<Contestant> contestants(2);

  contestants[0].name = "ChunkPartition (kth_smallest_fast)";
  contestants[0].label = "fast";
  contestants[0].prefixes = kFastPrefixes;
  contestants[0].build = [&] {
    auto t0 = Clock::now();
    fast_seq = ChunkSequenceOps::tabulate<uint64_t>(n, "kthd_fast_in", key_at);
    return elapsed(t0);
  };
  contestants[0].select = [&] {
    auto t0 = Clock::now();
    contestants[0].result =
        ChunkSequenceOps::kth_smallest_fast<uint64_t>(fast_seq, (long)k);
    return elapsed(t0);
  };

  contestants[1].name = "delayed::lazy_filter (kth_smallest_delayed)";
  contestants[1].label = "delayed";
  contestants[1].prefixes = kDelayedPrefixes;
  contestants[1].build = [&] {
    auto t0 = Clock::now();
    delayed_seq =
        ChunkSequenceOps::tabulate<uint64_t>(n, "kthd_delayed_in", key_at);
    return elapsed(t0);
  };
  contestants[1].select = [&] {
    auto t0 = Clock::now();
    contestants[1].result = ChunkSequenceOps::kth_smallest_delayed<uint64_t>(
        delayed_seq, (long)k, delayed_write_bytes, eager_write_bytes);
    return elapsed(t0);
  };

  // In-memory baseline / cross-check reference: same keys, DRAM only, not
  // part of the disk rotation.
  uint64_t inmem_result = 0;
  double inmem_select_s = 0;
  if (inmem_ok) {
    std::cout << "[DRAM] in-mem parlaylib kth_smallest: generating keys..."
              << std::flush;
    auto keys_mem = parlay::tabulate(n, key_at);
    std::cout << " selecting..." << std::flush;
    auto t0 = Clock::now();
    inmem_result = kth_smallest(keys_mem, (long)k);
    inmem_select_s = elapsed(t0);
    std::cout << " " << std::setprecision(4) << inmem_select_s << "s\n";
  } else {
    std::cout << "[DRAM] in-mem parlaylib kth_smallest: skipped (~16n "
              << "exceeds RAM budget " << std::setprecision(2) << to_gb(budget)
              << " GB)\n";
  }

  for (const Contestant& c : contestants) clear_drives(c.prefixes);

  bool agree = true;
  for (size_t i = 0; i < contestants.size(); i++) {
    Contestant& c = contestants[(first + i) % contestants.size()];
    std::cout << "[" << (i + 1) << "/2] " << c.name << ": building input..."
              << std::flush;
    trace_mark(("build_start_" + c.label).c_str());
    c.build_s = c.build();
    trace_mark(("build_end_" + c.label).c_str());
    std::cout << " " << std::setprecision(4) << c.build_s
              << "s, selecting k=" << k << "..." << std::flush;
    settle_drives();  // the build's writeback must not land inside the timer

    trace_mark(("op_start_" + c.label).c_str());
    c.select_s = c.select();
    trace_mark(("op_end_" + c.label).c_str());
    std::cout << " " << std::setprecision(4) << c.select_s << "s -> "
              << c.result << "\n";

    if (inmem_ok && c.result != inmem_result) {
      std::cout << "    *** MISMATCH: " << c.name << " = " << c.result
                << " != in-mem " << inmem_result << " ***\n";
      agree = false;
    }

    fast_seq = delayed_seq = chunk_seq{};
    clear_drives(c.prefixes);
  }

  std::cout << "\nDisk bytes written across the whole recursion:\n"
            << "  ChunkPartition (all 32 buckets, every level):   "
            << std::setprecision(3) << to_gb(eager_write_bytes) << " GB\n"
            << "  lazy_filter (only the surviving bucket):        "
            << std::setprecision(3) << to_gb(delayed_write_bytes) << " GB\n";
  if (eager_write_bytes > 0)
    std::cout << "  reduction: " << std::setprecision(2)
              << (double)eager_write_bytes / std::max<size_t>(1, delayed_write_bytes)
              << "x fewer bytes written\n";

  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  const double gb_s = to_gb(n * sizeof(uint64_t)) / contestants[1].select_s;
  std::cout << "CSV," << n << ',' << k << ',' << f9(contestants[0].build_s)
            << ',' << f9(contestants[0].select_s) << ','
            << f9(contestants[1].build_s) << ',' << f9(contestants[1].select_s)
            << ',' << (inmem_ok ? f9(inmem_select_s) : std::string()) << ','
            << contestants[1].result << ',' << eager_write_bytes << ','
            << delayed_write_bytes << ',' << f9(gb_s) << '\n';

  return agree ? 0 : 1;
}
