// Benchmark: the two out-of-core kth-smallest selections head-to-head on one
// key set, against in-memory parlay::kth_smallest as the yardstick.
//
//   1. ChunkPartition  kth_smallest_fast (ExternalKthSmallest.h): routes+writes
//                      ALL 32 oversampled-pivot buckets in one combined
//                      reader+writer pass, then recurses into the one bucket
//                      it needs and discards the other 31's data it just wrote.
//   2. delayed         kth_smallest_delayed (kth_smallest_delayed.h): same
//                      pivot sampling and read-only histogram pass, but the
//                      winning bucket is a delayed::lazy_filter node with no
//                      writes -- if it already fits the DRAM budget it goes
//                      straight into a parlay::sequence via materialize (zero
//                      disk writes for that level), otherwise it is force()'d,
//                      writing ONLY that one bucket.
//   3. in-memory       parlay::kth_smallest on the same keys in DRAM -- the
//                      yardstick the other two are chasing. It stops at the
//                      RAM cliff (~16n; see the budget below), so its line
//                      ends partway across the sweep while the out-of-core
//                      ones keep going.
//
// (1) vs (2) isolates what lazy_filter buys over ChunkPartition for a split
// where most branches are discarded: kth-smallest's 32-way split keeps only 1
// of 32 buckets every level, so skipping the other 31's writes is a real
// saving -- the positive counterpart to chunk_convex_hull_lazy_filter.h's
// quickhull rewrite, where a 2-way split keeps BOTH branches and lazy_filter
// is pure overhead (see kth_smallest_delayed.h's file banner).
//
// All three select from key_at(i) = parlay::hash64(i) for i in [0,n). The
// keys are distinct, so the k-th smallest is unique: every out-of-core result
// must equal the in-mem reference exactly.
//
// Fairness -- each contestant runs exactly ONCE, and the drives are made
// quiet between (same rationale as samplesort_three_way.cpp): each
// contestant writes its own input and recursion intermediates, so running
// both back to back would have the second selecting on drives the first left
// dirty. settle_drives() syncs every mount and lets it sit before the next
// one starts; KTHD_FIRST lets you verify the order doesn't move the times.
//
//   usage: kth_smallest_delayedExample [global --flags] [n] [k]
//     n                number of keys (default 1e6)
//     k                rank to select, 0-based (default n/2, the median)
//     BENCH_SETTLE_MS  how long the drives must sit idle after a sync
//                      (default 2000; see examples/external/bench_drives.h)
//     KTH_SMALLEST_DRAM_BUDGET_BYTES  DRAM base-case budget shared by both
//                      out-of-core algorithms
//     EXAMPLE_INMEM_BUDGET_BYTES
//                      RAM budget for the in-memory selection + cross-check
//                      (~16n; default: physical RAM)
//     KTHD_FIRST       which selection goes first, 0..1 (default 0): a knob
//                      for *checking* that the teardown works -- rotate it
//                      and the two times should not move. It does not change
//                      what is measured.
//
// CSV line:
//   CSV,<n>,<k>,<fast_build_s>,<fast_select_s>,<delayed_build_s>,
//       <delayed_select_s>,<inmem_select_s>,<result>,<eager_write_bytes>,
//       <delayed_write_bytes>,<throughput_gb_s>
//   throughput is input bytes (n*8) over kth_smallest_delayed's own select
//   time. <inmem_select_s> is left BLANK past the RAM budget, so the plotted
//   DRAM line stops at the cliff.
//
// Dual-purpose like the other examples: prints human-readable output AND the
// machine-readable CSV line benchmarks/run_benches.py greps.

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/examples/external/ExternalKthSmallest.h"
#include "ChunkSequence/helper/bench_drives.h"
#include "benchmarks/benchmark_files/kth_smallest_delayed.h"
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

using bench_drives::clear_drives;   // remove a contestant's files, then settle
using bench_drives::settle_drives;  // sync the mounts and let them go idle

static double to_gb(size_t bytes) {
  return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

// Each contestant's on-disk file family: its input plus its recursion
// intermediates.
//   fast:     kthd_fast_in + kth_next_ (kth_smallest_fast's ChunkPartition
//             scratch, from ExternalKthSmallest.h).
//   delayed:  kthd_delayed_in + kdl_next_ (kth_smallest_delayed's force()
//             scratch, from kth_smallest_delayed.h).
static const std::vector<std::string> kFastPrefixes = {"kthd_fast_in",
                                                        "kth_next_"};
static const std::vector<std::string> kDelayedPrefixes = {"kthd_delayed_in",
                                                           "kdl_next_"};

// Same key both contestants and the in-mem baseline select from.
static uint64_t key_at(size_t i) { return parlay::hash64(i); }

namespace {

// One contestant: build its input, select from it (timed).
struct Selector {
  std::string name;
  std::string label;  // short slug for trace_mark, e.g. op_start_<label>
  std::vector<std::string> prefixes;
  std::function<double()> build;   // -> build seconds
  std::function<double()> select;  // -> select seconds

  double build_s = 0, select_s = 0;
  uint64_t result = 0;
};

}  // namespace

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  // Both selections fan out one io_uring instance + one open file per drive
  // per worker, well past the 1024 soft fd limit; lift it before any I/O
  // starts.
  RaiseFdLimit();

  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
  const size_t k = (argc > 2) ? std::stoull(argv[2]) : n / 2;
  CHECK(n > 0 && k < n) << "need n > 0 and 0 <= k < n (n=" << n << ", k=" << k
                        << ")";

  // Which selection goes first; the other follows. A check knob, not a
  // measurement one -- with the teardown doing its job the times must not
  // depend on it (see the fairness note at the top).
  size_t first = 0;
  if (const char* e = getenv("KTHD_FIRST")) first = std::stoull(e) % 2;

  // RAM budget for the in-memory baseline + cross-check: the n-key input
  // (8n bytes) plus the per-recursion bucket ids (~n bytes) and packed
  // survivors, with sort/pack roughly doubling the top level -- ~16n, the
  // same gate kth_smallest.cpp's single-algorithm driver uses.
  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys;
  if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  const bool inmem_ok = n <= budget / 16;

  std::cout << std::fixed;

  // Held between a selection's build and its op (both algorithms mutate the
  // chunk_seq they're handed, recursing into fresh files each level).
  chunk_seq fast_seq, delayed_seq;
  size_t eager_write_bytes = 0, delayed_write_bytes = 0;

  std::vector<Selector> selectors(2);

  selectors[0].name = "kth_smallest_fast (ChunkPartition)";
  selectors[0].label = "fast";
  selectors[0].prefixes = kFastPrefixes;
  selectors[0].build = [&] {
    auto t0 = Clock::now();
    fast_seq = ChunkSequenceOps::tabulate<uint64_t>(n, "kthd_fast_in", key_at);
    return elapsed(t0);
  };
  selectors[0].select = [&] {
    auto t0 = Clock::now();
    selectors[0].result =
        ChunkSequenceOps::kth_smallest_fast<uint64_t>(fast_seq, (long)k);
    return elapsed(t0);
  };

  selectors[1].name = "kth_smallest_delayed (lazy_filter)";
  selectors[1].label = "delayed";
  selectors[1].prefixes = kDelayedPrefixes;
  selectors[1].build = [&] {
    auto t0 = Clock::now();
    delayed_seq =
        ChunkSequenceOps::tabulate<uint64_t>(n, "kthd_delayed_in", key_at);
    return elapsed(t0);
  };
  selectors[1].select = [&] {
    auto t0 = Clock::now();
    selectors[1].result = ChunkSequenceOps::kth_smallest_delayed<uint64_t>(
        delayed_seq, (long)k, delayed_write_bytes, eager_write_bytes);
    return elapsed(t0);
  };

  // ── the third contestant: the same selection, in DRAM ───────────────────
  // parlay::kth_smallest on the identical keys, held to the same n -- the
  // yardstick the out-of-core selections are trying to approach. Not on the
  // drives at all, so no teardown and no place in the rotation; only key
  // generation ("build") and the selection itself ("op") are timed.
  //
  // Bracketed with trace_mark like the two disk contestants (label "inmem")
  // so io_trace.py's per-algorithm breakdown draws it as a third window on
  // the same timeline.
  //
  // The selected value is also the cross-check reference: every out-of-core
  // result must equal it exactly (keys are distinct, so the k-th smallest is
  // unique).
  uint64_t inmem_result = 0;
  double inmem_select_s = 0;
  if (inmem_ok) {
    std::cout << "  [DRAM] in-mem parlaylib kth_smallest: generating keys..."
              << std::flush;
    trace_mark("build_start_inmem");
    auto keys_mem = parlay::tabulate(n, key_at);
    trace_mark("build_end_inmem");
    std::cout << " selecting..." << std::flush;
    auto t0 = Clock::now();
    trace_mark("op_start_inmem");
    inmem_result = kth_smallest(keys_mem, (long)k);
    trace_mark("op_end_inmem");
    inmem_select_s = elapsed(t0);
    std::cout << " " << std::setprecision(3) << inmem_select_s << "s\n";
  } else {
    std::cout
        << "  [DRAM] in-mem parlaylib kth_smallest: skipped (~16n exceeds "
           "the RAM budget "
        << std::setprecision(2) << to_gb(budget)
        << " GB) -- cross-check skipped with it\n";
  }

  // Start from drives clear of every contestant's files (including a stale
  // run's), and clear of the freeing work that removing them just queued.
  for (const Selector& s : selectors) clear_drives(s.prefixes);

  bool agree = true;
  for (size_t i = 0; i < selectors.size(); i++) {
    Selector& s = selectors[(first + i) % selectors.size()];

    std::cout << "  [" << (i + 1) << "/2] " << s.name << ": building input..."
              << std::flush;
    trace_mark(("build_start_" + s.label).c_str());
    s.build_s = s.build();
    trace_mark(("build_end_" + s.label).c_str());
    std::cout << " " << std::setprecision(3) << s.build_s
              << "s, selecting k=" << k << "..." << std::flush;
    settle_drives();  // the build's writeback must not land inside the timer

    trace_mark(("op_start_" + s.label).c_str());
    s.select_s = s.select();
    trace_mark(("op_end_" + s.label).c_str());
    std::cout << " " << std::setprecision(3) << s.select_s << "s   ("
              << std::setprecision(2)
              << to_gb(n * sizeof(uint64_t)) / s.select_s << " GB/s)\n";

    if (inmem_ok) {
      if (s.result == inmem_result) {
        std::cout << "      cross-check: matches the in-mem reference\n";
      } else {
        std::cout << "      *** MISMATCH (" << s.label << "): " << s.result
                  << " != in-mem " << inmem_result << " ***\n";
        agree = false;
      }
    }

    // Hand the drives to the next selection in the state this one found them
    // in: drop the chunk_seqs pointing at the files, remove every file this
    // selection wrote, and wait for the file system to finish freeing them.
    fast_seq = delayed_seq = chunk_seq{};
    clear_drives(s.prefixes);
  }

  // ── results ──────────────────────────────────────────────────────────────
  const double fast_s = selectors[0].select_s;
  const double delayed_s = selectors[1].select_s;

  std::cout << "\n"
            << n << " keys / " << std::setprecision(2)
            << to_gb(n * sizeof(uint64_t)) << " GB, one run each:\n";
  for (const Selector& s : selectors)
    std::cout << "  " << std::left << std::setw(36) << s.name << std::right
              << std::setprecision(3) << std::setw(8) << s.select_s << " s   "
              << std::setprecision(2) << std::setw(6)
              << to_gb(n * sizeof(uint64_t)) / s.select_s << " GB/s\n";
  if (inmem_ok)
    std::cout << "  " << std::left << std::setw(36)
              << "in-memory parlay::kth_smallest (DRAM)" << std::right
              << std::setprecision(3) << std::setw(8) << inmem_select_s
              << " s   " << std::setprecision(2) << std::setw(6)
              << to_gb(n * sizeof(uint64_t)) / inmem_select_s << " GB/s\n";
  else
    std::cout << "  " << std::left << std::setw(36)
              << "in-memory parlay::kth_smallest (DRAM)" << std::right
              << std::setw(8) << "-" << "     (past the RAM budget)\n";
  std::cout << std::setprecision(2)
            << "  cost of lazy_filter (delayed / ChunkPartition):     "
            << (delayed_s / fast_s) << "x\n";
  if (inmem_ok)
    std::cout << std::setprecision(2)
              << "  cost of going out of core (ChunkPartition / DRAM):  "
              << (fast_s / inmem_select_s) << "x\n";

  // The headline this benchmark exists to show: ChunkPartition writes every
  // one of its 32 buckets every level (31 immediately discarded);
  // lazy_filter writes only the surviving bucket, or nothing at all once it
  // already fits the DRAM budget.
  std::cout << "\n  disk bytes written across the whole recursion:\n"
            << "  " << std::left << std::setw(36)
            << "ChunkPartition (all 32 buckets)" << std::right
            << std::setprecision(3) << std::setw(8)
            << to_gb(eager_write_bytes) << " GB\n"
            << "  " << std::left << std::setw(36)
            << "lazy_filter (only the survivor)" << std::right
            << std::setprecision(3) << std::setw(8)
            << to_gb(delayed_write_bytes) << " GB\n";
  if (delayed_write_bytes > 0)
    std::cout << std::setprecision(2) << "  reduction: "
              << (double)eager_write_bytes / (double)delayed_write_bytes
              << "x fewer bytes written\n";
  else if (eager_write_bytes > 0)
    std::cout << "  reduction: every level's winning bucket already fit the "
                 "DRAM budget -- lazy_filter wrote nothing at all\n";

  // Machine-readable line for benchmarks/run_benches.py.
  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  const double gb = to_gb(n * sizeof(uint64_t));
  std::cout << "CSV," << n << ',' << k << ',' << f9(selectors[0].build_s)
            << ',' << f9(fast_s) << ',' << f9(selectors[1].build_s) << ','
            << f9(delayed_s) << ','
            << (inmem_ok ? f9(inmem_select_s) : std::string()) << ','
            << selectors[1].result << ',' << eager_write_bytes << ','
            << delayed_write_bytes << ',' << f9(gb / delayed_s) << '\n';

  return agree ? 0 : 1;
}
