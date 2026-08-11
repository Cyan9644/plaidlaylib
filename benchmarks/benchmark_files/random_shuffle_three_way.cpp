// Benchmark: the two out-of-core random shuffles head-to-head on one key set,
// against in-memory parlay::random_shuffle as the yardstick.
//
//   1. our primitives  external_random_shuffle.h (ChunkSequenceOps::random_shuffle):
//                      random per-index bucket assignment via a delayed map,
//                      count_sort routes the elements into per-bucket
//                      external sequences, each bucket is shuffled in place
//                      via apply<ChunkOperation::Shuffle> (process_inplace),
//                      flatten concatenates the buckets.
//   2. our direct      direct_random_shuffle.h (ChunkSequenceOps::direct_random_shuffle):
//                      the same algorithm, written straight against
//                      io_uring/O_DIRECT -- direct_sample_sort's scatter/
//                      gather shape with the pivot phase dropped (see that
//                      file's header comment).
//   3. in-memory       parlay::random_shuffle on the same keys in DRAM -- the
//                      yardstick the other two are chasing.  It stops at the
//                      RAM cliff (~32n; see the budget below), so its line
//                      ends partway across the sweep while the out-of-core
//                      ones keep going.
//
// (1) vs (2) isolates *the primitives*: same algorithm, same data model, so
// the gap is what the library's generality (delayed map, count_sort,
// process_inplace_budgeted) costs over hand-written I/O -- the same gap
// samplesort_three_way.cpp measures for sort ("our direct vs our
// primitives"). There is no vendored third-party shuffle to compare against
// (unlike samplesort_three_way's Peter's leg), so this driver has one fewer
// contestant.
//
// All three shuffle key_at(i) = parlay::hash64(i) for i in [0,n). The keys
// are distinct, so correctness is a *permutation* check, not element-wise
// equality: an output is a valid shuffle iff, once sorted, it equals the
// sorted key set. Both out-of-core outputs and the in-mem baseline are
// checked that way when the input fits the RAM budget.
//
// Fairness -- each shuffle runs exactly ONCE, and the drives are made quiet
// between (same rationale as samplesort_three_way.cpp): every shuffle writes
// its own input, intermediates and output, so running two back to back would
// have the second sorting on drives the first left dirty. settle_drives()
// syncs every mount and lets it sit before the next one starts; RS3_FIRST
// lets you verify the order doesn't move the times.
//
//   usage: random_shuffle_three_wayExample [global --flags] [n]
//     n                number of keys (default 1e6)
//     BENCH_SETTLE_MS  how long the drives must sit idle after a sync
//                      (default 2000; see examples/external/bench_drives.h)
//     EXAMPLE_INMEM_BUDGET_BYTES
//                      RAM budget for the in-memory shuffle + cross-check
//                      (~32n; default: physical RAM)
//     RS3_FIRST        which shuffle goes first, 0..1 (default 0): a knob for
//                      *checking* that the teardown works -- rotate it and
//                      the two times should not move. It does not change
//                      what is measured.
//
// CSV line:
//   CSV,<n>,<prim_s>,<direct_s>,<inmem_s>,<prim_build_s>,<direct_build_s>,
//       <prim_gb_s>,<direct_gb_s>
//   throughput is input bytes (n*8) over the method's own time. <inmem_s> is
//   left BLANK past the RAM budget, so the plotted DRAM line stops at the
//   cliff.
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

#include "ChunkSequence/Primitives/materialize.h"
#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/helper/bench_drives.h"
#include "benchmarks/benchmark_files/direct_random_shuffle.h"
#include "ChunkSequence/examples/external_TODO/external_random_shuffle.h"
#include "absl/log/check.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "utils/trace_marker.h"

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

using bench_drives::clear_drives;   // remove a shuffle's files, then settle
using bench_drives::settle_drives;  // sync the mounts and let them go idle

static double to_gb(size_t bytes) {
  return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

// Each contestant's on-disk file family: its input, its intermediates, and
// its output (both of ours hand back a chunk_seq whose files *are* the
// shuffled result, so those must be swept too or the whole output leaks
// every sweep point).
//   primitives:  rs3_in + ChunkSequenceOps::random_shuffle's internal bucket
//                files, which are unconditionally named "ss_bucket_"+tag --
//                that function takes no result-prefix argument, so this is
//                the only prefix that can be swept for it (a quirk of
//                external_random_shuffle.h, shared with sample sort's own
//                naming; left as-is here).
//   direct:      drs3_in + drs3<tag>_tmp<b> (scatter buckets) and
//                drs3<tag>_<b> (the shuffled result).
static const std::vector<std::string> kPrimPrefixes = {"rs3_in", "ss_bucket_"};
static const std::vector<std::string> kDirectPrefixes = {"drs3_in", "drs3"};

// Same key both contestants and the in-mem baseline shuffle.
static uint64_t key_at(size_t i) { return parlay::hash64(i); }

namespace {

// One contestant: build its input, shuffle it (timed), read the shuffled
// output back, and sweep every file it put on the drives.
struct Shuffler {
  std::string name;
  std::string label;  // short slug for trace_mark, e.g. op_start_<label>
  std::vector<std::string> prefixes;
  std::function<double()> build;                     // -> build seconds
  std::function<double()> shuffle;                   // -> shuffle seconds
  std::function<std::vector<uint64_t>()> read_back;  // the shuffled output

  double shuffle_s = 0;
  double build_s = 0;
};

}  // namespace

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  // Both shuffles fan out one io_uring instance + one open file per drive per
  // worker, well past the 1024 soft fd limit; lift it before any I/O starts.
  RaiseFdLimit();

  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
  CHECK(n > 0) << "need n > 0 (n=" << n << ")";

  // Which shuffle goes first; the other follows. A check knob, not a
  // measurement one -- with the teardown doing its job the times must not
  // depend on it (see the fairness note at the top).
  size_t first = 0;
  if (const char* e = getenv("RS3_FIRST")) first = std::stoull(e) % 2;

  // RAM budget for the in-memory baseline + cross-check: the keys/sorted
  // reference (8n, aliased in place), the in-mem shuffled output (8n), and
  // both out-of-core outputs read back one at a time for the check (8n each,
  // ~16n) -- ~32n, the same gate external_random_shuffle.cpp uses.
  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys;
  if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  const bool inmem_ok = n <= budget / 32;

  std::cout << std::fixed;

  // Held between a shuffle and its read-back (both of ours hand back a
  // chunk_seq whose chunks point at files the sweep is about to delete).
  chunk_seq prim_in, prim_out, direct_in, direct_out;

  std::vector<Shuffler> shufflers(2);

  shufflers[0].name = "ours, primitives (chunk_seq)";
  shufflers[0].label = "primitives";
  shufflers[0].prefixes = kPrimPrefixes;
  shufflers[0].build = [&] {
    auto t0 = Clock::now();
    prim_in = ChunkSequenceOps::tabulate<uint64_t>(n, "rs3_in", key_at);
    return elapsed(t0);
  };
  shufflers[0].shuffle = [&] {
    auto t0 = Clock::now();
    prim_out = ChunkSequenceOps::random_shuffle<uint64_t>(prim_in);
    return elapsed(t0);
  };
  shufflers[0].read_back = [&] {
    auto s = ChunkSequenceOps::materialize<uint64_t>(prim_out);
    return std::vector<uint64_t>(s.begin(), s.end());
  };

  shufflers[1].name = "ours, direct I/O (chunk_seq)";
  shufflers[1].label = "direct";
  shufflers[1].prefixes = kDirectPrefixes;
  shufflers[1].build = [&] {
    auto t0 = Clock::now();
    direct_in = ChunkSequenceOps::tabulate<uint64_t>(n, "drs3_in", key_at);
    return elapsed(t0);
  };
  shufflers[1].shuffle = [&] {
    auto t0 = Clock::now();
    direct_out = ChunkSequenceOps::direct_random_shuffle<uint64_t>(
        direct_in, /*seed=*/0, "drs3");
    return elapsed(t0);
  };
  shufflers[1].read_back = [&] {
    auto s = ChunkSequenceOps::materialize<uint64_t>(direct_out);
    return std::vector<uint64_t>(s.begin(), s.end());
  };

  // ── the third contestant: the same shuffle, in DRAM ─────────────────────
  // parlay::random_shuffle on the identical keys, held to the same n -- the
  // yardstick the out-of-core shuffles are trying to approach. Not on the
  // drives at all, so no teardown and no place in the rotation; only key
  // generation ("build") and the shuffle itself ("op") are timed.
  //
  // Bracketed with trace_mark like the two disk contestants (label "inmem")
  // so io_trace.py's per-algorithm breakdown draws it as a third window on
  // the same timeline -- a CPU-pinned, disk-idle band next to the disk
  // contestants' IO-bound ones is exactly the IO-bound-vs-CPU-bound contrast
  // io_trace.py's header comment says it exists to show.
  //
  // The sorted key set is also the cross-check reference: every out-of-core
  // output, once sorted, must equal it exactly.
  parlay::sequence<uint64_t> ref;
  double inmem_shuffle_s = 0;
  parlay::sequence<uint64_t> inmem_shuffled;
  if (inmem_ok) {
    std::cout << "  [DRAM] in-memory parlay::random_shuffle: generating keys..."
              << std::flush;
    trace_mark("build_start_inmem");
    auto keys_mem = parlay::tabulate(n, key_at);
    trace_mark("build_end_inmem");
    std::cout << " shuffling..." << std::flush;
    auto t0 = Clock::now();
    trace_mark("op_start_inmem");
    inmem_shuffled = parlay::random_shuffle(keys_mem);
    trace_mark("op_end_inmem");
    inmem_shuffle_s = elapsed(t0);
    std::cout << " " << std::setprecision(3) << inmem_shuffle_s << "s   ("
              << std::setprecision(2)
              << to_gb(n * sizeof(uint64_t)) / inmem_shuffle_s << " GB/s)\n";

    ref = keys_mem;
    parlay::sort_inplace(ref);
  } else {
    std::cout
        << "  [DRAM] in-memory parlay::random_shuffle: skipped (~32n exceeds "
           "the RAM budget "
        << std::setprecision(2) << to_gb(budget)
        << " GB) -- cross-check skipped with it\n";
  }

  // Start from drives clear of every shuffle's files (including a stale
  // run's), and clear of the freeing work that removing them just queued.
  for (const Shuffler& s : shufflers) clear_drives(s.prefixes);

  // Sorting a valid shuffle of distinct keys must reproduce the sorted key
  // set exactly -- catches a dropped, duplicated, or corrupted element.
  auto is_permutation = [&](const char* who, std::vector<uint64_t> got) {
    if (got.size() != ref.size()) {
      std::cout << "      *** MISMATCH (" << who << "): produced " << got.size()
                << " keys, expected " << ref.size() << " ***\n";
      return false;
    }
    std::sort(got.begin(), got.end());
    for (size_t i = 0; i < ref.size(); i++) {
      if (got[i] != ref[i]) {
        std::cout << "      *** MISMATCH (" << who << ") at sorted index " << i
                  << ": " << got[i] << " != " << ref[i] << " ***\n";
        return false;
      }
    }
    return true;
  };

  bool agree = true;
  for (size_t k = 0; k < shufflers.size(); k++) {
    Shuffler& s = shufflers[(first + k) % shufflers.size()];

    std::cout << "  [" << (k + 1) << "/2] " << s.name << ": building input..."
              << std::flush;
    trace_mark(("build_start_" + s.label).c_str());
    s.build_s = s.build();
    trace_mark(("build_end_" + s.label).c_str());
    std::cout << " " << std::setprecision(3) << s.build_s << "s, shuffling..."
              << std::flush;
    // The build's writeback must not land inside the shuffle's timer.
    settle_drives();

    trace_mark(("op_start_" + s.label).c_str());
    s.shuffle_s = s.shuffle();
    trace_mark(("op_end_" + s.label).c_str());
    std::cout << " " << std::setprecision(3) << s.shuffle_s << "s   ("
              << std::setprecision(2)
              << to_gb(n * sizeof(uint64_t)) / s.shuffle_s << " GB/s)\n";

    if (inmem_ok) {
      if (is_permutation(s.label.c_str(), s.read_back()))
        std::cout << "      cross-check: a valid permutation of the input\n";
      else
        agree = false;
    }

    // Hand the drives to the next shuffle in the state this one found them
    // in: drop the chunk_seqs pointing at the files, remove every file this
    // shuffle wrote, and wait for the file system to finish freeing them.
    prim_in = prim_out = direct_in = direct_out = chunk_seq{};
    clear_drives(s.prefixes);
  }

  if (inmem_ok) {
    if (!is_permutation("in-mem parlay",
                        std::vector<uint64_t>(inmem_shuffled.begin(),
                                               inmem_shuffled.end())))
      agree = false;
    else
      std::cout
          << "cross-check: all three outputs are permutations of the input\n";
  }

  // ── results ──────────────────────────────────────────────────────────────
  const double prim_s = shufflers[0].shuffle_s;
  const double direct_s = shufflers[1].shuffle_s;

  std::cout << "\n"
            << n << " keys / " << std::setprecision(2)
            << to_gb(n * sizeof(uint64_t)) << " GB, one run each:\n";
  for (const Shuffler& s : shufflers)
    std::cout << "  " << std::left << std::setw(36) << s.name << std::right
              << std::setprecision(3) << std::setw(8) << s.shuffle_s << " s   "
              << std::setprecision(2) << std::setw(6)
              << to_gb(n * sizeof(uint64_t)) / s.shuffle_s << " GB/s\n";
  if (inmem_ok)
    std::cout << "  " << std::left << std::setw(36)
              << "in-memory parlay::random_shuffle (DRAM)" << std::right
              << std::setprecision(3) << std::setw(8) << inmem_shuffle_s
              << " s   " << std::setprecision(2) << std::setw(6)
              << to_gb(n * sizeof(uint64_t)) / inmem_shuffle_s << " GB/s\n";
  else
    std::cout << "  " << std::left << std::setw(36)
              << "in-memory parlay::random_shuffle (DRAM)" << std::right
              << std::setw(8) << "-" << "     (past the RAM budget)\n";
  std::cout << std::setprecision(2)
            << "  cost of the primitives (ours-prims / ours-direct):  "
            << (prim_s / direct_s) << "x\n";
  if (inmem_ok)
    std::cout << std::setprecision(2)
              << "  cost of going out of core (ours-direct / DRAM):     "
              << (direct_s / inmem_shuffle_s) << "x\n";

  // Machine-readable line for benchmarks/run_benches.py.
  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  const double gb = to_gb(n * sizeof(uint64_t));
  std::cout << "CSV," << n << ',' << f9(prim_s) << ',' << f9(direct_s) << ','
            << (inmem_ok ? f9(inmem_shuffle_s) : std::string()) << ','
            << f9(shufflers[0].build_s) << ',' << f9(shufflers[1].build_s)
            << ',' << f9(gb / prim_s) << ',' << f9(gb / direct_s) << '\n';

  return agree ? 0 : 1;
}
