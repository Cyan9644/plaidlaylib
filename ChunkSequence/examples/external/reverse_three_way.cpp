// Benchmark: the two out-of-core reverses head-to-head on one key set,
// against in-memory std::reverse as the yardstick.
//
//   1. legacy   ChunkSequenceOps::reverse (ExternalPrimitives/reverse.h):
//               reverses each chunk's bytes in place with hand-rolled
//               O_DIRECT pread/pwrite (small_sequence_ops::reverse), then
//               reverses the vector of chunk headers -- one random-access
//               read pass and one random-access write pass, chunk by chunk.
//   2. fast     ChunkSequenceOps::reverse_fast (same header): built on
//               process_inplace_budgeted, the engine sort_inplace uses --
//               reads the whole sequence into one flat, DRAM-budget-aware
//               buffer, std::reverses it, and writes it back over the same
//               chunks at the same per-chunk sizes. One streaming read pass,
//               one streaming write pass.
//   3. in-memory std::reverse on the same keys in DRAM -- the yardstick the
//               other two are chasing, and also the correctness reference
//               (both disk outputs must match it element-wise). Stops at the
//               RAM cliff like every other example's in-mem series.
//
// (1) vs (2) isolates *the primitives*: same on-disk result, so the gap is
// what hand-rolled random-access I/O costs relative to one streaming
// read+write pass over a DRAM-budgeted buffer -- the same question
// samplesort_three_way.cpp / random_shuffle_three_way.cpp ask of sort and
// shuffle. There is no vendored third-party reverse to compare against, so
// (like random_shuffle_three_way) this driver has one fewer contestant than
// a Peter-style three-way.
//
// Reverse is deterministic (unlike shuffle), so correctness is element-wise
// equality against the reversed key array, not a permutation check:
// reversed[i] must equal key_at(n-1-i) for every i.
//
// Both reverse() and reverse_fast() mutate their chunk_seq in place rather
// than returning a fresh one, so each contestant rebuilds its own input
// (its "build" phase) and then reverses that same chunk_seq (its "op"
// phase) -- there is no separate output prefix to sweep, only the input's.
//
// Fairness -- each reverse runs exactly ONCE, and the drives are made quiet
// between contestants (same rationale as samplesort_three_way.cpp /
// random_shuffle_three_way.cpp): settle_drives() syncs every mount and lets
// it sit before the next contestant starts; RV3_FIRST lets you verify the
// order doesn't move the times.
//
//   usage: reverse_three_wayExample [global --flags] [n]
//     n                number of keys (default 1e6)
//     BENCH_SETTLE_MS  how long the drives must sit idle after a sync
//                      (default 2000; see examples/external/bench_drives.h)
//     EXAMPLE_INMEM_BUDGET_BYTES
//                      RAM budget for the in-memory reverse + cross-check
//                      (~16n; default: physical RAM)
//     RV3_FIRST        which disk contestant goes first, 0..1 (default 0):
//                      a knob for *checking* that the teardown works --
//                      rotate it and the two times should not move. It does
//                      not change what is measured. (The in-memory
//                      contestant is not on the drives, so it is not part
//                      of this rotation -- same as RS3_FIRST.)
//
// CSV line:
//   CSV,<n>,<legacy_s>,<fast_s>,<inmem_s>,<legacy_build_s>,<fast_build_s>,
//       <legacy_gb_s>,<fast_gb_s>
//   throughput is input bytes (n*8) over the contestant's own time. <inmem_s>
//   is left BLANK past the RAM budget, so the plotted DRAM line stops at the
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

#include "ChunkSequence/ExternalPrimitives/materialize.h"
#include "ChunkSequence/ExternalPrimitives/reverse.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/examples/external/bench_drives.h"
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

using bench_drives::clear_drives;   // remove a contestant's files, then settle
using bench_drives::settle_drives;  // sync the mounts and let them go idle

static double to_gb(size_t bytes) {
  return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

// Each contestant's on-disk file family is just its own input -- reverse()
// and reverse_fast() both mutate that chunk_seq's own chunks in place, so
// there is no separate output prefix to sweep.
static const std::vector<std::string> kLegacyPrefixes = {"rv3_legacy_in"};
static const std::vector<std::string> kFastPrefixes = {"rv3_fast_in"};

// Same key both contestants and the in-mem baseline reverse.
static uint64_t key_at(size_t i) { return parlay::hash64(i); }

namespace {

// One contestant: build its input, reverse it in place (timed), read the
// reversed output back, and sweep every file it put on the drives.
struct Reverser {
  std::string name;
  std::string label;  // short slug for trace_mark, e.g. op_start_<label>
  std::vector<std::string> prefixes;
  std::function<double()> build;                     // -> build seconds
  std::function<double()> reverse;                   // -> reverse seconds
  std::function<std::vector<uint64_t>()> read_back;  // the reversed output

  double reverse_s = 0;
  double build_s = 0;
};

}  // namespace

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  // Both reverses fan out one io_uring instance + one open file per drive per
  // worker, well past the 1024 soft fd limit; lift it before any I/O starts.
  RaiseFdLimit();

  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
  CHECK(n > 0) << "need n > 0 (n=" << n << ")";

  // Which disk contestant goes first; the other follows. A check knob, not a
  // measurement one -- with the teardown doing its job the times must not
  // depend on it (see the fairness note at the top).
  size_t first = 0;
  if (const char* e = getenv("RV3_FIRST")) first = std::stoull(e) % 2;

  // RAM budget for the in-memory baseline + cross-check: the reversed
  // reference (8n, generated and reversed in place), and both out-of-core
  // outputs read back one at a time for the check (8n each) -- ~16n, smaller
  // than shuffle/unique's ~32n since there is no separate sorted copy to
  // hold.
  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys;
  if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  const bool inmem_ok = n <= budget / 16;

  std::cout << std::fixed;

  // Held between a contestant's reverse and its read-back (both point at
  // files the sweep is about to delete).
  chunk_seq legacy_seq, fast_seq;

  std::vector<Reverser> reversers(2);

  reversers[0].name = "ours, legacy (chunkwise pread/pwrite)";
  reversers[0].label = "legacy";
  reversers[0].prefixes = kLegacyPrefixes;
  reversers[0].build = [&] {
    auto t0 = Clock::now();
    legacy_seq =
        ChunkSequenceOps::tabulate<uint64_t>(n, "rv3_legacy_in", key_at);
    return elapsed(t0);
  };
  reversers[0].reverse = [&] {
    auto t0 = Clock::now();
    legacy_seq = ChunkSequenceOps::reverse<uint64_t>(legacy_seq);
    return elapsed(t0);
  };
  reversers[0].read_back = [&] {
    auto s = ChunkSequenceOps::materialize<uint64_t>(legacy_seq);
    return std::vector<uint64_t>(s.begin(), s.end());
  };

  reversers[1].name = "ours, fast (process_inplace_budgeted)";
  reversers[1].label = "fast";
  reversers[1].prefixes = kFastPrefixes;
  reversers[1].build = [&] {
    auto t0 = Clock::now();
    fast_seq = ChunkSequenceOps::tabulate<uint64_t>(n, "rv3_fast_in", key_at);
    return elapsed(t0);
  };
  reversers[1].reverse = [&] {
    auto t0 = Clock::now();
    ChunkSequenceOps::reverse_fast<uint64_t>(fast_seq);
    return elapsed(t0);
  };
  reversers[1].read_back = [&] {
    auto s = ChunkSequenceOps::materialize<uint64_t>(fast_seq);
    return std::vector<uint64_t>(s.begin(), s.end());
  };

  // ── the third contestant: the same reverse, in DRAM ─────────────────────
  // std::reverse on the identical keys, held to the same n -- the yardstick
  // the out-of-core reverses are trying to approach, and also the
  // correctness reference both disk outputs are checked against. Not on the
  // drives at all, so no teardown and no place in the rotation; only key
  // generation ("build") and the reverse itself ("op") are timed.
  parlay::sequence<uint64_t> ref;
  double inmem_reverse_s = 0;
  if (inmem_ok) {
    std::cout << "  [DRAM] in-memory std::reverse: generating keys..."
              << std::flush;
    trace_mark("build_start_inmem");
    ref = parlay::tabulate(n, key_at);
    trace_mark("build_end_inmem");
    std::cout << " reversing..." << std::flush;
    auto t0 = Clock::now();
    trace_mark("op_start_inmem");
    std::reverse(ref.begin(), ref.end());
    trace_mark("op_end_inmem");
    inmem_reverse_s = elapsed(t0);
    std::cout << " " << std::setprecision(3) << inmem_reverse_s << "s   ("
              << std::setprecision(2)
              << to_gb(n * sizeof(uint64_t)) / inmem_reverse_s << " GB/s)\n";
  } else {
    std::cout << "  [DRAM] in-memory std::reverse: skipped (~16n exceeds "
                  "the RAM budget "
              << std::setprecision(2) << to_gb(budget)
              << " GB) -- cross-check skipped with it\n";
  }

  // Start from drives clear of every contestant's files (including a stale
  // run's), and clear of the freeing work that removing them just queued.
  for (const Reverser& r : reversers) clear_drives(r.prefixes);

  auto matches_ref = [&](const char* who, const std::vector<uint64_t>& got) {
    if (got.size() != ref.size()) {
      std::cout << "      *** MISMATCH (" << who << "): produced " << got.size()
                << " keys, expected " << ref.size() << " ***\n";
      return false;
    }
    for (size_t i = 0; i < ref.size(); i++) {
      if (got[i] != ref[i]) {
        std::cout << "      *** MISMATCH (" << who << ") at index " << i
                  << ": " << got[i] << " != " << ref[i] << " ***\n";
        return false;
      }
    }
    return true;
  };

  bool agree = true;
  for (size_t k = 0; k < reversers.size(); k++) {
    Reverser& r = reversers[(first + k) % reversers.size()];

    std::cout << "  [" << (k + 1) << "/2] " << r.name << ": building input..."
              << std::flush;
    trace_mark(("build_start_" + r.label).c_str());
    r.build_s = r.build();
    trace_mark(("build_end_" + r.label).c_str());
    std::cout << " " << std::setprecision(3) << r.build_s << "s, reversing..."
              << std::flush;
    // The build's writeback must not land inside the reverse's timer.
    settle_drives();

    trace_mark(("op_start_" + r.label).c_str());
    r.reverse_s = r.reverse();
    trace_mark(("op_end_" + r.label).c_str());
    std::cout << " " << std::setprecision(3) << r.reverse_s << "s   ("
              << std::setprecision(2)
              << to_gb(n * sizeof(uint64_t)) / r.reverse_s << " GB/s)\n";

    if (inmem_ok) {
      if (matches_ref(r.label.c_str(), r.read_back()))
        std::cout << "      cross-check: matches the reversed reference\n";
      else
        agree = false;
    }

    // Hand the drives to the next contestant in the state this one found
    // them in: drop the chunk_seqs pointing at the files, remove every file
    // this contestant wrote, and wait for the file system to finish freeing
    // them.
    legacy_seq = fast_seq = chunk_seq{};
    clear_drives(r.prefixes);
  }

  // ── results ──────────────────────────────────────────────────────────────
  const double legacy_s = reversers[0].reverse_s;
  const double fast_s = reversers[1].reverse_s;

  std::cout << "\n"
            << n << " keys / " << std::setprecision(2)
            << to_gb(n * sizeof(uint64_t)) << " GB, one run each:\n";
  for (const Reverser& r : reversers)
    std::cout << "  " << std::left << std::setw(36) << r.name << std::right
              << std::setprecision(3) << std::setw(8) << r.reverse_s << " s   "
              << std::setprecision(2) << std::setw(6)
              << to_gb(n * sizeof(uint64_t)) / r.reverse_s << " GB/s\n";
  if (inmem_ok)
    std::cout << "  " << std::left << std::setw(36)
              << "in-memory std::reverse (DRAM)" << std::right
              << std::setprecision(3) << std::setw(8) << inmem_reverse_s
              << " s   " << std::setprecision(2) << std::setw(6)
              << to_gb(n * sizeof(uint64_t)) / inmem_reverse_s << " GB/s\n";
  else
    std::cout << "  " << std::left << std::setw(36)
              << "in-memory std::reverse (DRAM)" << std::right
              << std::setw(8) << "-" << "     (past the RAM budget)\n";
  std::cout << std::setprecision(2)
            << "  cost of legacy chunkwise I/O (legacy / fast):       "
            << (legacy_s / fast_s) << "x\n";
  if (inmem_ok)
    std::cout << std::setprecision(2)
              << "  cost of going out of core (fast / DRAM):            "
              << (fast_s / inmem_reverse_s) << "x\n";

  // Machine-readable line for benchmarks/run_benches.py.
  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  const double gb = to_gb(n * sizeof(uint64_t));
  std::cout << "CSV," << n << ',' << f9(legacy_s) << ',' << f9(fast_s) << ','
            << (inmem_ok ? f9(inmem_reverse_s) : std::string()) << ','
            << f9(reversers[0].build_s) << ',' << f9(reversers[1].build_s)
            << ',' << f9(gb / legacy_s) << ',' << f9(gb / fast_s) << '\n';

  return agree ? 0 : 1;
}
