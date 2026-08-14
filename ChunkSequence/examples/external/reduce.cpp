// reduceExample — out-of-core reduce benchmark/demo.
//
// Sums plaid::iota-style uint64_t elements via plaid::ChunkReduce (a
// RemoveWorker fold, see Primitives/reduce.h) and compares against
// parlay::reduce with parlay::addm<uint64_t>() on the identical values built
// in DRAM.
//
// Dual-purpose like the other examples: prints human-readable timings and
// ends with a machine-readable `CSV,` line that benchmarks/run_benches.py
// greps.  Unsigned addition is associative/commutative bit-exact regardless
// of term order, so the cross-check is exact equality, not a tolerance
// compare.  The in-memory baseline is gated by a RAM budget (generous here:
// the baseline's only resident state is the n-element input itself, no
// separate output buffer), overridable via EXAMPLE_INMEM_BUDGET_BYTES; past
// it the run is skipped and the CSV field left blank.
//
//   usage: reduceExample [global --flags] [n]

#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/Primitives/reduce.h"
#include "absl/log/check.h"
#include "parlay/monoid.h"
#include "parlay/primitives.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "utils/io_backend.h"
#include "utils/trace_marker.h"

struct SumMonoid {
  uint64_t identity = 0;
  uint64_t operator()(uint64_t a, uint64_t b) const { return a + b; }
};

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}
static double to_gb(size_t bytes) {
  return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static void quiesce_drives() {
  sync();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

static void cleanup_prefix(const std::string& prefix) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    plaid::io::Unlink(GetFileName(prefix, d).c_str());
}

static uint64_t elem_at(size_t i) { return (uint64_t)i; }

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
  CHECK(n > 0) << "need n > 0 (n=" << n << ")";

  // RAM budget for the in-memory baseline: only the n-element input is
  // resident (parlay::reduce accumulates in registers, no output buffer), so
  // this is generous headroom (~16n) compared to the other new primitive
  // drivers.
  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys / 2;
  if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  const bool inmem_ok = n <= budget / 16;

  const std::string in_prefix = "red_in";

  std::cout << "Building " << n << "-element input..." << std::flush;
  trace_mark("build_start");
  auto t0 = Clock::now();
  chunk_seq seq = plaid::tabulate<uint64_t>(n, in_prefix, elem_at);
  const double build_s = elapsed(t0);
  trace_mark("build_end");
  std::cout << " done (" << std::fixed << std::setprecision(4) << build_s
            << "s)\n";
  quiesce_drives();

  std::cout << "Reducing " << n << " elements (sum)..." << std::flush;
  trace_mark("op_start");
  t0 = Clock::now();
  const uint64_t sum = plaid::ChunkReduce<uint64_t>(seq, SumMonoid{});
  const double reduce_s = elapsed(t0);
  trace_mark("op_end");
  const double gb_s = to_gb(n * sizeof(uint64_t)) / reduce_s;
  std::cout << " done   sum=" << sum << "   " << std::setprecision(4)
            << reduce_s << "s   " << std::setprecision(2) << gb_s
            << " GB/s (input read)\n";

  bool agree = true;
  double inmem_reduce_s = 0;
  if (inmem_ok) {
    auto seq_mem = parlay::tabulate(n, elem_at);
    t0 = Clock::now();
    const uint64_t sum_mem = parlay::reduce(seq_mem, parlay::addm<uint64_t>());
    inmem_reduce_s = elapsed(t0);
    std::cout << "in-mem parlay::reduce: sum=" << sum_mem << "   "
              << std::setprecision(4) << inmem_reduce_s << "s\n";
    if (sum != sum_mem) {
      std::cout << "*** MISMATCH: out-of-core sum " << sum
                << " != in-mem sum " << sum_mem << " ***\n";
      agree = false;
    } else {
      std::cout << "cross-check: sums match exactly\n";
    }
  } else {
    std::cout << "in-mem parlay::reduce: skipped (~16n footprint exceeds "
              << "RAM budget " << std::setprecision(2) << to_gb(budget)
              << " GB)\n";
  }

  // Machine-readable line for benchmarks/run_benches.py (examples sweep).
  // Columns: n,build_s,reduce_s,inmem_reduce_s,result,throughput_gb_s
  // (inmem_reduce_s blank when the input exceeds the RAM budget).
  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  std::cout << "CSV," << n << ',' << f9(build_s) << ',' << f9(reduce_s) << ','
            << (inmem_ok ? f9(inmem_reduce_s) : std::string()) << ',' << sum
            << ',' << f9(gb_s) << '\n';

  cleanup_prefix(in_prefix);
  return agree ? 0 : 1;
}
