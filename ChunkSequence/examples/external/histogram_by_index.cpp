// histogram_by_indexExample — out-of-core histogram benchmark/demo.
//
// Builds an n-element uint64_t sequence of bucket ids (hash64(i) % 4096) and
// counts them via plaid::ChunkHistogramByIndex (Primitives/histogram_by_index.h,
// a RemoveWorker per-worker bucket-count fold), comparing against
// parlay::histogram_by_index on the identical ids built in DRAM.
//
// Dual-purpose like the other examples: prints human-readable timings and
// ends with a machine-readable `CSV,` line that benchmarks/run_benches.py
// greps.  ChunkHistogramByIndex returns its result directly as a small
// (num_buckets-length) parlay::sequence<size_t> reduction — not a chunk_seq —
// so no readback is needed for the cross-check, and the in-memory baseline's
// resident footprint is dominated by the n-element id input alone, giving
// generous headroom (~16n) for the RAM budget gate, overridable via
// EXAMPLE_INMEM_BUDGET_BYTES.
//
//   usage: histogram_by_indexExample [global --flags] [n]

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
#include "ChunkSequence/Primitives/histogram_by_index.h"
#include "absl/log/check.h"
#include "parlay/primitives.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "utils/trace_marker.h"

static constexpr size_t NUM_BUCKETS = 4096;

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
    unlink(GetFileName(prefix, d).c_str());
}

static uint64_t id_at(size_t i) { return parlay::hash64(i) % NUM_BUCKETS; }

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
  CHECK(n > 0) << "need n > 0 (n=" << n << ")";

  // RAM budget: dominated by the n-element id input (8n); the baseline's
  // output is only NUM_BUCKETS*8 bytes, negligible in comparison.
  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys / 2;
  if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  const bool inmem_ok = n <= budget / 16;

  const std::string in_prefix = "hist_in";

  std::cout << "Building " << n << "-element id input (mod " << NUM_BUCKETS
            << ")..." << std::flush;
  trace_mark("build_start");
  auto t0 = Clock::now();
  chunk_seq ids = plaid::tabulate<uint64_t>(n, in_prefix, id_at);
  const double build_s = elapsed(t0);
  trace_mark("build_end");
  std::cout << " done (" << std::fixed << std::setprecision(4) << build_s
            << "s)\n";
  quiesce_drives();

  std::cout << "Histogramming " << n << " ids into " << NUM_BUCKETS
            << " buckets..." << std::flush;
  trace_mark("op_start");
  t0 = Clock::now();
  auto counts = plaid::ChunkHistogramByIndex<uint64_t>(ids, NUM_BUCKETS);
  const double hist_s = elapsed(t0);
  trace_mark("op_end");
  const double gb_s = to_gb(n * sizeof(uint64_t)) / hist_s;
  std::cout << " done   " << std::setprecision(4) << hist_s << "s   "
            << std::setprecision(2) << gb_s << " GB/s (input read)\n";

  bool agree = true;
  double inmem_hist_s = 0;
  if (inmem_ok) {
    auto ids_mem = parlay::tabulate(n, id_at);
    t0 = Clock::now();
    auto counts_mem = parlay::histogram_by_index(ids_mem, NUM_BUCKETS);
    inmem_hist_s = elapsed(t0);
    std::cout << "in-mem parlay::histogram_by_index: " << std::setprecision(4)
              << inmem_hist_s << "s\n";

    if (counts.size() != counts_mem.size()) {
      std::cout << "*** MISMATCH: out-of-core produced " << counts.size()
                << " buckets, expected " << counts_mem.size() << " ***\n";
      agree = false;
    } else {
      for (size_t b = 0; b < counts.size() && agree; b++) {
        if (counts[b] != counts_mem[b]) {
          std::cout << "*** MISMATCH at bucket " << b << ": " << counts[b]
                    << " != " << counts_mem[b] << " ***\n";
          agree = false;
        }
      }
      if (agree)
        std::cout << "cross-check: out-of-core histogram matches in-mem "
                     "histogram exactly\n";
    }
  } else {
    std::cout << "in-mem parlay::histogram_by_index: skipped (~16n footprint "
              << "exceeds RAM budget " << std::setprecision(2)
              << to_gb(budget) << " GB)\n";
  }

  // Machine-readable line for benchmarks/run_benches.py (examples sweep).
  // Columns: n,build_s,hist_s,inmem_hist_s,num_buckets,throughput_gb_s
  // (inmem_hist_s blank when the input exceeds the RAM budget).
  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  std::cout << "CSV," << n << ',' << f9(build_s) << ',' << f9(hist_s) << ','
            << (inmem_ok ? f9(inmem_hist_s) : std::string()) << ','
            << NUM_BUCKETS << ',' << f9(gb_s) << '\n';

  cleanup_prefix(in_prefix);
  return agree ? 0 : 1;
}
