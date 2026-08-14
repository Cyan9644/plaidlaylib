// scanExample — out-of-core exclusive prefix-sum benchmark/demo.
//
// Runs an exclusive sum scan over an n-element uint64_t input via
// plaid::ChunkScan (Primitives/scan.h: pass 1 RemoveWorker per-chunk sums,
// pass 2 ExternalTransform seeded per chunk) and compares against
// parlay::scan with parlay::addm<uint64_t>() on the identical input built in
// DRAM.
//
// Dual-purpose like the other examples: prints human-readable timings and
// ends with a machine-readable `CSV,` line that benchmarks/run_benches.py
// greps.  Unsigned addition is associative/commutative bit-exact regardless
// of term order, so both the per-element scan and the grand total are
// checked with exact equality (after reading the out-of-core result back via
// plaid::materialize).  The in-memory baseline is gated by a RAM budget
// (input + baseline output + readback = ~24n), overridable via
// EXAMPLE_INMEM_BUDGET_BYTES.
//
//   usage: scanExample [global --flags] [n]

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
#include "ChunkSequence/Primitives/materialize.h"
#include "ChunkSequence/Primitives/scan.h"
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

static uint64_t elem_at(size_t i) { return parlay::hash64(i); }

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
  CHECK(n > 0) << "need n > 0 (n=" << n << ")";

  // RAM budget: baseline input (8n) + baseline scanned output (8n) +
  // out-of-core readback for the cross-check (8n) = ~24n.
  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys / 2;
  if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  const bool inmem_ok = n <= budget / 24;

  const std::string in_prefix = "scn_in";
  const std::string out_prefix = "scn_out";

  std::cout << "Building " << n << "-element input..." << std::flush;
  trace_mark("build_start");
  auto t0 = Clock::now();
  chunk_seq seq = plaid::tabulate<uint64_t>(n, in_prefix, elem_at);
  const double build_s = elapsed(t0);
  trace_mark("build_end");
  std::cout << " done (" << std::fixed << std::setprecision(4) << build_s
            << "s)\n";
  quiesce_drives();

  std::cout << "Scanning " << n << " elements (exclusive sum)..."
            << std::flush;
  trace_mark("op_start");
  t0 = Clock::now();
  auto [scanned, total] =
      plaid::ChunkScan<uint64_t>(seq, out_prefix, SumMonoid{});
  const double scan_s = elapsed(t0);
  trace_mark("op_end");
  const double gb_s = to_gb(n * sizeof(uint64_t)) / scan_s;
  std::cout << " done   total=" << total << "   " << std::setprecision(4)
            << scan_s << "s   " << std::setprecision(2) << gb_s
            << " GB/s (input read)\n";

  bool agree = true;
  double inmem_scan_s = 0;
  if (inmem_ok) {
    auto seq_mem = parlay::tabulate(n, elem_at);
    t0 = Clock::now();
    auto [scanned_mem, total_mem] =
        parlay::scan(seq_mem, parlay::addm<uint64_t>());
    inmem_scan_s = elapsed(t0);
    std::cout << "in-mem parlay::scan: total=" << total_mem << "   "
              << std::setprecision(4) << inmem_scan_s << "s\n";

    if (total != total_mem) {
      std::cout << "*** MISMATCH: out-of-core total " << total
                << " != in-mem total " << total_mem << " ***\n";
      agree = false;
    } else {
      auto ours = plaid::materialize<uint64_t>(scanned);
      if (ours.size() != scanned_mem.size()) {
        std::cout << "*** MISMATCH: out-of-core produced " << ours.size()
                  << " elements, expected " << scanned_mem.size() << " ***\n";
        agree = false;
      } else {
        for (size_t i = 0; i < ours.size() && agree; i++) {
          if (ours[i] != scanned_mem[i]) {
            std::cout << "*** MISMATCH at index " << i << ": " << ours[i]
                      << " != " << scanned_mem[i] << " ***\n";
            agree = false;
          }
        }
      }
      if (agree)
        std::cout << "cross-check: out-of-core scan matches in-mem scan "
                     "exactly\n";
    }
  } else {
    std::cout << "in-mem parlay::scan: skipped (~24n footprint exceeds "
              << "RAM budget " << std::setprecision(2) << to_gb(budget)
              << " GB)\n";
  }

  // Machine-readable line for benchmarks/run_benches.py (examples sweep).
  // Columns: n,build_s,scan_s,inmem_scan_s,total,throughput_gb_s
  // (inmem_scan_s blank when the input exceeds the RAM budget).
  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  std::cout << "CSV," << n << ',' << f9(build_s) << ',' << f9(scan_s) << ','
            << (inmem_ok ? f9(inmem_scan_s) : std::string()) << ',' << total
            << ',' << f9(gb_s) << '\n';

  cleanup_prefix(in_prefix);
  cleanup_prefix(out_prefix);
  return agree ? 0 : 1;
}
