// tabulateExample — out-of-core tabulate benchmark/demo.
//
// Generates an n-element uint64_t sequence f(i) = i*i+1 directly to disk via
// plaid::ChunkFlatTabulate (Primitives/flat_tabulate.h, a DensePack
// generator-only producer -- no input chunk_seq, unlike map/filter/pack) and
// compares against parlay::tabulate on the identical index function.
//
// This is a synthetic microbenchmark of the primitive itself, distinct from
// primes.cpp's Eratosthenes sieve (which is also built on ChunkFlatTabulate
// but is separately requested/benchmarked as its own example): generation is
// the entire operation here, so -- like primes.cpp -- there is no separate
// build phase.
//
// Dual-purpose like the other examples: prints human-readable timings and
// ends with a machine-readable `CSV,` line that benchmarks/run_benches.py
// greps.  The out-of-core output is read back (plaid::materialize) and
// compared element-for-element against the in-memory result; the in-memory
// baseline is gated by a RAM budget (baseline output + readback = ~16n),
// overridable via EXAMPLE_INMEM_BUDGET_BYTES.
//
//   usage: tabulateExample [global --flags] [n]

#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/Primitives/flat_tabulate.h"
#include "ChunkSequence/Primitives/materialize.h"
#include "absl/log/check.h"
#include "parlay/primitives.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "utils/trace_marker.h"

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}
static double to_gb(size_t bytes) {
  return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static void cleanup_prefix(const std::string& prefix) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    unlink(GetFileName(prefix, d).c_str());
}

static uint64_t f_scalar(size_t i) { return (uint64_t)i * (uint64_t)i + 1; }

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
  CHECK(n > 0) << "need n > 0 (n=" << n << ")";

  // RAM budget: baseline output (8n) + out-of-core readback for the
  // cross-check (8n) = ~16n. No separate input build to isolate the op
  // timer from, so no quiesce_drives() call here either.
  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys / 2;
  if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  const bool inmem_ok = n <= budget / 16;

  const std::string out_prefix = "tab_out";

  std::cout << "Tabulating " << n << " elements (i*i+1)..." << std::flush;
  trace_mark("op_start");
  auto t0 = Clock::now();
  chunk_seq tab = plaid::ChunkFlatTabulate<uint64_t>(
      n, out_prefix, [](size_t start, size_t end) {
        parlay::sequence<uint64_t> out(end - start);
        for (size_t i = start; i < end; i++) out[i - start] = f_scalar(i);
        return out;
      });
  const double tabulate_s = elapsed(t0);
  trace_mark("op_end");
  const double gb_s = to_gb(n * sizeof(uint64_t)) / tabulate_s;
  std::cout << " done   " << std::fixed << std::setprecision(4) << tabulate_s
            << "s   " << std::setprecision(2) << gb_s << " GB/s (output write)\n";

  bool agree = true;
  double inmem_tabulate_s = 0;
  if (inmem_ok) {
    t0 = Clock::now();
    auto tab_mem = parlay::tabulate(n, f_scalar);
    inmem_tabulate_s = elapsed(t0);
    std::cout << "in-mem parlay::tabulate: " << std::setprecision(4)
              << inmem_tabulate_s << "s\n";

    auto ours = plaid::materialize<uint64_t>(tab);
    if (ours.size() != tab_mem.size()) {
      std::cout << "*** MISMATCH: out-of-core produced " << ours.size()
                << " elements, expected " << tab_mem.size() << " ***\n";
      agree = false;
    } else {
      for (size_t i = 0; i < ours.size() && agree; i++) {
        if (ours[i] != tab_mem[i]) {
          std::cout << "*** MISMATCH at index " << i << ": " << ours[i]
                    << " != " << tab_mem[i] << " ***\n";
          agree = false;
        }
      }
      if (agree)
        std::cout << "cross-check: out-of-core tabulate matches in-mem "
                     "tabulate exactly\n";
    }
  } else {
    std::cout << "in-mem parlay::tabulate: skipped (~16n footprint exceeds "
              << "RAM budget " << std::setprecision(2) << to_gb(budget)
              << " GB)\n";
  }

  // Machine-readable line for benchmarks/run_benches.py (examples sweep).
  // Columns: n,tabulate_s,inmem_tabulate_s,throughput_gb_s
  // (inmem_tabulate_s blank when the output exceeds the RAM budget).
  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  std::cout << "CSV," << n << ',' << f9(tabulate_s) << ','
            << (inmem_ok ? f9(inmem_tabulate_s) : std::string()) << ','
            << f9(gb_s) << '\n';

  cleanup_prefix(out_prefix);
  return agree ? 0 : 1;
}
