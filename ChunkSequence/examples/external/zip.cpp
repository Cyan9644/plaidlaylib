// zipExample — out-of-core zip+reduce benchmark/demo.
//
// Computes the dot product of two n-element uint64_t operand sequences via
// the delayed (fused) layer (Primitives/delayed.h): delay both operands ->
// zip -> map(multiply) -> reduce(sum), a single streaming read pass over
// both operands with zero intermediate writes -- the point of the delayed
// layer (see the file-level comment there).  Compares against parlay::zip +
// parlay::reduce on the identical operands built in DRAM.
//
// parlay::zip has no standalone dot-product helper, so the in-memory
// baseline zips the two operand sequences (producing std::tuple<uint64_t,
// uint64_t> elements), maps the product with parlay::tabulate, and reduces
// with parlay::addm<uint64_t>() -- the direct in-memory analogue of the
// out-of-core delayed chain.
//
// Dual-purpose like the other examples: prints human-readable timings and
// ends with a machine-readable `CSV,` line that benchmarks/run_benches.py
// greps.  Unsigned multiply-then-add is associative/commutative bit-exact
// regardless of term order, so the cross-check is exact scalar equality.
// The in-memory baseline's resident footprint is both operands (16n) + the
// materialized zipped-pair sequence (16n) + the product sequence (8n) =
// ~40n, gated via EXAMPLE_INMEM_BUDGET_BYTES.
//
//   usage: zipExample [global --flags] [n]

#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>

#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/Primitives/delayed.h"
#include "absl/log/check.h"
#include "parlay/monoid.h"
#include "parlay/primitives.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"
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
    unlink(GetFileName(prefix, d).c_str());
}

static uint64_t a_at(size_t i) { return parlay::hash64(i); }
static uint64_t b_at(size_t i) {
  return parlay::hash64(i ^ 0x9e3779b97f4a7c15ULL);
}

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
  CHECK(n > 0) << "need n > 0 (n=" << n << ")";

  // RAM budget: both baseline operands (16n) + materialized zipped pairs
  // (16n) + product sequence (8n) = ~40n.
  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys / 2;
  if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  const bool inmem_ok = n <= budget / 40;

  const std::string a_prefix = "zip_a";
  const std::string b_prefix = "zip_b";

  std::cout << "Building two " << n << "-element operands..." << std::flush;
  trace_mark("build_start");
  auto t0 = Clock::now();
  chunk_seq a = plaid::tabulate<uint64_t>(n, a_prefix, a_at);
  chunk_seq b = plaid::tabulate<uint64_t>(n, b_prefix, b_at);
  const double build_s = elapsed(t0);
  trace_mark("build_end");
  std::cout << " done (" << std::fixed << std::setprecision(4) << build_s
            << "s)\n";
  quiesce_drives();

  std::cout << "Zipping + reducing " << n << " element pairs (dot product)..."
            << std::flush;
  trace_mark("op_start");
  t0 = Clock::now();
  namespace d = plaid::delayed;
  auto za = d::delay<uint64_t>(a);
  auto zb = d::delay<uint64_t>(b);
  auto zipped = d::zip(za, zb);
  auto prod = d::map(zipped, [](std::pair<uint64_t, uint64_t> p) {
    return p.first * p.second;
  });
  const uint64_t dot = d::reduce(prod, SumMonoid{});
  const double zip_s = elapsed(t0);
  trace_mark("op_end");
  const double gb_s = to_gb(2 * n * sizeof(uint64_t)) / zip_s;
  std::cout << " done   dot=" << dot << "   " << std::setprecision(4) << zip_s
            << "s   " << std::setprecision(2) << gb_s
            << " GB/s (operands read)\n";

  bool agree = true;
  double inmem_zip_s = 0;
  if (inmem_ok) {
    auto a_mem = parlay::tabulate(n, a_at);
    auto b_mem = parlay::tabulate(n, b_at);
    t0 = Clock::now();
    auto zipped_mem = parlay::zip(a_mem, b_mem);
    auto prod_mem = parlay::tabulate(n, [&](size_t i) {
      return std::get<0>(zipped_mem[i]) * std::get<1>(zipped_mem[i]);
    });
    const uint64_t dot_mem = parlay::reduce(prod_mem, parlay::addm<uint64_t>());
    inmem_zip_s = elapsed(t0);
    std::cout << "in-mem parlay::zip+reduce: dot=" << dot_mem << "   "
              << std::setprecision(4) << inmem_zip_s << "s\n";
    if (dot != dot_mem) {
      std::cout << "*** MISMATCH: out-of-core dot " << dot
                << " != in-mem dot " << dot_mem << " ***\n";
      agree = false;
    } else {
      std::cout << "cross-check: dot products match exactly\n";
    }
  } else {
    std::cout << "in-mem parlay::zip+reduce: skipped (~40n footprint exceeds "
              << "RAM budget " << std::setprecision(2) << to_gb(budget)
              << " GB)\n";
  }

  // Machine-readable line for benchmarks/run_benches.py (examples sweep).
  // Columns: n,build_s,zip_s,inmem_zip_s,result,throughput_gb_s
  // (inmem_zip_s blank when the operands exceed the RAM budget).
  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  std::cout << "CSV," << n << ',' << f9(build_s) << ',' << f9(zip_s) << ','
            << (inmem_ok ? f9(inmem_zip_s) : std::string()) << ',' << dot
            << ',' << f9(gb_s) << '\n';

  cleanup_prefix(a_prefix);
  cleanup_prefix(b_prefix);
  return agree ? 0 : 1;
}
