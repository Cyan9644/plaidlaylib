// primitive_demos.cpp -- one binary holding every per-primitive demo.
//
//   usage: primitive_demosExample <primitive> [global --flags] [n ...]
//
// Each subcommand is the driver that used to be its own ChunkSequence/examples/
// external/<name>.cpp binary, moved here verbatim inside its own namespace with
// main() renamed to run().  Behaviour is unchanged: every subcommand still
// prints human-readable timings, cross-checks the out-of-core result against an
// in-DRAM baseline, ends with the same machine-readable `CSV,` line that
// benchmarks/run_benches.py greps, and exits non-zero on a mismatch.
//
// Subcommands: map, reduce, scan, tabulate, zip, filter, pack, count_sort,
// histogram_by_index, cut, random_shuffle

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/Primitives/delayed.h"
#include "ChunkSequence/Primitives/secondary_primitives.h"
#include "ChunkSequence/Primitives/sort.h"
#include "absl/log/check.h"
#include "parlay/monoid.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "utils/file_utils.h"
#include "utils/trace_marker.h"

// ============================================================================
// map -- ChunkMap over an ExternalTransform body
//
// (was ChunkSequence/examples/external/map.cpp)
// ============================================================================

namespace demo_map {

// mapExample — out-of-core map benchmark/demo.
//
// Applies f(x) = x*2+1 to an n-element uint64_t input via plaid::ChunkMap
// (Primitives/map.h, an ExternalTransform body) and compares against
// parlaylib's own map idiom: parlaylib has no standalone `map` primitive, so
// the in-memory baseline is parlay::tabulate(n, [&](i){ return f(seq[i]); }),
// the same idiom this repo's own examples use wherever they need an in-memory
// elementwise map.
//
// Dual-purpose like the other examples: prints human-readable timings and
// ends with a machine-readable `CSV,` line that benchmarks/run_benches.py
// greps.  The out-of-core output is read back (plaid::materialize) and
// compared element-for-element against the in-memory result; the in-memory
// baseline is gated by a RAM budget (input + baseline output + readback =
// ~24n), overridable via EXAMPLE_INMEM_BUDGET_BYTES.
//
//   usage: mapExample [global --flags] [n]

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

static uint64_t elem_at(size_t i) { return parlay::hash64(i); }
static uint64_t f(uint64_t x) { return x * 2 + 1; }

int run(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
  CHECK(n > 0) << "need n > 0 (n=" << n << ")";

  // RAM budget: baseline input (8n) + baseline output (8n) + out-of-core
  // readback for the cross-check (8n) = ~24n.
  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys / 2;
  if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  const bool inmem_ok = n <= budget / 24;

  const std::string in_prefix = "map_in";
  const std::string out_prefix = "map_out";

  std::cout << "Building " << n << "-element input..." << std::flush;
  trace_mark("build_start");
  auto t0 = Clock::now();
  chunk_seq seq = plaid::tabulate<uint64_t>(n, in_prefix, elem_at);
  const double build_s = elapsed(t0);
  trace_mark("build_end");
  std::cout << " done (" << std::fixed << std::setprecision(4) << build_s
            << "s)\n";
  quiesce_drives();

  std::cout << "Mapping " << n << " elements (x*2+1)..." << std::flush;
  trace_mark("op_start");
  t0 = Clock::now();
  chunk_seq mapped = plaid::ChunkMap<uint64_t>(seq, out_prefix, f);
  const double map_s = elapsed(t0);
  trace_mark("op_end");
  const double gb_s = to_gb(n * sizeof(uint64_t)) / map_s;
  std::cout << " done   " << std::setprecision(4) << map_s << "s   "
            << std::setprecision(2) << gb_s << " GB/s (input read)\n";

  bool agree = true;
  double inmem_map_s = 0;
  if (inmem_ok) {
    auto seq_mem = parlay::tabulate(n, elem_at);
    t0 = Clock::now();
    auto mapped_mem =
        parlay::tabulate(n, [&](size_t i) { return f(seq_mem[i]); });
    inmem_map_s = elapsed(t0);
    std::cout << "in-mem parlay::tabulate map: " << std::setprecision(4)
              << inmem_map_s << "s\n";

    auto ours = plaid::materialize<uint64_t>(mapped);
    if (ours.size() != mapped_mem.size()) {
      std::cout << "*** MISMATCH: out-of-core produced " << ours.size()
                << " elements, expected " << mapped_mem.size() << " ***\n";
      agree = false;
    } else {
      for (size_t i = 0; i < ours.size() && agree; i++) {
        if (ours[i] != mapped_mem[i]) {
          std::cout << "*** MISMATCH at index " << i << ": " << ours[i]
                    << " != " << mapped_mem[i] << " ***\n";
          agree = false;
        }
      }
      if (agree)
        std::cout << "cross-check: out-of-core map matches in-mem map "
                     "exactly\n";
    }
  } else {
    std::cout << "in-mem parlay::tabulate map: skipped (~24n footprint "
              << "exceeds RAM budget " << std::setprecision(2) << to_gb(budget)
              << " GB)\n";
  }

  // Machine-readable line for benchmarks/run_benches.py (examples sweep).
  // Columns: n,build_s,map_s,inmem_map_s,throughput_gb_s
  // (inmem_map_s blank when the input exceeds the RAM budget).
  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  std::cout << "CSV," << n << ',' << f9(build_s) << ',' << f9(map_s) << ','
            << (inmem_ok ? f9(inmem_map_s) : std::string()) << ',' << f9(gb_s)
            << '\n';

  cleanup_prefix(in_prefix);
  cleanup_prefix(out_prefix);
  return agree ? 0 : 1;
}

}  // namespace demo_map

// ============================================================================
// reduce -- ChunkReduce over a RemoveWorker fold
//
// (was ChunkSequence/examples/external/reduce.cpp)
// ============================================================================

namespace demo_reduce {

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

static uint64_t elem_at(size_t i) { return (uint64_t)i; }

int run(int argc, char* argv[]) {
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
      std::cout << "*** MISMATCH: out-of-core sum " << sum << " != in-mem sum "
                << sum_mem << " ***\n";
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

}  // namespace demo_reduce

// ============================================================================
// scan -- ChunkScan (RemoveWorker pass 1 + ExternalTransform pass 2)
//
// (was ChunkSequence/examples/external/scan.cpp)
// ============================================================================

namespace demo_scan {

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

static uint64_t elem_at(size_t i) { return parlay::hash64(i); }

int run(int argc, char* argv[]) {
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

  std::cout << "Scanning " << n << " elements (exclusive sum)..." << std::flush;
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

}  // namespace demo_scan

// ============================================================================
// tabulate -- tabulate / iota -- the writer-only pipeline
//
// (was ChunkSequence/examples/external/tabulate.cpp)
// ============================================================================

namespace demo_tabulate {

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

int run(int argc, char* argv[]) {
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
            << "s   " << std::setprecision(2) << gb_s
            << " GB/s (output write)\n";

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

}  // namespace demo_tabulate

// ============================================================================
// zip -- delayed::zip of two chunk_seqs, fused
//
// (was ChunkSequence/examples/external/zip.cpp)
// ============================================================================

namespace demo_zip {

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

int run(int argc, char* argv[]) {
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
      std::cout << "*** MISMATCH: out-of-core dot " << dot << " != in-mem dot "
                << dot_mem << " ***\n";
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
            << (inmem_ok ? f9(inmem_zip_s) : std::string()) << ',' << dot << ','
            << f9(gb_s) << '\n';

  cleanup_prefix(a_prefix);
  cleanup_prefix(b_prefix);
  return agree ? 0 : 1;
}

}  // namespace demo_zip

// ============================================================================
// filter -- ChunkFilter over DensePackStream
//
// (was ChunkSequence/examples/external/filter.cpp)
// ============================================================================

namespace demo_filter {

// filterExample — out-of-core filter benchmark/demo.
//
// Keeps the even elements of an n-element uint64_t input via
// plaid::ChunkFilter (Primitives/filter.h, a DensePack producer) and compares
// against parlay::filter on the identical input built in DRAM.
//
// Dual-purpose like the other examples: prints human-readable timings and
// ends with a machine-readable `CSV,` line that benchmarks/run_benches.py
// greps.  Both implementations preserve input order, so the cross-check is a
// direct element-for-element equality (not just a count/sum check) after
// reading the out-of-core survivors back with plaid::materialize.  The
// in-memory baseline is gated by a RAM budget (input + baseline output +
// readback = ~24n), overridable via EXAMPLE_INMEM_BUDGET_BYTES.
//
//   usage: filterExample [global --flags] [n]

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

static uint64_t elem_at(size_t i) { return (uint64_t)i; }
static bool pred(uint64_t v) { return (v & 1) == 0; }

int run(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
  CHECK(n > 0) << "need n > 0 (n=" << n << ")";

  // RAM budget: baseline input (8n) + baseline output (up to 8n) +
  // out-of-core readback for the cross-check (up to 8n) = ~24n worst case.
  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys / 2;
  if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  const bool inmem_ok = n <= budget / 24;

  const std::string in_prefix = "flt_in";
  const std::string out_prefix = "flt_out";

  std::cout << "Building " << n << "-element input..." << std::flush;
  trace_mark("build_start");
  auto t0 = Clock::now();
  chunk_seq seq = plaid::tabulate<uint64_t>(n, in_prefix, elem_at);
  const double build_s = elapsed(t0);
  trace_mark("build_end");
  std::cout << " done (" << std::fixed << std::setprecision(4) << build_s
            << "s)\n";
  quiesce_drives();

  std::cout << "Filtering " << n << " elements (keep even)..." << std::flush;
  trace_mark("op_start");
  t0 = Clock::now();
  chunk_seq survivors = plaid::ChunkFilter<uint64_t>(seq, out_prefix, pred);
  const double filter_s = elapsed(t0);
  trace_mark("op_end");
  size_t count = 0;
  for (const auto& c : survivors.chunks) count += c.used / sizeof(uint64_t);
  const double gb_s = to_gb(n * sizeof(uint64_t)) / filter_s;
  std::cout << " done   " << count << " survivor(s)   " << std::setprecision(4)
            << filter_s << "s   " << std::setprecision(2) << gb_s
            << " GB/s (input read)\n";

  bool agree = true;
  double inmem_filter_s = 0;
  if (inmem_ok) {
    auto seq_mem = parlay::tabulate(n, elem_at);
    t0 = Clock::now();
    auto survivors_mem = parlay::filter(seq_mem, pred);
    inmem_filter_s = elapsed(t0);
    std::cout << "in-mem parlay::filter: " << survivors_mem.size()
              << " survivor(s)   " << std::setprecision(4) << inmem_filter_s
              << "s\n";

    if (count != survivors_mem.size()) {
      std::cout << "*** MISMATCH: out-of-core count " << count
                << " != in-mem count " << survivors_mem.size() << " ***\n";
      agree = false;
    } else {
      auto ours = plaid::materialize<uint64_t>(survivors);
      for (size_t i = 0; i < ours.size() && agree; i++) {
        if (ours[i] != survivors_mem[i]) {
          std::cout << "*** MISMATCH at index " << i << ": " << ours[i]
                    << " != " << survivors_mem[i] << " ***\n";
          agree = false;
        }
      }
      if (agree)
        std::cout << "cross-check: out-of-core filter matches in-mem filter "
                     "exactly (order-preserving)\n";
    }
  } else {
    std::cout << "in-mem parlay::filter: skipped (~24n footprint exceeds "
              << "RAM budget " << std::setprecision(2) << to_gb(budget)
              << " GB)\n";
  }

  // Machine-readable line for benchmarks/run_benches.py (examples sweep).
  // Columns: n,build_s,filter_s,inmem_filter_s,count,throughput_gb_s
  // (inmem_filter_s blank when the input exceeds the RAM budget).
  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  std::cout << "CSV," << n << ',' << f9(build_s) << ',' << f9(filter_s) << ','
            << (inmem_ok ? f9(inmem_filter_s) : std::string()) << ',' << count
            << ',' << f9(gb_s) << '\n';

  cleanup_prefix(in_prefix);
  cleanup_prefix(out_prefix);
  return agree ? 0 : 1;
}

}  // namespace demo_filter

// ============================================================================
// pack -- pack -- boolean-array gate on DensePack
//
// (was ChunkSequence/examples/external/pack.cpp)
// ============================================================================

namespace demo_pack {

// packExample — out-of-core pack benchmark/demo.
//
// Packs the elements of an n-element uint64_t input flagged true by a
// deterministic boolean sequence (i%3 != 0, ~2/3 survive) via plaid::pack
// (Primitives/pack.h, the DRAM-boolean-sequence overload, a DensePack
// producer) and compares against parlay::pack on the identical input and
// flags built in DRAM.
//
// Dual-purpose like the other examples: prints human-readable timings and
// ends with a machine-readable `CSV,` line that benchmarks/run_benches.py
// greps.  Both implementations preserve input order, so the cross-check is a
// direct element-for-element equality (after reading the out-of-core result
// back with plaid::materialize).  The in-memory baseline is gated by a RAM
// budget (input + flags + baseline output + readback = ~25n, rounded to /32
// for headroom), overridable via EXAMPLE_INMEM_BUDGET_BYTES.
//
//   usage: packExample [global --flags] [n]

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

static uint64_t elem_at(size_t i) { return parlay::hash64(i); }
static bool flag_at(size_t i) { return (i % 3) != 0; }

int run(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
  CHECK(n > 0) << "need n > 0 (n=" << n << ")";

  // RAM budget: baseline input (8n) + flags (~1n, bool) + baseline output
  // (up to 8n) + out-of-core readback for the cross-check (up to 8n) =~ 25n;
  // rounded down to /32 for headroom.
  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys / 2;
  if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  const bool inmem_ok = n <= budget / 32;

  const std::string in_prefix = "pck_in";
  const std::string out_prefix = "pck_out";

  std::cout << "Building " << n << "-element input..." << std::flush;
  trace_mark("build_start");
  auto t0 = Clock::now();
  chunk_seq seq = plaid::tabulate<uint64_t>(n, in_prefix, elem_at);
  parlay::sequence<bool> flags = parlay::tabulate(n, flag_at);
  const double build_s = elapsed(t0);
  trace_mark("build_end");
  std::cout << " done (" << std::fixed << std::setprecision(4) << build_s
            << "s)\n";
  quiesce_drives();

  std::cout << "Packing " << n << " elements (keep i%3!=0)..." << std::flush;
  trace_mark("op_start");
  t0 = Clock::now();
  chunk_seq packed = plaid::pack<uint64_t>(seq, out_prefix, flags);
  const double pack_s = elapsed(t0);
  trace_mark("op_end");
  size_t out_elems = 0;
  for (const auto& c : packed.chunks) out_elems += c.used / sizeof(uint64_t);
  const double gb_s = to_gb(n * sizeof(uint64_t)) / pack_s;
  std::cout << " done   " << out_elems << " kept   " << std::setprecision(4)
            << pack_s << "s   " << std::setprecision(2) << gb_s
            << " GB/s (input read)\n";

  bool agree = true;
  double inmem_pack_s = 0;
  if (inmem_ok) {
    auto seq_mem = parlay::tabulate(n, elem_at);
    t0 = Clock::now();
    auto packed_mem = parlay::pack(seq_mem, flags);
    inmem_pack_s = elapsed(t0);
    std::cout << "in-mem parlay::pack: " << packed_mem.size() << " kept   "
              << std::setprecision(4) << inmem_pack_s << "s\n";

    if (out_elems != packed_mem.size()) {
      std::cout << "*** MISMATCH: out-of-core count " << out_elems
                << " != in-mem count " << packed_mem.size() << " ***\n";
      agree = false;
    } else {
      auto ours = plaid::materialize<uint64_t>(packed);
      for (size_t i = 0; i < ours.size() && agree; i++) {
        if (ours[i] != packed_mem[i]) {
          std::cout << "*** MISMATCH at index " << i << ": " << ours[i]
                    << " != " << packed_mem[i] << " ***\n";
          agree = false;
        }
      }
      if (agree)
        std::cout << "cross-check: out-of-core pack matches in-mem pack "
                     "exactly (order-preserving)\n";
    }
  } else {
    std::cout << "in-mem parlay::pack: skipped (~25n footprint exceeds "
              << "RAM budget " << std::setprecision(2) << to_gb(budget)
              << " GB)\n";
  }

  // Machine-readable line for benchmarks/run_benches.py (examples sweep).
  // Columns: n,build_s,pack_s,inmem_pack_s,out_elems,throughput_gb_s
  // (inmem_pack_s blank when the input exceeds the RAM budget).
  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  std::cout << "CSV," << n << ',' << f9(build_s) << ',' << f9(pack_s) << ','
            << (inmem_ok ? f9(inmem_pack_s) : std::string()) << ',' << out_elems
            << ',' << f9(gb_s) << '\n';

  cleanup_prefix(in_prefix);
  cleanup_prefix(out_prefix);
  return agree ? 0 : 1;
}

}  // namespace demo_pack

// ============================================================================
// count_sort -- count_sort -- BucketWriter scatter into per-bucket sequences
//
// (was ChunkSequence/examples/external/count_sort.cpp)
// ============================================================================

namespace demo_group_by_index {

// group_by_indexExample — out-of-core group-by benchmark/demo.
//
// Buckets an n-element uint64_t input via plaid::group_by_index
// (Primitives/sort.h, a single streaming-reader pass that routes each
// element to its bucket's write stream via a key function), then reads each
// bucket's chunk_seq back with plaid::materialize individually and lays the
// results out in bucket order -- group_by_index's own contract forbids
// concatenating its per-bucket chunk_seqs (plaid::fuse), since that would
// bury each bucket's trailing partial chunk mid-sequence.  Compares against
// parlay::counting_sort (its identity-key overload, which requires elements
// already in [0, num_buckets)) on the identical input built in DRAM.
//
// The key IS the value here (hash64(i) % num_buckets, so every element in
// bucket b is literally the integer b, repeated) -- this makes the
// bucket-grouped output fully determined regardless of within-bucket order,
// so the cross-check is a direct element-for-element equality (after reading
// the out-of-core result back with plaid::materialize) rather than a
// permutation-only check.
//
// Dual-purpose like the other examples: prints human-readable timings and
// ends with a machine-readable `CSV,` line that benchmarks/run_benches.py
// greps.  The in-memory baseline is gated by a RAM budget (input + baseline
// output + readback = ~24n), overridable via EXAMPLE_INMEM_BUDGET_BYTES.
//
//   usage: group_by_indexExample [global --flags] [n]

// Upper bound on bucket count once n is large enough to amortize
// group_by_index's fixed per-bucket setup cost -- see its derivation from
// n/RAM in run() below (a fixed 4096 regardless of n used to blow past
// available RAM at modest n: group_by_index's BucketWriter provisions a
// kRequestsPerBucket * IO_VECTOR_SIZE-iovec Request pool PER BUCKET
// (Primitives/chunk_seq.h) plus one live SAMPLE_SORT_BUCKET_SIZE scatter
// buffer per (worker, bucket) -- overhead that scales with num_buckets, not
// with n, so a small n can't amortize it the way sample_sort's num_buckets
// (~filer / 128 MiB, Primitives/sort.h) does).
static constexpr size_t kMaxBuckets = 4096;

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

// group_by_index's BucketWriter creates one file per bucket (GetFileName(
// bucket_prefix, i) for i in [0, num_buckets)), not one per drive -- unlike
// cleanup_prefix above, which only unlinks SSD_COUNT files and is correct for
// in_prefix (tabulate spreads its few chunks round-robin over the drives).
// Reusing cleanup_prefix for bucket_prefix would strand num_buckets -
// SSD_COUNT files on every run.
static void cleanup_bucket_prefix(const std::string& prefix,
                                  size_t num_buckets) {
  for (size_t i = 0; i < num_buckets; i++) unlink(GetFileName(prefix, i).c_str());
}

int run(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  // group_by_index's BucketWriter opens one fd per bucket (up to
  // kMaxBuckets), past the 1024 soft fd limit; lift it to the hard limit
  // before any I/O starts.
  RaiseFdLimit();
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
  CHECK(n > 0) << "need n > 0 (n=" << n << ")";

  // RAM budget: baseline input (8n) + baseline output (8n) + out-of-core
  // readback for the cross-check (8n) = ~24n.
  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys / 2;
  if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  const bool inmem_ok = n <= budget / 24;

  // num_buckets: capped at kMaxBuckets, but also bounded so
  // group_by_index's fixed per-bucket setup cost (a kRequestsPerBucket *
  // IO_VECTOR_SIZE-iovec Request pool entry, plus one live
  // SAMPLE_SORT_BUCKET_SIZE scatter buffer per worker -- see
  // Primitives/chunk_seq.h's BucketWriter / Primitives/sort.h's
  // group_by_index) can't dwarf the RAM budget at small n.  Mirrors
  // sample_sort's own bucket count scaling with input size
  // (Primitives/sort.h) instead of a fixed constant.
  const size_t per_bucket_overhead_bytes =
      plaid::kRequestsPerBucket * IO_VECTOR_SIZE * sizeof(struct iovec) +
      parlay::num_workers() * SAMPLE_SORT_BUCKET_SIZE;
  const size_t max_buckets_by_ram =
      std::max<size_t>(1, budget / 4 / per_bucket_overhead_bytes);
  const size_t num_buckets =
      std::max<size_t>(1, std::min({kMaxBuckets, max_buckets_by_ram, n}));

  const std::string in_prefix = "gbi_ex_in";
  const std::string bucket_prefix = "gbi_ex_bucket";

  const auto key_at = [num_buckets](size_t i) {
    return parlay::hash64(i) % num_buckets;
  };

  std::cout << "Building " << n << "-element input (mod " << num_buckets
            << ")..." << std::flush;
  trace_mark("build_start");
  auto t0 = Clock::now();
  chunk_seq keys = plaid::tabulate<uint64_t>(n, in_prefix, key_at);
  const double build_s = elapsed(t0);
  trace_mark("build_end");
  std::cout << " done (" << std::fixed << std::setprecision(4) << build_s
            << "s)\n";
  quiesce_drives();

  std::cout << "Grouping " << n << " elements into " << num_buckets
            << " buckets..." << std::flush;
  trace_mark("op_start");
  t0 = Clock::now();
  std::vector<chunk_seq> buckets = plaid::group_by_index<uint64_t>(
      keys, num_buckets, bucket_prefix,
      [](uint64_t v) { return (size_t)v; });
  const double sort_s = elapsed(t0);
  trace_mark("op_end");
  const double gb_s = to_gb(n * sizeof(uint64_t)) / sort_s;
  std::cout << " done   " << std::setprecision(4) << sort_s << "s   "
            << std::setprecision(2) << gb_s << " GB/s (input read)\n";

  bool agree = true;
  double inmem_sort_s = 0;
  if (inmem_ok) {
    auto keys_mem = parlay::tabulate(n, key_at);
    t0 = Clock::now();
    auto sorted_mem = parlay::counting_sort(keys_mem, num_buckets).first;
    inmem_sort_s = elapsed(t0);
    std::cout << "in-mem parlay::counting_sort: " << std::setprecision(4)
              << inmem_sort_s << "s\n";

    // group_by_index's per-bucket chunk_seqs must not be concatenated
    // (Primitives/sort.h) -- materialize each bucket separately and lay the
    // results out in bucket order, matching counting_sort's layout, instead
    // of fusing the chunk lists first.
    parlay::sequence<uint64_t> ours(n);
    size_t offset = 0;
    for (size_t b = 0; b < num_buckets && offset <= n; b++) {
      auto part = plaid::materialize<uint64_t>(buckets[b]);
      if (offset + part.size() > n) {
        offset = n + 1;  // force the size mismatch branch below
        break;
      }
      std::copy(part.begin(), part.end(), ours.begin() + offset);
      offset += part.size();
    }
    if (offset != sorted_mem.size()) {
      std::cout << "*** MISMATCH: out-of-core produced " << offset
                << " elements, expected " << sorted_mem.size() << " ***\n";
      agree = false;
    } else {
      for (size_t i = 0; i < offset && agree; i++) {
        if (ours[i] != sorted_mem[i]) {
          std::cout << "*** MISMATCH at index " << i << ": " << ours[i]
                    << " != " << sorted_mem[i] << " ***\n";
          agree = false;
        }
      }
      if (agree)
        std::cout << "cross-check: out-of-core counting sort matches in-mem "
                     "counting sort exactly\n";
    }
  } else {
    std::cout << "in-mem parlay::counting_sort: skipped (~24n footprint "
              << "exceeds RAM budget " << std::setprecision(2) << to_gb(budget)
              << " GB)\n";
  }

  // Machine-readable line for benchmarks/run_benches.py (examples sweep).
  // Columns: n,build_s,sort_s,inmem_sort_s,throughput_gb_s
  // (inmem_sort_s blank when the input exceeds the RAM budget).
  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  std::cout << "CSV," << n << ',' << f9(build_s) << ',' << f9(sort_s) << ','
            << (inmem_ok ? f9(inmem_sort_s) : std::string()) << ',' << f9(gb_s)
            << '\n';

  cleanup_prefix(in_prefix);
  cleanup_bucket_prefix(bucket_prefix, num_buckets);
  return agree ? 0 : 1;
}

}  // namespace demo_group_by_index

// ============================================================================
// histogram_by_index -- ChunkHistogramByIndex -- per-worker bucket-count fold
//
// (was ChunkSequence/examples/external/histogram_by_index.cpp)
// ============================================================================

namespace demo_histogram_by_index {

// histogram_by_indexExample — out-of-core histogram benchmark/demo.
//
// Builds an n-element uint64_t sequence of bucket ids (hash64(i) % 4096) and
// counts them via plaid::ChunkHistogramByIndex
// (Primitives/histogram_by_index.h, a RemoveWorker per-worker bucket-count
// fold), comparing against parlay::histogram_by_index on the identical ids
// built in DRAM.
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

int run(int argc, char* argv[]) {
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
              << "exceeds RAM budget " << std::setprecision(2) << to_gb(budget)
              << " GB)\n";
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

}  // namespace demo_histogram_by_index

// ============================================================================
// cut -- cut -- chunk-aligned slice/shift
//
// (was ChunkSequence/examples/external/cut.cpp)
// ============================================================================

namespace demo_cut {

// Example: out-of-core slice/cut via
// plaid::sequential_cut_no_compression.
//
// Builds an n-element sequence of pseudo-random uint64 keys across the SSDs
// (deterministic from parlay::hash64, so it is reproducible and
// duplicate-free), then extracts the half-open range [start, end) into a new,
// independent out-of-core sequence with the cut in
// ExternalPrimitives/chunk_cut.h.  The cut rewrites the (possibly unaligned)
// first/last partial chunks into fresh O_DIRECT seam files and threads the
// interior chunk headers through; from_chunks then materializes the whole
// result into fresh, independent on-disk files (a full copy of the range across
// the drives), so the returned sequence owns all its data -- symmetric with the
// in-memory baseline, which copies the range into a fresh DRAM sequence.
//
// Dual-purpose, like the benchmarks and the other examples: prints
// human-readable results AND a machine-readable "CSV," line that
// benchmarks/run_benches.py greps. The examples sweep (make bench-examples)
// times the cut across a sweep of n with the range held at the middle ~half (k
// = n/2), so the cut length scales with n. Both endpoints are placed in the
// MIDDLE of a chunk (offset ELEMS_PER_CHUNK/2), never on a chunk boundary, so
// every sweep point deterministically exercises the same real seam-rewrite work
// (see the default computation in main() for why the naive n/4, 3n/4 endpoints
// alias the chunk grid for power-of-two n and skew the timings).  The in-mem
// baseline cuts the identical [start, end).
//
// Baseline: parlaylib's slice/cut on the same keys in DRAM.  parlay::slice::cut
// itself is O(1) -- it returns a view, copying nothing -- so timing it alone is
// meaningless.  The out-of-core cut materializes an *independent* copy of the
// range, so the honest in-memory analog is copying the slice into a fresh
// sequence (parlay::to_sequence(keys.cut(first, last))); that is what we time.
// The result is cross-checked by reading the out-of-core cut back in logical
// order and comparing it element-wise against the in-memory copy (keys are
// distinct, so the range is unique and the two substrates must agree exactly).
// Budget: half of physical RAM, override via EXAMPLE_INMEM_BUDGET_BYTES; when
// skipped the CSV field is left blank so the plotted in-mem line stops at the
// RAM cliff (as in the other examples).
//
//   usage: chunk_cutExample [global --flags] [n] [start] [end]
//     n       number of keys (default 1e6)
//     start   0-based first index of the range to cut (default n/4)
//     end     0-based one-past-last index of the range   (default 3n/4)
//
// CSV line:
// CSV,<n>,<start>,<end>,<build_s>,<cut_s>,<inmem_cut_s>,<out_elems>,<throughput_gb_s>
//   throughput = cut-range bytes ((end-start)*8) / cut_s.
//
// Complexity: O(k) work for a k-element range -- from_chunks copies the whole
// range (interior chunks + the two rewritten seam chunks) to fresh on-disk
// files.

// Out-of-core operation under test.  The in-memory baseline is just parlaylib's
// slice::cut, pulled in with the rest of parlay/primitives.h above.

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Quiesce the drives between the input build and the timed operation: flush the
// build's just-written dirty pages and let the devices settle, so its still-
// draining writeback doesn't queue behind (and inflate) the operation's own
// I/O. Called outside every timed region, so its cost never enters a
// measurement -- it isolates the op timer from the build, the biggest source of
// run-to-run timing noise on a shared substrate.
static void quiesce_drives() {
  sync();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
}
static double to_gb(size_t bytes) {
  return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

// Deterministic, duplicate-free key i, computable anywhere so the out-of-core
// input and the in-memory baseline hold the identical multiset.
static uint64_t key_at(size_t i) { return parlay::hash64(i); }

// Read a chunk_seq back into DRAM in logical (vector) order, concatenating each
// chunk's `used` bytes.  Unlike chunk_seq::to_vector / operator[], this does
// NOT assume the index-ordered invariant or that every chunk but the last is
// full: the cut's output has a (possibly partial) rewritten head chunk and
// reuses interior chunks with their original index fields, so it must be read
// strictly by position.  Sequential and only used for the cross-check.
template <typename T>
static std::vector<T> read_in_order(const chunk_seq& seq) {
  std::vector<T> out;
  T* buf = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
  CHECK(buf != nullptr) << "read_in_order: buffer allocation failed";
  for (const chunk& c : seq.chunks) {
    int fd = open(c.filename.c_str(), O_RDONLY | O_DIRECT);
    SYSCALL(fd);
    SYSCALL(pread(fd, buf, AlignUp(c.used), (off_t)c.begin_addr));
    close(fd);
    const size_t cnt = c.used / sizeof(T);
    out.insert(out.end(), buf, buf + cnt);
  }
  free(buf);
  return out;
}

int run(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;

  // Default range: the middle ~half of the sequence ([start, end) with k =
  // n/2), with BOTH endpoints landing in the MIDDLE of a chunk (offset
  // ELEMS_PER_CHUNK/2) rather than on a chunk boundary.  This is deterministic
  // regardless of how n aligns to the CHUNK grid.
  //
  // Why it matters: the old defaults (n/4, 3n/4) land *exactly* on chunk
  // boundaries for every power-of-two sweep size >= 2^21 (n/4 is then a
  // multiple of ELEMS_PER_CHUNK).  A boundary-aligned cut hits a degenerate
  // seam path -- the start seam rewrites a whole chunk and the end seam is an
  // empty (used=0) chunk -- so it exercises little of the real per-seam
  // read/rewrite work and its cost jumps around vs. the one non-aligned point
  // (2^20).  Forcing each seam to split its chunk ~half-half makes every sweep
  // point do the same real work.  The in-mem baseline below uses this identical
  // [start, end), so both substrates always cut the same range.
  constexpr size_t EPC = ELEMS_PER_CHUNK;
  size_t def_start = (n / 4 / EPC) * EPC + EPC / 2;
  size_t def_end = (3 * n / 4 / EPC) * EPC + EPC / 2;
  if (def_start >= def_end ||
      def_end > n) {  // tiny n: fall back to a valid range
    def_start = n / 4;
    def_end = (3 * n) / 4;
  }
  const size_t start = (argc > 2) ? std::stoull(argv[2]) : def_start;
  const size_t end = (argc > 3) ? std::stoull(argv[3]) : def_end;
  CHECK(n > 0 && start < end && end <= n)
      << "need 0 <= start < end <= n (n=" << n << ", start=" << start
      << ", end=" << end << ")";
  const size_t k = end - start;  // cut length

  // RAM budget for the in-memory baseline (as in the other examples): when it
  // runs we hold the n-key input (8n bytes), the copied slice (8k <= 8n), and
  // the read-back of the out-of-core cut (8k) for the element-wise cross-check
  // -- call it ~24n worst case.
  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys;
  if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  const bool inmem_ok = n <= budget / 24;

  const std::string in_prefix = "cut_in";

  std::cout << "Building " << n << "-key input..." << std::flush;
  auto t0 = Clock::now();
  chunk_seq seq = plaid::tabulate<uint64_t>(n, in_prefix, key_at);
  const double build_s = elapsed(t0);
  std::cout << " done (" << std::fixed << std::setprecision(4) << build_s
            << "s)\n";
  quiesce_drives();  // isolate the op timer from the build's writeback

  std::cout << "Cutting [" << start << ", " << end << ") (" << k
            << " elems) of " << n << "..." << std::flush;
  t0 = Clock::now();
  chunk_seq cut =
      plaid::sequential_cut_no_compression<uint64_t>(seq, start, end);
  const double cut_s = elapsed(t0);
  std::cout << " done\n";

  // True logical length: sum each chunk's used bytes.  (plaid::size
  // can't be used -- it assumes every chunk but the last is full, but the cut's
  // rewritten head chunk is partial.)
  size_t out_elems = 0;
  for (const chunk& c : cut.chunks) out_elems += c.used / sizeof(uint64_t);
  const double gb_s = to_gb(k * sizeof(uint64_t)) / cut_s;
  std::cout << "cut " << out_elems << " elems   " << std::setprecision(4)
            << cut_s << "s   " << std::setprecision(2) << gb_s
            << " GB/s (cut range)\n";

  // In-memory baseline: parlaylib's slice::cut on the same keys, materialized
  // into an independent sequence (the cut view itself copies nothing), timed
  // outside the input build.  Cross-checked by reading the out-of-core cut back
  // in logical order and comparing it element-wise.
  bool agree = true;
  double inmem_cut_s = 0;
  if (inmem_ok) {
    auto keys_mem = parlay::tabulate(n, key_at);  // parlay::sequence<uint64_t>
    t0 = Clock::now();
    auto slice_mem = parlay::to_sequence(keys_mem.cut(start, end));
    inmem_cut_s = elapsed(t0);
    std::cout << "in-mem parlaylib cut (materialized copy)   "
              << std::setprecision(4) << inmem_cut_s << "s\n";

    auto out_mem = read_in_order<uint64_t>(cut);
    if (out_mem.size() != slice_mem.size()) {
      std::cout << "*** MISMATCH: out-of-core cut produced " << out_mem.size()
                << " elems, expected " << slice_mem.size() << " ***\n";
      agree = false;
    } else {
      for (size_t i = 0; i < slice_mem.size(); i++) {
        if (out_mem[i] != slice_mem[i]) {
          std::cout << "*** MISMATCH at index " << i << ": out-of-core "
                    << out_mem[i] << " != in-mem " << slice_mem[i] << " ***\n";
          agree = false;
          break;
        }
      }
    }
    if (agree)
      std::cout << "cross-check: out-of-core cut matches in-mem slice\n";
  } else {
    std::cout << "in-mem parlaylib cut: skipped (~24n footprint exceeds RAM "
              << "budget " << std::setprecision(2) << to_gb(budget) << " GB)\n";
  }

  // Machine-readable line for benchmarks/run_benches.py (examples sweep).
  // Columns: n,start,end,build_s,cut_s,inmem_cut_s,out_elems,throughput_gb_s
  // (inmem_cut_s blank when the input exceeds the RAM budget).
  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  std::cout << "CSV," << n << ',' << start << ',' << end << ',' << f9(build_s)
            << ',' << f9(cut_s) << ','
            << (inmem_ok ? f9(inmem_cut_s) : std::string()) << ',' << out_elems
            << ',' << f9(gb_s) << '\n';

  // Don't leave data on the drives across sweep points.  Four sets of files
  // exist: the input ("cut_in<d>"), the two seam scratch files the cut writes
  // ("cut_in<d>_cut_start" / "cut_in<d>_cut_end"), and the materialized,
  // independent cut output that from_chunks now writes ("cut_out<d>").  Unlink
  // all of them (run_benches.py's "cut_in*"/"cut_out*" globs cover them across
  // sweep points).
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++) {
    const std::string f = GetFileName(in_prefix, d);
    unlink(f.c_str());
    unlink((f + "_cut_start").c_str());
    unlink((f + "_cut_end").c_str());
    unlink(GetFileName("cut_out", d).c_str());
  }
  return agree ? 0 : 1;
}

}  // namespace demo_cut

// ============================================================================
// random_shuffle -- random_shuffle -- count-sort bucketing + per-bucket shuffle
//
// (was ChunkSequence/examples/external/external_random_shuffle.cpp)
// ============================================================================

namespace demo_random_shuffle {

// Benchmark: out-of-core random shuffle, three contestants on the identical key
// multiset --
//
//   1. random_shuffle_method  (ExternalPrimitives/random_shuffle.h) -- the
//      bucketing shuffle written on the high-level abstractions: a fused
//      delayed map draws each element's bucket, count_sort routes the elements
//      into per-bucket external sequences, each bucket is read back / shuffled
//      in DRAM / written out as *fresh* files, flatten concatenates them.
//   2. plaid::Permutation::Permute -- the same algorithm on the
//      low-level reader/writer paradigm (ported from Peter's scatter-gather):
//      identical bucketing, but each bucket is rewritten **in place** over the
//      count-sort's own chunks (process_inplace), so it moves one fewer
//      copy of the data.
//   3. parlay::random_shuffle -- the in-memory parlaylib baseline (stops at the
//      RAM cliff, like every other example's in-mem series).
//
// Same shape as external_samplesort_vs_peter.cpp (two out-of-core substrates
// timed head-to-head, each on drives holding only its own input), extended with
// the in-DRAM baseline the plain examples carry.
//
// Correctness for a shuffle is a *permutation* check, not element-wise
// equality: the keys key_at(i)=parlay::hash64(i) are distinct, so an output is
// a valid shuffle iff, once sorted, it equals the sorted key set.  Both
// out-of-core outputs (and the in-mem baseline) are checked that way when the
// input fits the RAM budget.
//
//   usage: external_random_shuffleExample [global --flags] [n]
//     n   number of keys (default 1e6)
//
// CSV line:
//   CSV,<n>,<build_s>,<shuffle_s>,<perm_s>,<inmem_shuffle_s>,
//       <shuffle_gb_s>,<perm_gb_s>
//   throughput = input bytes (n*8) / that method's own time; inmem_shuffle_s is
//   blank past the RAM budget, so the plotted in-mem line stops at the cliff.
//
// Dual-purpose like the other examples: prints human-readable output AND the
// machine-readable "CSV," line benchmarks/run_benches.py greps.

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Flush dirty pages and let the drives settle between a write phase and a timed
// shuffle, so still-draining writeback doesn't queue behind (and inflate) the
// shuffle's own I/O.  Always outside a timed region — see
// external_samplesort.cpp.
static void quiesce_drives() {
  sync();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
}
static double to_gb(size_t bytes) {
  return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

// ── drive hygiene ───────────────────────────────────────────────────────────
// Two mechanisms, because the point of this benchmark is that neither method
// may leave anything behind:
//
//   remove_prefixes()  — the working sweep.  A generic name-prefix scan (not
//     GetFileName(prefix, d)) because both methods tag their intermediates with
//     a per-call counter and a bucket id (rs_bucket_<tag><d>,
//     rs_out_<tag>_<b><d>), which a fixed 0..num_drives enumeration would miss.
//     Used to isolate each timed method on drives that hold only the shared
//     input.
//
//   snapshot_drives() / sweep_new_files() — the backstop.  Everything on the
//     drives is listed before the run; at the end, ANY file that appeared since
//     and is still there is a leak by definition, whatever it is named.  It is
//     removed and reported, and the run exits non-zero — a silent leak would
//     otherwise accumulate across sweep points and quietly change what later
//     points measure.
static void remove_prefixes(const std::vector<std::string>& prefixes) {
  for (const std::string& dir : GetSSDList()) {
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
      const std::string name = e.path().filename().string();
      for (const std::string& p : prefixes) {
        if (name.rfind(p, 0) == 0) {  // name starts with p
          std::filesystem::remove(e.path(), ec);
          break;
        }
      }
    }
  }
}

static std::set<std::string> snapshot_drives() {
  std::set<std::string> files;
  for (const std::string& dir : GetSSDList()) {
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec))
      files.insert(e.path().string());
  }
  return files;
}

// Remove every file that appeared on the drives since `before`; returns them.
static std::vector<std::string> sweep_new_files(
    const std::set<std::string>& before) {
  std::vector<std::string> leaked;
  for (const std::string& path : snapshot_drives()) {
    if (before.count(path)) continue;
    std::error_code ec;
    std::filesystem::remove(path, ec);
    leaked.push_back(path);
  }
  return leaked;
}

// The input, and each method's on-disk file family.  random_shuffle_method's
// result *is* its rs_out_ files and Permutation's result *is* its perm files
// (rewritten in place over the count sort's own chunks), so those prefixes must
// be swept too — otherwise the entire output of every sweep point stays on the
// drives.
static const std::string kInPrefix = "rs_in";
static const std::vector<std::string> kMethodPrefixes = {
    "rs_bucket_", "rs_out_", "rs_base_"};  // random_shuffle_method(seq, "rs")
static const std::vector<std::string> kPermPrefixes = {
    "perm"};  // Permute(seq, "perm")

// Deterministic, duplicate-free key i, so a permutation check is exact.
static uint64_t key_at(size_t i) { return parlay::hash64(i); }

int run(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  // Both shuffles fan out one io_uring instance + one open file per drive for
  // every concurrent reader/writer, past the 1024 soft fd limit; lift it to the
  // hard limit before any I/O starts.
  RaiseFdLimit();

  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
  CHECK(n > 0) << "need n > 0 (n=" << n << ")";

  // RAM budget for the in-memory baseline + cross-check: the keys (8n), the
  // baseline's shuffled output (8n), the sorted reference (8n), and BOTH
  // out-of-core outputs read back (16n) — call it ~32n, as in
  // external_samplesort_vs_peter.  Past the budget the out-of-core shuffles
  // still run and are timed; only the baseline and the checks are skipped.
  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys;
  if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  const bool check_ok = n <= budget / 32;

  std::cout << std::fixed;
  std::cout << "Clearing stale shuffle files from the drives..." << std::flush;
  std::vector<std::string> all_prefixes = {kInPrefix};
  all_prefixes.insert(all_prefixes.end(), kMethodPrefixes.begin(),
                      kMethodPrefixes.end());
  all_prefixes.insert(all_prefixes.end(), kPermPrefixes.begin(),
                      kPermPrefixes.end());
  remove_prefixes(all_prefixes);
  std::cout << " done\n";

  // Everything on the drives right now is somebody else's; anything else that
  // survives to the end of this run is ours and is a leak.
  const std::set<std::string> pre_run = snapshot_drives();

  // ── the shared input ────────────────────────────────────────────────────
  std::cout << "Building " << n << "-key chunk_seq input..." << std::flush;
  auto t0 = Clock::now();
  chunk_seq seq = plaid::tabulate<uint64_t>(n, kInPrefix, key_at);
  const double build_s = elapsed(t0);
  std::cout << " done (" << std::setprecision(4) << build_s << "s)\n";
  quiesce_drives();  // isolate the op timers from the build's writeback

  // ── 1. random_shuffle_method (high-level abstractions) ──────────────────
  std::cout << "Shuffling " << n << " keys (random_shuffle_method)..."
            << std::flush;
  t0 = Clock::now();
  chunk_seq shuffled = random_shuffle_method<uint64_t>(seq, "rs");
  const double shuffle_s = elapsed(t0);
  const double shuffle_gb_s = to_gb(n * sizeof(uint64_t)) / shuffle_s;
  std::cout << " done   " << std::setprecision(4) << shuffle_s << "s   "
            << std::setprecision(2) << shuffle_gb_s << " GB/s (input read)\n";

  // Snapshot its output (under budget), then clear every file it left so the
  // next method is timed on drives holding only the shared input.
  parlay::sequence<uint64_t> ours;
  if (check_ok) ours = plaid::materialize<uint64_t>(shuffled);
  remove_prefixes(kMethodPrefixes);
  std::cout
      << "random_shuffle_method's files cleared before the next method runs\n";
  quiesce_drives();

  // ── 2. Permutation::Permute (low-level reader/writer, in-place buckets) ──
  std::cout << "Permuting " << n << " keys (plaid::Permutation)..."
            << std::flush;
  plaid::Permutation<uint64_t> permuter;
  t0 = Clock::now();
  chunk_seq permuted = permuter.Permute(seq, "perm");
  const double perm_s = elapsed(t0);
  const double perm_gb_s = to_gb(n * sizeof(uint64_t)) / perm_s;
  std::cout << " done   " << std::setprecision(4) << perm_s << "s   "
            << std::setprecision(2) << perm_gb_s << " GB/s (input read)\n";

  parlay::sequence<uint64_t> theirs;
  if (check_ok) theirs = plaid::materialize<uint64_t>(permuted);
  remove_prefixes(kPermPrefixes);
  quiesce_drives();

  std::cout << "speedup (random_shuffle_method / Permutation): "
            << std::setprecision(2) << (shuffle_s / perm_s) << "x\n";

  // ── 3. in-memory baseline + permutation cross-check ─────────────────────
  bool agree = true;
  double inmem_shuffle_s = 0;
  if (check_ok) {
    auto keys_mem = parlay::tabulate(n, key_at);
    t0 = Clock::now();
    auto shuffled_mem = parlay::random_shuffle(keys_mem);
    inmem_shuffle_s = elapsed(t0);
    std::cout << "in-mem parlay::random_shuffle   " << std::setprecision(4)
              << inmem_shuffle_s << "s\n";

    auto ref = keys_mem;  // the key set, sorted: the permutation target
    parlay::sort_inplace(ref);

    // Sorting a valid shuffle of the input must reproduce the key set exactly
    // (the keys are distinct), so this catches a dropped, duplicated, or
    // corrupted element in any of the three outputs.
    auto is_permutation = [&](const char* who, parlay::sequence<uint64_t> got) {
      if (got.size() != ref.size()) {
        std::cout << "*** MISMATCH (" << who << "): produced " << got.size()
                  << " keys, expected " << ref.size() << " ***\n";
        return false;
      }
      parlay::sort_inplace(got);
      for (size_t i = 0; i < ref.size(); i++) {
        if (got[i] != ref[i]) {
          std::cout << "*** MISMATCH (" << who << ") at sorted index " << i
                    << ": " << got[i] << " != " << ref[i] << " ***\n";
          return false;
        }
      }
      return true;
    };
    agree = is_permutation("random_shuffle_method", std::move(ours));
    agree &= is_permutation("Permutation", std::move(theirs));
    agree &= is_permutation("in-mem parlay", std::move(shuffled_mem));
    if (agree)
      std::cout
          << "cross-check: all three outputs are permutations of the input\n";
  } else {
    std::cout
        << "in-mem baseline + cross-check: skipped (~32n footprint exceeds "
        << "RAM budget " << std::setprecision(2) << to_gb(budget) << " GB)\n";
  }

  // Machine-readable line for benchmarks/run_benches.py (examples sweep).
  // Columns: n,build_s,shuffle_s,perm_s,inmem_shuffle_s,shuffle_gb_s,perm_gb_s
  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  std::cout << "CSV," << n << ',' << f9(build_s) << ',' << f9(shuffle_s) << ','
            << f9(perm_s) << ','
            << (check_ok ? f9(inmem_shuffle_s) : std::string()) << ','
            << f9(shuffle_gb_s) << ',' << f9(perm_gb_s) << '\n';

  // ── leave the drives exactly as we found them ───────────────────────────
  // Each method's files are already gone; this drops the input, and then the
  // snapshot diff catches anything either method wrote under a name this
  // driver's prefix list does not know about.
  remove_prefixes(all_prefixes);
  const std::vector<std::string> leaked = sweep_new_files(pre_run);
  if (!leaked.empty()) {
    std::cout << "*** LEAK: " << leaked.size()
              << " file(s) were left on the drives by this run (removed now); "
                 "the driver's prefix list is incomplete ***\n";
    for (size_t i = 0; i < leaked.size() && i < 20; i++)
      std::cout << "      " << leaked[i] << '\n';
    if (leaked.size() > 20)
      std::cout << "      ... and " << (leaked.size() - 20) << " more\n";
    agree = false;
  } else {
    std::cout << "drives clean: no files left behind by this run\n";
  }
  return agree ? 0 : 1;
}

}  // namespace demo_random_shuffle

namespace demo_reverse {

// reverseExample — in-place out-of-core reversal benchmark/demo.
//
// Builds an n-element uint64_t input (value i at index i), reverses it in
// place via plaid::reverse (Primitives/secondary_primitives.h — one
// O_DIRECT read + write per chunk, no extra output files), and cross-checks
// the readback against std::reverse on an identical in-memory copy.  Note
// plaid::reverse does not preserve the dense-except-last chunk invariant (see
// its doc comment) — the demo only reads the result back via to_vector, which
// is safe.
//
// Dual-purpose like the other examples: prints human-readable timings and
// ends with a machine-readable `CSV,` line that benchmarks/run_benches.py
// greps.  The in-memory baseline is gated by a RAM budget (input + baseline
// copy + readback = ~24n), overridable via EXAMPLE_INMEM_BUDGET_BYTES.
//
//   usage: reverseExample [global --flags] [n]

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

int run(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
  CHECK(n > 0) << "need n > 0 (n=" << n << ")";

  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys / 2;
  if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  const bool inmem_ok = n <= budget / 24;

  const std::string in_prefix = "rev_in";

  std::cout << "Building " << n << "-element input..." << std::flush;
  trace_mark("build_start");
  auto t0 = Clock::now();
  chunk_seq seq = plaid::tabulate<uint64_t>(
      n, in_prefix, [](size_t i) { return (uint64_t)i; });
  const double build_s = elapsed(t0);
  trace_mark("build_end");
  std::cout << " done (" << std::fixed << std::setprecision(4) << build_s
            << "s)\n";
  quiesce_drives();

  std::cout << "Reversing " << n << " elements in place..." << std::flush;
  trace_mark("op_start");
  t0 = Clock::now();
  plaid::reverse<uint64_t>(seq);
  const double rev_s = elapsed(t0);
  trace_mark("op_end");
  const double gb_s = to_gb(n * sizeof(uint64_t)) / rev_s;
  std::cout << " done   " << std::setprecision(4) << rev_s << "s   "
            << std::setprecision(2) << gb_s << " GB/s\n";

  bool agree = true;
  double inmem_rev_s = 0;
  if (inmem_ok) {
    auto mem = parlay::tabulate(n, [](size_t i) { return (uint64_t)i; });
    t0 = Clock::now();
    std::reverse(mem.begin(), mem.end());
    inmem_rev_s = elapsed(t0);
    std::cout << "in-mem std::reverse: " << std::setprecision(4)
              << inmem_rev_s << "s\n";

    std::vector<uint64_t> ours = seq.to_vector<uint64_t>();
    if (ours.size() != mem.size()) {
      std::cout << "*** MISMATCH: out-of-core produced " << ours.size()
                << " elements, expected " << mem.size() << " ***\n";
      agree = false;
    } else {
      for (size_t i = 0; i < ours.size() && agree; i++) {
        if (ours[i] != mem[i]) {
          std::cout << "*** MISMATCH at index " << i << ": " << ours[i]
                    << " != " << mem[i] << " ***\n";
          agree = false;
        }
      }
      if (agree)
        std::cout << "cross-check: out-of-core reverse matches in-mem "
                     "std::reverse exactly\n";
    }
  } else {
    std::cout << "in-mem std::reverse: skipped (~24n footprint exceeds RAM "
                 "budget "
              << std::setprecision(2) << to_gb(budget) << " GB)\n";
  }

  // Machine-readable line for benchmarks/run_benches.py (examples sweep).
  // Columns: n,build_s,reverse_s,inmem_reverse_s,throughput_gb_s
  // (inmem_reverse_s blank when the input exceeds the RAM budget).
  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  std::cout << "CSV," << n << ',' << f9(build_s) << ',' << f9(rev_s) << ','
            << (inmem_ok ? f9(inmem_rev_s) : std::string()) << ',' << f9(gb_s)
            << '\n';

  cleanup_prefix(in_prefix);
  return agree ? 0 : 1;
}

}  // namespace demo_reverse

// ============================================================================
// dispatch
// ============================================================================

int main(int argc, char* argv[]) {
  static const std::map<std::string, int (*)(int, char**)> kDemos = {
      {"map", &demo_map::run},
      {"reduce", &demo_reduce::run},
      {"scan", &demo_scan::run},
      {"tabulate", &demo_tabulate::run},
      {"zip", &demo_zip::run},
      {"filter", &demo_filter::run},
      {"pack", &demo_pack::run},
      {"group_by_index", &demo_group_by_index::run},
      {"histogram_by_index", &demo_histogram_by_index::run},
      {"cut", &demo_cut::run},
      {"random_shuffle", &demo_random_shuffle::run},
      {"reverse", &demo_reverse::run},
  };

  if (argc < 2 || kDemos.find(argv[1]) == kDemos.end()) {
    std::cerr << "usage: " << argv[0] << " <primitive> [flags] [n ...]\n"
              << "primitives:";
    for (const auto& kv : kDemos) std::cerr << ' ' << kv.first;
    std::cerr << '\n';
    return 2;
  }

  // Shift argv so each demo's run() sees exactly the argv its own main()
  // used to see: argv[0] = program name, argv[1..] = its own arguments.
  auto* fn = kDemos.at(argv[1]);
  argv[1] = argv[0];
  return fn(argc - 1, argv + 1);
}
