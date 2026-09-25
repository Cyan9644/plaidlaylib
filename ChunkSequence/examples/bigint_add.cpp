// bigint_addExample — out-of-core big-integer addition benchmark/demo.
//
// Adds two n-limb (base-2^64, two's-complement) big integers stored across the
// SSDs using the delayed primitives (see chunk_bigint_add.h): the fused chain
// zip -> map(classify) -> scan(carry) -> zip -> map(add) -> force moves the
// operands past DRAM without spilling any intermediate to disk.
//
// Dual-purpose like the other examples: run by hand it prints human-readable
// timings; it always ends with a machine-readable `CSV,` line that
// benchmarks/run_benches.py greps.  The in-memory baseline is our own parlaylib
// reference (bigint_reference::add) rather than an upstream parlaylib example —
// there is no upstream big-integer example — and it doubles as a differential
// test: the out-of-core sum is read back and compared limb-for-limb, exiting
// non-zero on any mismatch.  The baseline is gated by a RAM budget (half of
// physical RAM, overridable via EXAMPLE_INMEM_BUDGET_BYTES); past it the run is
// skipped and the CSV field left blank, so the plotted in-mem line stops at the
// RAM cliff.
//
// With BIGINT_ADD_EAGER=1 it also times plaid::ChunkBigIntAddEager -- the same
// algorithm, but with the classify and carry-scan stages materialized to disk
// between primitives instead of fused into one pass (see chunk_bigint_add.h)
// -- and cross-checks it against the fused result bit-exactly via a streaming
// disk-to-disk compare (chunk_contents_equal) that needs no DRAM budget, so it
// runs at every n.  Off by default so traces show only the fused pass.
//
// Usage: bigint_addExample [global --flags] [n]      (n = number of 64-bit
// limbs)

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/examples/chunk_bigint_add.h"
#include "absl/log/check.h"
#include "parlay/primitives.h"
#include "utils/file_utils.h"
#include "utils/trace_marker.h"

using digit = plaid::bigint_detail::digit;

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

// Element-wise check of an out-of-core uint64_t result against the in-mem
// baseline's sequence: read each output chunk back off the drives in index
// order and compare every limb.  Only called when the baseline ran, so
// `expected` fits in RAM by construction.
template <typename Seq>
static bool contents_equal(const chunk_seq& cs, const Seq& expected) {
  void* buf = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
  CHECK(buf != nullptr);
  bool ok = true;
  size_t j = 0;
  for (const chunk& c : cs.chunks) {
    if (!ok || c.used == 0) continue;
    int fd = open(c.filename.c_str(), O_DIRECT | O_RDONLY);
    SYSCALL(fd);
    SYSCALL(pread(fd, buf, AlignUp(c.used), (off_t)c.begin_addr));
    close(fd);
    const uint64_t* elems = reinterpret_cast<const uint64_t*>(buf);
    const size_t cnt = c.used / sizeof(uint64_t);
    for (size_t i = 0; i < cnt && ok; i++, j++)
      ok = j < expected.size() && elems[i] == (uint64_t)expected[j];
  }
  free(buf);
  return ok && j == expected.size();
}

// Bit-exact, chunk-by-chunk compare of two on-disk chunk_seqs' contents, with
// only two chunk-sized buffers in DRAM regardless of n.  Used to cross-check
// the delayed and eager adds against each other: unlike contents_equal (which
// needs an in-DRAM `expected` sequence and only runs under the RAM budget),
// this must scale past it, since both operands here are already out-of-core.
static bool chunk_contents_equal(const chunk_seq& x, const chunk_seq& y) {
  if (x.chunks.size() != y.chunks.size()) return false;
  void* bx = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
  void* by = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
  CHECK(bx != nullptr && by != nullptr);
  bool ok = true;
  for (size_t i = 0; ok && i < x.chunks.size(); i++) {
    const chunk& cx = x.chunks[i];
    const chunk& cy = y.chunks[i];
    if (cx.used != cy.used) {
      ok = false;
      break;
    }
    if (cx.used == 0) continue;
    int fdx = open(cx.filename.c_str(), O_DIRECT | O_RDONLY);
    SYSCALL(fdx);
    SYSCALL(pread(fdx, bx, AlignUp(cx.used), (off_t)cx.begin_addr));
    close(fdx);
    int fdy = open(cy.filename.c_str(), O_DIRECT | O_RDONLY);
    SYSCALL(fdy);
    SYSCALL(pread(fdy, by, AlignUp(cy.used), (off_t)cy.begin_addr));
    close(fdy);
    ok = memcmp(bx, by, cx.used) == 0;
  }
  free(bx);
  free(by);
  return ok;
}

// Deterministic, full-width limbs (random sign bits, so the sign-extension /
// overflow path is exercised), computable anywhere.  Distinct seeds per
// operand.
static digit limb_a(size_t i) { return (digit)parlay::hash64(i); }
static digit limb_b(size_t i) {
  return (digit)parlay::hash64(i ^ 0x9e3779b97f4a7c15ULL);
}

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;

  // RAM budget for the in-memory baseline.  Footprint is ≈ two materialized
  // operands + the result + transient fused state ≈ 4 * n * sizeof(digit).
  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys / 2;
  if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  const bool inmem_ok = n <= budget / (4 * sizeof(digit));

  const std::string a_prefix = "bi_a";
  const std::string b_prefix = "bi_b";
  const std::string sum_prefix = "bi_sum";

  std::cout << "Building two " << n << "-limb operands..." << std::flush;
  trace_mark("build_start");
  auto t0 = Clock::now();
  chunk_seq a = plaid::tabulate<digit>(n, a_prefix, limb_a);
  chunk_seq b = plaid::tabulate<digit>(n, b_prefix, limb_b);
  const double build_s = elapsed(t0);
  trace_mark("build_end");
  std::cout << " done (" << std::fixed << std::setprecision(4) << build_s
            << "s)\n";

  std::cout << "Adding..." << std::flush;
  trace_mark("op_start");
  t0 = Clock::now();
  chunk_seq sum = plaid::ChunkBigIntAdd(a, b, sum_prefix);
  const double add_s = elapsed(t0);
  trace_mark("op_end");
  std::cout << " done\n";

  size_t result_limbs = 0;
  for (const auto& c : sum.chunks) result_limbs += c.used / sizeof(digit);
  // Bytes moved ≈ both operands read once (the fused pass; scan re-reads them,
  // but we report the logical operand size like the other examples).
  const double gb_s = to_gb(2 * n * sizeof(digit)) / add_s;

  std::cout << result_limbs << " result limb(s)   " << std::setprecision(4)
            << add_s << "s   " << std::setprecision(2) << gb_s
            << " GB/s (operands read)\n";

  bool agree = true;

  // Eager out-of-core baseline (opt-in: BIGINT_ADD_EAGER=1): same algorithm,
  // but the classify and carry-scan stages are materialized to disk between
  // primitives instead of fused into one pass (chunk_bigint_add.h's
  // ChunkBigIntAddEager).  Off by default so a plain run (and an io_trace.py
  // trace) shows only the fused pass; the delayed-vs-eager ablation entry in
  // run_benches.py turns it on.  Cross-checked against the fused result
  // bit-exactly via a streaming disk-to-disk compare, since at this
  // benchmark's scale (up to 2^36 limbs) neither result fits in DRAM.
  const char* eager_env = getenv("BIGINT_ADD_EAGER");
  const bool run_eager = eager_env && std::string(eager_env) == "1";
  double eager_add_s = 0, eager_gb_s = 0;
  size_t eager_result_limbs = 0;
  if (run_eager) {
    std::cout << "Adding (eager, materialized intermediates)..." << std::flush;
    trace_mark("op_start_eager");
    t0 = Clock::now();
    chunk_seq eager_sum = plaid::ChunkBigIntAddEager(a, b, "bi_sum_eager");
    eager_add_s = elapsed(t0);
    trace_mark("op_end_eager");
    std::cout << " done\n";

    for (const auto& c : eager_sum.chunks)
      eager_result_limbs += c.used / sizeof(digit);
    eager_gb_s = to_gb(2 * n * sizeof(digit)) / eager_add_s;

    std::cout << "eager: " << eager_result_limbs << " result limb(s)   "
              << std::setprecision(4) << eager_add_s << "s   "
              << std::setprecision(2) << eager_gb_s
              << " GB/s (operands read)\n";

    if (eager_result_limbs != result_limbs ||
        !chunk_contents_equal(sum, eager_sum)) {
      std::cout << "*** MISMATCH: eager sum differs from delayed (fused) "
                   "sum ***\n";
      agree = false;
    } else {
      std::cout << "cross-check: eager sum matches delayed (fused) sum "
                   "(bit-exact)\n";
    }
    cleanup_prefix("bi_sum_eager");
  }

  // In-memory baseline: our parlaylib reference on the same operands (built in
  // DRAM outside the timed region), cross-checked limb-for-limb.
  double inmem_add_s = 0;
  if (inmem_ok) {
    auto a_mem = parlay::tabulate(n, limb_a);  // parlay::sequence<digit>
    auto b_mem = parlay::tabulate(n, limb_b);
    trace_mark("op_start_inmem");
    t0 = Clock::now();
    auto sum_mem = plaid::bigint_reference::add(a_mem, b_mem);
    inmem_add_s = elapsed(t0);
    trace_mark("op_end_inmem");
    std::cout << "in-mem parlaylib add: " << sum_mem.size() << " limb(s)   "
              << std::setprecision(4) << inmem_add_s << "s\n";
    if (sum_mem.size() != result_limbs) {
      std::cout << "*** MISMATCH: in-mem " << sum_mem.size()
                << " limbs != out-of-core " << result_limbs << " ***\n";
      agree = false;
    } else if (!contents_equal(sum, sum_mem)) {
      std::cout << "*** MISMATCH: in-mem sum differs from out-of-core "
                << "output ***\n";
      agree = false;
    }
  } else {
    std::cout << "in-mem parlaylib add: skipped (operands exceed RAM budget "
              << std::setprecision(2) << to_gb(budget) << " GB)\n";
  }

  // Machine-readable line for benchmarks/run_benches.py (examples sweep).
  // Columns: n,build_s,add_s,inmem_add_s,result_limbs,throughput_gb_s,
  //          eager_add_s,eager_result_limbs,eager_throughput_gb_s
  // (inmem_add_s blank when the operands exceed the RAM budget; the eager_*
  // columns blank unless BIGINT_ADD_EAGER=1).
  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  std::cout << "CSV," << n << ',' << f9(build_s) << ',' << f9(add_s) << ','
            << (inmem_ok ? f9(inmem_add_s) : std::string()) << ','
            << result_limbs << ',' << f9(gb_s) << ','
            << (run_eager ? f9(eager_add_s) : std::string()) << ','
            << (run_eager ? std::to_string(eager_result_limbs) : std::string())
            << ',' << (run_eager ? f9(eager_gb_s) : std::string()) << '\n';

  // Don't leave operands/output on the drives across sweep points.
  cleanup_prefix(a_prefix);
  cleanup_prefix(b_prefix);
  cleanup_prefix(sum_prefix);
  return agree ? 0 : 1;
}
