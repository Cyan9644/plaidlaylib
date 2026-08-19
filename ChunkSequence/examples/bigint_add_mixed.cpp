// bigint_add_mixedExample — out-of-core big-integer addition across operands
// stored at DIFFERENT element widths.
//
// Same base-2^64 addition as bigint_addExample, but the second operand's limbs
// are stored as uint32_t (legitimate whenever every limb of b is known to fit
// in 32 bits — a compressed operand, half the bytes on disk).  See
// chunk_bigint_add.h's ChunkBigIntAddNarrow.
//
// The point for the library: a uint32_t chunk_seq holds CHUNK_SIZE/4 elements
// per chunk against a uint64_t one's CHUNK_SIZE/8, so the two operands do NOT
// share a chunk partition.  The delayed layer's zip re-grids onto the union of
// the two partitions and addresses each side by its own chunk index plus an
// element offset — so the identical fused chain
//   zip -> map(classify) -> scan(carry) -> zip -> map(add) -> force
// runs over mismatched grids with no repacking pass, and every physical chunk
// is still read exactly once (plan_chunks dedups the reads a straddling chunk
// generates and refcounts its buffer).
//
// Against bigint_addExample at the same limb count this reads 12 B/limb rather
// than 16 B/limb, so the interesting comparison is time at equal *limb count*
// (this should win on I/O) as well as at equal input bytes.
//
// Dual-purpose like the other examples: run by hand it prints human-readable
// timings; it always ends with a machine-readable `CSV,` line that
// benchmarks/run_benches.py greps.  The in-memory baseline is our own parlaylib
// reference (bigint_reference::add) on the widened operands, and it doubles as
// a differential test: the out-of-core sum is read back and compared
// limb-for-limb, exiting non-zero on any mismatch.  The baseline is gated by a
// RAM budget (half of physical RAM, overridable via
// EXAMPLE_INMEM_BUDGET_BYTES); past it the run is skipped and the CSV field
// left blank, so the plotted in-mem line stops at the RAM cliff.
//
// Usage: bigint_add_mixedExample [global --flags] [n]   (n = limb count; a is
// stored as n uint64_t limbs, b as n uint32_t limbs)

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
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
// order and compare every limb.  Sizes by each chunk's own `used`, so it is
// correct even when the result is ragged (which a re-gridding zip's force
// output can be).  Only called when the baseline ran, so `expected` fits in
// RAM by construction.
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

// Deterministic limbs, computable anywhere.  `a` is full-width (random sign
// bits, so the sign-extension / overflow path is exercised); `b` is narrow by
// construction, which is what makes the uint32_t storage valid.
static digit limb_a(size_t i) { return (digit)parlay::hash64(i); }
static uint32_t limb_b32(size_t i) {
  return (uint32_t)parlay::hash64(i ^ 0x9e3779b97f4a7c15ULL);
}

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;

  // RAM budget for the in-memory baseline.  Footprint is ≈ both operands
  // materialized as full-width digits + the result + transient fused state ≈
  // 4 * n * sizeof(digit) — the same as bigint_add, since the baseline widens
  // b to compare against the same reference adder.
  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys / 2;
  if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  const bool inmem_ok = n <= budget / (4 * sizeof(digit));

  const std::string a_prefix = "bim_a";
  const std::string b_prefix = "bim_b32";
  const std::string sum_prefix = "bim_sum";

  std::cout << "Building a: " << n << " x uint64_t limbs, b: " << n
            << " x uint32_t limbs..." << std::flush;
  trace_mark("build_start");
  auto t0 = Clock::now();
  chunk_seq a = plaid::tabulate<digit>(n, a_prefix, limb_a);
  chunk_seq b32 = plaid::tabulate<uint32_t>(n, b_prefix, limb_b32);
  const double build_s = elapsed(t0);
  trace_mark("build_end");
  std::cout << " done (" << std::fixed << std::setprecision(4) << build_s
            << "s)\n";
  std::cout << "  a: " << a.chunks.size() << " chunks of "
            << CHUNK_SIZE / sizeof(digit) << " limbs;  b: " << b32.chunks.size()
            << " chunks of " << CHUNK_SIZE / sizeof(uint32_t)
            << " limbs  (partitions disagree -> zip re-grids)\n";

  std::cout << "Adding..." << std::flush;
  trace_mark("op_start");
  t0 = Clock::now();
  chunk_seq sum = plaid::ChunkBigIntAddNarrow(a, b32, sum_prefix);
  const double add_s = elapsed(t0);
  trace_mark("op_end");
  std::cout << " done\n";

  size_t result_limbs = 0;
  for (const auto& c : sum.chunks) result_limbs += c.used / sizeof(digit);
  // Bytes moved ≈ both operands read once: 8 B/limb for a, 4 B/limb for b.
  const double gb_s = to_gb(n * sizeof(digit) + n * sizeof(uint32_t)) / add_s;

  std::cout << result_limbs << " result limb(s)   " << std::setprecision(4)
            << add_s << "s   " << std::setprecision(2) << gb_s
            << " GB/s (operands read)\n";

  // In-memory baseline: our parlaylib reference on the same operands, with b
  // widened to full digits (built in DRAM outside the timed region), then
  // cross-checked limb-for-limb.
  bool agree = true;
  double inmem_add_s = 0;
  if (inmem_ok) {
    auto a_mem = parlay::tabulate(n, limb_a);  // parlay::sequence<digit>
    auto b_mem =
        parlay::tabulate(n, [](size_t i) { return (digit)limb_b32(i); });
    t0 = Clock::now();
    auto sum_mem = plaid::bigint_reference::add(a_mem, b_mem);
    inmem_add_s = elapsed(t0);
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
  // Columns: n,build_s,add_s,inmem_add_s,result_limbs,throughput_gb_s
  // (inmem_add_s blank when the operands exceed the RAM budget).
  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  std::cout << "CSV," << n << ',' << f9(build_s) << ',' << f9(add_s) << ','
            << (inmem_ok ? f9(inmem_add_s) : std::string()) << ','
            << result_limbs << ',' << f9(gb_s) << '\n';

  // Don't leave operands/output on the drives across sweep points.
  cleanup_prefix(a_prefix);
  cleanup_prefix(b_prefix);
  cleanup_prefix(sum_prefix);
  return agree ? 0 : 1;
}
