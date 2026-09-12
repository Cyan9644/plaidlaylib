#include <fcntl.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/examples/chunk_linefit.h"
#include "ChunkSequence/examples/chunk_linefit_eager.h"
#include "ChunkSequence/examples/in_memory_baselines.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "utils/file_utils.h"

// Correctness test for plaid::linefit (chunk_linefit.h, a fully-delayed
// least-squares fit -- zip x/y and reduce, never materializing the zipped
// points), previously untested anywhere in ChunkSequence/tests/.  Cross-checks
// against the independent DRAM reference `::linefit` (in_memory_baselines.h,
// a hand-ported Numerical-Recipes-style fit, not derived from the out-of-core
// code under test) within a relative tolerance -- the out-of-core fit sums
// x/y in chunk-grouped order while the reference sums linearly, so the two
// are numerically close but not bit-identical (same tolerance/rationale as
// examples/linefit.cpp's own driver).
//
// Also cross-checks plaid::linefit_eager (chunk_linefit_eager.h) -- the same
// centered-sums algorithm built on NRemoveWorker's explicit lockstep fold
// instead of the delayed engine -- against the same DRAM reference, at the
// same tolerance.

static constexpr double OFFSET = 3.0;
static constexpr double SLOPE = 1.0;

// Deterministic, per-index point coordinates (same shape as linefit.cpp's
// x_at/y_at): computable from the index alone, so the out-of-core x/y
// sequences and the in-memory reference points hold identical values.
static double x_at(size_t i) {
  parlay::random_generator gen;
  auto r = gen[i];
  std::uniform_real_distribution<double> dis(0.0, 1.0);
  return dis(r);
}
static double y_at(size_t i) { return OFFSET + SLOPE * x_at(i); }

static bool close(double a, double b, double rel = 1e-6) {
  return std::fabs(a - b) <=
         rel * std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
}

static void cleanup_prefix(const std::string& prefix) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    unlink(GetFileName(prefix, d).c_str());
}

static bool check_linefit(const std::string& label, size_t n) {
  const std::string x_prefix = "lf_test_x";
  const std::string y_prefix = "lf_test_y";

  chunk_seq x = plaid::tabulate<double>(n, x_prefix, x_at);
  chunk_seq y = plaid::tabulate<double>(n, y_prefix, y_at);
  auto [offset, slope] = plaid::linefit(x, y);
  auto [eager_offset, eager_slope] = plaid::linefit_eager(x, y);
  cleanup_prefix(x_prefix);
  cleanup_prefix(y_prefix);

  auto points_ref = parlay::tabulate(
      n, [](size_t i) { return plaid::point(x_at(i), y_at(i)); });
  auto [offset_ref, slope_ref] = linefit(points_ref);  // global DRAM reference

  bool ok = close(offset, offset_ref) && close(slope, slope_ref) &&
            close(eager_offset, offset_ref) && close(eager_slope, slope_ref);
  std::cout << "  " << (ok ? "OK" : "FAIL") << " " << label << ": n=" << n
            << "  delayed=(" << offset << ", " << slope << ")"
            << "  eager=(" << eager_offset << ", " << eager_slope << ")"
            << "  reference=(" << offset_ref << ", " << slope_ref << ")\n";
  return ok;
}

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  const size_t n_arg = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;

  bool all_pass = true;

  // Small sizes, including the minimum meaningful n (a line needs 2 points).
  for (size_t n : {(size_t)2, (size_t)3, (size_t)100, (size_t)10000})
    all_pass &= check_linefit("small", n);

  // Chunk-boundary-straddling sizes (ELEMS_PER_CHUNK doubles as the delayed
  // layer's per-chunk grid; these exercise the multi-chunk zip+reduce path).
  for (size_t n : {ELEMS_PER_CHUNK - 1, ELEMS_PER_CHUNK, ELEMS_PER_CHUNK + 1,
                   2 * ELEMS_PER_CHUNK + 12345})
    all_pass &= check_linefit("chunk_boundary", n);

  // argv-provided size, matching the example's own default scale.
  all_pass &= check_linefit("large", n_arg);

  std::cout << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
  return all_pass ? 0 : 1;
}
