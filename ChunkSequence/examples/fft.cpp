// Example: out-of-core 1-D FFT of a power-of-two complex<double> sequence, done
// transpose-free as two passes with contrasting I/O shapes -- a streaming
// length-A pass over contiguous blocks (ExternalTransform) and a random
// length-B pass over strided columns (small io_uring 4 KiB gather/scatter). See
// examples/chunk_fft.h for the algorithm and the transpose-free four-step math.
//
// The point is to explore SSD random-read performance: the external-memory FFT
// literature treats the matrix transpose as the dominant cost (an HDD premise),
// but across many SSDs the strided column reads are cheap enough to skip the
// transpose entirely.  The two stages are separately timed and separately
// marked for benchmarks/io_trace.py so the streaming vs random I/O contrast is
// visible.
//
// Dual-purpose like the other examples: prints human-readable output AND a
// machine-readable "CSV," line that benchmarks/run_benches.py greps.  When it
// fits RAM it times an in-memory baseline (the inmem_s the `make
// bench-examples` comparison plot needs) -- and that baseline is the SAME
// transpose-free four-step FFT run in DRAM (same fft_inplace kernel and
// parallel structure as the out-of-core stages), so the comparison isolates the
// I/O cost rather than pitting our algorithm against a different FFT
// implementation.  Unlike the frozen pre-cleanup driver (which gated it behind
// FFT_VERIFY=1), the full-spectrum correctness cross-check against an
// independent oracle (parlaylib complex_fft) -- reading the entire result back
// and comparing every bin -- runs by default whenever the baseline itself runs,
// matching every other example's always-on cross-check convention; it exits
// non-zero on a mismatch.  Both the baseline and the compare are skipped under
// PLAID_TRACE (they must not inflate a trace) and are gated by a RAM budget of
// half physical RAM; override via EXAMPLE_INMEM_BUDGET_BYTES (0 disables both).
//
//   usage: fftExample [global --flags] [n]
//     n   sequence length (default 1<<20); rounded down to a power of two, >=
//     2^16.
//
// CSV line:
//   CSV,<N>,<build_s>,<stage1_s>,<stage2_s>,<total_s>,<inmem_s>,<count>,<throughput_gb_s>
//   total_s = stage1_s + stage2_s (the sweep plots this against inmem_s);
//   throughput = bytes moved (stage1 r+w + stage2 r+w = 4*N*16) / total_s.

#include <unistd.h>

#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "parlay/primitives.h"

// Upstream parlaylib example (fetched by `make deps`): the in-memory baseline
// complex_fft.  Global symbols, no include guard -- the only upstream header in
// this TU (chunk_fft.h intentionally uses its own kernel to avoid coupling).
#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/examples/chunk_fft.h"
#include "parlaylib-examples/fast_fourier_transform.h"
#include "utils/file_utils.h"
#include "utils/trace_marker.h"

using ChunkFFT::cd;
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

// Deterministic logical input value at logical index m: pseudo-random complex
// in
// [-0.5,0.5]^2, computable anywhere.
static cd input_val(size_t m) {
  const double inv = 1.0 / 18446744073709551616.0;  // 1 / 2^64
  const double re = (double)parlay::hash64(2 * m) * inv - 0.5;
  const double im = (double)parlay::hash64(2 * m + 1) * inv - 0.5;
  return cd{re, im};
}

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  const size_t req_n = (argc > 1) ? std::stoull(argv[1]) : (size_t(1) << 20);
  const ChunkFFT::Dims d = ChunkFFT::choose_dims(req_n);
  const size_t N = d.N, A = d.A, B = d.B;
  CHECK(N >= (size_t(1) << 16)) << "n must be at least 2^16";

  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys / 2;
  if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  const bool inmem_ok = N * sizeof(cd) <= budget;

  // The in-mem parlaylib complex_fft plays two roles once do_baseline is true:
  //   * baseline TIMING (inmem_s) -- what the `make bench-examples` comparison
  //     plot needs; run by default when it fits RAM, like the other examples.
  //   * full-spectrum correctness COMPARE (read the whole result back and scan
  //     all N bins) -- heavy (its output is as big as its input), but run by
  //     default (not opt-in) to match every other example's built-in
  //     cross-check convention.
  // Both are skipped under PLAID_TRACE (must not inflate a trace) and both are
  // RAM-budget gated; set EXAMPLE_INMEM_BUDGET_BYTES=0 to force both off.
  const bool do_baseline = inmem_ok && !trace_enabled();
  const bool do_verify = do_baseline;

  const std::string in_prefix = "fft_in";
  const std::string s1_prefix = "fft_s1";

  std::cout << "FFT N=" << N << " = A(" << A << ") x B(" << B << ")\n";

  // Build the input in the column-major placement stage 1 expects: physical
  // position p = b*A + a holds logical element m = a*B + b.
  std::cout << "Building input..." << std::flush;
  trace_mark("build_start");
  auto t0 = Clock::now();
  chunk_seq input =
      plaid::tabulate<cd>(N, in_prefix, [A, B](size_t p) {
        const size_t a = p % A;
        const size_t b = p / A;
        return input_val(a * B + b);
      });
  const double build_s = elapsed(t0);
  trace_mark("build_end");
  std::cout << " done (" << std::fixed << std::setprecision(4) << build_s
            << "s)\n";

  // Stage 1: streaming length-A block FFTs.
  std::cout << "Stage 1 (streaming rows)..." << std::flush;
  trace_mark("stage1_start");
  t0 = Clock::now();
  chunk_seq s1 = ChunkFFT::stage1_rows(input, d, s1_prefix);
  const double stage1_s = elapsed(t0);
  trace_mark("stage1_end");
  std::cout << " done (" << std::setprecision(4) << stage1_s << "s)\n";

  // Stage 2: random strided column FFTs (in place on s1).
  std::cout << "Stage 2 (random columns)..." << std::flush;
  trace_mark("stage2_start");
  t0 = Clock::now();
  ChunkFFT::stage2_cols(s1, d, s1_prefix);
  const double stage2_s = elapsed(t0);
  trace_mark("stage2_end");
  std::cout << " done (" << std::setprecision(4) << stage2_s << "s)\n";

  const double total_s = stage1_s + stage2_s;
  const size_t bytes_moved = 4 * N * sizeof(cd);  // s1 r+w + s2 r+w
  const double gb_s = to_gb(bytes_moved) / total_s;
  std::cout << "stage1 " << std::setprecision(4) << stage1_s << "s   stage2 "
            << stage2_s << "s   " << std::setprecision(2) << gb_s
            << " GB/s (bytes moved)\n";

  // In-memory baseline: parlaylib complex_fft on the same logical input,
  // cross-checked over the full spectrum with the transpose-free permutation.
  bool agree = true;
  double inmem_s = 0;
  if (do_baseline) {
    // Comparable in-mem baseline: the SAME transpose-free four-step (same
    // fft_inplace kernel + parallel structure as the out-of-core stages), on a
    // DRAM array with no I/O.  The column-major placement is the untimed
    // "build" equivalent; only the two transform passes are timed, matching the
    // out-of-core total_s = stage1_s + stage2_s, so inmem_s isolates I/O cost.
    parlay::sequence<cd> x =
        parlay::tabulate(N, [](size_t m) { return input_val(m); });
    parlay::sequence<cd> Xmem = ChunkFFT::in_mem_place(x, d);
    t0 = Clock::now();
    ChunkFFT::in_mem_transform(Xmem, d);
    inmem_s = elapsed(t0);
    std::cout << "in-mem four-step FFT: " << std::setprecision(4) << inmem_s
              << "s";

    if (do_verify) {
      // Differential correctness against an INDEPENDENT oracle (parlaylib
      // complex_fft): check both the out-of-core result and the in-mem
      // baseline over the full spectrum (with the transpose-free permutation).
      auto Xref = complex_fft(x);
      std::vector<cd> out = s1.to_vector<cd>();
      auto e = ChunkFFT::spectrum_errs(out, Xmem, Xref, d, ChunkFFT::out_perm);
      const double tol = 1e-6 * (e.max_ref > 0 ? e.max_ref : 1.0);
      agree = (e.err_oc <= tol) && (e.err_mem <= tol);
      std::cout << "   out-of-core err " << std::scientific << e.err_oc
                << "   in-mem err " << e.err_mem << "   tol " << tol
                << std::fixed << (agree ? "   OK" : "   *** MISMATCH ***");
    }
    std::cout << "\n";
  } else if (trace_enabled()) {
    std::cout << "in-mem complex_fft: skipped (tracing)\n";
  } else {
    std::cout << "in-mem complex_fft: skipped (spectrum exceeds RAM budget "
              << std::setprecision(2) << to_gb(budget) << " GB)\n";
  }

  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  std::cout << "CSV," << N << ',' << f9(build_s) << ',' << f9(stage1_s) << ','
            << f9(stage2_s) << ',' << f9(total_s) << ','
            << (do_baseline ? f9(inmem_s) : std::string()) << ',' << N << ','
            << f9(gb_s) << '\n';

  cleanup_prefix(in_prefix);
  cleanup_prefix(s1_prefix);
  return agree ? 0 : 1;
}
