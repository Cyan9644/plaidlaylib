// Example: out-of-core 1-D FFT of a power-of-two complex<double> sequence, done
// the CLASSIC external-memory way -- with an explicit on-disk matrix TRANSPOSE
// between the two passes, so both length-A and length-B FFT passes are
// contiguous streaming.  This is the counterpart to fftExample (which skips the
// transpose and does the length-B pass by random 4 KiB gather/scatter): the two
// binaries let you measure the classic external-memory tradeoff on SSDs
// directly --
//
//   fftExample (transpose-free): stage1 (2N) + stage2 random (2N) = 4N, half
//   random fft_transposeExample (this): stage1 (2N) + transpose (2N) + stage2T
//   (2N) = 6N,
//                                all streaming.
//
// On HDDs the sequential 6N wins hugely; on SSDs the 4N-with-random path may
// win. The three stages are separately timed and marked for
// benchmarks/io_trace.py.  See examples/chunk_fft.h for the algorithm, the
// transpose, and the four-step math.
//
// Dual-purpose like the other examples: prints human-readable output AND a
// "CSV," line run_benches.py greps.  When it fits RAM it times the SAME
// in-memory four-step baseline as fftExample (inmem_s, for the make
// bench-examples plot); FFT_VERIFY=1 additionally cross-checks the full
// spectrum against the independent parlaylib complex_fft oracle (using the
// transpose variant's output permutation) and exits non-zero on a mismatch.
// Both are skipped under PLAID_TRACE and gated by a RAM budget of half physical
// RAM (EXAMPLE_INMEM_BUDGET_BYTES, 0 disables).
//
//   usage: fft_transposeExample [global --flags] [n]   (FFT_VERIFY=1 to
//   cross-check)
//     n   sequence length (default 1<<20); power of two, 2^16 <= N <= 2^36 (B
//     <= EPC).
//
// CSV line:
//   CSV,<N>,<build_s>,<stage1_s>,<transpose_s>,<stage2t_s>,<total_s>,<inmem_s>,<count>,<gb_s>
//   total_s = stage1_s + transpose_s + stage2t_s; throughput = 6*N*16 /
//   total_s.

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

// Upstream parlaylib example (fetched by `make deps`): the independent
// complex_fft oracle used only under FFT_VERIFY.  Global symbols, no include
// guard -- the only upstream header in this TU.
#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/examples/external/chunk_fft.h"
#include "parlaylib-examples/fast_fourier_transform.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "utils/io_backend.h"
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
    plaid::io::Unlink(GetFileName(prefix, d).c_str());
}

// Deterministic logical input value at logical index m (identical to
// fftExample).
static cd input_val(size_t m) {
  const double inv = 1.0 / 18446744073709551616.0;  // 1 / 2^64
  const double re = (double)parlay::hash64(2 * m) * inv - 0.5;
  const double im = (double)parlay::hash64(2 * m + 1) * inv - 0.5;
  return cd{re, im};
}

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  RaiseFdLimit();
  const size_t req_n = (argc > 1) ? std::stoull(argv[1]) : (size_t(1) << 20);
  const ChunkFFT::Dims d = ChunkFFT::choose_dims(req_n);
  const size_t N = d.N, A = d.A, B = d.B;
  CHECK(N >= (size_t(1) << 16)) << "n must be at least 2^16";
  CHECK(B <= ChunkFFT::EPC)
      << "transpose variant requires B <= EPC (N <= 2^36); "
         "use fftExample for larger N";

  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys / 2;
  if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  const bool inmem_ok = N * sizeof(cd) <= budget;
  const bool do_baseline = inmem_ok && !trace_enabled();
  const bool do_verify = do_baseline && (getenv("FFT_VERIFY") != nullptr);

  const std::string in_prefix = "fft_in";
  const std::string s1_prefix = "fft_s1";
  const std::string t_prefix = "fft_t";
  const std::string t2_prefix = "fft_t2";

  std::cout << "FFT (transpose) N=" << N << " = A(" << A << ") x B(" << B
            << ")\n";

  // Build input in the column-major placement stage 1 expects.
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

  // Stage 1: streaming length-A block FFTs (shared with fftExample).
  std::cout << "Stage 1 (streaming rows)..." << std::flush;
  trace_mark("stage1_start");
  t0 = Clock::now();
  chunk_seq s1 = ChunkFFT::stage1_rows(input, d, s1_prefix);
  const double stage1_s = elapsed(t0);
  trace_mark("stage1_end");
  std::cout << " done (" << std::setprecision(4) << stage1_s << "s)\n";

  // Transpose: materialize the A x B transpose on disk (streaming band
  // transpose).
  std::cout << "Transpose (streaming)..." << std::flush;
  trace_mark("transpose_start");
  t0 = Clock::now();
  chunk_seq sT = ChunkFFT::transpose_pass(s1, d, t_prefix);
  const double transpose_s = elapsed(t0);
  trace_mark("transpose_end");
  std::cout << " done (" << std::setprecision(4) << transpose_s << "s)\n";

  // Stage 2T: streaming length-B column FFTs (now contiguous) + twiddle.
  std::cout << "Stage 2T (streaming columns)..." << std::flush;
  trace_mark("stage2t_start");
  t0 = Clock::now();
  chunk_seq s2 = ChunkFFT::stage2t_cols(sT, d, t2_prefix);
  const double stage2t_s = elapsed(t0);
  trace_mark("stage2t_end");
  std::cout << " done (" << std::setprecision(4) << stage2t_s << "s)\n";

  const double total_s = stage1_s + transpose_s + stage2t_s;
  const size_t bytes_moved =
      6 * N * sizeof(cd);  // stage1 + transpose + stage2t, r+w each
  const double gb_s = to_gb(bytes_moved) / total_s;
  std::cout << "stage1 " << std::setprecision(4) << stage1_s << "s   transpose "
            << transpose_s << "s   stage2t " << stage2t_s << "s   "
            << std::setprecision(2) << gb_s << " GB/s (bytes moved)\n";

  // In-memory baseline: the SAME four-step in DRAM (identical to fftExample),
  // and under FFT_VERIFY the independent complex_fft oracle over the full
  // spectrum.
  bool agree = true;
  double inmem_s = 0;
  if (do_baseline) {
    parlay::sequence<cd> x =
        parlay::tabulate(N, [](size_t m) { return input_val(m); });
    parlay::sequence<cd> Xmem = ChunkFFT::in_mem_place(x, d);
    t0 = Clock::now();
    ChunkFFT::in_mem_transform(Xmem, d);
    inmem_s = elapsed(t0);
    std::cout << "in-mem four-step FFT: " << std::setprecision(4) << inmem_s
              << "s";

    if (do_verify) {
      auto Xref = complex_fft(x);
      std::vector<cd> out = s2.to_vector<cd>();
      auto e = ChunkFFT::spectrum_errs(out, Xmem, Xref, d,
                                       ChunkFFT::out_perm_transpose);
      const double tol = 1e-6 * (e.max_ref > 0 ? e.max_ref : 1.0);
      agree = (e.err_oc <= tol) && (e.err_mem <= tol);
      std::cout << "   out-of-core err " << std::scientific << e.err_oc
                << "   in-mem err " << e.err_mem << "   tol " << tol
                << std::fixed << (agree ? "   OK" : "   *** MISMATCH ***");
    } else {
      std::cout << "   (timing baseline only; FFT_VERIFY=1 to cross-check)";
    }
    std::cout << "\n";
  } else if (trace_enabled()) {
    std::cout << "in-mem four-step FFT: skipped (tracing)\n";
  } else {
    std::cout << "in-mem four-step FFT: skipped (spectrum exceeds RAM budget "
              << std::setprecision(2) << to_gb(budget) << " GB)\n";
  }

  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  std::cout << "CSV," << N << ',' << f9(build_s) << ',' << f9(stage1_s) << ','
            << f9(transpose_s) << ',' << f9(stage2t_s) << ',' << f9(total_s)
            << ',' << (do_baseline ? f9(inmem_s) : std::string()) << ',' << N
            << ',' << f9(gb_s) << '\n';

  cleanup_prefix(in_prefix);
  cleanup_prefix(s1_prefix);
  cleanup_prefix(t_prefix);
  cleanup_prefix(t2_prefix);
  return agree ? 0 : 1;
}
