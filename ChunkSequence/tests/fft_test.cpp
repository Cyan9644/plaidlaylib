#include <fcntl.h>
#include <unistd.h>

#include <complex>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/examples/chunk_fft.h"
#include "parlay/primitives.h"
#include "parlaylib-examples/fast_fourier_transform.h"
#include "utils/file_utils.h"

// Correctness test for the out-of-core transpose-free FFT (chunk_fft.h,
// ChunkFFT::stage1_rows + stage2_cols), previously untested anywhere in
// ChunkSequence/tests/ -- examples/fft.cpp's own inline cross-check was the
// only correctness signal.  Reuses chunk_fft.h's own ChunkFFT::spectrum_errs /
// ChunkFFT::out_perm helpers (the same ones fft.cpp's driver calls) rather
// than reimplementing the error/permutation math, and cross-checks against
// the SAME independent upstream oracle the driver uses --
// parlaylib-examples/fast_fourier_transform.h's complex_fft -- at the same
// 1e-6 relative tolerance.

using ChunkFFT::cd;

static cd input_val(size_t m) {
  const double inv = 1.0 / 18446744073709551616.0;  // 1 / 2^64
  const double re = (double)parlay::hash64(2 * m) * inv - 0.5;
  const double im = (double)parlay::hash64(2 * m + 1) * inv - 0.5;
  return cd{re, im};
}

static void cleanup_prefix(const std::string& prefix) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    unlink(GetFileName(prefix, d).c_str());
}

static bool check_fft(const std::string& label, size_t req_n) {
  const ChunkFFT::Dims d = ChunkFFT::choose_dims(req_n);
  const size_t N = d.N, A = d.A, B = d.B;

  const std::string in_prefix = "fft_test_in";
  const std::string s1_prefix = "fft_test_s1";

  // Column-major placement stage 1 expects: physical p = b*A + a holds
  // logical element m = a*B + b (matches fft.cpp's own build).
  chunk_seq input = plaid::tabulate<cd>(N, in_prefix, [A, B](size_t p) {
    const size_t a = p % A;
    const size_t b = p / A;
    return input_val(a * B + b);
  });

  chunk_seq s1 = ChunkFFT::stage1_rows(input, d, s1_prefix);
  ChunkFFT::stage2_cols(s1, d, s1_prefix);
  std::vector<cd> out = s1.to_vector<cd>();

  cleanup_prefix(in_prefix);
  cleanup_prefix(s1_prefix);

  // Independent oracle + same-algorithm DRAM baseline, exactly as fft.cpp's
  // own driver computes them.
  parlay::sequence<cd> x =
      parlay::tabulate(N, [](size_t m) { return input_val(m); });
  auto Xref = complex_fft(x);
  parlay::sequence<cd> Xmem = ChunkFFT::in_mem_place(x, d);
  ChunkFFT::in_mem_transform(Xmem, d);

  auto e = ChunkFFT::spectrum_errs(out, Xmem, Xref, d, ChunkFFT::out_perm);
  const double tol = 1e-6 * (e.max_ref > 0 ? e.max_ref : 1.0);
  const bool ok = (e.err_oc <= tol) && (e.err_mem <= tol);
  std::cout << "  " << (ok ? "OK" : "FAIL") << " " << label
            << ": req_n=" << req_n << " -> N=" << N << " (A=" << A
            << " B=" << B << ")  out-of-core err=" << e.err_oc
            << "  in-mem err=" << e.err_mem << "  tol=" << tol << "\n";
  return ok;
}

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  const size_t n_arg = (argc > 1) ? std::stoull(argv[1]) : (size_t(1) << 18);

  bool all_pass = true;

  // Exact powers of two, at fft.cpp's own required minimum (N >= 2^16).
  all_pass &= check_fft("pow2_min", size_t(1) << 16);
  all_pass &= check_fft("pow2", size_t(1) << 17);

  // Non-power-of-two n: choose_dims truncates to the largest power of two
  // <= n (100000 -> N=65536=2^16).
  all_pass &= check_fft("non_pow2_truncation", 100000);

  // argv-provided size, matching the example's own scale.
  all_pass &= check_fft("large", n_arg);

  std::cout << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
  return all_pass ? 0 : 1;
}
