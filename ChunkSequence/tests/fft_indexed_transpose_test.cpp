// Compare the out-of-core FFT's TWO transpose implementations inside the real
// four-step pipeline:
//
//   path A: ChunkFFT::transpose_pass  — the hand-rolled band + RandomRing on-disk
//                                        transpose (examples/chunk_fft.h).
//   path B: ChunkTranspose::transpose_rect — the same rectangular B×A → A×B
//                                        transpose expressed as a plain get/set loop
//                                        on the direct-indexing view (chunk_indexed.h).
//
// Both feed the identical stage1 (stage1_rows) input and the identical stage2
// (stage2t_cols) consumer, so this isolates the transpose.  The FFT's transpose is
// a pure data movement, so the two transposed sequences must be BIT-IDENTICAL, and
// therefore so must the two final spectra.  When it fits in RAM the result is also
// cross-checked against the in-memory four-step (ChunkFFT::in_mem_transform) so the
// comparison rests on an independently-correct answer, not just self-consistency.
//
// The transpose_pass permutation is p = b*A + k1  ->  q = k1*B + b, i.e. a
// row-major transpose of the B×A matrix M[b][k1] into A×B: out[k1*B+b] = in[b*A+k1].
// That is exactly transpose_rect(in, R=B, C=A): out[c*R+r] = in[r*C+c] with r=b, c=k1.
//
//   usage: fftIndexedTest [global --flags] [n]   (n rounded to a power of two; default 2^21)
//
// The default 2^21 gives A≠B (a genuinely RECTANGULAR transpose), which the
// square-only demo path would not exercise.

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

#include "absl/log/check.h"
#include "parlay/primitives.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/examples/chunk_fft.h"
#include "ChunkSequence/examples/chunk_transpose.h"

using ChunkFFT::cd;
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}
static void cleanup_prefix(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++) unlink(GetFileName(prefix, d).c_str());
}
// Same deterministic logical input as the fft example drivers.
static cd input_val(size_t m) {
    const double inv = 1.0 / 18446744073709551616.0;   // 1 / 2^64
    const double re = (double)parlay::hash64(2 * m)     * inv - 0.5;
    const double im = (double)parlay::hash64(2 * m + 1) * inv - 0.5;
    return cd{re, im};
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    RaiseFdLimit();
    namespace ops = ChunkSequenceOps;

    const size_t req_n = (argc > 1) ? std::stoull(argv[1]) : (size_t(1) << 21);
    const ChunkFFT::Dims d = ChunkFFT::choose_dims(req_n);
    const size_t N = d.N, A = d.A, B = d.B;
    CHECK(N >= (size_t(1) << 16)) << "n must be at least 2^16";
    CHECK(B <= ChunkFFT::EPC) << "transpose variant requires B <= EPC";
    // The indexed transpose tiles on the block grid: A and B must be multiples of
    // one O_DIRECT block's element count (256 for 16-byte cd).  A,B are powers of
    // two >= 256 for N >= 2^16, so this holds.
    const size_t epb = O_DIRECT_MULTIPLE / sizeof(cd);
    CHECK(A % epb == 0 && B % epb == 0) << "A,B must be multiples of " << epb;

    std::cout << "fftIndexedTest: N=" << N << " = A(" << A << ") x B(" << B << ")"
              << (A == B ? "  [square]" : "  [rectangular]") << "\n";

    int fails = 0;
    auto expect = [&](bool ok, const std::string& msg) {
        if (!ok) { std::cerr << "FAIL: " << msg << "\n"; fails++; }
    };

    // Build input in the column-major placement stage 1 expects, then stage 1.
    chunk_seq input = ops::tabulate<cd>(N, "ffti_in", [A, B](size_t p) {
        return input_val((p % A) * B + (p / A));
    });
    chunk_seq s1 = ChunkFFT::stage1_rows(input, d, "ffti_s1");

    // Path A: existing hand-rolled transpose.
    auto t0 = Clock::now();
    chunk_seq sT_a = ChunkFFT::transpose_pass(s1, d, "ffti_ta");
    const double ta = elapsed(t0);

    // Path B: the same transpose via the direct-indexing view (R=B, C=A).
    t0 = Clock::now();
    chunk_seq sT_b = ChunkTranspose::transpose_rect<cd>(s1, B, A, "ffti_tb");
    const double tb = elapsed(t0);

    // The transpose is a pure copy, so the two results must be bit-identical.
    {
        auto va = sT_a.to_vector<cd>();
        auto vb = sT_b.to_vector<cd>();
        expect(va.size() == N && vb.size() == N, "transposed size wrong");
        bool ok = va.size() == vb.size();
        for (size_t i = 0; i < va.size() && ok; i++) ok = (va[i] == vb[i]);
        expect(ok, "transpose_pass and indexed transpose_rect differ");
    }

    // Feed each transpose through the identical stage 2, compare the spectra, and
    // (when it fits) cross-check path A against the in-memory four-step.
    chunk_seq s2a = ChunkFFT::stage2t_cols(sT_a, d, "ffti_s2a");
    chunk_seq s2b = ChunkFFT::stage2t_cols(sT_b, d, "ffti_s2b");
    auto oa = s2a.to_vector<cd>();
    auto ob = s2b.to_vector<cd>();
    {
        bool ok = oa.size() == N && ob.size() == N && oa.size() == ob.size();
        for (size_t i = 0; i < oa.size() && ok; i++) ok = (oa[i] == ob[i]);
        expect(ok, "final spectra of the two transpose-FFTs differ");
    }

    const size_t phys = (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
    size_t budget = phys / 2;
    if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES")) budget = std::stoull(e);
    if (N * sizeof(cd) <= budget) {
        parlay::sequence<cd> x = parlay::tabulate(N, [](size_t m) { return input_val(m); });
        parlay::sequence<cd> Xmem = ChunkFFT::in_mem_place(x, d);
        ChunkFFT::in_mem_transform(Xmem, d);
        // s2 is in transposed physical order: X[k] lives at out_perm_transpose(k).
        double max_ref = 0, err = 0;
        for (size_t k = 0; k < N; k++) {
            max_ref = std::max(max_ref, std::abs(Xmem[k]));
            err = std::max(err, std::abs(oa[ChunkFFT::out_perm_transpose(k, d)] - Xmem[k]));
        }
        const double tol = 1e-6 * (max_ref > 0 ? max_ref : 1.0);
        std::cout << "  vs in-mem four-step: err " << std::scientific << err
                  << "  tol " << tol << std::fixed << "\n";
        expect(err <= tol, "transpose-FFT disagrees with in-mem four-step");
    } else {
        std::cout << "  in-mem cross-check skipped (spectrum exceeds RAM budget)\n";
    }

    std::cout << std::fixed << std::setprecision(4)
              << "transpose timings: transpose_pass " << ta << "s   indexed transpose_rect "
              << tb << "s   (" << std::setprecision(2)
              << (tb > 0 ? ta / tb : 0.0) << "x)\n";

    for (const char* p : {"ffti_in", "ffti_s1", "ffti_ta", "ffti_tb", "ffti_s2a", "ffti_s2b"})
        cleanup_prefix(p);

    std::cout << (fails == 0 ? "PASS" : "FAIL") << "  fails=" << fails << "\n";
    return fails == 0 ? 0 : 1;
}
