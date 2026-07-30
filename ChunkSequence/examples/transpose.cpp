// Example: out-of-core M×M matrix transpose via the direct-indexing view.
//
// Demonstrates IndexedChunkSeq (ChunkSequence/chunk_indexed.h): an out-of-core
// transpose written as a plain index-based loop (out[c*M+r] = in[r*M+c]) with
// per-worker block caching, instead of a bespoke transpose primitive or
// hand-rolled io_uring (cf. the FFT's band + RandomRing transpose_pass).  The
// algorithm itself lives in examples/chunk_transpose.h (tested by indexedTest).
//
// Dual-purpose like the other examples: prints human-readable results AND a
// machine-readable "CSV," line the examples sweep greps.
//
//   usage: transposeExample [global --flags] [n]
//     n   number of matrix elements (M = floor(sqrt(n)), N = M*M actually moved)
//
// Baselines (both cross-checked for identical output, exit non-zero on mismatch):
//   * in-mem parlay transpose over a DRAM copy (gated by EXAMPLE_INMEM_BUDGET_BYTES;
//     ~16N-byte footprint), the honest DRAM ceiling, timed as inmem_s.
//   * naive per-element chunk_seq::operator[] transpose (open/pread/close per
//     element) — the pre-abstraction cost; run only for small n (gated by
//     TRANSPOSE_NAIVE_MAX_ELEMS, default 2^20) since it is orders of magnitude
//     slower, and printed to human output only (not a CSV column).
//
// CSV line: CSV,<n>,<build_s>,<transpose_s>,<inmem_s>,<count>,<throughput_gb_s>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

#include "absl/log/check.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_indexed.h"
#include "ChunkSequence/examples/chunk_transpose.h"

using Clock = std::chrono::steady_clock;
using T = uint64_t;

static double elapsed(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}
static double to_gb(size_t bytes) { return (double)bytes / (1024.0 * 1024.0 * 1024.0); }

static void cleanup_prefix(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++) unlink(GetFileName(prefix, d).c_str());
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    namespace ops = ChunkSequenceOps;

    const size_t n = (argc > 1) ? std::stoull(argv[1]) : (size_t)1'000'000;
    // The cache-blocked transpose (chunk_transpose.h) requires M on the block grid
    // (a multiple of one O_DIRECT block's element count), so round sqrt(n) to the
    // nearest such multiple; N = M*M is what actually moves.
    const size_t epb = O_DIRECT_MULTIPLE / sizeof(T);        // elements per block (TILE)
    size_t M = (size_t)(std::sqrt((double)n) / epb + 0.5) * epb;
    if (M < epb) M = epb;
    const size_t N = M * M;

    std::cout << "Transpose " << M << "x" << M << " (" << N << " elements, TILE=" << epb << ")\n";

    // ── build the input matrix in[r*M+c] = r*M+c (untimed) ─────────────────────
    const std::string in_prefix = "transpose_in";
    auto tb = Clock::now();
    chunk_seq in = ops::tabulate<T>(N, in_prefix, [](size_t i) { return (T)i; });
    const double build_s = elapsed(tb);

    // ── the out-of-core transpose (timed) ──────────────────────────────────────
    const std::string out_prefix = "transpose_out";
    auto t0 = Clock::now();
    chunk_seq out = ChunkTranspose::transpose<T>(in, M, out_prefix);
    const double transpose_s = elapsed(t0);

    const size_t bytes = N * sizeof(T);
    std::cout << std::fixed << std::setprecision(4)
              << "out-of-core transpose: " << transpose_s << "s   "
              << std::setprecision(2) << to_gb(bytes) / transpose_s
              << " GB/s (matrix size)\n";

    bool agree = true;

    // Expected transposed value at output index q: out[q] = in[(q%M)*M + q/M],
    // and in is the identity iota, so out[q] == (q%M)*M + (q/M).
    auto expect = [M](size_t q) -> T { return (T)((q % M) * M + (q / M)); };

    // ── in-mem parlay transpose baseline (gated by RAM budget) ─────────────────
    const size_t phys = (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
    size_t budget = phys / 2;
    if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES")) budget = std::stoull(e);
    const bool inmem_ok = (N * 2 * sizeof(T)) <= budget;   // input copy + output

    double inmem_s = 0;
    if (inmem_ok) {
        auto a = parlay::tabulate(N, [](size_t i) { return (T)i; });
        auto t1 = Clock::now();
        auto b = parlay::tabulate(N, [&](size_t q) { return a[(q % M) * M + (q / M)]; });
        inmem_s = elapsed(t1);
        std::cout << "in-mem parlay transpose: " << std::setprecision(4) << inmem_s << "s\n";

        // Full contents cross-check: read the out-of-core result back and compare.
        auto got = out.to_vector<T>();
        if (got.size() != N) { std::cout << "*** MISMATCH: size ***\n"; agree = false; }
        else {
            bool ok = true;
            parlay::parallel_for(0, N, [&](size_t q) { if (got[q] != b[q]) ok = false; });
            if (!ok) { std::cout << "*** MISMATCH: contents vs in-mem ***\n"; agree = false; }
        }
    } else {
        std::cout << "in-mem parlay transpose: skipped (~16N footprint exceeds budget "
                  << std::setprecision(2) << to_gb(budget) << " GB)\n";
        // Spot-check a spread of output indices via chunk_seq::operator[].
        bool ok = true;
        for (size_t s = 0; s < 64 && ok; s++) {
            const size_t q = (size_t)((__uint128_t)s * N / 64);
            ok = out[q] == expect(q);
        }
        if (!ok) { std::cout << "*** MISMATCH: spot-check ***\n"; agree = false; }
    }

    // ── naive per-element operator[] transpose (small n only, human output) ─────
    size_t naive_max = (size_t)1 << 20;
    if (const char* e = getenv("TRANSPOSE_NAIVE_MAX_ELEMS")) naive_max = std::stoull(e);
    if (N <= naive_max) {
        const std::string nv_prefix = "transpose_naive";
        auto t2 = Clock::now();
        // Read each source element with chunk_seq::operator[] (open/pread/close per
        // element) — the pre-abstraction cost.  Writes go through tabulate's writer,
        // same as the view path, so this isolates the random-READ strategy.
        chunk_seq nv = ops::tabulate<T>(N, nv_prefix,
            [&](size_t q) { return in[(q % M) * M + (q / M)]; });
        const double naive_s = elapsed(t2);
        std::cout << "naive operator[] transpose: " << std::setprecision(4) << naive_s
                  << "s   (" << std::setprecision(1) << naive_s / transpose_s
                  << "x the cached view)\n";
        // Cross-check naive == expected.
        bool ok = true;
        for (size_t s = 0; s < 64 && ok; s++) {
            const size_t q = (size_t)((__uint128_t)s * N / 64);
            ok = nv[q] == expect(q);
        }
        if (!ok) { std::cout << "*** MISMATCH: naive ***\n"; agree = false; }
        cleanup_prefix(nv_prefix);
    }

    // Machine-readable line (columns: n,build_s,transpose_s,inmem_s,count,gb_s).
    auto f9 = [](double v) { std::ostringstream o; o << std::setprecision(9) << v; return o.str(); };
    std::cout << "CSV," << n << ',' << f9(build_s) << ',' << f9(transpose_s) << ','
              << (inmem_ok ? f9(inmem_s) : std::string()) << ',' << N << ','
              << f9(to_gb(bytes) / transpose_s) << '\n';

    cleanup_prefix(in_prefix);
    cleanup_prefix(out_prefix);
    return agree ? 0 : 1;
}
