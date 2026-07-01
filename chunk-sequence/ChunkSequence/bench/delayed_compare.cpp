// Scaling harness: in-memory delayed vs chunk-eager vs chunk-delayed.
//
// For one problem size n, times the SAME logical pipeline three ways and checks
// they agree:
//   * in-memory parlay::delayed  — the DRAM "speed-of-light" line (the paper's
//     technique in RAM); skipped once the input exceeds a RAM budget;
//   * chunk-eager                — ChunkMap/ChunkReduce, round-tripping every
//     intermediate through the SSDs;
//   * chunk-delayed              — ChunkSequenceOps::delayed, fused.
// Plus a raw-read ceiling (a bare ChunkReduce = one read pass).
//
// Pipelines (data generated outside the timed region):
//   raw read         ChunkReduce(perm)                          (device-read ceiling)
//   map|reduce       eager 2r+1w (3n)  delayed 1r (n)  in-mem delayed::reduce(map)
//   map|map|reduce   eager 3r+2w (5n)  delayed 1r (n)  in-mem delayed::reduce(map(map))
//   force(map|map)   eager 2r+2w (4n)  delayed 1r+1w (2n)  in-mem delayed::to_sequence(map(map))
//
// Every point runs all available substrates on the same input, so the harness
// doubles as a cross-substrate differential correctness check: a mismatch sets
// agree=0 and the program exits non-zero.  Eager intermediates are written under
// bw_dl_* and cleaned up after each step.
//
// Emits a machine-readable "CSV," line for bench_delayed_scale.sh.  In-memory
// fields are left blank past the RAM cliff so the plotter stops that line.

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <string>
#include <unistd.h>

#include "parlay/primitives.h"
#include "parlay/delayed.h"
#include "parlay/monoid.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_map.h"
#include "ChunkSequence/chunk_reduce.h"
#include "ChunkSequence/chunk_delayed.h"

namespace cd = ChunkSequenceOps::delayed;

// Chunk-side monoid (operator()-style, as ChunkReduce/cd::reduce expect).
struct SumMonoid {
    uint64_t identity = 0;
    uint64_t operator()(uint64_t a, uint64_t b) const { return a + b; }
};

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

static size_t chunk_seq_bytes(const chunk_seq& seq) {
    size_t total = 0;
    for (const auto& c : seq.chunks) total += c.used;
    return total;
}
static double to_gb(size_t bytes) { return (double)bytes / (1024.0 * 1024.0 * 1024.0); }

static void cleanup_prefix(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++) unlink(GetFileName(prefix, d).c_str());
}

// One timed row: wall time + effective-input throughput (logical input bytes /
// time) — the apples-to-apples metric across substrates (same logical work).
static void print_row(const std::string& label, size_t in_bytes, double secs) {
    std::cout << "  " << std::left << std::setw(16) << label << std::right
              << std::fixed << std::setprecision(4) << std::setw(9) << secs << "s   "
              << std::setprecision(2) << to_gb(in_bytes) / secs << " GB/s (eff. input)\n";
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 5'000'000ULL;

    auto add1 = [](uint64_t x) { return x + 1; };
    auto mul2 = [](uint64_t x) { return 2 * x; };
    const parlay::plus<uint64_t> psum{};                    // in-memory monoid

    // ── RAM budget for the in-memory baseline ────────────────────────────────
    const size_t phys = (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
    size_t budget = phys / 2;
    if (const char* e = getenv("DELAYED_INMEM_BUDGET_BYTES")) budget = std::stoull(e);
    const size_t in_bytes_logical = n * sizeof(uint64_t);
    const bool inmem_reduce_ok = in_bytes_logical <= budget;        // ~8n resident
    const bool inmem_force_ok  = in_bytes_logical * 2 <= budget;    // ~16n (input + output)

    std::cout << "n=" << n << "   RAM budget for in-memory baseline " << std::fixed
              << std::setprecision(2) << to_gb(budget) << " GB (phys " << to_gb(phys) << " GB)\n"
              << "in-mem reduce lines: " << (inmem_reduce_ok ? "ON" : "OFF (n exceeds budget)")
              << ";  in-mem force line: " << (inmem_force_ok ? "ON" : "OFF") << "\n\n";

    // ── data generation (outside every timed region) ─────────────────────────
    std::cout << "Generating chunk_seq perm(" << n << ")..." << std::flush;
    const chunk_seq cseq = ChunkSequenceOps::perm(n);
    const size_t in_bytes = chunk_seq_bytes(cseq);
    std::cout << " " << cseq.chunks.size() << " chunks, " << to_gb(in_bytes) << " GB\n";

    parlay::sequence<uint64_t> A;   // in-memory source (materialized, data in RAM)
    if (inmem_reduce_ok) {
        std::cout << "Generating in-memory perm(" << n << ")..." << std::flush;
        A = parlay::tabulate(n, [](size_t i) { return (uint64_t)i; });
        std::cout << " done\n";
    }
    std::cout << "\n";

    double raw_s = 0, e_mr = 0, d_mr = 0, i_mr = 0, e_mmr = 0, d_mmr = 0, i_mmr = 0,
           e_f = 0, d_f = 0, i_f = 0;
    bool agree = true;
    auto check = [&](const char* what, bool ok) {
        if (!ok) { std::cout << "  *** MISMATCH: " << what << " ***\n"; agree = false; }
    };

    // ── raw read (device-read ceiling) ───────────────────────────────────────
    std::cout << "--- raw read: ChunkReduce(sum) — device-read ceiling ---\n";
    {
        auto t0 = Clock::now();
        uint64_t r = ChunkSequenceOps::ChunkReduce<uint64_t>(cseq, SumMonoid{});
        raw_s = elapsed(t0);
        (void)r;
        print_row("raw read", in_bytes, raw_s);
    }

    // ── map(x+1) | reduce(sum) ───────────────────────────────────────────────
    std::cout << "\n--- map(x+1) | reduce(sum) ---\n";
    {
        auto t0 = Clock::now();
        e_mr = 0;
        uint64_t r = ChunkSequenceOps::ChunkReduce<uint64_t>(
            ChunkSequenceOps::ChunkMap<uint64_t>(cseq, "bw_dl_m", add1), SumMonoid{});
        e_mr = elapsed(t0);
        cleanup_prefix("bw_dl_m");
        uint64_t mr_e = r;
        print_row("chunk-eager", in_bytes, e_mr);

        auto t1 = Clock::now();
        uint64_t mr_d = cd::reduce(cd::map(cd::delay(cseq), add1), SumMonoid{});
        d_mr = elapsed(t1);
        print_row("chunk-delayed", in_bytes, d_mr);
        check("map|reduce  eager == delayed", mr_e == mr_d);

        if (inmem_reduce_ok) {
            auto t2 = Clock::now();
            uint64_t mr_i = parlay::delayed::reduce(parlay::delayed::map(A, add1), psum);
            i_mr = elapsed(t2);
            print_row("in-mem delayed", in_bytes, i_mr);
            check("map|reduce  in-mem == eager", mr_i == mr_e);
        }
    }

    // ── map(x+1) | map(2x) | reduce(sum) ─────────────────────────────────────
    uint64_t mmr_e = 0;
    std::cout << "\n--- map(x+1) | map(2x) | reduce(sum) ---\n";
    {
        auto t0 = Clock::now();
        mmr_e = ChunkSequenceOps::ChunkReduce<uint64_t>(
            ChunkSequenceOps::ChunkMap<uint64_t>(
                ChunkSequenceOps::ChunkMap<uint64_t>(cseq, "bw_dl_m1", add1),
                "bw_dl_m2", mul2),
            SumMonoid{});
        e_mmr = elapsed(t0);
        cleanup_prefix("bw_dl_m1"); cleanup_prefix("bw_dl_m2");
        print_row("chunk-eager", in_bytes, e_mmr);

        auto t1 = Clock::now();
        uint64_t mmr_d = cd::reduce(cd::map(cd::map(cd::delay(cseq), add1), mul2), SumMonoid{});
        d_mmr = elapsed(t1);
        print_row("chunk-delayed", in_bytes, d_mmr);
        check("map|map|reduce  eager == delayed", mmr_e == mmr_d);

        if (inmem_reduce_ok) {
            auto t2 = Clock::now();
            uint64_t mmr_i = parlay::delayed::reduce(
                parlay::delayed::map(parlay::delayed::map(A, add1), mul2), psum);
            i_mmr = elapsed(t2);
            print_row("in-mem delayed", in_bytes, i_mmr);
            check("map|map|reduce  in-mem == eager", mmr_i == mmr_e);
        }
    }

    // ── force(map(x+1) | map(2x)) — write-terminated ─────────────────────────
    // Time only the materialization; the verifying reduce of each output is untimed.
    std::cout << "\n--- force(map(x+1) | map(2x)) — write-terminated ---\n";
    {
        auto t0 = Clock::now();
        chunk_seq m1 = ChunkSequenceOps::ChunkMap<uint64_t>(cseq, "bw_dl_g1", add1);
        chunk_seq m2 = ChunkSequenceOps::ChunkMap<uint64_t>(m1, "bw_dl_g2", mul2);
        e_f = elapsed(t0);
        uint64_t f_e = ChunkSequenceOps::ChunkReduce<uint64_t>(m2, SumMonoid{});  // verify (untimed)
        cleanup_prefix("bw_dl_g1"); cleanup_prefix("bw_dl_g2");
        print_row("chunk-eager", in_bytes, e_f);

        auto t1 = Clock::now();
        chunk_seq out = cd::force(cd::map(cd::map(cd::delay(cseq), add1), mul2), "bw_dl_gf");
        d_f = elapsed(t1);
        uint64_t f_d = ChunkSequenceOps::ChunkReduce<uint64_t>(out, SumMonoid{});
        cleanup_prefix("bw_dl_gf");
        print_row("chunk-delayed", in_bytes, d_f);
        check("force  eager == delayed", f_e == f_d);
        check("force sum == map|map|reduce sum", f_e == mmr_e);

        if (inmem_force_ok) {
            auto t2 = Clock::now();
            parlay::sequence<uint64_t> outm = parlay::delayed::to_sequence(
                parlay::delayed::map(parlay::delayed::map(A, add1), mul2));
            i_f = elapsed(t2);
            uint64_t f_i = parlay::reduce(outm, psum);
            print_row("in-mem force", in_bytes, i_f);
            check("force  in-mem == eager", f_i == f_e);
        }
    }

    std::cout << "\n" << (agree ? "agree=1 (all substrates match)"
                                : "agree=0 (MISMATCH — see above)") << "\n";

    // Machine-readable line for bench_delayed_scale.sh.  In-mem fields blank when
    // skipped (past the RAM cliff).  Columns:
    //   n,raw_read_s,eager_mr,delayed_mr,inmem_mr,eager_mmr,delayed_mmr,inmem_mmr,
    //     eager_fmm,delayed_fmm,inmem_fmm,agree
    auto f9 = [](double v) { std::ostringstream o; o << std::setprecision(9) << v; return o.str(); };
    auto opt = [&](bool has, double v) { return has ? f9(v) : std::string(); };
    std::cout << "CSV," << n
              << ',' << f9(raw_s)
              << ',' << f9(e_mr)  << ',' << f9(d_mr)  << ',' << opt(inmem_reduce_ok, i_mr)
              << ',' << f9(e_mmr) << ',' << f9(d_mmr) << ',' << opt(inmem_reduce_ok, i_mmr)
              << ',' << f9(e_f)   << ',' << f9(d_f)   << ',' << opt(inmem_force_ok,  i_f)
              << ',' << (agree ? 1 : 0) << '\n';
    return agree ? 0 : 1;
}
