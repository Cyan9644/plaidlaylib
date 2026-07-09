// Chunk-size sensitivity benchmark: eager vs delayed across varying CHUNK_SIZE.
//
// Compiled once per chunk size with -DCHUNK_SIZE_BYTES=N (see Makefile pattern
// rule for chunkSizeCompare_%).  For a fixed problem size n, times the same
// logical pipeline four ways:
//
//   raw read          ChunkReduce(iota)                   — device-read ceiling (1n I/O)
//   map|reduce        eager 2r+1w (3n)  delayed 1r (1n)
//   map|map|reduce    eager 3r+2w (5n)  delayed 1r (1n)  — clearest fusion win
//   force(map|map)    eager 2r+2w (4n)  delayed 1r+1w (2n)
//
// No in-memory baseline: that comparison is orthogonal to chunk size and is
// covered by delayed_compare.cpp / bench_delayed_scale.sh.
//
// Cross-substrate correctness check on every run; exits non-zero on mismatch.
// Intermediates are cleaned up after each pipeline step.
//
// CSV line emitted for bench_chunk_size.sh:
//   CSV,<chunk_size_bytes>,<n>,<raw_s>,<eager_mr_s>,<delayed_mr_s>,
//       <eager_mmr_s>,<delayed_mmr_s>,<eager_fmm_s>,<delayed_fmm_s>,<agree>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <string>
#include <unistd.h>

#include "parlay/primitives.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_map.h"
#include "ChunkSequence/chunk_reduce.h"
#include "ChunkSequence/chunk_delayed.h"

namespace cd = ChunkSequenceOps::delayed;

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
    for (size_t d = 0; d < GetSSDList().size(); d++)
        unlink(GetFileName(prefix, d).c_str());
}

static void print_row(const std::string& label, size_t in_bytes, double secs) {
    std::cout << "  " << std::left << std::setw(16) << label << std::right
              << std::fixed << std::setprecision(4) << std::setw(9) << secs << "s   "
              << std::setprecision(2) << to_gb(in_bytes) / secs << " GB/s (eff. input)\n";
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    const size_t n = (argc > 1) ? std::stoull(argv[1]) : (size_t(1) << 24);

    auto add1 = [](uint64_t x) { return x + 1; };
    auto mul2 = [](uint64_t x) { return 2 * x; };

    std::cout << "CHUNK_SIZE=" << CHUNK_SIZE << " bytes (" << CHUNK_SIZE / 1024 << " KB)"
              << "  n=" << n << "\n";

    std::cout << "Generating chunk_seq iota(" << n << ")..." << std::flush;
    const chunk_seq cseq = ChunkSequenceOps::iota(n);
    const size_t in_bytes = chunk_seq_bytes(cseq);
    std::cout << " " << cseq.chunks.size() << " chunks, "
              << std::fixed << std::setprecision(3) << to_gb(in_bytes) << " GB\n\n";

    double raw_s = 0, e_mr = 0, d_mr = 0, e_mmr = 0, d_mmr = 0, e_f = 0, d_f = 0;
    bool agree = true;
    auto check = [&](const char* what, bool ok) {
        if (!ok) { std::cout << "  *** MISMATCH: " << what << " ***\n"; agree = false; }
    };

    // ── raw read ceiling ─────────────────────────────────────────────────────
    std::cout << "--- raw read: ChunkReduce(sum) ---\n";
    {
        auto t0 = Clock::now();
        volatile uint64_t r = ChunkSequenceOps::ChunkReduce<uint64_t>(cseq, SumMonoid{});
        raw_s = elapsed(t0);
        (void)r;
        print_row("raw read", in_bytes, raw_s);
    }

    // ── map(x+1) | reduce(sum) ───────────────────────────────────────────────
    std::cout << "\n--- map(x+1) | reduce(sum) ---\n";
    {
        auto t0 = Clock::now();
        uint64_t mr_e = ChunkSequenceOps::ChunkReduce<uint64_t>(
            ChunkSequenceOps::ChunkMap<uint64_t>(cseq, "bw_cs_m", add1), SumMonoid{});
        e_mr = elapsed(t0);
        cleanup_prefix("bw_cs_m");
        print_row("chunk-eager", in_bytes, e_mr);

        auto t1 = Clock::now();
        uint64_t mr_d = cd::reduce(cd::map(cd::delay(cseq), add1), SumMonoid{});
        d_mr = elapsed(t1);
        print_row("chunk-delayed", in_bytes, d_mr);
        check("map|reduce  eager == delayed", mr_e == mr_d);
    }

    // ── map(x+1) | map(2x) | reduce(sum) ─────────────────────────────────────
    uint64_t mmr_ref = 0;
    std::cout << "\n--- map(x+1) | map(2x) | reduce(sum) ---\n";
    {
        auto t0 = Clock::now();
        uint64_t mmr_e = ChunkSequenceOps::ChunkReduce<uint64_t>(
            ChunkSequenceOps::ChunkMap<uint64_t>(
                ChunkSequenceOps::ChunkMap<uint64_t>(cseq, "bw_cs_m1", add1),
                "bw_cs_m2", mul2),
            SumMonoid{});
        e_mmr = elapsed(t0);
        cleanup_prefix("bw_cs_m1");
        cleanup_prefix("bw_cs_m2");
        mmr_ref = mmr_e;
        print_row("chunk-eager", in_bytes, e_mmr);

        auto t1 = Clock::now();
        uint64_t mmr_d = cd::reduce(cd::map(cd::map(cd::delay(cseq), add1), mul2), SumMonoid{});
        d_mmr = elapsed(t1);
        print_row("chunk-delayed", in_bytes, d_mmr);
        check("map|map|reduce  eager == delayed", mmr_e == mmr_d);
    }

    // ── force(map(x+1) | map(2x)) ────────────────────────────────────────────
    std::cout << "\n--- force(map(x+1) | map(2x)) ---\n";
    {
        auto t0 = Clock::now();
        chunk_seq g1 = ChunkSequenceOps::ChunkMap<uint64_t>(cseq, "bw_cs_g1", add1);
        chunk_seq g2 = ChunkSequenceOps::ChunkMap<uint64_t>(g1, "bw_cs_g2", mul2);
        e_f = elapsed(t0);
        uint64_t f_e = ChunkSequenceOps::ChunkReduce<uint64_t>(g2, SumMonoid{});
        cleanup_prefix("bw_cs_g1");
        cleanup_prefix("bw_cs_g2");
        print_row("chunk-eager", in_bytes, e_f);
        check("force  sum == map|map|reduce sum", f_e == mmr_ref);

        auto t1 = Clock::now();
        chunk_seq out = cd::force(cd::map(cd::map(cd::delay(cseq), add1), mul2), "bw_cs_gf");
        d_f = elapsed(t1);
        uint64_t f_d = ChunkSequenceOps::ChunkReduce<uint64_t>(out, SumMonoid{});
        cleanup_prefix("bw_cs_gf");
        print_row("chunk-delayed", in_bytes, d_f);
        check("force  eager == delayed", f_e == f_d);
    }

    // ── cleanup input ────────────────────────────────────────────────────────
    cleanup_prefix("iota");

    std::cout << "\n" << (agree ? "agree=1 (all substrates match)"
                                : "agree=0 (MISMATCH — see above)") << "\n";

    // Machine-readable line for bench_chunk_size.sh.
    // Columns: chunk_size_bytes,n,raw_s,eager_mr_s,delayed_mr_s,
    //          eager_mmr_s,delayed_mmr_s,eager_fmm_s,delayed_fmm_s,agree
    auto f9 = [](double v) {
        std::ostringstream o;
        o << std::setprecision(9) << v;
        return o.str();
    };
    std::cout << "CSV," << CHUNK_SIZE << ',' << n
              << ',' << f9(raw_s)
              << ',' << f9(e_mr)  << ',' << f9(d_mr)
              << ',' << f9(e_mmr) << ',' << f9(d_mmr)
              << ',' << f9(e_f)   << ',' << f9(d_f)
              << ',' << (agree ? 1 : 0) << '\n';
    return agree ? 0 : 1;
}
