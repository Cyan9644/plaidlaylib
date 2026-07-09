// Example: out-of-core slice/cut via ChunkSequenceOps::sequential_cut_no_compression.
//
// Builds an n-element sequence of pseudo-random uint64 keys across the SSDs
// (deterministic from parlay::hash64, so it is reproducible and duplicate-free),
// then extracts the half-open range [start, end) into a new, independent
// out-of-core sequence with the cut in ExternalPrimitives/chunk_cut.h.  The cut
// rewrites the (possibly unaligned) first/last partial chunks into fresh O_DIRECT
// seam files and threads the interior chunk headers through; from_chunks then
// materializes the whole result into fresh, independent on-disk files (a full
// copy of the range across the drives), so the returned sequence owns all its
// data -- symmetric with the in-memory baseline, which copies the range into a
// fresh DRAM sequence.
//
// Dual-purpose, like the benchmarks and the other examples: prints human-readable
// results AND a machine-readable "CSV," line that benchmarks/run_benches.py greps.
// The examples sweep (make bench-examples) times the cut across a sweep of n with
// the range held at the middle ~half (k = n/2), so the cut length scales with n.
// Both endpoints are placed in the MIDDLE of a chunk (offset ELEMS_PER_CHUNK/2),
// never on a chunk boundary, so every sweep point deterministically exercises the
// same real seam-rewrite work (see the default computation in main() for why the
// naive n/4, 3n/4 endpoints alias the chunk grid for power-of-two n and skew the
// timings).  The in-mem baseline cuts the identical [start, end).
//
// Baseline: parlaylib's slice/cut on the same keys in DRAM.  parlay::slice::cut
// itself is O(1) -- it returns a view, copying nothing -- so timing it alone is
// meaningless.  The out-of-core cut materializes an *independent* copy of the
// range, so the honest in-memory analog is copying the slice into a fresh
// sequence (parlay::to_sequence(keys.cut(first, last))); that is what we time.
// The result is cross-checked by reading the out-of-core cut back in logical
// order and comparing it element-wise against the in-memory copy (keys are
// distinct, so the range is unique and the two substrates must agree exactly).
// Budget: half of physical RAM, override via EXAMPLE_INMEM_BUDGET_BYTES; when
// skipped the CSV field is left blank so the plotted in-mem line stops at the
// RAM cliff (as in the other examples).
//
//   usage: chunk_cutExample [global --flags] [n] [start] [end]
//     n       number of keys (default 1e6)
//     start   0-based first index of the range to cut (default n/4)
//     end     0-based one-past-last index of the range   (default 3n/4)
//
// CSV line: CSV,<n>,<start>,<end>,<build_s>,<cut_s>,<inmem_cut_s>,<out_elems>,<throughput_gb_s>
//   throughput = cut-range bytes ((end-start)*8) / cut_s.
//
// Complexity: O(k) work for a k-element range -- from_chunks copies the whole
// range (interior chunks + the two rewritten seam chunks) to fresh on-disk files.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "absl/log/check.h"

#include "parlay/primitives.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "utils/logger.h"
#include "ChunkSequence/chunk_seq.h"
// Out-of-core operation under test.  The in-memory baseline is just parlaylib's
// slice::cut, pulled in with the rest of parlay/primitives.h above.
#include "ChunkSequence/ExternalPrimitives/chunk_cut.h"

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}
static double to_gb(size_t bytes) { return (double)bytes / (1024.0 * 1024.0 * 1024.0); }

// Deterministic, duplicate-free key i, computable anywhere so the out-of-core
// input and the in-memory baseline hold the identical multiset.
static uint64_t key_at(size_t i) { return parlay::hash64(i); }

// Read a chunk_seq back into DRAM in logical (vector) order, concatenating each
// chunk's `used` bytes.  Unlike ChunkSequenceOps::materialize / peek, this does
// NOT assume the index-ordered invariant or that every chunk but the last is
// full: the cut's output has a (possibly partial) rewritten head chunk and reuses
// interior chunks with their original index fields, so it must be read strictly
// by position.  Sequential and only used for the cross-check.
template <typename T>
static std::vector<T> read_in_order(const chunk_seq& seq) {
    std::vector<T> out;
    T* buf = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
    CHECK(buf != nullptr) << "read_in_order: buffer allocation failed";
    for (const chunk& c : seq.chunks) {
        int fd = open(c.filename.c_str(), O_RDONLY | O_DIRECT);
        SYSCALL(fd);
        SYSCALL(pread(fd, buf, AlignUp(c.used), (off_t)c.begin_addr));
        close(fd);
        const size_t cnt = c.used / sizeof(T);
        out.insert(out.end(), buf, buf + cnt);
    }
    free(buf);
    return out;
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;

    // Default range: the middle ~half of the sequence ([start, end) with k = n/2),
    // with BOTH endpoints landing in the MIDDLE of a chunk (offset
    // ELEMS_PER_CHUNK/2) rather than on a chunk boundary.  This is deterministic
    // regardless of how n aligns to the CHUNK grid.
    //
    // Why it matters: the old defaults (n/4, 3n/4) land *exactly* on chunk
    // boundaries for every power-of-two sweep size >= 2^21 (n/4 is then a multiple
    // of ELEMS_PER_CHUNK).  A boundary-aligned cut hits a degenerate seam path --
    // the start seam rewrites a whole chunk and the end seam is an empty (used=0)
    // chunk -- so it exercises little of the real per-seam read/rewrite work and
    // its cost jumps around vs. the one non-aligned point (2^20).  Forcing each
    // seam to split its chunk ~half-half makes every sweep point do the same real
    // work.  The in-mem baseline below uses this identical [start, end), so both
    // substrates always cut the same range.
    constexpr size_t EPC = ELEMS_PER_CHUNK;
    size_t def_start = (n / 4 / EPC) * EPC + EPC / 2;
    size_t def_end   = (3 * n / 4 / EPC) * EPC + EPC / 2;
    if (def_start >= def_end || def_end > n) {   // tiny n: fall back to a valid range
        def_start = n / 4;
        def_end   = (3 * n) / 4;
    }
    const size_t start = (argc > 2) ? std::stoull(argv[2]) : def_start;
    const size_t end   = (argc > 3) ? std::stoull(argv[3]) : def_end;
    CHECK(n > 0 && start < end && end <= n)
        << "need 0 <= start < end <= n (n=" << n << ", start=" << start
        << ", end=" << end << ")";
    const size_t k = end - start;   // cut length

    // RAM budget for the in-memory baseline (as in the other examples): when it
    // runs we hold the n-key input (8n bytes), the copied slice (8k <= 8n), and
    // the read-back of the out-of-core cut (8k) for the element-wise cross-check
    // -- call it ~24n worst case.
    const size_t phys = (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
    size_t budget = phys;
    if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES")) budget = std::stoull(e);
    const bool inmem_ok = n <= budget / 24;

    const std::string in_prefix = "cut_in";

    std::cout << "Building " << n << "-key input..." << std::flush;
    auto t0 = Clock::now();
    chunk_seq seq = ChunkSequenceOps::tabulate<uint64_t>(n, in_prefix, key_at);
    const double build_s = elapsed(t0);
    std::cout << " done (" << std::fixed << std::setprecision(4) << build_s
              << "s)\n";

    std::cout << "Cutting [" << start << ", " << end << ") (" << k
              << " elems) of " << n << "..." << std::flush;
    t0 = Clock::now();
    chunk_seq cut =
        ChunkSequenceOps::sequential_cut_no_compression<uint64_t>(seq, start, end);
    const double cut_s = elapsed(t0);
    std::cout << " done\n";

    // True logical length: sum each chunk's used bytes.  (ChunkSequenceOps::size
    // can't be used -- it assumes every chunk but the last is full, but the cut's
    // rewritten head chunk is partial.)
    size_t out_elems = 0;
    for (const chunk& c : cut.chunks) out_elems += c.used / sizeof(uint64_t);
    const double gb_s = to_gb(k * sizeof(uint64_t)) / cut_s;
    std::cout << "cut " << out_elems << " elems   "
              << std::setprecision(4) << cut_s << "s   "
              << std::setprecision(2) << gb_s << " GB/s (cut range)\n";

    // In-memory baseline: parlaylib's slice::cut on the same keys, materialized
    // into an independent sequence (the cut view itself copies nothing), timed
    // outside the input build.  Cross-checked by reading the out-of-core cut back
    // in logical order and comparing it element-wise.
    bool agree = true;
    double inmem_cut_s = 0;
    if (inmem_ok) {
        auto keys_mem = parlay::tabulate(n, key_at);   // parlay::sequence<uint64_t>
        t0 = Clock::now();
        auto slice_mem = parlay::to_sequence(keys_mem.cut(start, end));
        inmem_cut_s = elapsed(t0);
        std::cout << "in-mem parlaylib cut (materialized copy)   "
                  << std::setprecision(4) << inmem_cut_s << "s\n";

        auto out_mem = read_in_order<uint64_t>(cut);
        if (out_mem.size() != slice_mem.size()) {
            std::cout << "*** MISMATCH: out-of-core cut produced " << out_mem.size()
                      << " elems, expected " << slice_mem.size() << " ***\n";
            agree = false;
        } else {
            for (size_t i = 0; i < slice_mem.size(); i++) {
                if (out_mem[i] != slice_mem[i]) {
                    std::cout << "*** MISMATCH at index " << i << ": out-of-core "
                              << out_mem[i] << " != in-mem " << slice_mem[i]
                              << " ***\n";
                    agree = false;
                    break;
                }
            }
        }
        if (agree) std::cout << "cross-check: out-of-core cut matches in-mem slice\n";
    } else {
        std::cout << "in-mem parlaylib cut: skipped (~24n footprint exceeds RAM "
                  << "budget " << std::setprecision(2) << to_gb(budget) << " GB)\n";
    }

    // Machine-readable line for benchmarks/run_benches.py (examples sweep).
    // Columns: n,start,end,build_s,cut_s,inmem_cut_s,out_elems,throughput_gb_s
    // (inmem_cut_s blank when the input exceeds the RAM budget).
    auto f9 = [](double v) { std::ostringstream o; o << std::setprecision(9) << v; return o.str(); };
    std::cout << "CSV," << n << ',' << start << ',' << end << ','
              << f9(build_s) << ',' << f9(cut_s) << ','
              << (inmem_ok ? f9(inmem_cut_s) : std::string()) << ','
              << out_elems << ',' << f9(gb_s) << '\n';

    // Don't leave data on the drives across sweep points.  Four sets of files
    // exist: the input ("cut_in<d>"), the two seam scratch files the cut writes
    // ("cut_in<d>_cut_start" / "cut_in<d>_cut_end"), and the materialized,
    // independent cut output that from_chunks now writes ("cut_out<d>").  Unlink
    // all of them (run_benches.py's "cut_in*"/"cut_out*" globs cover them across
    // sweep points).
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++) {
        const std::string f = GetFileName(in_prefix, d);
        unlink(f.c_str());
        unlink((f + "_cut_start").c_str());
        unlink((f + "_cut_end").c_str());
        unlink(GetFileName("cut_out", d).c_str());
    }
    return agree ? 0 : 1;
}
