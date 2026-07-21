// Benchmark: out-of-core random shuffle, three contestants on the identical key
// multiset --
//
//   1. random_shuffle_method  (ExternalPrimitives/random_shuffle.h) -- the
//      bucketing shuffle written on the high-level abstractions: a fused delayed
//      map draws each element's bucket, count_sort routes the elements into
//      per-bucket external sequences, each bucket is read back / shuffled in DRAM
//      / written out as *fresh* files, flatten concatenates them.
//   2. ChunkSequenceOps::Permutation::Permute -- the same algorithm on the
//      low-level reader/writer paradigm (ported from Peter's scatter-gather):
//      identical bucketing, but each bucket is rewritten **in place** over the
//      count-sort's own chunks (process_buckets_inplace), so it moves one fewer
//      copy of the data.
//   3. parlay::random_shuffle -- the in-memory parlaylib baseline (stops at the
//      RAM cliff, like every other example's in-mem series).
//
// Same shape as external_samplesort_vs_peter.cpp (two out-of-core substrates
// timed head-to-head, each on drives holding only its own input), extended with
// the in-DRAM baseline the plain examples carry.
//
// Correctness for a shuffle is a *permutation* check, not element-wise equality:
// the keys key_at(i)=parlay::hash64(i) are distinct, so an output is a valid
// shuffle iff, once sorted, it equals the sorted key set.  Both out-of-core
// outputs (and the in-mem baseline) are checked that way when the input fits the
// RAM budget.
//
//   usage: external_random_shuffleExample [global --flags] [n]
//     n   number of keys (default 1e6)
//
// CSV line:
//   CSV,<n>,<build_s>,<shuffle_s>,<perm_s>,<inmem_shuffle_s>,
//       <shuffle_gb_s>,<perm_gb_s>
//   throughput = input bytes (n*8) / that method's own time; inmem_shuffle_s is
//   blank past the RAM budget, so the plotted in-mem line stops at the cliff.
//
// Dual-purpose like the other examples: prints human-readable output AND the
// machine-readable "CSV," line benchmarks/run_benches.py greps.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "absl/log/check.h"

#include "parlay/primitives.h"
#include "parlay/random.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/ExternalPrimitives/materialize.h"
#include "ChunkSequence/ExternalPrimitives/random_shuffle.h"

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Flush dirty pages and let the drives settle between a write phase and a timed
// shuffle, so still-draining writeback doesn't queue behind (and inflate) the
// shuffle's own I/O.  Always outside a timed region — see external_samplesort.cpp.
static void quiesce_drives() {
    sync();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
}
static double to_gb(size_t bytes) { return (double)bytes / (1024.0 * 1024.0 * 1024.0); }

// ── drive hygiene ───────────────────────────────────────────────────────────
// Two mechanisms, because the point of this benchmark is that neither method may
// leave anything behind:
//
//   remove_prefixes()  — the working sweep.  A generic name-prefix scan (not
//     GetFileName(prefix, d)) because both methods tag their intermediates with a
//     per-call counter and a bucket id (rs_bucket_<tag><d>, rs_out_<tag>_<b><d>),
//     which a fixed 0..num_drives enumeration would miss.  Used to isolate each
//     timed method on drives that hold only the shared input.
//
//   snapshot_drives() / sweep_new_files() — the backstop.  Everything on the
//     drives is listed before the run; at the end, ANY file that appeared since
//     and is still there is a leak by definition, whatever it is named.  It is
//     removed and reported, and the run exits non-zero — a silent leak would
//     otherwise accumulate across sweep points and quietly change what later
//     points measure.
static void remove_prefixes(const std::vector<std::string>& prefixes) {
    for (const std::string& dir : GetSSDList()) {
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
            const std::string name = e.path().filename().string();
            for (const std::string& p : prefixes) {
                if (name.rfind(p, 0) == 0) {  // name starts with p
                    std::filesystem::remove(e.path(), ec);
                    break;
                }
            }
        }
    }
}

static std::set<std::string> snapshot_drives() {
    std::set<std::string> files;
    for (const std::string& dir : GetSSDList()) {
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(dir, ec))
            files.insert(e.path().string());
    }
    return files;
}

// Remove every file that appeared on the drives since `before`; returns them.
static std::vector<std::string> sweep_new_files(const std::set<std::string>& before) {
    std::vector<std::string> leaked;
    for (const std::string& path : snapshot_drives()) {
        if (before.count(path)) continue;
        std::error_code ec;
        std::filesystem::remove(path, ec);
        leaked.push_back(path);
    }
    return leaked;
}

// The input, and each method's on-disk file family.  random_shuffle_method's
// result *is* its rs_out_ files and Permutation's result *is* its perm files
// (rewritten in place over the count sort's own chunks), so those prefixes must
// be swept too — otherwise the entire output of every sweep point stays on the
// drives.
static const std::string kInPrefix = "rs_in";
static const std::vector<std::string> kMethodPrefixes =
    {"rs_bucket_", "rs_out_", "rs_base_"};   // random_shuffle_method(seq, "rs")
static const std::vector<std::string> kPermPrefixes = {"perm"};  // Permute(seq, "perm")

// Deterministic, duplicate-free key i, so a permutation check is exact.
static uint64_t key_at(size_t i) { return parlay::hash64(i); }

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    // Both shuffles fan out one io_uring instance + one open file per drive for
    // every concurrent reader/writer, past the 1024 soft fd limit; lift it to the
    // hard limit before any I/O starts.
    RaiseFdLimit();

    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
    CHECK(n > 0) << "need n > 0 (n=" << n << ")";

    // RAM budget for the in-memory baseline + cross-check: the keys (8n), the
    // baseline's shuffled output (8n), the sorted reference (8n), and BOTH
    // out-of-core outputs read back (16n) — call it ~32n, as in
    // external_samplesort_vs_peter.  Past the budget the out-of-core shuffles
    // still run and are timed; only the baseline and the checks are skipped.
    const size_t phys = (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
    size_t budget = phys;
    if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES")) budget = std::stoull(e);
    const bool check_ok = n <= budget / 32;

    std::cout << std::fixed;
    std::cout << "Clearing stale shuffle files from the drives..." << std::flush;
    std::vector<std::string> all_prefixes = {kInPrefix};
    all_prefixes.insert(all_prefixes.end(), kMethodPrefixes.begin(), kMethodPrefixes.end());
    all_prefixes.insert(all_prefixes.end(), kPermPrefixes.begin(), kPermPrefixes.end());
    remove_prefixes(all_prefixes);
    std::cout << " done\n";

    // Everything on the drives right now is somebody else's; anything else that
    // survives to the end of this run is ours and is a leak.
    const std::set<std::string> pre_run = snapshot_drives();

    // ── the shared input ────────────────────────────────────────────────────
    std::cout << "Building " << n << "-key chunk_seq input..." << std::flush;
    auto t0 = Clock::now();
    chunk_seq seq = ChunkSequenceOps::tabulate<uint64_t>(n, kInPrefix, key_at);
    const double build_s = elapsed(t0);
    std::cout << " done (" << std::setprecision(4) << build_s << "s)\n";
    quiesce_drives();   // isolate the op timers from the build's writeback

    // ── 1. random_shuffle_method (high-level abstractions) ──────────────────
    std::cout << "Shuffling " << n << " keys (random_shuffle_method)..." << std::flush;
    t0 = Clock::now();
    chunk_seq shuffled = random_shuffle_method<uint64_t>(seq, "rs");
    const double shuffle_s = elapsed(t0);
    const double shuffle_gb_s = to_gb(n * sizeof(uint64_t)) / shuffle_s;
    std::cout << " done   " << std::setprecision(4) << shuffle_s << "s   "
              << std::setprecision(2) << shuffle_gb_s << " GB/s (input read)\n";

    // Snapshot its output (under budget), then clear every file it left so the
    // next method is timed on drives holding only the shared input.
    parlay::sequence<uint64_t> ours;
    if (check_ok) ours = ChunkSequenceOps::materialize<uint64_t>(shuffled);
    remove_prefixes(kMethodPrefixes);
    std::cout << "random_shuffle_method's files cleared before the next method runs\n";
    quiesce_drives();

    // ── 2. Permutation::Permute (low-level reader/writer, in-place buckets) ──
    std::cout << "Permuting " << n << " keys (ChunkSequenceOps::Permutation)..." << std::flush;
    ChunkSequenceOps::Permutation<uint64_t> permuter;
    t0 = Clock::now();
    chunk_seq permuted = permuter.Permute(seq, "perm");
    const double perm_s = elapsed(t0);
    const double perm_gb_s = to_gb(n * sizeof(uint64_t)) / perm_s;
    std::cout << " done   " << std::setprecision(4) << perm_s << "s   "
              << std::setprecision(2) << perm_gb_s << " GB/s (input read)\n";

    parlay::sequence<uint64_t> theirs;
    if (check_ok) theirs = ChunkSequenceOps::materialize<uint64_t>(permuted);
    remove_prefixes(kPermPrefixes);
    quiesce_drives();

    std::cout << "speedup (random_shuffle_method / Permutation): " << std::setprecision(2)
              << (shuffle_s / perm_s) << "x\n";

    // ── 3. in-memory baseline + permutation cross-check ─────────────────────
    bool agree = true;
    double inmem_shuffle_s = 0;
    if (check_ok) {
        auto keys_mem = parlay::tabulate(n, key_at);
        t0 = Clock::now();
        auto shuffled_mem = parlay::random_shuffle(keys_mem);
        inmem_shuffle_s = elapsed(t0);
        std::cout << "in-mem parlay::random_shuffle   "
                  << std::setprecision(4) << inmem_shuffle_s << "s\n";

        auto ref = keys_mem;              // the key set, sorted: the permutation target
        parlay::sort_inplace(ref);

        // Sorting a valid shuffle of the input must reproduce the key set exactly
        // (the keys are distinct), so this catches a dropped, duplicated, or
        // corrupted element in any of the three outputs.
        auto is_permutation = [&](const char* who, parlay::sequence<uint64_t> got) {
            if (got.size() != ref.size()) {
                std::cout << "*** MISMATCH (" << who << "): produced " << got.size()
                          << " keys, expected " << ref.size() << " ***\n";
                return false;
            }
            parlay::sort_inplace(got);
            for (size_t i = 0; i < ref.size(); i++) {
                if (got[i] != ref[i]) {
                    std::cout << "*** MISMATCH (" << who << ") at sorted index " << i
                              << ": " << got[i] << " != " << ref[i] << " ***\n";
                    return false;
                }
            }
            return true;
        };
        agree  = is_permutation("random_shuffle_method", std::move(ours));
        agree &= is_permutation("Permutation", std::move(theirs));
        agree &= is_permutation("in-mem parlay", std::move(shuffled_mem));
        if (agree)
            std::cout << "cross-check: all three outputs are permutations of the input\n";
    } else {
        std::cout << "in-mem baseline + cross-check: skipped (~32n footprint exceeds "
                  << "RAM budget " << std::setprecision(2) << to_gb(budget) << " GB)\n";
    }

    // Machine-readable line for benchmarks/run_benches.py (examples sweep).
    // Columns: n,build_s,shuffle_s,perm_s,inmem_shuffle_s,shuffle_gb_s,perm_gb_s
    auto f9 = [](double v) { std::ostringstream o; o << std::setprecision(9) << v; return o.str(); };
    std::cout << "CSV," << n << ',' << f9(build_s) << ',' << f9(shuffle_s) << ','
              << f9(perm_s) << ',' << (check_ok ? f9(inmem_shuffle_s) : std::string())
              << ',' << f9(shuffle_gb_s) << ',' << f9(perm_gb_s) << '\n';

    // ── leave the drives exactly as we found them ───────────────────────────
    // Each method's files are already gone; this drops the input, and then the
    // snapshot diff catches anything either method wrote under a name this
    // driver's prefix list does not know about.
    remove_prefixes(all_prefixes);
    const std::vector<std::string> leaked = sweep_new_files(pre_run);
    if (!leaked.empty()) {
        std::cout << "*** LEAK: " << leaked.size()
                  << " file(s) were left on the drives by this run (removed now); "
                     "the driver's prefix list is incomplete ***\n";
        for (size_t i = 0; i < leaked.size() && i < 20; i++)
            std::cout << "      " << leaked[i] << '\n';
        if (leaked.size() > 20)
            std::cout << "      ... and " << (leaked.size() - 20) << " more\n";
        agree = false;
    } else {
        std::cout << "drives clean: no files left behind by this run\n";
    }
    return agree ? 0 : 1;
}
