// Example: out-of-core word count via a streaming hash-fold.
//
// Counts every space-separated word in a synthetic n-char text stored across the
// SSDs.  Unlike a sort-based group-by, this makes ONE streaming read of the text
// (no sort of the token stream, no intermediate writes): each parlay worker folds
// the words it sees into a local hash map keyed by a 64-bit word hash, and the
// caller merges the maps and stitches the few words that straddle a chunk
// boundary (plaid::WordCount, examples/chunk_word_count.h).  This is
// the out-of-core analogue of parlaylib's `word_counts`, which likewise groups by
// hashing (histogram_by_key) and only sorts the small distinct-pairs list.
//
// Dual-purpose, like the benchmarks: prints human-readable results AND a
// machine-readable "CSV," line that benchmarks/run_benches.py greps.
//
// When it fits in RAM the driver also times parlaylib's own in-memory word_counts
// (deps/parlaylib-examples/word_counts.h) over the same text as a DRAM baseline,
// and cross-checks the full distinct (word -> count) map by hash (exits non-zero
// on a mismatch).  Budget: half of physical RAM, override via
// EXAMPLE_INMEM_BUDGET_BYTES; when skipped the CSV field is left blank so the
// plotted in-mem line stops at the RAM cliff.
//
//   usage: word_countExample [global --flags] [n]
//     n   text length in characters (default 1e8)
//
// CSV line: CSV,<n>,<build_s>,<count_s>,<inmem_count_s>,<distinct_words>,<throughput_gb_s>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unistd.h>

#include "absl/log/check.h"

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "parlay/utilities.h"   // parlay::hash64

// Upstream parlaylib example (fetched by `make deps`), used only as the in-memory
// comparison baseline.  Defines global word_counts(sequence<char>) and pulls in
// parlay::tokens / histogram_by_key; the out-of-core algorithm below stays
// self-contained in plaid.  Only one upstream header is included here.
#include "parlaylib-examples/word_counts.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "utils/trace_marker.h"
#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/examples/external/chunk_word_count.h"

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}
static double to_gb(size_t bytes) { return (double)bytes / (1024.0 * 1024.0 * 1024.0); }

static void cleanup_prefix(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++) unlink(GetFileName(prefix, d).c_str());
}

// ── Deterministic synthetic text ────────────────────────────────────────────────
// Fixed 6-letter words separated by single spaces (SPAN = 7 = word + space).  A
// word's letters are a pure function of its dictionary id, and the id is drawn
// with a mild low-id skew (min of two hashes) so counts are non-uniform and the
// top-K list is interesting.  SPAN = 7 is coprime with CHUNK_SIZE (a power of two),
// so word boundaries fall at varying offsets and words genuinely straddle chunk
// seams -- exercising the seam-stitch path.  The generator is a pure function of
// the character index, so the out-of-core text and the DRAM baseline are byte-
// identical.
static constexpr size_t WC_WORD_LEN = 6;
static constexpr size_t WC_SPAN     = WC_WORD_LEN + 1;
static constexpr uint64_t WC_DICT   = 4096;

static inline uint64_t wc_word_id(size_t w) {
    const uint64_t a = parlay::hash64(2 * w)     % WC_DICT;
    const uint64_t b = parlay::hash64(2 * w + 1) % WC_DICT;
    return std::min(a, b);
}
static inline char wc_gen(size_t i) {
    const size_t off = i % WC_SPAN;
    if (off == WC_WORD_LEN) return ' ';
    const uint64_t id = wc_word_id(i / WC_SPAN);
    return (char)('a' + (parlay::hash64(id * WC_SPAN + off) % 26));
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 100'000'000ULL;

    // RAM budget for the in-memory baseline.  Upstream word_counts materializes
    // the text, an iota(n+1) of longs, and one sequence per token; ~12n bytes.
    const size_t phys = (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
    size_t budget = phys / 2;
    if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES")) budget = std::stoull(e);
    const bool inmem_ok = n * 12 <= budget;

    // ── Build the text out-of-core ──────────────────────────────────────────────
    std::cout << "Building text of " << n << " chars..." << std::flush;
    const std::string prefix = "wc_text";
    auto tb = Clock::now();
    chunk_seq text = plaid::tabulate<char>(n, prefix, wc_gen);
    const double build_s = elapsed(tb);
    std::cout << " done (" << text.chunks.size() << " chunks across "
              << GetSSDList().size() << " drives)\n";

    // ── Count ───────────────────────────────────────────────────────────────────
    trace_mark("op_start");
    auto t0 = Clock::now();
    plaid::WordCounts counts = plaid::WordCount(text);
    const double count_s = elapsed(t0);
    trace_mark("op_end");

    size_t distinct = counts.size();
    size_t total_words = 0;
    for (const auto& kv : counts) total_words += kv.second.first;
    size_t text_bytes = 0;
    for (const auto& c : text.chunks) text_bytes += c.used;

    std::cout << "distinct words = " << distinct << ", total words = " << total_words
              << "   " << std::fixed << std::setprecision(4) << count_s << "s   "
              << std::setprecision(2) << to_gb(text_bytes) / count_s
              << " GB/s (text read)\n";

    // Top few words by frequency (upstream's final sorted output).
    auto sorted = plaid::SortedByCount(counts);
    const size_t show = std::min<size_t>(sorted.size(), 10);
    std::cout << "top " << show << " word(s):";
    for (size_t i = 0; i < show; i++)
        std::cout << ' ' << sorted[i].first << '(' << sorted[i].second << ')';
    std::cout << "\n";

    // ── In-memory parlaylib baseline + differential check ───────────────────────
    bool agree = true;
    double inmem_s = 0;
    if (inmem_ok) {
        parlay::sequence<char> text_dram = parlay::tabulate(n, [](size_t i) { return wc_gen(i); });
        auto t1 = Clock::now();
        auto pairs = word_counts(text_dram);   // upstream: sorted (word, count)
        inmem_s = elapsed(t1);

        // Expected distinct (hash -> count), hashed identically to the fold.
        std::unordered_map<uint64_t, uint64_t> expected;
        for (const auto& pr : pairs)
            expected[plaid::HashWord(pr.first.data(), pr.first.size())] +=
                (uint64_t)pr.second;

        std::cout << "in-mem parlaylib word_counts: distinct = " << expected.size()
                  << "   " << std::setprecision(4) << inmem_s << "s\n";

        if (expected.size() != distinct) {
            std::cout << "*** MISMATCH: distinct in-mem " << expected.size()
                      << " != out-of-core " << distinct << " ***\n";
            agree = false;
        } else {
            for (const auto& kv : expected) {
                auto it = counts.find(kv.first);
                if (it == counts.end() || it->second.first != kv.second) {
                    std::cout << "*** MISMATCH: count for a word differs "
                              << "(in-mem " << kv.second << ") ***\n";
                    agree = false;
                    break;
                }
            }
        }
    } else {
        std::cout << "in-mem parlaylib word_counts: skipped (~12n footprint exceeds "
                  << "RAM budget " << std::setprecision(2) << to_gb(budget) << " GB)\n";
    }

    // Machine-readable line for benchmarks/run_benches.py (examples sweep).
    // Columns: n,build_s,count_s,inmem_count_s,distinct_words,throughput_gb_s
    // (inmem_count_s blank when the footprint exceeds the RAM budget).
    auto f9 = [](double v) { std::ostringstream o; o << std::setprecision(9) << v; return o.str(); };
    std::cout << "CSV," << n << ',' << f9(build_s) << ',' << f9(count_s) << ','
              << (inmem_ok ? f9(inmem_s) : std::string()) << ',' << distinct
              << ',' << f9(to_gb(text_bytes) / count_s) << '\n';

    cleanup_prefix(prefix);
    return agree ? 0 : 1;
}
