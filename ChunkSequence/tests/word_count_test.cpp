#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unistd.h>

#include "parlay/primitives.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/examples/external/chunk_word_count.h"

// One chunk of char text holds CHUNK_SIZE elements.
static constexpr size_t CHARS_PER_CHUNK = CHUNK_SIZE / sizeof(char);

static void cleanup_prefix(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++)
        unlink(GetFileName(prefix, d).c_str());
}

// Sequential reference word count: regenerate the text char-by-char from f(i)
// and split on spaces, exactly as WordCount should (no O(n) text buffer needed).
static std::unordered_map<std::string, uint64_t> reference_wc(
    size_t n, const std::function<char(size_t)>& f) {
    std::unordered_map<std::string, uint64_t> m;
    std::string cur;
    for (size_t i = 0; i < n; i++) {
        const char c = f(i);
        if (c == ' ') {
            if (!cur.empty()) { m[cur]++; cur.clear(); }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) m[cur]++;   // trailing partial word (bounded by end-of-text)
    return m;
}

// Build a char text of length n via tabulate<char>(f), run WordCount, and verify
// the full distinct (word -> count) map against the streaming reference.
static bool run_case(const std::string& name, size_t n,
                     const std::function<char(size_t)>& f) {
    std::cout << "  " << name << "  (n=" << n << ", ~" << (n / CHARS_PER_CHUNK + 1)
              << " chunks)\n" << std::flush;

    const std::string prefix = "wc_test_text";
    chunk_seq text = plaid::tabulate<char>(n, prefix, f);
    plaid::WordCounts counts = plaid::WordCount(text);

    const auto expected = reference_wc(n, f);
    bool pass = true;

    // Rebuild an actual (word -> count) map from the representatives so the
    // comparison also exercises the hash -> word representative bookkeeping.
    std::unordered_map<std::string, uint64_t> actual;
    for (const auto& kv : counts) actual[kv.second.second] = kv.second.first;

    if (actual.size() != expected.size()) {
        std::cout << "    FAIL distinct: got=" << actual.size()
                  << " expected=" << expected.size() << "\n";
        pass = false;
    } else {
        for (const auto& kv : expected) {
            auto it = actual.find(kv.first);
            if (it == actual.end() || it->second != kv.second) {
                std::cout << "    FAIL count for \"" << kv.first << "\": got="
                          << (it == actual.end() ? std::string("<none>")
                                                 : std::to_string(it->second))
                          << " expected=" << kv.second << "\n";
                pass = false;
                break;
            }
        }
    }

    if (pass) std::cout << "    OK (" << actual.size() << " distinct)\n";
    std::cout << "    => " << (pass ? "PASS" : "FAIL") << "\n\n";
    cleanup_prefix(prefix);
    return pass;
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);

    bool all_pass = true;

    // Fixed 6-letter words separated by single spaces.  SPAN=7 is coprime with
    // CHUNK_SIZE (power of two), so words straddle chunk boundaries.
    auto span7 = [](size_t i, uint64_t dict) -> char {
        const size_t off = i % 7;
        if (off == 6) return ' ';
        const uint64_t id = parlay::hash64(i / 7) % dict;
        return (char)('a' + (parlay::hash64(id * 7 + off) % 26));
    };

    // 1. Multi-chunk varied vocabulary, trailing partial word (n not a multiple
    //    of the span nor the chunk size).
    all_pass &= run_case("multi_chunk_words", 3 * CHARS_PER_CHUNK + 7,
                         [&](size_t i) { return span7(i, 512); });

    // 2. A single repeated word (max hash contention, dense count) whose
    //    occurrences straddle every chunk boundary.
    all_pass &= run_case("single_word_dense", 2 * CHARS_PER_CHUNK + 13,
                         [&](size_t i) { return span7(i, 1); });

    // 3. A recognizable word planted straight across the chunk-0/1 boundary,
    //    inside an otherwise span-7 background.  "needle" starts 3 chars before
    //    the boundary; spaces are forced on both sides so it is its own token.
    {
        const size_t n = 2 * CHARS_PER_CHUNK + 20;
        const std::string needle = "needle";
        const size_t plant = CHARS_PER_CHUNK - 3;   // spans chunks 0 and 1
        auto f = [&, n](size_t i) -> char {
            if (i == plant - 1 || i == plant + needle.size()) return ' ';
            if (i >= plant && i < plant + needle.size()) return needle[i - plant];
            return span7(i, 512);
        };
        all_pass &= run_case("planted_boundary", n, f);
    }

    // 4. Text whose length is a multiple of the span, so the last char is a
    //    space and there is no trailing partial word (> 1 chunk: 4200000 > 4 MiB).
    all_pass &= run_case("ends_on_space", 7 * 600000,
                         [&](size_t i) { return span7(i, 256); });

    std::cout << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
    return all_pass ? 0 : 1;
}
