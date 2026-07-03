#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

#include "parlay/primitives.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/examples/chunk_kmp.h"

// One chunk of char text holds CHUNK_SIZE elements.
static constexpr size_t CHARS_PER_CHUNK = CHUNK_SIZE / sizeof(char);

// Remove all per-drive files created under a given prefix (one file per drive).
static void cleanup_prefix(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++)
        unlink(GetFileName(prefix, d).c_str());
}

// Sequential reference KMP that regenerates the text char-by-char from f(i),
// so even multi-batch cases need no O(n) text buffer.  Returns all (possibly
// overlapping) match-start positions in order.
static std::vector<uint64_t> reference_kmp(
    size_t n, const std::function<char(size_t)>& f, const std::string& pat)
{
    const long m = (long)pat.size();
    std::vector<long> failure(m, -1);
    for (long r = 1, l = -1; r < m; r++) {
        while (l != -1 && pat[l + 1] != pat[r]) l = failure[l];
        if (pat[l + 1] == pat[r]) failure[r] = ++l;
    }
    std::vector<uint64_t> out;
    long tail = -1;
    for (size_t i = 0; i < n; i++) {
        const char c = f(i);
        while (tail != -1 && c != pat[tail + 1]) tail = failure[tail];
        if (c == pat[tail + 1]) tail++;
        if (tail == m - 1) {
            out.push_back((uint64_t)(i - (size_t)tail));
            tail = failure[tail];
        }
    }
    return out;
}

// Builds a char text of length n via tabulate<char>(f), runs ChunkKmp with
// `pat`, and verifies count / packing / index order / exact positions against
// the streaming reference.  Cleans up all files and returns true iff PASS.
static bool run_kmp_test(const std::string& name, size_t n,
                         const std::function<char(size_t)>& f,
                         const std::string& pat)
{
    std::cout << "  " << name << "  (n=" << n << ", m=" << pat.size() << ")\n"
              << std::flush;

    const std::string text_prefix = "kmp_text";
    const std::string out_prefix  = "kmp_out";
    const std::string consolidated = "kmp_test_consolidated";

    chunk_seq text = ChunkSequenceOps::tabulate<char>(n, text_prefix, f);
    chunk_seq matches = ChunkSequenceOps::ChunkKmp<char>(text, out_prefix, pat);

    const std::vector<uint64_t> expected = reference_kmp(n, f, pat);
    bool pass = true;

    // 1. Element count.
    size_t actual_count = 0;
    for (const auto& c : matches.chunks)
        actual_count += c.used / sizeof(uint64_t);
    if (actual_count != expected.size()) {
        std::cout << "    FAIL count: got=" << actual_count
                  << " expected=" << expected.size() << "\n";
        pass = false;
    } else {
        std::cout << "    count  OK (" << actual_count << ")\n";
    }

    // 2. Tight packing: all chunks except the last must be full.
    {
        bool ok = true;
        for (size_t i = 0; i + 1 < matches.chunks.size() && ok; i++) {
            if (matches.chunks[i].used != CHUNK_SIZE) {
                std::cout << "    FAIL packing: chunk " << i
                          << " used=" << matches.chunks[i].used << "\n";
                pass = ok = false;
            }
        }
        if (ok) std::cout << "    packing OK\n";
    }

    // 3. Index-ordered invariant: chunks[i].index == i.
    {
        bool ok = true;
        for (size_t i = 0; i < matches.chunks.size() && ok; i++) {
            if (matches.chunks[i].index != i) {
                std::cout << "    FAIL index order: chunks[" << i
                          << "].index=" << matches.chunks[i].index << "\n";
                pass = ok = false;
            }
        }
        if (ok) std::cout << "    index  OK\n";
    }

    // 4. Exact positions: consolidate to a local file, stream-compare.
    if (pass && actual_count > 0) {
        matches.consolidate(consolidated);
        int fd = open(consolidated.c_str(), O_RDONLY);
        if (fd < 0) {
            std::cout << "    FAIL open(" << consolidated << "): "
                      << strerror(errno) << "\n";
            pass = false;
        } else {
            constexpr size_t BUF_ELEMS = (1 << 20);
            std::vector<uint64_t> buf(BUF_ELEMS);
            size_t j = 0;
            bool ok = true;
            while (ok) {
                const ssize_t got = read(fd, buf.data(), BUF_ELEMS * sizeof(uint64_t));
                if (got < 0) {
                    std::cout << "    FAIL read: " << strerror(errno) << "\n";
                    pass = ok = false;
                    break;
                }
                if (got == 0) break;  // EOF
                const size_t count = (size_t)got / sizeof(uint64_t);
                for (size_t i = 0; i < count && ok; i++, j++) {
                    if (j >= expected.size() || buf[i] != expected[j]) {
                        std::cout << "    FAIL position: element " << j
                                  << " got " << buf[i] << " expected "
                                  << (j < expected.size()
                                          ? std::to_string(expected[j])
                                          : std::string("<none>")) << "\n";
                        pass = ok = false;
                    }
                }
            }
            close(fd);
            if (ok && j != expected.size()) {
                std::cout << "    FAIL positions: read " << j
                          << " elements, expected " << expected.size() << "\n";
                pass = false;
            } else if (ok) {
                std::cout << "    positions OK\n";
            }
        }
    }

    std::cout << "    => " << (pass ? "PASS" : "FAIL") << "\n\n";

    cleanup_prefix(text_prefix);
    cleanup_prefix(out_prefix);
    unlink(consolidated.c_str());
    return pass;
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);

    bool all_pass = true;

    // Deterministic pseudo-random 4-letter text.
    auto random4 = [](size_t i) -> char {
        return (char)('a' + parlay::hash64(i) % 4);
    };

    // ── 1. Random text, self-overlapping pattern, partial last chunk ─────────
    // "abab" has a non-trivial failure function; n is not a chunk multiple.
    all_pass &= run_kmp_test("random_abab",
                             3 * CHARS_PER_CHUNK + 7, random4, "abab");

    // ── 2. All-'a' text, pattern "aaaa": overlapping matches everywhere ──────
    // Every position 0..n-m matches; stresses boundary overlap + dense packing
    // (n - m + 1 positions = many full uint64_t output chunks).
    all_pass &= run_kmp_test("all_a_dense",
                             2 * CHARS_PER_CHUNK + 5,
                             [](size_t) { return 'a'; }, "aaaa");

    // ── 3. No matches: pattern contains a letter outside the alphabet ────────
    all_pass &= run_kmp_test("no_match",
                             CHARS_PER_CHUNK + 3, random4, "abz");

    // ── 4. Planted matches straddling a chunk boundary and at the very end ───
    // Text is 'a'/'b' noise with "needle" written across the chunk-0/1 boundary
    // and flush against the end of the text.
    {
        const size_t n = 2 * CHARS_PER_CHUNK + 100;
        const std::string pat = "needle";
        const size_t plant1 = CHARS_PER_CHUNK - 3;  // spans chunks 0 and 1
        const size_t plant2 = n - pat.size();       // match at the very end
        auto f = [&, n](size_t i) -> char {
            if (i >= plant1 && i < plant1 + pat.size()) return pat[i - plant1];
            if (i >= plant2) return pat[i - plant2];
            return (char)('a' + parlay::hash64(i) % 2);
        };
        all_pass &= run_kmp_test("planted_boundary_and_end", n, f, pat);
    }

    // ── 5. Multi-batch seam ───────────────────────────────────────────────────
    // > DENSE_PACK_BATCH_SIZE (128) chunks so the producer runs twice, with a
    // match planted across the chunk-127/128 boundary — exercises the
    // synchronous seam-head read.  argv[1] overrides n (min 2 chunks).
    {
        size_t n = (argc > 1) ? std::stoull(argv[1]) : 129 * CHARS_PER_CHUNK;
        n = std::max(n, 2 * CHARS_PER_CHUNK);
        const std::string pat = "seamseam";
        const size_t seam_chunk =
            std::min<size_t>(ChunkSequenceOps::DENSE_PACK_BATCH_SIZE,
                             n / CHARS_PER_CHUNK) - 1;
        const size_t plant = (seam_chunk + 1) * CHARS_PER_CHUNK - pat.size() / 2;
        auto f = [&](size_t i) -> char {
            if (i >= plant && i < plant + pat.size()) return pat[i - plant];
            return (char)('a' + parlay::hash64(i) % 2);
        };
        all_pass &= run_kmp_test("multi_batch_seam", n, f, pat);
    }

    std::cout << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
    return all_pass ? 0 : 1;
}
