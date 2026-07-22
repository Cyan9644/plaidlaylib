// Correctness test for ChunkDC3 (examples/external/chunk_dc3.h).
//
// Validates the out-of-core DC3 suffix array two ways:
//   1. vs brute force on many tiny texts (random small alphabets, all-equal,
//      binary, with heavy repeats) — the duplicate-heavy stress that exercises
//      the triple-naming and the DC3 comparator's tie handling.  Each text is
//      built out-of-core and its suffix array compared element-wise to a
//      std::sort of all suffixes.
//   2. a multi-chunk differential: the fully out-of-core path (DRAM budget
//      forced to 0, so every recursion level streams off the drives) must agree
//      with the pure in-memory Kärkkäinen–Sanders base case (budget forced huge)
//      on a text spanning several chunks.
//
// Every tiny case is run under BOTH budgets, so both the streaming recursion and
// the DRAM base case are checked against brute force.  Exits 0 iff all pass.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <unistd.h>

#include "absl/log/check.h"
#include "parlay/primitives.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/examples/external/chunk_dc3.h"

namespace {

void set_budget(const char* v) { setenv("DC3_DRAM_BUDGET_BYTES", v, 1); }

void cleanup_prefix(const std::string& prefix) {
    for (const std::string& dir : GetSSDList()) {
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
            const std::string name = e.path().filename().string();
            if (name.rfind(prefix, 0) == 0) std::filesystem::remove(e.path(), ec);
        }
    }
}

// Reference suffix array: sort all suffix start positions lexicographically.
std::vector<uint32_t> brute_sa(const std::string& s) {
    const size_t n = s.size();
    std::vector<uint32_t> sa(n);
    std::iota(sa.begin(), sa.end(), 0u);
    std::sort(sa.begin(), sa.end(), [&](uint32_t a, uint32_t b) {
        while (a < n && b < n) {
            if (s[a] != s[b]) return (unsigned char)s[a] < (unsigned char)s[b];
            a++; b++;
        }
        return a > b;   // the shorter (ran off the end) suffix is smaller
    });
    return sa;
}

// Read an index-ordered chunk_seq<uint32> back off the drives into a vector.
std::vector<uint32_t> read_sa(const chunk_seq& sa) {
    std::vector<const chunk*> ord;
    for (const auto& c : sa.chunks) ord.push_back(&c);
    std::sort(ord.begin(), ord.end(),
              [](const chunk* a, const chunk* b) { return a->index < b->index; });
    void* buf = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
    CHECK(buf != nullptr);
    std::vector<uint32_t> out;
    for (const chunk* c : ord) {
        if (c->used == 0) continue;
        int fd = open(c->filename.c_str(), O_RDONLY | O_DIRECT);
        SYSCALL(fd);
        SYSCALL(pread(fd, buf, AlignUp(c->used), (off_t)c->begin_addr));
        close(fd);
        const auto* e = (const uint32_t*)buf;
        for (size_t i = 0; i < c->used / sizeof(uint32_t); i++) out.push_back(e[i]);
    }
    free(buf);
    return out;
}

std::vector<uint32_t> dc3_of(const std::string& s, const char* budget) {
    set_budget(budget);
    const std::string tin = "dc3t_in", tout = "dc3t_out";
    chunk_seq text = ChunkSequenceOps::tabulate<char>(
        s.size(), tin, [&s](size_t i) { return s[i]; });
    chunk_seq sa = ChunkSequenceOps::ChunkDC3(text, tout);
    std::vector<uint32_t> res = read_sa(sa);
    cleanup_prefix(tin);
    cleanup_prefix(tout);
    return res;
}

bool eq(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

int failures = 0;
void check(const std::string& s, const std::string& label) {
    const std::vector<uint32_t> ref = brute_sa(s);
    for (const char* budget : {"0", "1000000000000"}) {   // out-of-core, then DRAM
        const std::vector<uint32_t> got = dc3_of(s, budget);
        if (!eq(got, ref)) {
            std::cout << "  FAIL [" << label << " n=" << s.size()
                      << " budget=" << budget << "]\n";
            if (s.size() <= 40) {
                std::cout << "    text: " << s << "\n    ref: ";
                for (auto x : ref) std::cout << x << ' ';
                std::cout << "\n    got: ";
                for (auto x : got) std::cout << x << ' ';
                std::cout << "\n";
            }
            failures++;
        }
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    RaiseFdLimit();

    std::mt19937_64 rng(12345);

    // Tiny handcrafted / structured cases.
    check("a", "single");
    check("aa", "double");
    check("banana", "banana");
    check("mississippi", "mississippi");
    check("abracadabra", "abracadabra");
    for (size_t n : {3, 4, 5, 6, 7, 8, 15, 33, 64, 100})
        check(std::string(n, 'a'), "all-equal");   // maximal duplicate stress

    // Random small texts over several alphabets.
    for (int alpha : {2, 3, 4, 26}) {
        for (int rep = 0; rep < 20; rep++) {
            const size_t n = 1 + rng() % 300;
            std::string s(n, 'a');
            for (auto& ch : s) ch = (char)('a' + rng() % alpha);
            check(s, "rand-a" + std::to_string(alpha));
        }
    }

    // Larger random cases (still brute-force checkable), boundary lengths mod 3.
    for (size_t n : {998, 999, 1000, 1001, 4096, 20000}) {
        std::string s(n, 'a');
        for (auto& ch : s) ch = (char)('a' + rng() % 4);
        check(s, "rand4");
    }

    // Multi-chunk differential: out-of-core recursion vs in-memory KS base case.
    if (failures == 0) {
        const size_t n = (argc > 1) ? std::stoull(argv[1]) : 5'000'000;   // > 1 chunk
        std::cout << "multi-chunk differential n=" << n << " ..." << std::flush;
        std::string s(n, 'a');
        for (auto& ch : s) ch = (char)('a' + rng() % 4);
        const std::vector<uint32_t> a = dc3_of(s, "0");                 // full out-of-core
        const std::vector<uint32_t> b = dc3_of(s, "1000000000000");     // in-memory base
        if (!eq(a, b)) { std::cout << " FAIL (streaming vs DRAM disagree)\n"; failures++; }
        else std::cout << " ok\n";
    }

    if (failures == 0) std::cout << "dc3Test: ALL PASS\n";
    else std::cout << "dc3Test: " << failures << " FAILURES\n";
    return failures == 0 ? 0 : 1;
}
