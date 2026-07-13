#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

#include "absl/log/check.h"
#include "parlay/primitives.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_flat_map.h"
#include "ChunkSequence/dense_pack.h"

// One chunk of uint64_t holds EPCT elements.
static constexpr size_t EPCT = CHUNK_SIZE / sizeof(uint64_t);

// Remove all per-drive files created under a given prefix (one file per drive).
static void cleanup_prefix(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++)
        unlink(GetFileName(prefix, d).c_str());
}

// Verify `matches` against `expected`: element count, tight packing, the
// index-ordered invariant, and exact contents via consolidate + stream compare.
// Mirrors kmp_test's verifier.  Returns true iff PASS.
static bool verify(const std::string& name, const chunk_seq& matches,
                   const std::vector<uint64_t>& expected) {
    bool pass = true;

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

    {
        bool ok = true;
        for (size_t i = 0; i + 1 < matches.chunks.size() && ok; i++)
            if (matches.chunks[i].used != CHUNK_SIZE) {
                std::cout << "    FAIL packing: chunk " << i
                          << " used=" << matches.chunks[i].used << "\n";
                pass = ok = false;
            }
        if (ok) std::cout << "    packing OK\n";
    }

    {
        bool ok = true;
        for (size_t i = 0; i < matches.chunks.size() && ok; i++)
            if (matches.chunks[i].index != i) {
                std::cout << "    FAIL index order: chunks[" << i
                          << "].index=" << matches.chunks[i].index << "\n";
                pass = ok = false;
            }
        if (ok) std::cout << "    index  OK\n";
    }

    if (pass && actual_count > 0) {
        const std::string consolidated = "flatmap_test_consolidated";
        matches.consolidate(consolidated);
        std::vector<uint64_t> got = [&] {
            std::vector<uint64_t> v(actual_count);
            int fd = open(consolidated.c_str(), O_RDONLY);
            CHECK(fd >= 0) << "open(" << consolidated << "): " << strerror(errno);
            size_t off = 0;
            while (off < actual_count * sizeof(uint64_t)) {
                ssize_t g = read(fd, (char*)v.data() + off,
                                 actual_count * sizeof(uint64_t) - off);
                CHECK(g > 0) << "short read";
                off += (size_t)g;
            }
            close(fd);
            return v;
        }();
        unlink(consolidated.c_str());

        bool ok = true;
        for (size_t i = 0; i < actual_count && ok; i++)
            if (got[i] != expected[i]) {
                std::cout << "    FAIL position " << i << ": got " << got[i]
                          << " expected " << expected[i] << "\n";
                pass = ok = false;
            }
        if (ok) std::cout << "    positions OK\n";
    }

    std::cout << "    => " << (pass ? "PASS" : "FAIL") << "\n\n";
    return pass;
}

// ── Test 1: halo == 0, variable-length per-chunk output, no neighbor ─────────
// Body keeps even input values, doubled.  Asserts the engine passes a null,
// empty halo for every call when halo == 0.  Differential against an in-memory
// filter+map over the same generator.
static bool test_halo0(size_t n, const std::function<uint64_t(size_t)>& gen) {
    std::cout << "  halo0_even_doubled  (n=" << n << ")\n" << std::flush;
    const std::string in_prefix = "flatmap_in";
    const std::string out_prefix = "flatmap_out";

    chunk_seq input = ChunkSequenceOps::tabulate<uint64_t>(n, in_prefix, gen);
    std::atomic<bool> saw_bad_halo{false};
    chunk_seq matches = ChunkSequenceOps::ChunkFlatMap<uint64_t, uint64_t>(
        input, out_prefix, /*halo=*/0,
        [&](const uint64_t* data, size_t cnt, uint64_t /*gpos*/,
            const uint64_t* halo, size_t halo_n) {
            if (halo != nullptr || halo_n != 0) saw_bad_halo.store(true);
            parlay::sequence<uint64_t> out;
            for (size_t i = 0; i < cnt; i++)
                if (data[i] % 2 == 0) out.push_back(data[i] * 2);
            return out;
        });

    std::vector<uint64_t> expected;
    for (size_t i = 0; i < n; i++)
        if (gen(i) % 2 == 0) expected.push_back(gen(i) * 2);

    bool pass = verify("halo0_even_doubled", matches, expected);
    if (saw_bad_halo.load()) {
        std::cout << "    FAIL: body saw non-null/non-empty halo with halo==0\n";
        pass = false;
    } else {
        std::cout << "    halo-null OK\n";
    }

    cleanup_prefix(in_prefix);
    cleanup_prefix(out_prefix);
    return pass;
}

// ── Test 2/3: halo == 1, a 2-element "pattern" search over uint64_t ──────────
// The body reports every global position i (starting in its own chunk) where
// text[i]==P0 && text[i+1]==P1; the second element may live in the forward
// halo.  Also asserts the final chunk's halo is empty (halo_n == 0).  Planting
// a match across a chunk boundary exercises the in-batch neighbor; planting one
// across the 127/128 batch seam exercises the synchronous seam read.
static constexpr uint64_t P0 = 100, P1 = 101;

static bool test_halo1(const std::string& name, size_t n,
                       const std::function<uint64_t(size_t)>& gen) {
    std::cout << "  " << name << "  (n=" << n << ")\n" << std::flush;
    const std::string in_prefix = "flatmap_in";
    const std::string out_prefix = "flatmap_out";

    chunk_seq input = ChunkSequenceOps::tabulate<uint64_t>(n, in_prefix, gen);
    std::atomic<bool> bad_final{false};
    chunk_seq matches = ChunkSequenceOps::ChunkFlatMap<uint64_t, uint64_t>(
        input, out_prefix, /*halo=*/1,
        [&](const uint64_t* data, size_t cnt, uint64_t gpos,
            const uint64_t* halo, size_t halo_n) {
            // The very last chunk of the whole sequence must get an empty halo.
            if (gpos + cnt == n && halo_n != 0) bad_final.store(true);
            const size_t avail = cnt + halo_n;
            auto at = [&](size_t i) { return i < cnt ? data[i] : halo[i - cnt]; };
            parlay::sequence<uint64_t> out;
            for (size_t i = 0; i + 1 < avail && i < cnt; i++)
                if (at(i) == P0 && at(i + 1) == P1) out.push_back(gpos + i);
            return out;
        });

    std::vector<uint64_t> expected;
    for (size_t i = 0; i + 1 < n; i++)
        if (gen(i) == P0 && gen(i + 1) == P1) expected.push_back(i);

    bool pass = verify(name, matches, expected);
    if (bad_final.load()) {
        std::cout << "    FAIL: final chunk received a non-empty halo\n";
        pass = false;
    } else {
        std::cout << "    final-halo-empty OK\n";
    }

    cleanup_prefix(in_prefix);
    cleanup_prefix(out_prefix);
    return pass;
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    bool all_pass = true;

    // Deterministic small-range noise (always < P0, so never a spurious match).
    auto noise = [](size_t i) -> uint64_t { return parlay::hash64(i) % 4; };

    // 1. halo == 0: variable-length flat-map with a partial last chunk.
    all_pass &= test_halo0(2 * EPCT + 5, noise);

    // 2. halo == 1, in-batch: matches planted inside a chunk and straddling the
    //    chunk-0/1 and chunk-1/2 boundaries (< DENSE_PACK_BATCH_SIZE chunks).
    {
        const size_t n = 3 * EPCT + 7;
        auto gen = [&](size_t i) -> uint64_t {
            if (i == EPCT / 2)     return P0;   // fully inside chunk 0
            if (i == EPCT / 2 + 1) return P1;
            if (i == EPCT - 1)     return P0;   // straddles chunk 0/1
            if (i == EPCT)         return P1;
            if (i == 2 * EPCT - 1) return P0;   // straddles chunk 1/2
            if (i == 2 * EPCT)     return P1;
            return noise(i);
        };
        all_pass &= test_halo1("halo1_inbatch", n, gen);
    }

    // 3. halo == 1, batch seam: > DENSE_PACK_BATCH_SIZE chunks with a match
    //    planted across the chunk-127/128 boundary, exercising the O_DIRECT
    //    seam read.  argv[1] overrides n (min 2 chunks).
    {
        const size_t seam = ChunkSequenceOps::DENSE_PACK_BATCH_SIZE;  // 128
        size_t n = (argc > 1) ? std::stoull(argv[1]) : (seam + 1) * EPCT + 9;
        n = std::max(n, 2 * EPCT);
        const size_t last_of_batch = std::min<size_t>(seam, n / EPCT) - 1;
        const size_t plant = (last_of_batch + 1) * EPCT - 1;  // P0 at seam tail
        auto gen = [&](size_t i) -> uint64_t {
            if (i == plant)     return P0;   // last elem of chunk `last_of_batch`
            if (i == plant + 1) return P1;   // first elem of next chunk (seam)
            return noise(i);
        };
        all_pass &= test_halo1("halo1_seam", n, gen);
    }

    std::cout << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
    return all_pass ? 0 : 1;
}
