#include <iostream>
#include <set>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <functional>

#include "absl/log/check.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_reduce.h"
#include "ChunkSequence/chunk_filter.h"

struct SumMonoid {
    uint64_t identity = 0;
    uint64_t operator()(uint64_t a, uint64_t b) const { return a + b; }
};

// Remove all per-drive files created under a given prefix (one file per drive).
// Used for both perm input files and filter output files.
static void cleanup_prefix(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++)
        unlink(GetFileName(prefix, d).c_str());
}

// Builds perm(n), applies ChunkFilter, consolidates the survivor stream in index
// order to a local file, and verifies every element equals expected_at(j) — i.e.
// that filter PRESERVES global element order across batch boundaries (which the
// order-insensitive sum check in run_filter_test cannot catch).
static bool run_order_test(
    const std::string& name,
    size_t n,
    std::function<bool(uint64_t)> pred,
    std::function<uint64_t(size_t)> expected_at,
    size_t expected_count)
{
    std::cout << "  " << name
              << "  (n=" << n << ", expected=" << expected_count << ")\n" << std::flush;

    chunk_seq seq = ChunkSequenceOps::perm(n);

    const std::string out_prefix    = "filter_test_out";
    const std::string consolidated  = "filter_test_order_consolidated";
    chunk_seq filtered = ChunkSequenceOps::ChunkFilter<uint64_t>(seq, out_prefix, pred);

    bool pass = true;

    // Count check up front (consolidate writes exactly the survivor stream).
    size_t actual_count = 0;
    for (const auto& c : filtered.chunks)
        actual_count += c.used / sizeof(uint64_t);
    if (actual_count != expected_count) {
        std::cout << "    FAIL count: got=" << actual_count
                  << " expected=" << expected_count << "\n";
        pass = false;
    } else {
        std::cout << "    count  OK\n";
    }

    // Write survivors to a local file in index order, then read back sequentially
    // and compare each element to the expected in-order value.
    filtered.consolidate(consolidated);

    int fd = open(consolidated.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cout << "    FAIL open(" << consolidated << "): " << strerror(errno) << "\n";
        pass = false;
    } else {
        constexpr size_t BUF_ELEMS = (1 << 20);  // 8 MiB worth of uint64_t per read
        std::vector<uint64_t> buf(BUF_ELEMS);
        size_t j = 0;
        bool order_ok = true;
        while (order_ok) {
            const ssize_t got = read(fd, buf.data(), BUF_ELEMS * sizeof(uint64_t));
            if (got < 0) {
                std::cout << "    FAIL read: " << strerror(errno) << "\n";
                pass = order_ok = false;
                break;
            }
            if (got == 0) break;  // EOF
            const size_t count = (size_t)got / sizeof(uint64_t);
            for (size_t i = 0; i < count; i++, j++) {
                const uint64_t expected = expected_at(j);
                if (buf[i] != expected) {
                    std::cout << "    FAIL order: element " << j
                              << " got " << buf[i] << " expected " << expected << "\n";
                    pass = order_ok = false;
                    break;
                }
            }
        }
        close(fd);
        if (order_ok && j != expected_count) {
            std::cout << "    FAIL order: read " << j
                      << " elements, expected " << expected_count << "\n";
            pass = false;
        } else if (order_ok) {
            std::cout << "    order  OK\n";
        }
    }

    std::cout << "    => " << (pass ? "PASS" : "FAIL") << "\n\n";

    cleanup_prefix("perm");
    cleanup_prefix(out_prefix);
    unlink(consolidated.c_str());

    return pass;
}

// Builds perm(n), applies ChunkFilter with pred, verifies count / packing /
// index-order / sum, cleans up everything, and returns true iff all checks pass.
//
// expected_chunks: if >= 0, also verifies the exact number of output chunks.
static bool run_filter_test(
    const std::string& name,
    size_t n,
    std::function<bool(uint64_t)> pred,
    size_t expected_count,
    uint64_t expected_sum,
    int expected_chunks = -1)
{
    std::cout << "  " << name
              << "  (n=" << n << ", expected=" << expected_count << ")\n" << std::flush;

    chunk_seq seq = ChunkSequenceOps::perm(n);

    const std::string out_prefix = "filter_test_out";
    chunk_seq filtered = ChunkSequenceOps::ChunkFilter<uint64_t>(seq, out_prefix, pred);

    bool pass = true;

    // 1. Element count.
    {
        size_t actual = 0;
        for (const auto& c : filtered.chunks)
            actual += c.used / sizeof(uint64_t);
        if (actual != expected_count) {
            std::cout << "    FAIL count: got=" << actual
                      << " expected=" << expected_count << "\n";
            pass = false;
        } else {
            std::cout << "    count  OK\n";
        }
    }

    // 2. Tight packing: all chunks except the last must be full.
    {
        bool ok = true;
        for (size_t i = 0; i + 1 < filtered.chunks.size() && ok; i++) {
            if (filtered.chunks[i].used != CHUNK_SIZE) {
                std::cout << "    FAIL packing: chunk " << i
                          << " used=" << filtered.chunks[i].used
                          << " (expected CHUNK_SIZE=" << CHUNK_SIZE << ")\n";
                pass = ok = false;
            }
        }
        if (ok)
            std::cout << "    packing OK\n";
    }

    // 3. Index-ordered invariant: chunks[i].index == i.
    {
        bool ok = true;
        for (size_t i = 0; i < filtered.chunks.size() && ok; i++) {
            if (filtered.chunks[i].index != i) {
                std::cout << "    FAIL index order: chunks[" << i
                          << "].index=" << filtered.chunks[i].index << "\n";
                pass = ok = false;
            }
        }
        if (ok)
            std::cout << "    index  OK\n";
    }

    // 4. Optional exact chunk count.
    if (expected_chunks >= 0) {
        const size_t ec = static_cast<size_t>(expected_chunks);
        if (filtered.chunks.size() != ec) {
            std::cout << "    FAIL #chunks: got=" << filtered.chunks.size()
                      << " expected=" << ec << "\n";
            pass = false;
        } else {
            std::cout << "    #chunks OK (" << ec << ")\n";
        }
    }

    // 5. Sum via ChunkReduce (skip for empty output to avoid reducing empty seq).
    if (expected_count > 0) {
        const uint64_t actual_sum =
            ChunkSequenceOps::ChunkReduce<uint64_t>(filtered, SumMonoid{});
        if (actual_sum != expected_sum) {
            std::cout << "    FAIL sum: got=" << actual_sum
                      << " expected=" << expected_sum << "\n";
            pass = false;
        } else {
            std::cout << "    sum    OK\n";
        }
    }

    std::cout << "    => " << (pass ? "PASS" : "FAIL") << "\n\n";

    // Always clean up so the next sub-test starts with a clean slate.
    cleanup_prefix("perm");
    cleanup_prefix(out_prefix);

    return pass;
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);

    bool all_pass = true;

    // ── 1. No survivors ──────────────────────────────────────────────────────
    // pred always false → empty output chunk_seq; output files exist but are empty.
    all_pass &= run_filter_test(
        "all_fail",
        ELEMS_PER_CHUNK,
        [](uint64_t) { return false; },
        /*expected_count=*/0,
        /*expected_sum=*/0,
        /*expected_chunks=*/0);

    // ── 2. All survivors, exactly one full output chunk ──────────────────────
    // n == ELEMS_PER_CHUNK → no carry.
    {
        const size_t n = ELEMS_PER_CHUNK;
        all_pass &= run_filter_test(
            "all_pass_exact_chunk",
            n,
            [](uint64_t) { return true; },
            /*expected_count=*/n,
            /*expected_sum=*/(uint64_t)(n - 1) * n / 2,
            /*expected_chunks=*/1);
    }

    // ── 3. Single survivor ───────────────────────────────────────────────────
    // Extreme sparsity: only element 0 passes.
    // Output: 1 partial chunk with used = sizeof(uint64_t) = 8 bytes.
    all_pass &= run_filter_test(
        "single_survivor",
        ELEMS_PER_CHUNK,
        [](uint64_t x) { return x == 0; },
        /*expected_count=*/1,
        /*expected_sum=*/0,
        /*expected_chunks=*/1);

    // ── 4. Partial last input chunk, all-pass ────────────────────────────────
    // n is not a multiple of ELEMS_PER_CHUNK: the second (last) input chunk
    // contains only 7 elements.  All-pass → output: 1 full + 1 partial (7 elems).
    {
        const size_t n = ELEMS_PER_CHUNK + 7;
        all_pass &= run_filter_test(
            "partial_input_all_pass",
            n,
            [](uint64_t) { return true; },
            /*expected_count=*/n,
            /*expected_sum=*/(uint64_t)(n - 1) * n / 2,
            /*expected_chunks=*/2);
    }

    // ── 5. Leftover carry at end ─────────────────────────────────────────────
    // 3 full input chunks (fits in one batch), x%2==0 keeps exactly half.
    // 3*ELEMS_PER_CHUNK/2 = 1.5*ELEMS_PER_CHUNK survivors
    // → 1 full output chunk + 1 partial (ELEMS_PER_CHUNK/2 elements).
    {
        const size_t n = 3 * ELEMS_PER_CHUNK;
        const size_t expected = n / 2;
        // sum of 0, 2, 4, …, n-2  =  expected*(expected-1)
        all_pass &= run_filter_test(
            "leftover_carry",
            n,
            [](uint64_t x) { return x % 2 == 0; },
            expected,
            /*expected_sum=*/(uint64_t)(expected - 1) * expected,
            /*expected_chunks=*/2);
    }

    // ── 6. Carry propagates across a batch boundary ──────────────────────────
    // 129 input chunks → 2 batches (128 + 1).
    // x%3==0: batch-1 survivors (22,369,622) % ELEMS_PER_CHUNK = 349,526 ≠ 0,
    // so a non-zero carry flows from batch 1 into batch 2.
    // 129 = 3×43, so 129*ELEMS_PER_CHUNK is divisible by 3 → m = n/3 is exact.
    // sum of 0, 3, 6, … = 3*(m*(m-1)/2).
    {
        const size_t n = 129 * ELEMS_PER_CHUNK;
        const uint64_t m = n / 3;
        all_pass &= run_filter_test(
            "cross_batch_carry",
            n,
            [](uint64_t x) { return x % 3 == 0; },
            /*expected_count=*/m,
            /*expected_sum=*/3ULL * (m * (m - 1) / 2));
    }

    // ── 7. Two-batch dense filter (original test) ────────────────────────────
    // 160 chunks across 2 batches, x%2==0 → exactly 80*ELEMS_PER_CHUNK survivors,
    // no carry.  Accepts argv[1] to override n (forced even).
    {
        const size_t n_raw =
            (argc > 1) ? std::stoull(argv[1]) : 160ULL * ELEMS_PER_CHUNK;
        const size_t n = n_raw & ~size_t(1);
        const size_t expected = n / 2;
        all_pass &= run_filter_test(
            "even_dense_2batch",
            n,
            [](uint64_t x) { return x % 2 == 0; },
            expected,
            /*expected_sum=*/(uint64_t)(expected - 1) * expected);
    }

    // ── 8. Order preservation across batch boundaries ────────────────────────
    // 256 input chunks = 2 full batches.  x%2==0 keeps exactly half; for perm the
    // in-order survivors are 0,2,4,… so element j must equal 2*j.  Any cross-batch
    // reordering (chunks arriving out of completion order) breaks this.
    {
        const size_t n = 256 * ELEMS_PER_CHUNK;
        all_pass &= run_order_test(
            "order_cross_batch",
            n,
            [](uint64_t x) { return x % 2 == 0; },
            [](size_t j) -> uint64_t { return 2ULL * j; },
            /*expected_count=*/n / 2);
    }

    std::cout << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
    return all_pass ? 0 : 1;
}
