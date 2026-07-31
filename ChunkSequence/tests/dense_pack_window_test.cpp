// Stress test for DensePackStream's worker/packer backpressure gate
// (dense_pack.h, DENSE_PACK_STREAM_WINDOW_CHUNKS).  Built with a tiny window
// override (see Makefile) so the gate binds on nearly every chunk instead of
// only under real I/O contention, exercising it far more than a default-
// window run would.  Reuses ChunkFilter (a thin producer on DensePackStream)
// with the same count/packing/index-order/sum/order checks as filter_test.cpp
// — a clean pass here is direct evidence the gate doesn't corrupt output or
// deadlock even when constantly engaged.

#include <iostream>
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

static void cleanup_prefix(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++)
        unlink(GetFileName(prefix, d).c_str());
}

// Order preservation across many chunks: for iota with x%2==0, in-order
// survivors are 0,2,4,… so element j must equal 2*j.  With a 4-chunk window,
// this is a long, sustained stream of gate blocks/wakeups, not a one-off.
static bool run_order_test(size_t n) {
    std::cout << "  window_order  (n=" << n << ")\n" << std::flush;

    chunk_seq seq = ChunkSequenceOps::iota(n);

    const std::string out_prefix   = "dense_pack_window_test_out";
    const std::string consolidated = "dense_pack_window_test_consolidated";
    chunk_seq filtered = ChunkSequenceOps::ChunkFilter<uint64_t>(
        seq, out_prefix, [](uint64_t x) { return x % 2 == 0; });

    bool pass = true;
    const size_t expected_count = n / 2;

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

    for (size_t i = 0; i + 1 < filtered.chunks.size() && pass; i++) {
        if (filtered.chunks[i].used != CHUNK_SIZE) {
            std::cout << "    FAIL packing: chunk " << i
                      << " used=" << filtered.chunks[i].used << "\n";
            pass = false;
        }
    }
    for (size_t i = 0; i < filtered.chunks.size() && pass; i++) {
        if (filtered.chunks[i].index != i) {
            std::cout << "    FAIL index order: chunks[" << i
                      << "].index=" << filtered.chunks[i].index << "\n";
            pass = false;
        }
    }
    if (pass) std::cout << "    packing/index OK\n";

    if (expected_count > 0) {
        const uint64_t expected_sum = (uint64_t)(expected_count - 1) * expected_count;
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

    filtered.consolidate(consolidated);
    int fd = open(consolidated.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cout << "    FAIL open(" << consolidated << "): " << strerror(errno) << "\n";
        pass = false;
    } else {
        constexpr size_t BUF_ELEMS = (1 << 20);
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
            if (got == 0) break;
            const size_t count = (size_t)got / sizeof(uint64_t);
            for (size_t i = 0; i < count; i++, j++) {
                const uint64_t expected = 2ULL * j;
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

    cleanup_prefix("iota");
    cleanup_prefix(out_prefix);
    unlink(consolidated.c_str());

    return pass;
}

int main(int argc, char* argv[]) {
    // DensePackStream clamps its window to >= 8 * parlay::num_workers() (a
    // liveness safety net, see dense_pack.h), which would swamp the tiny
    // -DDENSE_PACK_STREAM_WINDOW_CHUNKS build override on a many-core box.
    // Cap the worker pool here (before any parlay call lazily starts its
    // scheduler) so the gate actually binds tightly, as intended by this
    // test; respects an explicit override if the caller already set one.
    setenv("PARLAY_NUM_THREADS", "4", /*overwrite=*/0);

    ParseGlobalArguments(argc, argv);

    bool all_pass = true;

    // 256 input chunks by default (matches filter_test.cpp's order_cross_batch
    // scale); overridable via argv[1] to push the gate harder.
    const size_t n_raw =
        (argc > 1) ? std::stoull(argv[1]) : 256ULL * ELEMS_PER_CHUNK;
    const size_t n = n_raw & ~size_t(1);

    all_pass &= run_order_test(n);

    std::cout << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
    return all_pass ? 0 : 1;
}
