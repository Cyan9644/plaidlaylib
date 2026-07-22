#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include "parlay/primitives.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_segmented_reduce.h"

// ── monoids ──────────────────────────────────────────────────────────────────

struct SumMonoid {
    uint64_t identity = 0;
    uint64_t operator()(uint64_t a, uint64_t b) const { return a + b; }
};

struct MinMonoid {
    uint64_t identity = UINT64_MAX;
    uint64_t operator()(uint64_t a, uint64_t b) const { return std::min(a, b); }
};

struct MaxMonoid {
    uint64_t identity = 0;
    uint64_t operator()(uint64_t a, uint64_t b) const { return std::max(a, b); }
};

// ── helpers ──────────────────────────────────────────────────────────────────

// Cycle a small pattern of segment sizes (including 0, sub-chunk, and
// multi-chunk-spanning sizes relative to `ept`) until it covers exactly `n`
// elements, so the resulting bounds exercise: segments fully inside one
// chunk, segments that straddle a chunk seam, an empty segment, and a "mega"
// segment spanning several whole chunks (the chained-boundary-merge path).
static std::vector<size_t> make_bounds(size_t n, size_t ept) {
    const std::vector<size_t> pattern = {
        0, 1, ept / 2, 2 * ept + 137, ept - 100, 3,
        ept + 1, 0, 12345, ept * 3 + 7, 1, ept / 3 + 9
    };
    std::vector<size_t> bounds{0};
    size_t total = 0, p = 0;
    while (total < n) {
        size_t s = pattern[p % pattern.size()];
        if (total + s > n) s = n - total;
        total += s;
        bounds.push_back(total);
        p++;
    }
    return bounds;
}

// Verify a ChunkSegmentedReduce result against a per-segment closed form
// expected_fn(lo, hi) -> R, reporting up to 5 mismatches.
template<typename R, typename ExpectedFn>
static bool verify(const std::string& name, const parlay::sequence<R>& got,
                   const std::vector<size_t>& bounds, ExpectedFn expected_fn) {
    const size_t num_segments = bounds.size() - 1;
    size_t fails = 0;
    for (size_t v = 0; v < num_segments; v++) {
        const R expected = expected_fn(bounds[v], bounds[v + 1]);
        if (got[v] != expected) {
            if (fails < 5)
                std::cerr << "  FAIL segment " << v << " [" << bounds[v] << ","
                          << bounds[v + 1] << "): got " << (uint64_t)got[v]
                          << " expected " << (uint64_t)expected << "\n";
            fails++;
        }
    }
    std::cout << "  " << std::left << std::setw(32) << name
              << (fails == 0 ? "PASS" : "FAIL")
              << "  segments=" << num_segments << "  failed=" << fails << "\n";
    return fails == 0;
}

// Closed forms for reducing elem_to_val(x) = x over iota(n)'s values [lo,hi).
static uint64_t expected_sum(size_t lo, size_t hi) {
    return lo >= hi ? 0 : (uint64_t)(lo + hi - 1) * (hi - lo) / 2;
}
static uint64_t expected_min(size_t lo, size_t hi) {
    return lo < hi ? (uint64_t)lo : UINT64_MAX;
}
static uint64_t expected_max(size_t lo, size_t hi) {
    return lo < hi ? (uint64_t)(hi - 1) : 0ULL;
}

// A 32-byte record type (sizeof(T) != sizeof(uint64_t)) so this test also
// covers ChunkSegmentedReduce addressing chunks by CHUNK_SIZE/sizeof(T)
// rather than the global (uint64_t-sized) ELEMS_PER_CHUNK -- the exact bug
// class previously seen in the delayed::cut path for 32-byte elements.
struct rec32 {
    uint64_t v;
    unsigned char pad[24];
};
static_assert(sizeof(rec32) == 32, "rec32 must be 32 bytes");

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);

    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 5'000'000ULL;
    bool all_pass = true;

    {
        std::cout << "uint64_t elements, n=" << n << " (ELEMS_PER_CHUNK="
                  << ELEMS_PER_CHUNK << ")\n" << std::flush;
        const chunk_seq input = ChunkSequenceOps::iota(n);
        const auto bounds_v = make_bounds(n, ELEMS_PER_CHUNK);
        const parlay::sequence<size_t> bounds(bounds_v.begin(), bounds_v.end());
        auto id = [](uint64_t x) { return x; };

        all_pass &= verify<uint64_t>("sum",
            ChunkSequenceOps::ChunkSegmentedReduce<uint64_t, uint64_t>(input, bounds, id, SumMonoid{}),
            bounds_v, expected_sum);
        all_pass &= verify<uint64_t>("min",
            ChunkSequenceOps::ChunkSegmentedReduce<uint64_t, uint64_t>(input, bounds, id, MinMonoid{}),
            bounds_v, expected_min);
        all_pass &= verify<uint64_t>("max",
            ChunkSequenceOps::ChunkSegmentedReduce<uint64_t, uint64_t>(input, bounds, id, MaxMonoid{}),
            bounds_v, expected_max);
    }

    {
        const size_t n2 = std::min(n, (size_t) 2'000'000ULL);
        const size_t ept2 = CHUNK_SIZE / sizeof(rec32);
        std::cout << "\n32-byte elements, n=" << n2 << " (elems_per_chunk="
                  << ept2 << ", != ELEMS_PER_CHUNK=" << ELEMS_PER_CHUNK << ")\n"
                  << std::flush;
        const chunk_seq input = ChunkSequenceOps::tabulate<rec32>(
            n2, "segreduce_rec32", [](size_t i) { rec32 r; r.v = i; return r; });
        const auto bounds_v = make_bounds(n2, ept2);
        const parlay::sequence<size_t> bounds(bounds_v.begin(), bounds_v.end());
        auto id = [](const rec32& r) { return r.v; };

        all_pass &= verify<uint64_t>("sum (32B elems)",
            ChunkSequenceOps::ChunkSegmentedReduce<rec32, uint64_t>(input, bounds, id, SumMonoid{}),
            bounds_v, expected_sum);
        all_pass &= verify<uint64_t>("min (32B elems)",
            ChunkSequenceOps::ChunkSegmentedReduce<rec32, uint64_t>(input, bounds, id, MinMonoid{}),
            bounds_v, expected_min);
        all_pass &= verify<uint64_t>("max (32B elems)",
            ChunkSequenceOps::ChunkSegmentedReduce<rec32, uint64_t>(input, bounds, id, MaxMonoid{}),
            bounds_v, expected_max);

        for (const auto& c : input.chunks) unlink(c.filename.c_str());
    }

    std::cout << "\n" << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
    return all_pass ? 0 : 1;
}
