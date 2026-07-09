#include <iostream>
#include <iomanip>
#include <cstdint>
#include <algorithm>

#include "parlay/primitives.h"
#include "absl/log/check.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_reduce.h"

// ── monoids ──────────────────────────────────────────────────────────────────

struct SumMonoid {
    uint64_t identity = 0;
    uint64_t operator()(uint64_t a, uint64_t b) const { return a + b; }
};

struct MaxMonoid {
    // iota(n) starts at 0, so 0 is the correct min-identity for max.
    uint64_t identity = 0;
    uint64_t operator()(uint64_t a, uint64_t b) const { return std::max(a, b); }
};

struct MinMonoid {
    uint64_t identity = UINT64_MAX;
    uint64_t operator()(uint64_t a, uint64_t b) const { return std::min(a, b); }
};

struct XorMonoid {
    uint64_t identity = 0;
    uint64_t operator()(uint64_t a, uint64_t b) const { return a ^ b; }
};

// ── helpers ──────────────────────────────────────────────────────────────────

// XOR(0 ^ 1 ^ … ^ k) using the 4-cycle closed form.
static uint64_t xor_prefix(uint64_t k) {
    switch (k % 4) {
        case 0: return k;
        case 1: return 1;
        case 2: return k + 1;
        default: return 0;
    }
}

static bool report(const std::string& name, uint64_t got, uint64_t expected) {
    const bool ok = (got == expected);
    std::cout << "  " << std::left << std::setw(32) << name
              << (ok ? "PASS" : "FAIL")
              << "  got=" << got << " expected=" << expected << "\n";
    return ok;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);

    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 5'000'000ULL;

    std::cout << "Building iota(" << n << ")...\n" << std::flush;
    const chunk_seq input = ChunkSequenceOps::iota(n);
    std::cout << input.chunks.size() << " chunks across "
              << GetSSDList().size() << " drives\n\n";

    bool all_pass = true;

    // sum of 0+1+…+(n-1) = n*(n-1)/2
    all_pass &= report("sum  0+1+…+(n-1)",
                       ChunkSequenceOps::ChunkReduce<uint64_t>(input, SumMonoid{}),
                       (uint64_t)(n - 1) * n / 2);

    // max element of iota(n) = n-1
    all_pass &= report("max  element",
                       ChunkSequenceOps::ChunkReduce<uint64_t>(input, MaxMonoid{}),
                       (uint64_t)(n - 1));

    // min element of iota(n) = 0
    all_pass &= report("min  element",
                       ChunkSequenceOps::ChunkReduce<uint64_t>(input, MinMonoid{}),
                       0ULL);

    // XOR of 0^1^…^(n-1), computed via closed-form prefix formula
    all_pass &= report("xor  0^1^…^(n-1)",
                       ChunkSequenceOps::ChunkReduce<uint64_t>(input, XorMonoid{}),
                       xor_prefix((uint64_t)(n - 1)));

    std::cout << "\n" << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
    return all_pass ? 0 : 1;
}
