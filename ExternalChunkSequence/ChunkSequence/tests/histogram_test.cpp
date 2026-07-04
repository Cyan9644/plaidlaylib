#include <iostream>
#include <iomanip>
#include <cstdint>

#include "parlay/primitives.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_map.h"
#include "ChunkSequence/ExternalPrimitives/external_histogram_by_index.h"

static bool report(const std::string& name, bool ok) {
    std::cout << "  " << std::left << std::setw(40) << name
              << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 2'000'000ULL;

    std::cout << "Building perm(" << n << ")...\n" << std::flush;
    const chunk_seq perm = ChunkSequenceOps::perm(n);
    std::cout << perm.chunks.size() << " chunks across "
              << GetSSDList().size() << " drives\n\n";

    bool all_pass = true;

    // 1. Histogram of perm(n) with num_unique = n: every value 0..n-1 appears
    //    exactly once, so every bucket must be 1.
    {
        auto h = ChunkSequenceOps::histogram_by_index<uint64_t>(perm, n);
        bool ok = (h.size() == n);
        for (size_t b = 0; ok && b < n; b++) ok = (h[b] == 1);
        all_pass &= report("perm(n): all buckets == 1", ok);
    }

    // 2. Histogram of (x % k): bucket b gets ceil((n-b)/k) elements.  For k that
    //    divides n, that's exactly n/k per bucket.
    {
        const size_t k = 10;
        chunk_seq mod = ChunkSequenceOps::ChunkMap<uint64_t>(
            perm, "hist_mod", [k](uint64_t x) { return x % k; });
        auto h = ChunkSequenceOps::histogram_by_index<uint64_t>(mod, k);
        bool ok = (h.size() == k);
        size_t sum = 0;
        for (size_t b = 0; b < k; b++) {
            const size_t expected = (n - b + k - 1) / k;  // count of x<n with x%k==b
            if (h[b] != expected) ok = false;
            sum += h[b];
        }
        ok = ok && (sum == n);  // counts must total n
        all_pass &= report("(x % 10): per-bucket counts + total", ok);
    }

    std::cout << "\n" << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
    return all_pass ? 0 : 1;
}
