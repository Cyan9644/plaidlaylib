// Correctness test for ChunkPartition (chunk_partition.h).
//
// Builds a uint64_t sequence 0..n-1, partitions it into k buckets by a key
// function that also drops some elements, and verifies:
//   1. every element in bucket b really has key == b (and was not a drop),
//   2. every kept input value appears exactly once across the buckets and every
//      dropped value appears nowhere (union == input minus drops, no dupes),
//   3. each bucket is a valid chunk_seq: index-ordered and dense-except-last.
//
// Exits 0 iff all checks pass.

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

#include "parlay/primitives.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_partition.h"

static constexpr size_t U64_PER_CHUNK = CHUNK_SIZE / sizeof(uint64_t);

static void cleanup_prefix(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++) unlink(GetFileName(prefix, d).c_str());
}

// bucket for value v: drop multiples of 13, else v % k.
static size_t key_of(uint64_t v, size_t k) {
    if (v % 13 == 0) return ChunkSequenceOps::PARTITION_DROP;
    return (size_t)(v % k);
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 500'000;
    const size_t k = 4;

    const std::string in_prefix  = "pt_in";
    const std::string out_prefix = "pt_out";

    chunk_seq seq = ChunkSequenceOps::tabulate<uint64_t>(n, in_prefix,
        [](size_t i) { return (uint64_t)i; });

    std::vector<chunk_seq> parts = ChunkSequenceOps::ChunkPartition<uint64_t>(
        seq, k, out_prefix, [k](uint64_t v) { return key_of(v, k); });

    bool pass = true;

    if (parts.size() != k) {
        std::cout << "  FAIL: got " << parts.size() << " buckets, expected " << k << "\n";
        pass = false;
    }

    std::vector<char> seen(n, 0);   // which input values were returned
    size_t kept = 0;

    for (size_t b = 0; b < parts.size() && pass; b++) {
        const chunk_seq& bucket = parts[b];

        // Index-ordered + dense-except-last.
        for (size_t i = 0; i < bucket.chunks.size(); i++) {
            if (bucket.chunks[i].index != i) {
                std::cout << "  FAIL bucket " << b << ": chunk " << i
                          << " has index " << bucket.chunks[i].index << "\n";
                pass = false; break;
            }
            if (i + 1 < bucket.chunks.size() && bucket.chunks[i].used != CHUNK_SIZE) {
                std::cout << "  FAIL bucket " << b << ": non-last chunk " << i
                          << " used=" << bucket.chunks[i].used << " (not full)\n";
                pass = false; break;
            }
        }
        if (!pass) break;

        // Contents: every value routes to this bucket and is not a drop.
        std::vector<uint64_t> vals = bucket.to_vector<uint64_t>();
        for (uint64_t v : vals) {
            if (v >= n) {
                std::cout << "  FAIL bucket " << b << ": value " << v << " >= n\n";
                pass = false; break;
            }
            if (key_of(v, k) != b) {
                std::cout << "  FAIL bucket " << b << ": value " << v
                          << " has key " << key_of(v, k) << "\n";
                pass = false; break;
            }
            if (seen[v]) {
                std::cout << "  FAIL: value " << v << " appears more than once\n";
                pass = false; break;
            }
            seen[v] = 1;
            kept++;
        }
    }

    // Every kept input present exactly once; every dropped input absent.
    if (pass) {
        size_t expected_kept = 0;
        for (size_t v = 0; v < n; v++) {
            const bool dropped = (v % 13 == 0);
            if (!dropped) expected_kept++;
            if (seen[v] == (dropped ? 1 : 0)) {
                std::cout << "  FAIL: value " << v << (dropped ? " dropped but present"
                                                              : " kept but missing") << "\n";
                pass = false; break;
            }
        }
        if (pass && kept != expected_kept) {
            std::cout << "  FAIL: kept " << kept << " != expected " << expected_kept << "\n";
            pass = false;
        }
        if (pass) std::cout << "  OK: " << kept << " kept across " << k
                            << " buckets, drops absent, packing valid\n";
    }

    cleanup_prefix(in_prefix);
    cleanup_prefix(out_prefix);

    std::cout << (pass ? "PASS" : "FAIL") << "\n";
    return pass ? 0 : 1;
}
