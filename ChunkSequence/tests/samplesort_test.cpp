// Correctness test for ChunkSequenceOps::sample_sort (external_samplesort.h)
// at a bucket count and scatter/gather churn level that exercises the
// many-buckets regime real sweeps never reach cheaply.
//
// external_samplesort's benchmark drivers derive the pivot count from a
// hardcoded ~128 MiB target-bucket-size, so reaching thousands of buckets (as
// happens at n=2^38 in production) normally needs multi-TB of input. This
// test instead overrides the target bucket size AND the chunk size (both
// build-time knobs -- see configs.h's SS_TARGET_BUCKET_BYTES / CHUNK_SIZE_BYTES)
// down to values that reach hundreds of buckets, with real scatter/gather
// churn (SS_TARGET_BUCKET_BYTES/(8*num_workers) elements land in each
// (worker,bucket) buffer -- enough to fill and flush it many times over),
// at a data size (tens of MB) that runs in seconds.
//
// This is the regression test for the scatter-phase data race fixed in
// bucketed_file_writer.h: BucketWriter's I/O threads (plain std::thread,
// deliberately outside the parlay pool) used to free() scatter buffers
// through parlay::internal::block_allocator, whose free()/alloc() key a
// per-thread free list by parlay::worker_id() -- a thread_local that
// silently defaults to 0 on any thread the parlay scheduler never assigned
// an id to. The I/O threads therefore aliased real worker 0's free list,
// corrupting it under concurrent access once enough alloc/free churn hit a
// bad interleaving. Confirmed locally: the config below reliably segfaulted
// before the fix (a plain parlay::internal::block_allocator-backed
// bucket_allocator) and passes reliably after (a hazptr_stack-backed one).
//
// Exits 0 iff the out-of-core output matches parlay::sort element-for-element.

// Must precede any transitive include of configs.h: both macros are
// #ifndef-guarded there, so defining them first overrides the real ~128 MiB /
// 4 MiB production defaults for this translation unit only.
#define SS_TARGET_BUCKET_BYTES (1UL << 21)   // 2 MiB (vs. the real ~128 MiB)
#define CHUNK_SIZE_BYTES (1UL << 16)         // 64 KiB (vs. the real 4 MiB)

#include <cstdint>
#include <iostream>
#include <string>
#include <unistd.h>

#include "parlay/primitives.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/ExternalPrimitives/materialize.h"
#include "ChunkSequence/examples/external/external_samplesort.h"

static void cleanup_prefixes() {
    static const char* kPrefixes[] = {"sst_in", "ss_id_", "ss_bucket_", "ss_base_", "ss_deg_"};
    const auto& ssds = GetSSDList();
    for (const char* prefix : kPrefixes)
        for (size_t d = 0; d < ssds.size(); d++) unlink(GetFileName(prefix, d).c_str());
}

static uint64_t key_at(size_t i) { return parlay::hash64(i); }

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    RaiseFdLimit();   // thousands of buckets means thousands of open files
    // ~480 MB of input: SS_TARGET_BUCKET_BYTES/CHUNK_SIZE_BYTES = 32, so this
    // lands a couple hundred buckets, each spanning many chunks -- enough
    // scatter/gather traffic to exercise the allocator under real churn.
    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 60'000'000;

    const std::string in_prefix = "sst_in";
    chunk_seq seq = ChunkSequenceOps::tabulate<uint64_t>(n, in_prefix, key_at);

    chunk_seq sorted = ChunkSequenceOps::sample_sort<uint64_t>(seq);
    auto got = ChunkSequenceOps::materialize<uint64_t>(sorted);

    auto ref = parlay::tabulate(n, key_at);
    parlay::sort_inplace(ref);

    bool pass = true;
    if (got.size() != ref.size()) {
        std::cout << "  FAIL: got " << got.size() << " keys, expected " << ref.size() << "\n";
        pass = false;
    } else {
        for (size_t i = 0; i < ref.size(); i++) {
            if (got[i] != ref[i]) {
                std::cout << "  FAIL at index " << i << ": got " << got[i]
                          << " != expected " << ref[i] << "\n";
                pass = false;
                break;
            }
        }
    }
    if (pass) std::cout << "  OK: " << n << " keys sorted correctly\n";

    cleanup_prefixes();

    std::cout << (pass ? "PASS" : "FAIL") << "\n";
    return pass ? 0 : 1;
}
