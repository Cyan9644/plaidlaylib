// Correctness + placement test for direct_random_shuffle
// (examples/external/direct_random_shuffle.h) -- the shuffle counterpart to
// direct_samplesort_striped_test.cpp.
//
// Checks, at disk_span=1 (default) and disk_span=GetSSDList().size()):
//   1. the output is a valid permutation of the input: same element count,
//      and sorting it reproduces the sorted key set exactly (the keys are
//      distinct, so this catches a dropped, duplicated, or corrupted
//      element -- same check convention as external_random_shuffle.cpp and
//      chunk_operation_test.cpp's check_shuffle).
// and, on the placement side, that the striped run's output touches
// (strictly) more distinct drives overall than the plain run's -- same
// rationale as direct_samplesort_striped_test.cpp: with only a handful of
// buckets, GetBucketCount can produce unevenly sized buckets, so comparing
// whole-output drive counts (rather than assuming anything about individual
// buckets) avoids flakiness.
//
// Exits 0 iff all checks pass.

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "parlay/primitives.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/examples/external/bench_drives.h"
#include "ChunkSequence/examples/external/direct_random_shuffle.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"

using ChunkSequenceOps::direct_random_shuffle;

// Which drive a chunk's file lives on, by matching its GetFileName-derived
// path prefix against GetSSDList(); -1 if no drive matches (shouldn't happen).
static int drive_of(const std::string& filename) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++)
        if (filename.rfind(ssds[d] + "/", 0) == 0) return (int)d;
    return -1;
}

// Distinct drives touched anywhere in `seq`'s output.
static size_t distinct_drives(const chunk_seq& seq) {
    std::set<int> drives;
    for (const chunk& c : seq.chunks) drives.insert(drive_of(c.filename));
    return drives.size();
}

// Permutation check: sorting a valid shuffle of distinct keys must reproduce
// the sorted key set exactly.
static bool check_permutation(const std::string& name, chunk_seq& result,
                              const parlay::sequence<uint64_t>& ref) {
    std::vector<uint64_t> got = result.to_vector<uint64_t>();
    if (got.size() != ref.size()) {
        std::cout << "  FAIL " << name << ": got " << got.size()
                  << " elements, expected " << ref.size() << "\n";
        return false;
    }
    std::sort(got.begin(), got.end());
    for (size_t i = 0; i < got.size(); i++) {
        if (got[i] != ref[i]) {
            std::cout << "  FAIL " << name << ": sorted element " << i << " is "
                      << got[i] << ", expected " << ref[i] << "\n";
            return false;
        }
    }
    std::cout << "  OK " << name << ": " << got.size()
              << " elements, a valid permutation of the input\n";
    return true;
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 20'000'000;

    auto key_at = [](size_t i) { return parlay::hash64(i); };
    parlay::sequence<uint64_t> ref = parlay::tabulate(n, key_at);
    parlay::sort_inplace(ref);

    const size_t num_drives = GetSSDList().size();
    std::cout << "n=" << n << " drives=" << num_drives << "\n";

    bool pass = true;

    struct Case {
        const char* name;
        size_t disk_span;
        std::string prefix;
        size_t seed;
    };
    std::vector<Case> cases = {
        {"direct_random_shuffle (disk_span=1)",      1,          "drst_plain",   42},
        {"direct_random_shuffle (disk_span=drives)", num_drives, "drst_striped", 1234},
    };

    size_t plain_drives = 0, striped_drives = 0;

    for (size_t ci = 0; ci < cases.size(); ci++) {
        const Case& c = cases[ci];
        chunk_seq in = ChunkSequenceOps::tabulate<uint64_t>(n, c.prefix + "_in", key_at);
        chunk_seq out = direct_random_shuffle<uint64_t>(in, c.seed, c.prefix, c.disk_span);

        if (!check_permutation(c.name, out, ref)) pass = false;

        const size_t drives = distinct_drives(out);
        std::cout << "  .. " << c.name << ": output touches " << drives << " distinct drives\n";
        (ci == 0 ? plain_drives : striped_drives) = drives;

        // direct_random_shuffle derives all its filenames (scatter-phase temp
        // files and the final output files) from the single caller-supplied
        // prefix, same as direct_sample_sort, so sweeping just that prefix
        // (plus the input's, same string) is sufficient.
        bench_drives::clear_drives({c.prefix});
    }

    if (num_drives > 1) {
        if (striped_drives <= plain_drives) {
            std::cout << "  FAIL direct_random_shuffle: striped touched " << striped_drives
                      << " drives, plain touched " << plain_drives
                      << " -- expected striped strictly more\n";
            pass = false;
        } else {
            std::cout << "  OK direct_random_shuffle: striped (" << striped_drives
                      << " drives) > plain (" << plain_drives << " drives)\n";
        }
    }

    std::cout << (pass ? "PASS" : "FAIL") << "\n";
    return pass ? 0 : 1;
}
