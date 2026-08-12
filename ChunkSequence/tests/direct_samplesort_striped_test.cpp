// Correctness + placement test for direct_sample_sort's drive-striped
// bucketing (direct_samplesort.h's counterpart to samplesort_striped_test.cpp,
// which covers external_samplesort.h's sample_sort instead).
//
// direct_sample_sort's scatter phase routes through BucketWriter, which by
// default puts each bucket's entire on-disk data on exactly one drive (bucket
// b -> drive b % num_drives); its gather phase then reads that one file,
// sorts it, and writes the result back to one file per bucket -- so both the
// read AND the final sorted output are throttled to that one drive. Passing
// disk_span = GetSSDList().size() spreads both the scatter-phase temporary
// files and the gather-phase's final output across every drive instead (see
// BucketWriter's disk_span doc in bucketed_file_writer.h).
// sample_sort / sample_sort_singledrive are untouched by this change and are
// not exercised here.
//
// This test checks, for direct_sample_sort at disk_span=1 (default) and
// disk_span=GetSSDList().size()):
//   1. the output is fully sorted and element-for-element identical to
//      parlay::sort on the same (distinct) keys.
// and, on the placement side, that the striped run's output touches
// (strictly) more distinct drives overall than the plain run's -- the
// regression check a future change that quietly collapsed shards back to 1
// would fail.  This compares whole-output drive counts rather than assuming
// anything about individual bucket sizes: with only a handful of pivots,
// GetSampleSize/GetPivots can produce quite unevenly sized buckets, so a
// windowed "first bucket's worth of chunks" check would be flaky; a plain
// run's WHOLE output can only ever touch at most num_buckets drives (one per
// bucket), while a striped one touches up to num_buckets*disk_span -- so
// "striped touched strictly more drives than plain, for the same n" holds
// regardless of how the buckets happened to split.
//
// Exits 0 iff all checks pass.

#include <cstdint>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "parlay/primitives.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/examples/external_TODO/direct_samplesort.h"
#include "ChunkSequence/helper/bench_drives.h"

using plaid::direct_sample_sort;

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

static bool check_sorted(const std::string& name, chunk_seq& result,
                         const parlay::sequence<uint64_t>& ref) {
    std::vector<uint64_t> got = result.to_vector<uint64_t>();
    if (got.size() != ref.size()) {
        std::cout << "  FAIL " << name << ": got " << got.size()
                  << " elements, expected " << ref.size() << "\n";
        return false;
    }
    for (size_t i = 0; i < got.size(); i++) {
        if (got[i] != ref[i]) {
            std::cout << "  FAIL " << name << ": element " << i << " is " << got[i]
                      << ", expected " << ref[i] << "\n";
            return false;
        }
    }
    std::cout << "  OK " << name << ": " << got.size() << " elements, sorted correctly\n";
    return true;
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 50'000'000;

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
    };
    std::vector<Case> cases = {
        {"direct_sample_sort (disk_span=1)",      1,          "dsst_plain"},
        {"direct_sample_sort (disk_span=drives)", num_drives, "dsst_striped"},
    };

    size_t plain_drives = 0, striped_drives = 0;

    for (size_t ci = 0; ci < cases.size(); ci++) {
        const Case& c = cases[ci];
        chunk_seq in = plaid::tabulate<uint64_t>(n, c.prefix + "_in", key_at);
        chunk_seq out = direct_sample_sort<uint64_t>(in, std::less<>{}, c.prefix, c.disk_span);

        if (!check_sorted(c.name, out, ref)) pass = false;

        const size_t drives = distinct_drives(out);
        std::cout << "  .. " << c.name << ": output touches " << drives << " distinct drives\n";
        (ci == 0 ? plain_drives : striped_drives) = drives;

        // Sweep every file this sort touched before the next case runs.
        // direct_sample_sort derives all its filenames (scatter-phase temp
        // files and the final output files) from the single caller-supplied
        // prefix, so sweeping just that prefix (plus the input's, same
        // string) is sufficient -- no extra fixed prefixes like sample_sort's
        // "ss_bucket_"/"ss_base_" are needed here.
        bench_drives::clear_drives({c.prefix});
    }

    if (num_drives > 1) {
        if (striped_drives <= plain_drives) {
            std::cout << "  FAIL direct_sample_sort: striped touched " << striped_drives
                      << " drives, plain touched " << plain_drives
                      << " -- expected striped strictly more\n";
            pass = false;
        } else {
            std::cout << "  OK direct_sample_sort: striped (" << striped_drives
                      << " drives) > plain (" << plain_drives << " drives)\n";
        }
    }

    std::cout << (pass ? "PASS" : "FAIL") << "\n";
    return pass ? 0 : 1;
}
