// Correctness + placement test for external_samplesort.h's sample_sort.
//
// sample_sort's bucketing step now routes through group_by_index
// (Primitives/group_by.h), which writes one file per drive shared by every
// bucket -- so, unlike the old count_sort/BucketWriter substrate, a bucket's
// data is *always* spread across every drive; there is no longer a disk_span
// knob to compare plain vs. striped placement against.
//
// This test checks:
//   1. the output is fully sorted and element-for-element identical to
//      parlay::sort on the same (distinct) keys.
//   2. on the placement side, that the output touches more than one distinct
//      drive when more than one drive is available -- the regression check a
//      future change that quietly collapsed the output back onto a single
//      drive would fail.
//
// Exits 0 iff all checks pass.

#include <cstdint>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/helper/bench_drives.h"
#include "ChunkSequence/examples/external/external_samplesort.h"
#include "parlay/primitives.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"

using plaid::sample_sort;

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
  std::cout << "  OK " << name << ": " << got.size()
            << " elements, sorted correctly\n";
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
  const std::string prefix = "sst_prim_in";

  chunk_seq in = plaid::tabulate<uint64_t>(n, prefix, key_at);
  chunk_seq out = sample_sort<uint64_t>(in);

  if (!check_sorted("sample_sort", out, ref)) pass = false;

  const size_t drives = distinct_drives(out);
  std::cout << "  .. sample_sort: output touches " << drives
            << " distinct drives\n";
  if (num_drives > 1) {
    if (drives <= 1) {
      std::cout << "  FAIL sample_sort: output touched only " << drives
                << " drive(s), expected more than 1\n";
      pass = false;
    } else {
      std::cout << "  OK sample_sort: output spread across " << drives
                << " drives\n";
    }
  }

  // Sweep every file this sort touched.  sample_sort has no prefix parameter
  // of its own (it always uses "ss_bucket_"/"ss_base_" + a process-global
  // counter), so those two fixed prefixes are swept alongside the input's.
  bench_drives::clear_drives({prefix, "ss_bucket_", "ss_base_"});

  std::cout << (pass ? "PASS" : "FAIL") << "\n";
  return pass ? 0 : 1;
}
