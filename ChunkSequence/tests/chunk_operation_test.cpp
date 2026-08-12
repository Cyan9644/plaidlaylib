// Correctness test for ChunkOperation's dispatch front door
// (ExternalPrimitives/chunk_operation.h) and the DRAM-budget-checked,
// wave-batched process_inplace_budgeted engine it's built on
// (ExternalPrimitives/small_sequence_ops.h). Exercises Sort and Shuffle under
// both a default (single-wave) and an artificially small (forced multi-wave)
// DRAM budget.
//
// Exits 0 iff all checks pass.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "ChunkSequence/Primitives/operation.h"
#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/helper/bench_drives.h"
#include "parlay/primitives.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"

using plaid::ChunkOperation;
using plaid::apply;

namespace {

void set_budget(const char* v) { setenv("PROCESS_INPLACE_BUDGET_BYTES", v, 1); }
void clear_budget() { unsetenv("PROCESS_INPLACE_BUDGET_BYTES"); }

// Per-bucket key generator: a pure function of (bucket, local index), so a
// bucket's original content can be regenerated for comparison without having
// to snapshot it before the in-place operation runs.
uint64_t key_at(size_t bucket, size_t i) {
  return parlay::hash64(bucket * 1000000007ULL + i);
}

}  // namespace

// Build `num_buckets` independent chunk_seqs of `elems_per_bucket` elements
// each, via key_at.
static std::vector<chunk_seq> build_buckets(size_t num_buckets,
                                            size_t elems_per_bucket,
                                            const std::string& prefix) {
  std::vector<chunk_seq> buckets(num_buckets);
  for (size_t b = 0; b < num_buckets; b++) {
    buckets[b] = plaid::tabulate<uint64_t>(
        elems_per_bucket, prefix + "_" + std::to_string(b),
        [b](size_t i) { return key_at(b, i); });
  }
  return buckets;
}

static bool check_sort(std::vector<chunk_seq>& buckets, size_t elems_per_bucket,
                       const char* label) {
  bool ok = true;
  for (size_t b = 0; b < buckets.size(); b++) {
    std::vector<uint64_t> got = buckets[b].to_vector<uint64_t>();
    auto ref = parlay::tabulate(elems_per_bucket,
                                [b](size_t i) { return key_at(b, i); });
    parlay::sort_inplace(ref);
    if (got.size() != ref.size() ||
        !std::equal(got.begin(), got.end(), ref.begin())) {
      std::cout << "  FAIL " << label << ": bucket " << b
                << " not sorted correctly\n";
      ok = false;
    }
  }
  if (ok) std::cout << "  OK " << label << ": all buckets sorted correctly\n";
  return ok;
}

static bool check_shuffle(std::vector<chunk_seq>& buckets,
                          size_t elems_per_bucket, const char* label) {
  bool ok = true;
  for (size_t b = 0; b < buckets.size(); b++) {
    std::vector<uint64_t> got = buckets[b].to_vector<uint64_t>();
    auto ref = parlay::tabulate(elems_per_bucket,
                                [b](size_t i) { return key_at(b, i); });
    if (got.size() != ref.size()) {
      std::cout << "  FAIL " << label << ": bucket " << b << " has "
                << got.size() << " elements, expected " << ref.size() << "\n";
      ok = false;
      continue;
    }
    // Permutation check (exact order is randomized): a valid shuffle's sorted
    // content must match the sorted original content exactly.
    std::sort(got.begin(), got.end());
    parlay::sort_inplace(ref);
    if (!std::equal(got.begin(), got.end(), ref.begin())) {
      std::cout << "  FAIL " << label << ": bucket " << b
                << " is not a permutation of its original content\n";
      ok = false;
    }
  }
  if (ok)
    std::cout << "  OK " << label
              << ": all buckets are permutations of their original content\n";
  return ok;
}

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  const size_t num_buckets = 6;
  const size_t elems_per_bucket =
      (argc > 1) ? std::stoull(argv[1]) : 2'000'000;  // ~16MB/bucket (uint64_t)
  const size_t bucket_bytes = elems_per_bucket * sizeof(uint64_t);

  // A budget that fits exactly one bucket comfortably but not two -- forces
  // more than one wave across the 6 buckets built below, without ever
  // tripping process_inplace_budgeted's own-bucket-too-big CHECK (each
  // bucket alone is well under this budget).
  const size_t small_budget = bucket_bytes + bucket_bytes / 2;  // 1.5 buckets

  std::cout << "elems_per_bucket=" << elems_per_bucket
            << " bucket_bytes=" << bucket_bytes
            << " small_budget=" << small_budget << "\n";

  bool pass = true;

  // -- Sort, default (single-wave) budget --------------------------------
  clear_budget();
  {
    auto buckets = build_buckets(num_buckets, elems_per_bucket, "co_sort_default");
    apply<ChunkOperation::Sort, uint64_t>(buckets);
    pass &= check_sort(buckets, elems_per_bucket, "sort (default budget)");
    bench_drives::clear_drives({"co_sort_default"});
  }

  // -- Sort, forced multi-wave --------------------------------------------
  set_budget(std::to_string(small_budget).c_str());
  {
    auto buckets = build_buckets(num_buckets, elems_per_bucket, "co_sort_multiwave");
    apply<ChunkOperation::Sort, uint64_t>(buckets);
    pass &= check_sort(buckets, elems_per_bucket, "sort (forced multi-wave budget)");
    bench_drives::clear_drives({"co_sort_multiwave"});
  }

  // -- Shuffle, default (single-wave) budget ------------------------------
  clear_budget();
  {
    auto buckets = build_buckets(num_buckets, elems_per_bucket, "co_shuf_default");
    apply<ChunkOperation::Shuffle, uint64_t>(buckets, {}, /*seed=*/42);
    pass &= check_shuffle(buckets, elems_per_bucket, "shuffle (default budget)");
    bench_drives::clear_drives({"co_shuf_default"});
  }

  // -- Shuffle, forced multi-wave ------------------------------------------
  set_budget(std::to_string(small_budget).c_str());
  {
    auto buckets = build_buckets(num_buckets, elems_per_bucket, "co_shuf_multiwave");
    apply<ChunkOperation::Shuffle, uint64_t>(buckets, {}, /*seed=*/1234);
    pass &= check_shuffle(buckets, elems_per_bucket, "shuffle (forced multi-wave budget)");
    bench_drives::clear_drives({"co_shuf_multiwave"});
  }
  clear_budget();

  std::cout << (pass ? "PASS" : "FAIL") << "\n";
  return pass ? 0 : 1;
}
