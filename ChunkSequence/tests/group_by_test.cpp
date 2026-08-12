// Correctness test for group_by_index / group_by_key (group_by.h).
//
// group_by_index: builds a uint64_t sequence 0..n-1, groups it into k buckets
// by v % k (no drop -- group_by_index has no drop sentinel, unlike
// ChunkPartition), and verifies:
//   1. every element in bucket b really has v % k == b,
//   2. every input value appears exactly once across the buckets (full
//      coverage, since nothing can be dropped),
//   3. each bucket is a valid chunk_seq: index-ordered and dense-except-last.
//
// group_by_key: groups the same kind of sequence by an arbitrary key (first
// the identity key, then a coarser derived key v/1000), and verifies every
// returned value lands in the bucket its own hash(key)%num_buckets predicts,
// plus full coverage -- group_by_key is a thin hash-bucket wrapper over
// group_by_index, so it inherits the same no-drop/dense-except-last
// invariants.
//
// Exits 0 iff all checks pass.

#include <unistd.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "ChunkSequence/Primitives/group_by.h"
#include "ChunkSequence/Primitives/chunk_seq.h"
#include "parlay/primitives.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"

static void cleanup_prefix(const std::string& prefix) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    unlink(GetFileName(prefix, d).c_str());
}

// Checks that every bucket in `parts` is index-ordered + dense-except-last,
// every value satisfies `bucket_of(value) == bucket index`, and every input
// value in [0, n) is returned exactly once across all buckets. Returns
// whether all checks passed; prints FAIL lines for anything that didn't.
template <typename BucketOf>
static bool check_grouping(const std::string& label,
                           const std::vector<chunk_seq>& parts, size_t n,
                           BucketOf bucket_of) {
  bool pass = true;
  std::vector<char> seen(n, 0);

  for (size_t b = 0; b < parts.size() && pass; b++) {
    const chunk_seq& bucket = parts[b];

    for (size_t i = 0; i < bucket.chunks.size(); i++) {
      if (bucket.chunks[i].index != i) {
        std::cout << "  FAIL " << label << " bucket " << b << ": chunk " << i
                  << " has index " << bucket.chunks[i].index << "\n";
        pass = false;
        break;
      }
      if (i + 1 < bucket.chunks.size() && bucket.chunks[i].used != CHUNK_SIZE) {
        std::cout << "  FAIL " << label << " bucket " << b << ": non-last chunk "
                  << i << " used=" << bucket.chunks[i].used << " (not full)\n";
        pass = false;
        break;
      }
    }
    if (!pass) break;

    std::vector<uint64_t> vals = bucket.to_vector<uint64_t>();
    for (uint64_t v : vals) {
      if (v >= n) {
        std::cout << "  FAIL " << label << " bucket " << b << ": value " << v
                  << " >= n\n";
        pass = false;
        break;
      }
      if (bucket_of(v) != b) {
        std::cout << "  FAIL " << label << " bucket " << b << ": value " << v
                  << " expected bucket " << bucket_of(v) << "\n";
        pass = false;
        break;
      }
      if (seen[v]) {
        std::cout << "  FAIL " << label << ": value " << v
                  << " appears more than once\n";
        pass = false;
        break;
      }
      seen[v] = 1;
    }
  }

  if (pass) {
    for (size_t v = 0; v < n; v++) {
      if (!seen[v]) {
        std::cout << "  FAIL " << label << ": value " << v << " missing\n";
        pass = false;
        break;
      }
    }
  }
  if (pass)
    std::cout << "  OK " << label << ": " << n << " values covered across "
              << parts.size() << " buckets, packing valid\n";
  return pass;
}

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 500'000;
  const size_t k = 4;

  bool pass = true;

  // group_by_index: bucket = v % k.
  {
    const std::string in_prefix = "gbi_in";
    const std::string out_prefix = "gbi_out";
    chunk_seq seq = plaid::tabulate<uint64_t>(
        n, in_prefix, [](size_t i) { return (uint64_t)i; });

    std::vector<chunk_seq> parts = plaid::group_by_index<uint64_t>(
        seq, k, out_prefix, [k](uint64_t v) { return (size_t)(v % k); });

    if (parts.size() != k) {
      std::cout << "  FAIL group_by_index: got " << parts.size()
                << " buckets, expected " << k << "\n";
      pass = false;
    } else if (!check_grouping("group_by_index", parts, n,
                               [k](uint64_t v) { return (size_t)(v % k); })) {
      pass = false;
    }

    cleanup_prefix(in_prefix);
    cleanup_prefix(out_prefix);
  }

  // group_by_key: identity key, default Hash = std::hash<uint64_t>.
  {
    const std::string in_prefix = "gbk_id_in";
    const std::string out_prefix = "gbk_id_out";
    chunk_seq seq = plaid::tabulate<uint64_t>(
        n, in_prefix, [](size_t i) { return (uint64_t)i; });

    auto key_of = [](uint64_t v) { return v; };
    auto expected_bucket = [k](uint64_t v) {
      return (size_t)(std::hash<uint64_t>{}(v) % k);
    };

    std::vector<chunk_seq> parts =
        plaid::group_by_key<uint64_t>(seq, k, out_prefix, key_of);

    if (parts.size() != k) {
      std::cout << "  FAIL group_by_key(identity): got " << parts.size()
                << " buckets, expected " << k << "\n";
      pass = false;
    } else if (!check_grouping("group_by_key(identity)", parts, n,
                               expected_bucket)) {
      pass = false;
    }

    cleanup_prefix(in_prefix);
    cleanup_prefix(out_prefix);
  }

  // group_by_key: derived key (v / 1000), same default Hash.
  {
    const std::string in_prefix = "gbk_derived_in";
    const std::string out_prefix = "gbk_derived_out";
    chunk_seq seq = plaid::tabulate<uint64_t>(
        n, in_prefix, [](size_t i) { return (uint64_t)i; });

    auto key_of = [](uint64_t v) { return v / 1000; };
    auto expected_bucket = [k](uint64_t v) {
      return (size_t)(std::hash<uint64_t>{}(v / 1000) % k);
    };

    std::vector<chunk_seq> parts =
        plaid::group_by_key<uint64_t>(seq, k, out_prefix, key_of);

    if (parts.size() != k) {
      std::cout << "  FAIL group_by_key(derived): got " << parts.size()
                << " buckets, expected " << k << "\n";
      pass = false;
    } else if (!check_grouping("group_by_key(derived)", parts, n,
                               expected_bucket)) {
      pass = false;
    }

    cleanup_prefix(in_prefix);
    cleanup_prefix(out_prefix);
  }

  std::cout << (pass ? "PASS" : "FAIL") << "\n";
  return pass ? 0 : 1;
}
