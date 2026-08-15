#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/examples/chunk_kth_smallest.h"
#include "parlay/primitives.h"
#include "utils/file_utils.h"

// Correctness test for plaid::kth_smallest / plaid::kth_smallest_fast
// (chunk_kth_smallest.h): builds a shuffled-but-distinct uint64_t input,
// selects several ranks k out-of-core, and cross-checks each against
// std::nth_element on an identical DRAM copy.

static void cleanup_prefix(const std::string& prefix) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    unlink(GetFileName(prefix, d).c_str());
}

// Builds a distinct-valued, shuffled 0..n-1 sequence (both out-of-core and as
// a DRAM reference), selects rank k with both plaid::kth_smallest and
// plaid::kth_smallest_fast, and checks each against std::nth_element(k) on
// the DRAM copy.  Since values are distinct, the k-th smallest is unique.
static bool check_rank(const std::string& label, const std::vector<uint64_t>& shuffled,
                       size_t k) {
  const std::string prefix = "kth_test_in";
  const size_t n = shuffled.size();
  chunk_seq seq = plaid::tabulate<uint64_t>(
      n, prefix, [&](size_t i) { return shuffled[i]; });

  std::vector<uint64_t> ref = shuffled;
  std::nth_element(ref.begin(), ref.begin() + k, ref.end());
  const uint64_t expected = ref[k];

  chunk_seq seq_a = seq;
  const uint64_t got_slow = plaid::kth_smallest<uint64_t>(seq_a, (long)k);
  chunk_seq seq_b = seq;
  const uint64_t got_fast = plaid::kth_smallest_fast<uint64_t>(seq_b, (long)k);

  cleanup_prefix(prefix);

  bool pass = true;
  if (got_slow != expected) {
    std::cout << "  FAIL " << label << " (kth_smallest): n=" << n << " k=" << k
              << " got=" << got_slow << " expected=" << expected << "\n";
    pass = false;
  }
  if (got_fast != expected) {
    std::cout << "  FAIL " << label << " (kth_smallest_fast): n=" << n
              << " k=" << k << " got=" << got_fast
              << " expected=" << expected << "\n";
    pass = false;
  }
  if (pass)
    std::cout << "  OK " << label << ": n=" << n << " k=" << k
              << " -> " << expected << "\n";
  return pass;
}

static std::vector<uint64_t> shuffled_range(size_t n, uint64_t seed) {
  std::vector<uint64_t> v(n);
  for (size_t i = 0; i < n; i++) v[i] = i;
  std::mt19937_64 rng(seed);
  std::shuffle(v.begin(), v.end(), rng);
  return v;
}

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 500'000;

  bool all_pass = true;

  // ── 1. Single-chunk / DRAM-cutover case (n < 1536) ────────────────────────
  {
    auto small = shuffled_range(1000, 1);
    for (size_t k : {(size_t)0, small.size() / 2, small.size() - 1, (size_t)37})
      all_pass &= check_rank("single_chunk", small, k);
  }

  // ── 2. Multi-chunk / recursive case ────────────────────────────────────────
  {
    auto big = shuffled_range(n, 2);
    std::vector<size_t> ks = {0, big.size() / 2, big.size() - 1, big.size() / 4,
                              3 * big.size() / 4};
    std::mt19937_64 rng(3);
    std::uniform_int_distribution<size_t> dis(0, big.size() - 1);
    for (int i = 0; i < 3; i++) ks.push_back(dis(rng));
    for (size_t k : ks) all_pass &= check_rank("multi_chunk", big, k);
  }

  std::cout << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
  return all_pass ? 0 : 1;
}
