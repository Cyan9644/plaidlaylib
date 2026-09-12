#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/examples/chunk_primes.h"
#include "utils/file_utils.h"

// Correctness test for chunk_primes (chunk_primes.h, the out-of-core sieve
// examples/primes.cpp drives), previously untested anywhere in
// ChunkSequence/tests/ -- only the underlying ChunkFlatTabulate mechanism got
// indirect coverage, via primitives_test.cpp's own hand-rolled sieve, never
// chunk_primes's own code path (which additionally exercises in_mem_primes
// for the small-prime list).
//
// Deliberately does NOT use chunk_primes's own in_mem_primes as the oracle --
// that would test the algorithm against itself.  Instead compares against a
// plain, single-array Sieve of Eratosthenes written fresh here (a different
// implementation shape than chunk_primes's per-chunk-with-precomputed-small-
// primes approach), matching kmp_test/rabin_karp_test/bigint_add_test's
// convention of an independent reference.  O(n log log n), not O(n sqrt n)
// trial division, since `make test TEST_ARGS=<n>` can forward a large n here.
static std::vector<uint64_t> sieve_primes(uint64_t n) {
  std::vector<uint64_t> out;
  if (n < 2) return out;
  std::vector<bool> composite(n + 1, false);
  for (uint64_t p = 2; p * p <= n; p++) {
    if (composite[p]) continue;
    for (uint64_t m = p * p; m <= n; m += p) composite[m] = true;
  }
  for (uint64_t x = 2; x <= n; x++)
    if (!composite[x]) out.push_back(x);
  return out;
}

static void cleanup_prefix(const std::string& prefix) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    unlink(GetFileName(prefix, d).c_str());
}

static bool check_primes(const std::string& label, size_t n) {
  const std::string prefix = "primes_test";
  chunk_seq seq = chunk_primes(n, prefix);
  std::vector<uint64_t> got = seq.to_vector<uint64_t>();
  cleanup_prefix(prefix);

  std::vector<uint64_t> expected = sieve_primes(n);

  bool ok = (got.size() == expected.size()) &&
           std::equal(got.begin(), got.end(), expected.begin());
  std::cout << "  " << (ok ? "OK" : "FAIL") << " " << label << ": n=" << n
            << "  pi(n)=" << got.size() << " (expected " << expected.size()
            << ")\n";
  return ok;
}

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  const size_t n_arg = (argc > 1) ? std::stoull(argv[1]) : 100000;

  bool all_pass = true;

  // Small/edge n: no primes below 2, first primes, exact small primes.
  for (size_t n : {(size_t)0, (size_t)1, (size_t)2, (size_t)3, (size_t)10,
                   (size_t)100, (size_t)10000})
    all_pass &= check_primes("small", n);

  // Chunk-boundary-straddling and argv-provided sizes.
  for (size_t n : {ELEMS_PER_CHUNK - 1, ELEMS_PER_CHUNK, ELEMS_PER_CHUNK + 1})
    all_pass &= check_primes("chunk_boundary", n);
  all_pass &= check_primes("large", n_arg);

  std::cout << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
  return all_pass ? 0 : 1;
}
