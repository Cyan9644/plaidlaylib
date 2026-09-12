#ifndef CHUNK_PRIMES_H
#define CHUNK_PRIMES_H

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/Primitives/primitives.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

// **************************************************************
// In-memory sieve for small primes up to sqrt(n).
// At n = 2^40, sqrt(n) ~ 2^20: the small-primes list is a few MB — fits in RAM.
// **************************************************************
inline parlay::sequence<long> in_mem_primes(long n) {
  if (n < 2) return {};
  long sqrt_n = (long)std::sqrt((double)n);
  auto sqrt_primes = in_mem_primes(sqrt_n);
  parlay::sequence<bool> flags(n + 1, true);
  parlay::parallel_for(
      0, n / sqrt_n + 1,
      [&](long i) {
        long start = sqrt_n * i;
        long end = (std::min)(start + sqrt_n, n + 1);
        for (long j = 0; j < (long)sqrt_primes.size(); j++) {
          long p = sqrt_primes[j];
          long first = (std::max)(2 * p, (((start - 1) / p) + 1) * p);
          for (long k = first; k < end; k += p) flags[k] = false;
        }
      },
      1);
  flags[0] = flags[1] = false;
  return parlay::filter(parlay::iota<long>(n + 1),
                        [&](long i) { return flags[i]; });
}

// **************************************************************
// Out-of-core primes sieve via ChunkFlatTabulate.  Each 4 MB virtual chunk
// [start, end) is sieved independently against the small primes, and its
// surviving indices are collected as uint64_t; ChunkFlatTabulate packs the
// variable-length per-chunk lists into a dense, index-ordered chunk_seq.
// **************************************************************
inline chunk_seq chunk_primes(size_t n, const std::string& result_prefix) {
  long sqrt_n = (long)std::sqrt((double)n);
  // Correct for floating-point rounding: ensure sqrt_n^2 <= n < (sqrt_n+1)^2.
  while ((long long)(sqrt_n + 1) * (sqrt_n + 1) <= (long long)n) sqrt_n++;

  parlay::sequence<long> small = in_mem_primes(sqrt_n);

  return plaid::ChunkFlatTabulate<uint64_t>(
      n + 1, result_prefix, [&](size_t start, size_t end) {
        // Byte flags, not std::vector<bool>: the bit-packed form pays a
        // mask/shift on every strided mark and every survivor read, which
        // shows up on the CPU-bound profile; a byte per candidate is ~512 KB
        // per live chunk (fine) and turns those into plain stores/loads.
        std::vector<uint8_t> flags(end - start, 1);
        for (long p : small) {
          size_t first = std::max((size_t)(2 * p), (((start - 1) / p) + 1) * p);
          for (size_t k = first; k < end; k += (size_t)p) flags[k - start] = 0;
        }
        parlay::sequence<uint64_t> out;
        // Reserve by the prime-density estimate ~ (end-start)/ln(end) to cut
        // reallocation churn in the serial survivor scan below.
        const double lnb = std::log((double)std::max<size_t>(end, 3));
        out.reserve((size_t)((double)(end - start) / lnb) + 16);
        size_t lo = (start < 2) ? 2 : start;
        for (size_t i = lo; i < end; i++)
          if (flags[i - start]) out.push_back((uint64_t)i);
        return out;
      });
}

#endif  // CHUNK_PRIMES_H
