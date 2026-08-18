#ifndef EXTERNAL_KTH_SMALLESTH_H
#define EXTERNAL_KTH_SMALLESTH_H
#include <parlay/primitives.h>
#include <parlay/random.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <random>

#include "ChunkSequence/Primitives/secondary_primitives.h"
// parlay::internal::heap_tree comes in via <parlay/primitives.h> above; its
// header has no include guard, so do NOT include it a second time here.
#include "utils/file_utils.h"

// we are currently assuming that not all elements go into 1 bucket, for
// obvious reasons.
namespace plaid {

// Remove the one-file-per-drive scratch a pack_value call left under `prefix`
// once the sequence it produced has been fully consumed.
inline void kth_smallest_cleanup_prefix(const std::string& prefix) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    unlink(GetFileName(prefix, d).c_str());
}

// randomized ~O(n) algorithm
template <typename T, typename Less = std::less<>>
T kth_smallest(chunk_seq& seq, long k, Less less1 = {}) {
  size_t n = 0;
  for (size_t r = 0; r < seq.chunks.size(); r++) {
    n += seq.chunks[r]
             .used;  // add the used size of each chunk to the n; this tells us
                     // the number of total elements in the sequence
  }
  n /= sizeof(T);

  if (n < 1536) {  // 1536 elements is the point at which we say we can
                   // materialize and sort directly

    auto i = plaid::materialize<T>(seq);
    // no reason to sort over select
    std::nth_element(i.begin(), i.begin() + k, i.end(), less1);
    return i[k];
  }

  // pick 31 pivots by randomly choosing 8 * 31 keys, sorting them,
  // and taking every 8th key (i.e. oversampling)
  int sample_size = 31;
  int over = 8;
  parlay::random_generator gen;
  std::uniform_int_distribution<long> dis(0, n - 1);

  auto less2 = [&](std::pair<size_t, T> i, std::pair<size_t, T> j) {
    return less1(i.second, j.second);
  };
  // get a set of random indices
  parlay::sequence<std::pair<size_t, T>> pivots(sample_size * over);
  parlay::parallel_for(0, sample_size * over, [&](long o) {
    auto temp = gen[o];
    pivots[o].first = dis(temp);
  });

  // we now have the random indices, so we just need to figure out where these
  // live in the actual chunk sequence
  parlay::sequence<size_t> scan_seq(seq.chunks.size());
  scan_size<T>(seq, scan_seq);

  parlay::parallel_for(0, sample_size * over, [&](size_t count) {
    pivots[count].second = scan_find<T>(
        seq, scan_seq,
        pivots[count].first);  // this is intended to find the value in question
  });

  // we now have the values of the pivots in memory

  // take the oversampleth pivots
  pivots = parlay::sort(pivots, less2);
  pivots =
      parlay::tabulate(sample_size, [&](long i) { return pivots[i * over]; });

  // Determine which of the 32 buckets each key belongs in
  auto seconds = parlay::map(pivots, [](const auto& p) { return p.second; });
  parlay::internal::heap_tree ss(seconds);
  auto key_fn = [&](T e) { return (size_t)ss.rank(e, less1); };

  auto sums = plaid::ChunkHistogramByKey<T>(seq, sample_size + 1, key_fn);

  // find which bucket k belongs in, and pack the keys in that bucket into next
  auto [offsets, total] = parlay::scan(sums);
  auto id =
      std::upper_bound(offsets.begin(), offsets.end(), k) - offsets.begin() - 1;

  // Pack survivors straight off seq's values (single read pass, no selector).
  const std::string next_prefix = "next_" + std::to_string(n);
  auto next = plaid::pack_value<T>(
      seq, next_prefix, [&, id](T e) { return key_fn(e) == (size_t)id; });
  // recur on much smaller set, adjusting k as needed
  T result = kth_smallest<T>(next, k - offsets[id], less1);
  // `next` is fully consumed now -- unlink its scratch so recursion doesn't
  // strand a file set per level on every one of the SSDs.
  kth_smallest_cleanup_prefix(next_prefix);
  return result;
}

// Copy of kth_smallest that does not recurse out-of-core: once the winning
// bucket is packed, it is materialized into DRAM and finished with
// parlay::kth_smallest (in-memory selection) instead of another out-of-core
// recursion level.
template <typename T, typename Less = std::less<>>
T kth_smallest_fast(chunk_seq& seq, long k, Less less1 = {}) {
  size_t n = 0;
  for (size_t r = 0; r < seq.chunks.size(); r++) {
    n += seq.chunks[r]
             .used;  // add the used size of each chunk to the n; this tells us
                     // the number of total elements in the sequence
  }
  n /= sizeof(T);

  if (n < 1536) {  // 1536 elements is the point at which we say we can
                   // materialize and sort directly

    auto i = plaid::materialize<T>(seq);
    // no reason to sort over select
    std::nth_element(i.begin(), i.begin() + k, i.end(), less1);
    return i[k];
  }

  // pick 31 pivots by randomly choosing 8 * 31 keys, sorting them,
  // and taking every 8th key (i.e. oversampling)
  int sample_size = 31;
  int over = 8;
  parlay::random_generator gen;
  std::uniform_int_distribution<long> dis(0, n - 1);

  auto less2 = [&](std::pair<size_t, T> i, std::pair<size_t, T> j) {
    return less1(i.second, j.second);
  };
  // get a set of random indices
  parlay::sequence<std::pair<size_t, T>> pivots(sample_size * over);
  parlay::parallel_for(0, sample_size * over, [&](long o) {
    auto temp = gen[o];
    pivots[o].first = dis(temp);
  });

  // we now have the random indices, so we just need to figure out where these
  // live in the actual chunk sequence
  parlay::sequence<size_t> scan_seq(seq.chunks.size());
  scan_size<T>(seq, scan_seq);

  parlay::parallel_for(0, sample_size * over, [&](size_t count) {
    pivots[count].second = scan_find<T>(
        seq, scan_seq,
        pivots[count].first);  // this is intended to find the value in question
  });

  // we now have the values of the pivots in memory

  // take the oversampleth pivots
  pivots = parlay::sort(pivots, less2);
  pivots =
      parlay::tabulate(sample_size, [&](long i) { return pivots[i * over]; });

  // Determine which of the 32 buckets each key belongs in
  auto seconds = parlay::map(pivots, [](const auto& p) { return p.second; });
  parlay::internal::heap_tree ss(seconds);
  auto key_fn = [&](T e) { return (size_t)ss.rank(e, less1); };

  auto sums = plaid::ChunkHistogramByKey<T>(seq, sample_size + 1, key_fn);

  // find which bucket k belongs in, and pack the keys in that bucket into next
  auto [offsets, total] = parlay::scan(sums);
  auto id =
      std::upper_bound(offsets.begin(), offsets.end(), k) - offsets.begin() - 1;

  // Pack survivors straight off seq's values (single read pass, no selector).
  const std::string next_prefix = "next_" + std::to_string(n);
  auto next = plaid::pack_value<T>(
      seq, next_prefix, [&, id](T e) { return key_fn(e) == (size_t)id; });

  // Instead of recursing out-of-core again, materialize the winning bucket
  // into DRAM and finish with parlay::kth_smallest's in-memory selection.
  // parlay::kth_smallest returns an iterator into next_mem, not the value.
  auto next_mem = plaid::materialize<T>(next);
  T result = *parlay::kth_smallest(next_mem, (size_t)(k - offsets[id]), less1);
  // `next` is fully consumed now -- unlink its scratch.
  kth_smallest_cleanup_prefix(next_prefix);
  return result;
}

}  // namespace plaid

#endif
