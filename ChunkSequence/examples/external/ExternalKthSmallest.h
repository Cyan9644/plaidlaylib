#ifndef EXTERNAL_KTH_SMALLESTH_H
#define EXTERNAL_KTH_SMALLESTH_H
#include <parlay/primitives.h>
#include <parlay/random.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <functional>
#include <random>

#include "ChunkSequence/Primitives/materialize.h"
#include "ChunkSequence/Primitives/histogram_by_index.h"
#include "ChunkSequence/Primitives/pack.h"
#include "ChunkSequence/Primitives/partition.h"
// parlay::internal::heap_tree comes in via <parlay/primitives.h> above; its
// header has no include guard, so do NOT include it a second time here.
#include "ChunkSequence/Primitives/scan_find.h"
#include "utils/file_utils.h"

// we are currently assuming that not all elemsents go into 1 bucket, for
// obvious reasons.
namespace plaid {

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
  //   long n = seq.chunks.size();

  if (n < 1536) {  // 1536 elements is the point at which we say we can
                   // materialize and sort directly

    auto i = plaid::materialize<T>(
        seq);  // materialize external sequence to parlay sequence (not yet
               // cleanly implemented)
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
  //   auto pivots = parlay::sort(ChunkSequence::tabulate(sample_size*over, [&]
  //   (long i) {
  //     auto r = gen[i];
  //     return seq[dis(r)];}), less);
  //     //need to find some way around this random access
  //   pivots = parlay::tabulate(sample_size,[&] (long i) {return
  //   pivots[i*over];});

  auto less2 = [&](std::pair<size_t, T> i, std::pair<size_t, T> j) {
    return less1(i.second, j.second);
  };
  // get a set of random indices
  parlay::sequence<std::pair<size_t, T>> pivots(sample_size * over);
  parlay::parallel_for(0, sample_size * over, [&](long o) {
    auto temp = gen[o];
    pivots[o].first = dis(temp);
  });
  // auto pivots = parlay::tabulate(sample_size*over, [&] (long i) {
  // return = gen[i];
  // });

  // we now have the random indices, so we just need to figure out where these
  // live in the actual chunk sequence
  //  for(size_t count = 0; count < sample_size * over; count++){

  //     pivots[count].second = LinearFind<T>(seq, pivots[count].first); //this
  //     is intended to find the value in question
  // }
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

  auto sums =
      plaid::ChunkHistogramByKey<T>(seq, sample_size + 1, key_fn);

  // find which bucket k belongs in, and pack the keys in that bucket into next
  auto [offsets, total] = parlay::scan(sums);
  auto id =
      std::upper_bound(offsets.begin(), offsets.end(), k) - offsets.begin() - 1;

  // Pack survivors straight off seq's values (single read pass, no selector).
  auto next = plaid::pack_value<T>(
      seq, "next_" + std::to_string(n),
      [&, id](T e) { return key_fn(e) == (size_t)id; });
  // recur on much smaller set, adjusting k as needed
  return kth_smallest<T>(next, k - offsets[id], less1);
}

// Monotonic id source for unique per-call scratch prefixes -- same shape as
// chunk_convex_hull.h's prefix_counter, so two recursion levels (or two
// concurrent top-level calls) never share a ChunkPartition prefix.
inline std::atomic<size_t>& kth_smallest_prefix_counter() {
  static std::atomic<size_t> c{0};
  return c;
}

// Remove the one-file-per-drive scratch a ChunkPartition call left under
// `prefix` once every bucket it produced has been fully consumed.
inline void kth_smallest_cleanup_prefix(const std::string& prefix) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    unlink(GetFileName(prefix, d).c_str());
}

// randomized ~O(n) algorithm
template <typename T, typename Less = std::less<>>
T kth_smallest_fast(chunk_seq& seq, long k, Less less1 = {}) {
  size_t n = 0;
  for (size_t r = 0; r < seq.chunks.size(); r++) {
    n += seq.chunks[r]
             .used;  // add the used size of each chunk to the n; this tells us
                     // the number of total elements in the sequence
  }

  // Once the residual fits comfortably in DRAM, finish with one in-memory
  // nth_element instead of recursing again -- recursion only exists to shrink
  // an out-of-core set down to something that fits, so continuing to recurse
  // below this point buys nothing but more disk round trips.  Budget is
  // queried once (function-local static) and overridable, same shape as
  // sort_buckets.h's SORT_BUCKETS_BUDGET_BYTES.
  static const size_t dram_budget_bytes = [] {
    size_t budget =
        ((size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE)) / 4;
    if (const char* e = getenv("KTH_SMALLEST_DRAM_BUDGET_BYTES"))
      budget = std::stoull(e);
    return budget;
  }();

  if (n <= dram_budget_bytes) {  // the whole residual (in bytes) fits DRAM --
                                 // finish here

    auto i = plaid::materialize<T>(
        seq);  // materialize external sequence to parlay sequence (not yet
               // cleanly implemented)
    // no reason to sort over select
    std::nth_element(i.begin(), i.begin() + k, i.end(), less1);
    return i[k];
  }
  n /= sizeof(T);

  // pick 31 pivots by randomly choosing 8 * 31 keys, sorting them,
  // and taking every 8th key (i.e. oversampling)
  int sample_size = 31;
  int over = 8;
  parlay::random_generator gen;
  std::uniform_int_distribution<long> dis(0, n - 1);
  //   auto pivots = parlay::sort(ChunkSequence::tabulate(sample_size*over, [&]
  //   (long i) {
  //     auto r = gen[i];
  //     return seq[dis(r)];}), less);
  //     //need to find some way around this random access
  //   pivots = parlay::tabulate(sample_size,[&] (long i) {return
  //   pivots[i*over];});

  auto less2 = [&](std::pair<size_t, T> i, std::pair<size_t, T> j) {
    return less1(i.second, j.second);
  };
  // get a set of random indices
  parlay::sequence<std::pair<size_t, T>> pivots(sample_size * over);
  parlay::parallel_for(0, sample_size * over, [&](long o) {
    auto temp = gen[o];
    pivots[o].first = dis(temp);
  });
  // auto pivots = parlay::tabulate(sample_size*over, [&] (long i) {
  // return = gen[i];
  // });

  // we now have the random indices, so we just need to figure out where these
  // live in the actual chunk sequence
  //  for(size_t count = 0; count < sample_size * over; count++){

  //     pivots[count].second = LinearFind<T>(seq, pivots[count].first); //this
  //     is intended to find the value in question
  // }
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

  // Route every element to its bucket in ONE streaming read + one write --
  // ChunkPartition owns exactly one long-lived reader and one writer for the
  // whole pass, all 32 buckets included.  This replaces what used to be a
  // histogram pass (its own reader) followed by a pack_value pass whose
  // DensePack driver opens a fresh ChunkSequenceReader per 128-chunk batch --
  // at this scale (before hitting the DRAM base case above) that is
  // potentially thousands of io_uring rings built and torn down for a single
  // recursion level.  Same "one reader/one writer per level" shape as
  // chunk_convex_hull.h's UpperHull, which made the identical trade for its
  // own per-level split.  offsets/id below then come for free from each
  // bucket's own chunk count (size<T>), no extra I/O.
  const std::string part_prefix =
      "kth_next_" + std::to_string(kth_smallest_prefix_counter().fetch_add(1));
  std::vector<chunk_seq> buckets = plaid::ChunkPartition<T>(
      seq, sample_size + 1, part_prefix, key_fn);
 
  parlay::sequence<size_t> sums(sample_size + 1);
  for (size_t b = 0; b < (size_t)sample_size + 1; b++)
    sums[b] = plaid::size<T>(buckets[b]);

  // find which bucket k belongs in
  auto [offsets, total] = parlay::scan(sums);
  auto id =
      std::upper_bound(offsets.begin(), offsets.end(), k) - offsets.begin() - 1;

  // recur on much smaller set, adjusting k as needed
  T result = kth_smallest<T>(buckets[id], k - offsets[id], less1);
  // buckets[id] (and every sibling bucket) has now been fully consumed --
  // free the whole partition's shared files.
  kth_smallest_cleanup_prefix(part_prefix);
  return result;
}

}  // namespace plaid





// //will it work? Find out next week on the subsequent episode of interesting stuff that no one cares about
// template <typename T, typename Less = std::less<>>
// T kth_smallest_fast_delayed(chunk_seq& seq, long k, Less less1 = {}) {
//   size_t n = 0;
//   for (size_t r = 0; r < seq.chunks.size(); r++) {
//     n += seq.chunks[r]
//              .used;  // add the used size of each chunk to the n; this tells us
//                      // the number of total elements in the sequence
//   }

//   // Once the residual fits comfortably in DRAM, finish with one in-memory
//   // nth_element instead of recursing again -- recursion only exists to shrink
//   // an out-of-core set down to something that fits, so continuing to recurse
//   // below this point buys nothing but more disk round trips.  Budget is
//   // queried once (function-local static) and overridable, same shape as
//   // sort_buckets.h's SORT_BUCKETS_BUDGET_BYTES.
//   static const size_t dram_budget_bytes = [] {
//     size_t budget =
//         ((size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE)) / 4;
//     if (const char* e = getenv("KTH_SMALLEST_DRAM_BUDGET_BYTES"))
//       budget = std::stoull(e);
//     return budget;
//   }();

//   if (n <= dram_budget_bytes) {  // the whole residual (in bytes) fits DRAM --
//                                  // finish here

//     auto i = plaid::materialize<T>(
//         seq);  // materialize external sequence to parlay sequence (not yet
//                // cleanly implemented)
//     // no reason to sort over select
//     std::nth_element(i.begin(), i.begin() + k, i.end(), less1);
//     return i[k];
//   }
//   n /= sizeof(T);

//   // pick 31 pivots by randomly choosing 8 * 31 keys, sorting them,
//   // and taking every 8th key (i.e. oversampling)
//   int sample_size = 31;
//   int over = 8;
//   parlay::random_generator gen;
//   std::uniform_int_distribution<long> dis(0, n - 1);
//   parlay::sequence<std::pair<size_t, T>> pivots(sample_size * over);
//   parlay::parallel_for(0, sample_size * over, [&](long o) {
//     auto temp = gen[o];
//     pivots[o].first = dis(temp);
//   });
//     parlay::sequence<size_t> scan_seq(seq.chunks.size());
//   scan_size<T>(seq, scan_seq);

//   parlay::parallel_for(0, sample_size * over, [&](size_t count) {
//     pivots[count].second = scan_find<T>(
//         seq, scan_seq,
//         pivots[count].first);  // this is intended to find the value in question
//   });


//   pivots = parlay::sort(pivots, less2);
//   pivots =parlay::tabulate(sample_size, [&](long i) { return pivots[i * over]; });
//   auto sums =plaid::ChunkHistogramByKey<T>(seq, sample_size + 1, key_fn);
//   auto list = plaid::delayed::lazy_filter(plaid::delayed_map(plaid::delayed::delay<T>(seq), [&](T e)){

//     return ss.find()

//   }), );

//   parlay::sequence<size_t> sums(sample_size + 1);
//   for (size_t b = 0; b < (size_t)sample_size + 1; b++)
//     sums[b] = plaid::size<T>(buckets[b]);

//   // find which bucket k belongs in
//   auto [offsets, total] = parlay::scan(sums);
//   auto id =
//       std::upper_bound(offsets.begin(), offsets.end(), k) - offsets.begin() - 1;

//   // recur on much smaller set, adjusting k as needed
//   T result = kth_smallest<T>(buckets[id], k - offsets[id], less1);
//   // buckets[id] (and every sibling bucket) has now been fully consumed --
//   // free the whole partition's shared files.
//   kth_smallest_cleanup_prefix(part_prefix);

// }
#endif
