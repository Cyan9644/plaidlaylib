#ifndef EXTERNAL_KTH_SMALLEST_DELAYED_H
#define EXTERNAL_KTH_SMALLEST_DELAYED_H
#include <parlay/primitives.h>
#include <parlay/random.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>

#include "ChunkSequence/Primitives/materialize.h"
#include "ChunkSequence/Primitives/scan_find.h"
#include "ChunkSequence/Primitives/delayed.h"
#include "ChunkSequence/Primitives/histogram_by_index.h"
#include "ChunkSequence/examples/external/ExternalKthSmallest.h"
// parlay::internal::heap_tree comes in via <parlay/primitives.h> above; its
// header has no include guard, so do NOT include it a second time here.

// Variant of kth_smallest_fast (ExternalKthSmallest.h) that replaces
// ChunkPartition's all-32-buckets write with delayed::lazy_filter over just
// the ONE winning bucket. Same pivot sampling, same heap_tree ranking, same
// ChunkHistogramByKey read-only bucket-count pass as the original ("slow")
// kth_smallest -- the only change is how the winning bucket is turned into
// the next level's chunk_seq (or, at the DRAM-budget base case, straight into
// a parlay::sequence with no disk write at all).
//
// This is the positive counterpart to chunk_convex_hull_lazy_filter.h's
// UpperHullLazyFilter, which swaps ChunkPartition for lazy_filter in a 2-way
// split where BOTH branches are kept and recursed into -- there, lazy_filter
// only adds a second full pass with no offsetting savings (that file's own
// banner predicts it will be slower). Here the split is 32-way and 31 of the
// 32 buckets are discarded every level, so lazy_filter's "only pay to
// persist what you keep" property is a real saving: ChunkPartition always
// writes the WHOLE residual (all 32 buckets, `bytes_would_write_eager`
// below), while lazy_filter + materialize/force write only the survivors of
// bucket `id` (`bytes_written`), or nothing at all once that bucket already
// fits the DRAM budget -- a disk round trip kth_smallest_fast cannot skip,
// since its base-case check only runs *after* ChunkPartition has already
// written the bucket.
namespace ChunkSequenceOps {

namespace cd = ChunkSequenceOps::delayed;

// Monotonic id source for this variant's `force` scratch prefixes -- same
// shape as kth_smallest_prefix_counter (ExternalKthSmallest.h), kept
// separate so the two algorithms' recursions never share a prefix.
inline std::atomic<size_t>& kth_smallest_delayed_prefix_counter() {
  static std::atomic<size_t> c{0};
  return c;
}

// randomized ~O(n) algorithm; see the file banner above for why lazy_filter
// is a genuine win here (unlike the convex-hull 2-way-split case).
//
// `bytes_written` accumulates the actual disk bytes this call (and its
// recursive children) write via `force` -- zero for any level that instead
// hits the DRAM base case via `materialize`.
// `bytes_would_write_eager` accumulates what kth_smallest_fast's
// ChunkPartition would have written at each level (the whole residual, every
// time), computed analytically from the same histogram pass so the eager
// function itself never needs to be touched or instrumented.
template <typename T, typename Less = std::less<>>
T kth_smallest_delayed(chunk_seq& seq, long k, size_t& bytes_written,
                       size_t& bytes_would_write_eager, Less less1 = {}) {
  size_t n = 0;
  for (size_t r = 0; r < seq.chunks.size(); r++) n += seq.chunks[r].used;
  n /= sizeof(T);

  // Same env-override budget as kth_smallest_fast, so one knob tunes the
  // DRAM base case for both variants consistently.
  static const size_t dram_budget_bytes = [] {
    size_t budget =
        ((size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE)) / 4;
    if (const char* e = getenv("KTH_SMALLEST_DRAM_BUDGET_BYTES"))
      budget = std::stoull(e);
    return budget;
  }();

  if (n * sizeof(T) <= dram_budget_bytes) {
    auto i = ChunkSequenceOps::materialize<T>(seq);
    std::nth_element(i.begin(), i.begin() + k, i.end(), less1);
    return i[k];
  }

  // Oversampled pivot selection: identical to kth_smallest/kth_smallest_fast.
  int sample_size = 31;
  int over = 8;
  parlay::random_generator gen;
  std::uniform_int_distribution<long> dis(0, (long)n - 1);
  auto less2 = [&](std::pair<size_t, T> a, std::pair<size_t, T> b) {
    return less1(a.second, b.second);
  };
  parlay::sequence<std::pair<size_t, T>> pivots(sample_size * over);
  parlay::parallel_for(0, sample_size * over, [&](long o) {
    auto temp = gen[o];
    pivots[o].first = dis(temp);
  });
  parlay::sequence<size_t> scan_seq(seq.chunks.size());
  scan_size<T>(seq, scan_seq);
  parlay::parallel_for(0, sample_size * over, [&](size_t count) {
    pivots[count].second = scan_find<T>(seq, scan_seq, pivots[count].first);
  });
  pivots = parlay::sort(pivots, less2);
  pivots =
      parlay::tabulate(sample_size, [&](long i) { return pivots[i * over]; });

  auto seconds = parlay::map(pivots, [](const auto& p) { return p.second; });
  parlay::internal::heap_tree ss(seconds);
  auto key_fn = [&](T e) { return (size_t)ss.rank(e, less1); };

  // Read-only, one pass, no writes -- same call kth_smallest ("slow") makes.
  auto sums =
      ChunkSequenceOps::ChunkHistogramByKey<T>(seq, sample_size + 1, key_fn);
  auto [offsets, total] = parlay::scan(sums);
  auto id =
      std::upper_bound(offsets.begin(), offsets.end(), k) - offsets.begin() - 1;

  // What ChunkPartition would have written at this level: every element,
  // split across all 32 buckets, none dropped.
  bytes_would_write_eager += n * sizeof(T);

  auto fd = cd::lazy_filter(cd::delay<T>(seq), [&key_fn, id](T e) {
    return key_fn(e) == (size_t)id;
  });

  const long k_next = k - offsets[(size_t)id];

  if (fd.length() * sizeof(T) <= dram_budget_bytes) {
    // Fused single pass straight to DRAM -- no disk write at all for the
    // winning bucket, unlike kth_smallest_fast which always writes it via
    // ChunkPartition before its own recursive call's base case can fire.
    auto v = cd::materialize(fd);
    std::nth_element(v.begin(), v.begin() + k_next, v.end(), less1);
    return v[k_next];
  }

  const std::string next_prefix =
      "kdl_next_" +
      std::to_string(kth_smallest_delayed_prefix_counter().fetch_add(1));
  chunk_seq next = cd::force(fd, next_prefix);
  bytes_written += fd.length() * sizeof(T);

  T result = kth_smallest_delayed<T>(next, k_next, bytes_written,
                                     bytes_would_write_eager, less1);
  kth_smallest_cleanup_prefix(next_prefix);
  return result;
}

}  // namespace ChunkSequenceOps

#endif  // EXTERNAL_KTH_SMALLEST_DELAYED_H
