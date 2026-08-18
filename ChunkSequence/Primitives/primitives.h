// primitives.h -- the six core eager primitives.
//
//   ChunkMap           ExternalTransform body (FANOUT emits when sizeof(R) >
//   sizeof(T)) ChunkReduce        RemoveWorker fold + parlay::reduce ChunkScan
//   RemoveWorker pass 1 -> block prefix -> ExternalTransform pass 2 ChunkFilter
//   DensePackStream, predicate compaction ChunkFlatTabulate  DensePack,
//   generator source ChunkFlatMap       DensePackStream, flatten o map with an
//   optional forward halo
//
// These are the primitives an out-of-core program reaches for first; everything
// else in the eager layer lives in secondary_primitives.h, which includes this
// header.  Both are thin bodies over the three building blocks in chunk_seq.h.
//
// Depends on delayed.h only so that a caller including primitives.h alone still
// gets the fused layer -- the six bodies here are all eager.

#ifndef PLAID_PRIMITIVES_H
#define PLAID_PRIMITIVES_H

#include <fcntl.h>
#include <liburing.h>
#include <math.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/Primitives/delayed.h"
#include "absl/log/check.h"
#include "configs.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "utils/file_utils.h"

// ChunkMap
//
// (was ChunkSequence/Primitives/map.h)
// ============================================================================

namespace plaid {

/**
 * Apply f to every element across all chunks in seq, writing the results back
 * out as an index-ordered chunk_seq (out.chunks[i].index == i), so results are
 * directly chainable.
 *
 * Implemented as a thin body on ExternalTransform: each input chunk is mapped
 * into one or more output blocks.  When sizeof(R) > sizeof(T) an input chunk's
 * n elements may not fit in one output block, so up to FANOUT output blocks are
 * emitted per input; ExternalTransform sorts + densifies the emitted indices.
 *
 * Relies on the input's index-ordered invariant (seq.chunks[i].index == i).
 *
 * @tparam T  Input element type.
 * @tparam R  Output element type (defaults to T).
 */
template <typename T, typename R = T, typename F>
chunk_seq ChunkMap(const chunk_seq& seq, const std::string& result_prefix,
                   F f) {
  // Worst-case output blocks per input chunk: an input chunk holds at most
  // CHUNK_SIZE/sizeof(T) elements, each becoming one R; an output block holds
  // CHUNK_SIZE/sizeof(R).  ceil(sizeof(R)/sizeof(T)) bounds the ratio.
  constexpr size_t FANOUT = (sizeof(R) + sizeof(T) - 1) / sizeof(T);

  return ExternalTransform<T, R>(
      seq, result_prefix,
      [f](const T* in, size_t n, size_t index, const ChunkEmitter<R>& emit) {
        const size_t cap = emit.out_cap();
        size_t produced = 0, sub = 0;
        // do/while so empty input chunks (n == 0) still emit one empty
        // output chunk, preserving the chunk-for-chunk structure.
        do {
          const size_t cnt = std::min(cap, n - produced);
          R* out = emit.alloc();
          for (size_t i = 0; i < cnt; i++) out[i] = f(in[produced + i]);
          memset((char*)out + cnt * sizeof(R), 0, CHUNK_SIZE - cnt * sizeof(R));
          emit.emit(out, cnt, index * FANOUT + sub);
          produced += cnt;
          sub++;
        } while (produced < n);
      },
      /*max_out_per_input=*/FANOUT);
}

}  // namespace plaid

// ChunkReduce
//
// (was ChunkSequence/Primitives/reduce.h)
// ============================================================================

namespace plaid {

/**
 * Reduce all elements across every chunk in seq using a parlay-compatible
 * monoid (monoid.identity, monoid(a, b)).
 *
 * Each parlay worker accumulates a local partial via RemoveWorker, then
 * parlay::reduce combines the per-worker results with the same monoid.
 *
 * @tparam T       Element type stored in the chunks.
 * @tparam R       Accumulator type (defaults to T).
 * @tparam Monoid  Type providing identity and operator()(R, T) -> R.
 */
template <typename T, typename R = T, typename Monoid>
R ChunkReduce(const chunk_seq& seq, Monoid monoid) {
  auto locals = RemoveWorker<T>(seq, /*reader_threads=*/10,
                                [&](ChunkSequenceReader<T>& reader) {
                                  R local = monoid.identity;
                                  while (true) {
                                    auto [ptr, n, _] = reader.Poll();
                                    if (ptr == nullptr) break;
                                    for (size_t i = 0; i < n; i++)
                                      local = monoid(local, ptr[i]);
                                    reader.allocator.Free(ptr);
                                  }
                                  return local;
                                });
  return parlay::reduce(locals, monoid);
}

}  // namespace plaid

// ChunkScan
//
// (was ChunkSequence/Primitives/scan.h)
// ============================================================================

namespace plaid {

/**
 * Exclusive prefix scan of seq under a parlay-compatible monoid
 * (monoid.identity, monoid(a, b)): out[i] = monoid(in[0], …, in[i-1]), with
 * out[0] = monoid.identity.  Returns {result_seq, total} where total is the
 * grand reduction (the parlay scan convention).
 *
 * Out-of-core two-level (block) scan.  There is one accumulator per chunk, so
 * the O(c) block-sum array fits in DRAM:
 *   1. Pass 1 (RemoveWorker): reduce each chunk independently into
 *      chunk_sums[chunk_idx].
 *   2. Sequential exclusive prefix over chunk_sums -> offset[i] (the seed for
 *      chunk i); the running accumulator after the last chunk is the total.
 *   3. Pass 2 (ExternalTransform): re-read each chunk and run a sequential
 *      exclusive scan seeded with offset[chunk_idx], emitting the result.
 *
 * The output preserves the index-ordered invariant (out.chunks[i].index == i).
 *
 * @tparam T       Input element type.
 * @tparam R       Output/accumulator element type (defaults to T).
 * @tparam Monoid  Type providing identity and operator()(R, T) -> R.
 */
template <typename T, typename R = T, typename Monoid>
std::pair<chunk_seq, R> ChunkScan(const chunk_seq& seq,
                                  const std::string& result_prefix,
                                  Monoid monoid) {
  const size_t n_chunks = seq.chunks.size();

  // ── pass 1: per-chunk reductions into chunk_sums[chunk_idx] ───────────────
  std::vector<R> chunk_sums(n_chunks);
  RemoveWorker<T>(seq, /*reader_threads=*/10,
                  [&](ChunkSequenceReader<T>& reader) {
                    while (true) {
                      auto [ptr, n, chunk_idx] = reader.Poll();
                      if (ptr == nullptr) break;
                      R local = monoid.identity;
                      for (size_t i = 0; i < n; i++)
                        local = monoid(local, ptr[i]);
                      chunk_sums[chunk_idx] = local;
                      reader.allocator.Free(ptr);
                    }
                    return 0;  // side-effect worker; result unused
                  });

  // ── step 2: exclusive prefix over chunk sums (sequential, O(c) in RAM) ────
  std::vector<R> offset(n_chunks);
  R total = monoid.identity;
  {
    R run = monoid.identity;
    for (size_t i = 0; i < n_chunks; i++) {
      offset[i] = run;
      run = monoid(run, chunk_sums[i]);
    }
    total = run;
  }

  // ── pass 2: seeded exclusive scan within each chunk, write out ────────────
  // Scan preserves length, so (as in ChunkMap) an input chunk may span FANOUT
  // output blocks when sizeof(R) > sizeof(T); the running accumulator carries
  // across those sub-blocks.
  constexpr size_t FANOUT = (sizeof(R) + sizeof(T) - 1) / sizeof(T);

  chunk_seq result = ExternalTransform<T, R>(
      seq, result_prefix,
      [&monoid, &offset](const T* in, size_t n, size_t index,
                         const ChunkEmitter<R>& emit) {
        const size_t cap = emit.out_cap();
        R run = offset[index];
        size_t produced = 0, sub = 0;
        do {
          const size_t cnt = std::min(cap, n - produced);
          R* out = emit.alloc();
          for (size_t i = 0; i < cnt; i++) {
            out[i] = run;
            run = monoid(run, in[produced + i]);
          }
          memset((char*)out + cnt * sizeof(R), 0, CHUNK_SIZE - cnt * sizeof(R));
          emit.emit(out, cnt, index * FANOUT + sub);
          produced += cnt;
          sub++;
        } while (produced < n);
      },
      /*max_out_per_input=*/FANOUT);

  return {std::move(result), total};
}

}  // namespace plaid

// ChunkFilter
//
// (was ChunkSequence/Primitives/filter.h)
// ============================================================================

namespace plaid {

// FILTER_BATCH_SIZE now lives in delayed.h (see the note there).

/**
 * Filter every element across all chunks in seq, writing survivors as a tightly
 * packed, index-ordered chunk_seq (all output chunks but the last hold exactly
 * ELEMS_PER_CHUNK elements).  Element order is preserved.
 *
 * A thin producer on top of the streaming dense-pack driver (DensePackStream,
 * halo=0): the driver owns the persistent reader and the carry/scatter packing;
 * per chunk we copy survivors into a run in logical order.
 *
 * @tparam T  Element type (must match the type stored in the chunk_seq).
 * @tparam F  Predicate type; must be callable as bool(T).
 */
template <typename T, typename F>
chunk_seq ChunkFilter(const chunk_seq& seq, const std::string& result_prefix,
                      F pred) {
  if (seq.chunks.empty()) return {};

  return DensePackStream<T, T>(seq, result_prefix, /*halo=*/0,
                               [pred](const T* buf, size_t n, uint64_t /*gpos*/,
                                      const T* /*halo*/, size_t /*halo_n*/) {
                                 parlay::sequence<T> out;
                                 for (size_t j = 0; j < n; j++)
                                   if (pred(buf[j])) out.push_back(buf[j]);
                                 return out;
                               });
}

}  // namespace plaid

// ChunkFlatTabulate
//
// (was ChunkSequence/Primitives/flat_tabulate.h)
// ============================================================================

namespace plaid {
namespace detail {

/**
 * A produced batch for ChunkFlatTabulate: owns the per-virtual-chunk result
 * sequences.  run(b) reads results[b].data() from the settled Batch, which is
 * move-stable (the outer vector's element storage is heap-allocated), so the
 * pointer is valid even when parlay::sequence uses its small-buffer form.
 */
template <typename R>
struct FlatBatch {
  std::vector<parlay::sequence<R>> results;
  size_t size() const { return results.size(); }
  DensePackRun<R> run(size_t b) const {
    return {results[b].data(), results[b].size()};
  }
};

}  // namespace detail

/**
 * Out-of-core analogue of parlay::flatten(parlay::tabulate(num_chunks, f)).
 * The generative sibling of ChunkFlatMap (chunk_flat_map.h), which maps over a
 * stored input chunk_seq instead of an index range.
 *
 * Divides [0, n) into virtual chunks of size epct = CHUNK_SIZE / sizeof(R),
 * calling f(start, end) once per chunk in parallel.  f must return a
 * parlay::sequence<R> of that range's survivors in order.  Results are packed
 * densely (via DensePack) into an index-ordered chunk_seq.
 *
 * @tparam R  Output element type.
 * @tparam F  Callable: (size_t start, size_t end) -> parlay::sequence<R>
 */
template <typename R, typename F>
chunk_seq ChunkFlatTabulate(size_t n, const std::string& result_prefix, F f,
                            plaid::storage st = plaid::default_storage()) {
  if (n == 0) return {};

  const size_t epct = CHUNK_SIZE / sizeof(R);
  const size_t num_virtual = (n + epct - 1) / epct;

  return DensePack<R>(num_virtual, result_prefix,
                      [&, n, epct](size_t base, size_t batch_n) {
                        detail::FlatBatch<R> batch;
                        batch.results.resize(batch_n);
                        // Virtual chunks are produced in index order (no reader
                        // completion scrambling), so no sort is needed before
                        // packing.
                        parlay::parallel_for(
                            0, batch_n,
                            [&](size_t i) {
                              const size_t ci = base + i;
                              const size_t start = ci * epct;
                              const size_t end = std::min(start + epct, n);
                              batch.results[i] = f(start, end);
                            },
                            /*granularity=*/1);
                        return batch;
                      },
                      st);
}

}  // namespace plaid

// ChunkFlatMap
//
// (was ChunkSequence/Primitives/flat_map.h)
// ============================================================================

namespace plaid {

/**
 * Out-of-core analogue of parlay::flatten(parlay::map(seq, f)) with an optional
 * forward halo: the stored-input sibling of ChunkFlatTabulate (which generates
 * its input from an index range).
 *
 * Streams `seq` chunk-by-chunk off one persistent reader and calls f on each in
 * parallel (via DensePackStream), packing the returned per-chunk sequences
 * densely into an index-ordered chunk_seq.  Each call receives, besides its own
 * chunk, a read-only view of the following chunk's first `halo` elements — the
 * "forward halo" — so a body can catch events that straddle a chunk boundary
 * (e.g. a pattern match that starts in this chunk and finishes in the next).
 * The halo is the next chunk's head as delivered by the *same* streaming reader
 * (no separate seam read); a chunk's compute simply waits until both it and its
 * right neighbor have landed.  `halo == 0` gives a plain per-chunk flat-map
 * with no neighbor access.
 *
 * The body must report only outputs "belonging to" its own chunk — i.e. events
 * whose logical start falls in [global_start, global_start + n) — so the halo
 * is used purely as lookahead and no output is double-counted.
 *
 * Requires halo < CHUNK_SIZE/sizeof(T) (the halo is satisfiable from the single
 * next chunk) and a dense input (every chunk but the last full — the library
 * invariant), so only the final chunk can be short.  At the very last chunk of
 * the sequence the halo is empty (halo_n == 0, halo may be nullptr).
 *
 * @tparam T  Input element type (must match the chunk_seq).
 * @tparam R  Output element type (sizeof(R) <= 8, the DensePack on-disk limit).
 * @tparam F  Callable: (const T* data, size_t n, uint64_t global_start,
 *             const T* halo, size_t halo_n) -> parlay::sequence<R>
 */
template <typename T, typename R, typename F>
chunk_seq ChunkFlatMap(const chunk_seq& seq, const std::string& result_prefix,
                       size_t halo, F f) {
  if (seq.chunks.empty()) return {};
  CHECK(halo < CHUNK_SIZE / sizeof(T))
      << "ChunkFlatMap: halo must be smaller than one chunk";

  // Thin producer over the streaming dense-pack driver: it owns the persistent
  // reader, sources each chunk's halo from the next chunk in the same stream,
  // and packs the emitted runs.  `f` is the per-chunk body verbatim.
  return DensePackStream<T, R>(seq, result_prefix, halo, f);
}

/**
 * Elementwise sibling of the chunk-buffer ChunkFlatMap above: the literal
 * out-of-core analogue of parlay::flatten(parlay::map(seq, f)) -- the
 * stored-input counterpart to ChunkFlatTabulate (which generates its input
 * from an index range instead of reading a chunk_seq).
 *
 * Streams `seq` chunk-by-chunk (via DensePackStream, no halo -- an
 * elementwise mapping never needs cross-chunk lookahead) and calls f on each
 * element in logical order, concatenating every element's (possibly
 * variable-length, possibly empty) result into that chunk's output run before
 * dense-packing.
 *
 * @tparam T  Input element type (must match the chunk_seq).
 * @tparam R  Output element type (sizeof(R) <= 8, the DensePack on-disk limit).
 * @tparam F  Callable: T -> parlay::sequence<R>
 */
template <typename T, typename R, typename F>
chunk_seq ChunkFlatMap(const chunk_seq& seq, const std::string& result_prefix,
                       F f) {
  if (seq.chunks.empty()) return {};
  return DensePackStream<T, R>(seq, result_prefix, /*halo=*/0,
                               [f](const T* buf, size_t n, uint64_t /*gpos*/,
                                   const T* /*halo*/, size_t /*halo_n*/) {
                                 parlay::sequence<R> out;
                                 for (size_t j = 0; j < n; j++) {
                                   auto r = f(buf[j]);
                                   for (auto&& x : r)
                                     out.push_back(std::move(x));
                                 }
                                 return out;
                               });
}

}  // namespace plaid

#endif  // PLAID_PRIMITIVES_H
