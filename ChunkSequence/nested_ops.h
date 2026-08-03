#ifndef NESTED_OPS_H
#define NESTED_OPS_H

#include <unistd.h>

#include <cstring>
#include <string>

#include "ChunkSequence/dense_pack.h"
#include "ChunkSequence/external_engine.h"
#include "ChunkSequence/nested_seq.h"
#include "absl/log/check.h"
#include "configs.h"
#include "parlay/sequence.h"

namespace ChunkSequenceOps {

/**
 * NestedFlatten — concatenate every inner sequence's values into a single dense
 * flat chunk_seq (all output chunks but the last hold exactly ELEMS_PER_CHUNK
 * elements).  This is the `flatten` half of `flatten(map(...))`.
 *
 * Within a nested chunk the rows are already stored contiguously in row order,
 * and chunks are index-ordered, so streaming each chunk's `used` elements and
 * re-densifying across chunks (dropping each chunk's internal tail padding)
 * yields exactly the row-order concatenation.  Built on DensePackStream with an
 * identity body (return all n elements): one read pass + one write pass.
 */
template <typename T>
chunk_seq NestedFlatten(const nested_seq<T>& ns,
                        const std::string& result_prefix) {
  const chunk_seq view = ns.raw_view();
  if (view.chunks.empty()) return {};
  return DensePackStream<T, T>(
      view, result_prefix, /*halo=*/0,
      [](const T* buf, size_t n, uint64_t /*gpos*/, const T* /*halo*/,
         size_t /*halo_n*/) { return parlay::sequence<T>(buf, buf + n); });
}

/**
 * NestedMap — apply `g` to every inner sequence, producing a new nested_seq.
 *
 *   g(const T* data, size_t len) -> parlay::sequence<R>
 *
 * v1 is STRUCTURE-PRESERVING: g must return a sequence of the SAME length as its
 * input (asserted).  Because structure is preserved, the output shares the
 * input's seq_len_scan and every (first_seq, num_seqs), and each input chunk maps
 * to exactly one output chunk — so this is embarrassingly parallel per chunk,
 * built directly on the eager ExternalTransform engine (chunk = io_uring read =
 * unit of parallelism; the metadata only says how to carve a chunk into whole
 * inner sequences).  Requires sizeof(R) <= sizeof(T) so the transformed chunk
 * still fits.  (A length-changing map would need output re-packing — deferred.)
 */
template <typename T, typename R = T, typename G>
nested_seq<R> NestedMap(const nested_seq<T>& ns,
                        const std::string& result_prefix, G g) {
  static_assert(sizeof(R) <= sizeof(T),
                "NestedMap (v1) requires sizeof(R) <= sizeof(T) so the "
                "structure-preserving output chunk still fits");
  static_assert(CHUNK_SIZE % sizeof(R) == 0,
                "sizeof(R) must divide CHUNK_SIZE for O_DIRECT alignment");

  const chunk_seq in_view = ns.raw_view();
  chunk_seq out_flat = ExternalTransform<T, R>(
      in_view, result_prefix,
      [&ns, &g](const T* in, size_t n, size_t index,
                const ChunkEmitter<R>& emit) {
        const nested_chunk& nc = ns.chunks[index];
        const size_t base = ns.seq_len_scan[nc.first_seq];
        CHECK(n == ns.seq_len_scan[nc.first_seq + nc.num_seqs] - base)
            << "NestedMap: chunk " << index << " element count mismatch";

        R* out = emit.alloc();
        size_t out_fill = 0;
        for (size_t k = 0; k < nc.num_seqs; k++) {
          const size_t gseq = nc.first_seq + k;
          const size_t local_off = ns.seq_len_scan[gseq] - base;
          const size_t l = ns.seq_len_scan[gseq + 1] - ns.seq_len_scan[gseq];
          parlay::sequence<R> r = g(in + local_off, l);
          CHECK(r.size() == l)
              << "NestedMap (v1) is structure-preserving: g returned "
              << r.size() << " elements for an inner sequence of length " << l;
          if (l > 0) memcpy(out + out_fill, r.data(), l * sizeof(R));
          out_fill += l;
        }
        memset((char*)out + out_fill * sizeof(R), 0,
               CHUNK_SIZE - out_fill * sizeof(R));
        emit.emit(out, out_fill, index);
      },
      /*max_out_per_input=*/1, /*compact=*/true);

  // Structure is identical to the input; only the raw chunk headers changed.
  nested_seq<R> out;
  out.seq_len_scan = ns.seq_len_scan;
  out.chunks.resize(ns.chunks.size());
  CHECK(out_flat.chunks.size() == ns.chunks.size())
      << "NestedMap: output chunk count " << out_flat.chunks.size()
      << " != input " << ns.chunks.size();
  for (size_t i = 0; i < ns.chunks.size(); i++) {
    // ExternalTransform compacted indices to 0..k-1 in input order, so
    // out_flat.chunks[i] corresponds to input chunk i.
    out.chunks[i].raw = out_flat.chunks[i];
    out.chunks[i].first_seq = ns.chunks[i].first_seq;
    out.chunks[i].num_seqs = ns.chunks[i].num_seqs;
  }
  return out;
}

/**
 * NestedMapReduce is the fused inner map+reduce: map each element through
 * `map_fn` and fold with `monoid` in a SINGLE streaming read pass, so the mapped
 * values never materialize (not in DRAM, not on disk).  Contrast the unfused
 * `NestedReduce(NestedMap(ns, g), monoid)`, which writes a whole mapped
 * nested_seq to disk in between.  It consumes the nested_seq only as input; the
 * output is a flat per-inner-sequence array (result[i] = reduce over seq i), so
 * no packing/materialization is needed.
 *
 * Because every inner sequence lives wholly within one chunk (invariant (1)),
 * the per-sequence output indices touched by different chunks are DISJOINT, so
 * each worker writes out[first_seq + k] directly with NO locks and NO
 * boundary merge — the concrete simplification the balanced layout buys over
 * ChunkSegmentedReduce.  The result array is DRAM-resident (consistent with the
 * degree_scan assumption).
 */
template <typename T, typename R, typename MapFn, typename Monoid>
parlay::sequence<R> NestedMapReduce(const nested_seq<T>& ns, MapFn map_fn,
                                    Monoid monoid, size_t reader_threads = 10) {
  const size_t total = ns.total_seqs();
  parlay::sequence<R> out(total, monoid.identity);  // identity => empty seq handled
  const chunk_seq view = ns.raw_view();

  RemoveWorker<T>(view, reader_threads, [&](ChunkSequenceReader<T>& reader) {
    while (true) {
      auto [ptr, n, idx] = reader.Poll();
      if (ptr == nullptr) break;
      const nested_chunk& nc = ns.chunks[idx];
      const size_t base = ns.seq_len_scan[nc.first_seq];
      for (size_t k = 0; k < nc.num_seqs; k++) {
        const size_t gseq = nc.first_seq + k;
        const size_t local_off = ns.seq_len_scan[gseq] - base;
        const size_t l = ns.seq_len_scan[gseq + 1] - ns.seq_len_scan[gseq];
        R acc = monoid.identity;
        // fused: map each element and fold it immediately — no intermediate.
        for (size_t j = 0; j < l; j++) acc = monoid(acc, map_fn(ptr[local_off + j]));
        out[gseq] = acc;  // disjoint across chunks => no synchronization
      }
      reader.allocator.Free(ptr);
    }
    return (char)0;  // placeholder per-worker accumulator (unused)
  });

  return out;
}

// NestedReduce — reduce each inner sequence to a scalar, mapping each element
// through `elem_map` first.  That is exactly a fused inner map+reduce, so this
// is a thin alias for NestedMapReduce (kept for the "reduce" framing and the
// existing callers, e.g. the pull NestedBFS relaxation).
template <typename T, typename R, typename ElemFn, typename Monoid>
parlay::sequence<R> NestedReduce(const nested_seq<T>& ns, ElemFn elem_map,
                                 Monoid monoid, size_t reader_threads = 10) {
  return NestedMapReduce<T, R>(ns, elem_map, monoid, reader_threads);
}

/**
 * NestedMapReduceMaterialized — the UNFUSED spelling of the same computation:
 * write the mapped nested_seq to disk with NestedMap, then read it back and
 * reduce it with NestedReduce.  Same result as NestedMapReduce, but it moves the
 * mapped data across the SSDs (1 read + 1 write for the map, then 1 read for the
 * reduce) instead of the fused single read pass.  Exposed as its own function so
 * the two can be timed head-to-head (see benchmarks/nested_map_reduce_compare).
 * `scratch_prefix` names the intermediate's drive files, which are unlinked
 * before returning.  Requires sizeof(R) <= sizeof(T) (NestedMap's constraint).
 */
template <typename T, typename R, typename MapFn, typename Monoid>
parlay::sequence<R> NestedMapReduceMaterialized(const nested_seq<T>& ns,
                                                MapFn map_fn, Monoid monoid,
                                                const std::string& scratch_prefix,
                                                size_t reader_threads = 10) {
  static_assert(sizeof(R) <= sizeof(T),
                "the materialized path uses NestedMap, which needs "
                "sizeof(R) <= sizeof(T)");
  nested_seq<R> mapped = NestedMap<T, R>(
      ns, scratch_prefix, [map_fn](const T* d, size_t l) {
        parlay::sequence<R> r(l);
        for (size_t k = 0; k < l; k++) r[k] = map_fn(d[k]);
        return r;
      });
  parlay::sequence<R> out = NestedReduce<R, R>(
      mapped, [](R x) { return x; }, monoid, reader_threads);
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    unlink(GetFileName(scratch_prefix, d).c_str());
  return out;
}

}  // namespace ChunkSequenceOps

#endif  // NESTED_OPS_H
