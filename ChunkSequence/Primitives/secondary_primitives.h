// secondary_primitives.h -- the rest of the eager layer.
//
//   ChunkSegmentedReduce                RemoveWorker, arbitrary contiguous
//   segments pack / pack_if / pack_value         DensePack gates
//   ChunkHistogramByIndex / ByKey       RemoveWorker bucket-count fold
//   ChunkFindIf                         RemoveWorker, per-worker min matching
//   index ChunkPartition                      own reader+writer k-way split
//   flatten                             concatenate chunk_seqs by reindexing
//   materialize                         read a chunk_seq (or delayed source)
//   into DRAM cut                                 chunk-aligned slice / shift /
//   guard-limb scan_find / scan_size / linear_find single-element probes
//
// Split out from primitives.h so the six core primitives read on their own;
// this header includes primitives.h, so including it gets you the whole eager
// layer.

#ifndef PLAID_SECONDARY_PRIMITIVES_H
#define PLAID_SECONDARY_PRIMITIVES_H

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
#include "ChunkSequence/Primitives/primitives.h"
#include "absl/log/check.h"
#include "configs.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "utils/file_utils.h"

// ChunkSegmentedReduce
//
// (was ChunkSequence/Primitives/segmented_reduce.h)
// ============================================================================

namespace plaid {

// ChunkSegmentedReduce is somewhat hacky in that we would much prefer a more
// general streaming pass method

template <typename T, typename R, typename ElemFn, typename Monoid>
parlay::sequence<R> ChunkSegmentedReduce(const chunk_seq& seq,
                                         const parlay::sequence<size_t>& bounds,
                                         ElemFn elem_to_val, Monoid monoid,
                                         size_t reader_threads = 10) {
  static_assert(CHUNK_SIZE % sizeof(T) == 0,
                "sizeof(T) must divide CHUNK_SIZE");
  const size_t ept =
      CHUNK_SIZE / sizeof(T);  // per-type elements-per-chunk (NOT the global
                               // ELEMS_PER_CHUNK, which is sized for uint64_t)
  const size_t num_segments = bounds.size() - 1;
  const size_t n_chunks = seq.chunks.size();

  parlay::sequence<R> out(num_segments, monoid.identity);
  std::vector<std::vector<std::pair<size_t, R>>> boundary(n_chunks);

  RemoveWorker<T>(seq, reader_threads, [&](ChunkSequenceReader<T>& reader) {
    while (true) {
      auto [ptr, n, chunk_idx] = reader.Poll();
      if (ptr == nullptr) break;
      if (n > 0) {
        const size_t global_start = chunk_idx * ept;
        const size_t global_end = global_start + n;

        const size_t v_lo =
            (size_t)(std::upper_bound(bounds.begin(), bounds.end(),
                                      global_start) -
                     bounds.begin()) -
            1;
        const size_t v_hi =
            (size_t)(std::upper_bound(bounds.begin(), bounds.end(),
                                      global_end - 1) -
                     bounds.begin()) -
            1;

        // A segment is "boundary" if it started in an earlier chunk (v_lo,
        // when bounds[v_lo] < global_start) or continues into a later one
        // (v_hi, when bounds[v_hi+1] > global_end); such segments need the
        // sequential merge below. Everything else is fully contained in
        // this chunk and owned by no other chunk, so it's safe to write
        // directly with no synchronization.
        auto finalize = [&](size_t v, R val) {
          const bool is_boundary = (v == v_lo && bounds[v_lo] < global_start) ||
                                   (v == v_hi && bounds[v_hi + 1] > global_end);
          if (is_boundary)
            boundary[chunk_idx].push_back({v, val});
          else
            out[v] = val;
        };

        size_t cur_v = v_lo;
        R cur_val = monoid.identity;
        for (size_t i = 0; i < n; i++) {
          const size_t g = global_start + i;
          while (g >= bounds[cur_v + 1]) {
            finalize(cur_v, cur_val);
            cur_v++;
            cur_val = monoid.identity;
          }
          cur_val = monoid(cur_val, elem_to_val(ptr[i]));
        }
        finalize(cur_v, cur_val);
      }
      reader.allocator.Free(ptr);
    }
    return 0;  // side-effect worker; result unused
  });

  // Sequential O(n_chunks) merge (DRAM only): chain consecutive boundary
  // entries for the same segment in chunk order -- this is what correctly
  // handles a segment spanning many consecutive whole chunks, not just a
  // simple two-chunk seam.
  bool have_open = false;
  size_t open_v = 0;
  R open_val = monoid.identity;
  for (size_t c = 0; c < n_chunks; c++) {
    for (auto& [v, val] : boundary[c]) {
      if (have_open && v == open_v) {
        open_val = monoid(open_val, val);
      } else {
        if (have_open) out[open_v] = open_val;
        open_v = v;
        open_val = val;
        have_open = true;
      }
    }
  }
  if (have_open) out[open_v] = open_val;

  return out;
}

}  // namespace plaid

// pack / pack_if / pack_value
//
// (was ChunkSequence/Primitives/pack.h)
// ============================================================================

namespace plaid {

// Retained name for the input-chunk batch size (== the DensePack batch size).
// chunk_delayed.h's own Pack terminal reads this.
static constexpr size_t Pack_BATCH_SIZE = DENSE_PACK_BATCH_SIZE;

namespace detail {

/**
 * A produced batch for ChunkPack: owns the reader-pool buffers holding this
 * batch's survivors (compacted in place) and frees them back to the pool when
 * destroyed.  run(b) reads pointers from the settled Batch, so they stay valid
 * throughout the DensePack batch.
 */
template <typename T>
struct PackBatch {
  std::unique_ptr<ChunkSequenceReader<T>> reader;  // keeps the pool alive
  std::vector<T*> bufs;        // one per virtual chunk, index-sorted order
  std::vector<size_t> counts;  // survivors in each buf

  PackBatch() = default;
  PackBatch(PackBatch&&) = default;
  PackBatch& operator=(PackBatch&&) = default;
  PackBatch(const PackBatch&) = delete;
  PackBatch& operator=(const PackBatch&) = delete;
  ~PackBatch() {
    if (reader)
      for (T* b : bufs) reader->allocator.Free(b);
  }

  size_t size() const { return bufs.size(); }
  DensePackRun<T> run(size_t b) const { return {bufs[b], counts[b]}; }
};

}  // namespace detail

/**
 * Pack every element across all chunks in seq for which the corresponding flag
 * in boolean_seq is true, writing survivors as a tightly packed, index-ordered
 * chunk_seq (all output chunks but the last hold exactly ELEMS_PER_CHUNK
 * elements).  Element order is preserved.
 *
 * boolean_seq is a single flat boolean over every element of the sequence, in
 * logical (index) order: boolean_seq[g] gates the g-th element of seq, where g
 * runs across chunk boundaries (chunk 0's elements first, then chunk 1's, ...).
 * Its length must equal the total element count of seq.
 *
 * A thin producer on top of DensePack: each batch reads its index-contiguous
 * slice with its own reader, sorts by index (the reader is unordered), and
 * compacts survivors to the front of each buffer in parallel.  DensePack owns
 * the carry/prefix-sum/scatter packing and the writer.
 *
 * @tparam T  Element type (must match the type stored in the chunk_seq).
 */
template <typename T>
chunk_seq pack(const chunk_seq& seq, const std::string& result_prefix,
               const parlay::sequence<bool>& boolean_seq) {
  const size_t n_in = seq.chunks.size();
  if (n_in == 0) return {};

  // Global element offset of the first element of each chunk, in index order.
  // chunk.used is a byte count, so divide by sizeof(T) to get elements.
  // elem_offset[c] = number of elements in chunks[0 .. c-1].
  std::vector<size_t> elem_offset(n_in);
  size_t acc = 0;
  for (size_t c = 0; c < n_in; c++) {
    elem_offset[c] = acc;
    acc += seq.chunks[c].used / sizeof(T);
  }

  return DensePack<T>(n_in, result_prefix, [&](size_t base, size_t batch_n) {
    // Read this batch's contiguous slice [base, base+batch_n) with its
    // own reader, so completions can only belong to this batch.
    chunk_seq sub;
    sub.chunks.assign(seq.chunks.begin() + base,
                      seq.chunks.begin() + base + batch_n);
    auto reader = std::make_unique<ChunkSequenceReader<T>>();
    reader->PrepChunks(sub);
    reader->Start(5, 32, 16);

    struct FC {
      T* buf;
      size_t n;
      size_t idx;
    };
    std::vector<FC> fc(batch_n);
    for (size_t i = 0; i < batch_n; i++) {
      auto [ptr, n, cidx] = reader->Poll();
      fc[i] = {ptr, n, cidx};
    }
    // Restore logical order before packing.
    std::sort(fc.begin(), fc.end(),
              [](const FC& a, const FC& b) { return a.idx < b.idx; });

    // Compact survivors to the front of each buffer, in parallel.
    detail::PackBatch<T> batch;
    batch.reader = std::move(reader);
    batch.bufs.resize(batch_n);
    batch.counts.resize(batch_n);
    parlay::parallel_for(
        0, batch_n,
        [&](size_t b) {
          T* buf = fc[b].buf;
          const size_t n = fc[b].n;
          // fc[b].idx is the global chunk index; its elements map to the
          // slice of boolean_seq starting at this offset.
          const size_t g0 = elem_offset[fc[b].idx];
          size_t s = 0;
          for (size_t j = 0; j < n; j++)
            if (boolean_seq[g0 + j]) buf[s++] = buf[j];
          batch.bufs[b] = buf;
          batch.counts[b] = s;
        },
        /*granularity=*/1);

    return batch;
  });
}

/**
 * Overload of pack whose selector lives out-of-core: instead of a DRAM
 * parlay::sequence<bool> spanning every element, the flags are themselves a
 * chunk_seq of bool that is *chunk-parallel* to seq — same number of chunks,
 * and flag_seq.chunks[i] gates seq.chunks[i] element-for-element (identical
 * per-chunk element counts, index-aligned).  This is exactly the shape produced
 * by a ChunkMap<_, bool>(seq, ...) (or any transform that preserves seq's
 * chunking), so callers never have to materialize an n-element boolean in DRAM.
 *
 * Semantics are otherwise identical to the parlay::sequence<bool> overload:
 * survivors are written tightly packed and index-ordered, element order
 * preserved.
 *
 * A thin producer on top of DensePack, like the other overload, except each
 * batch reads *two* index-contiguous slices — the data slice from seq and its
 * parallel flag slice from flag_seq — each with its own reader.  The flag
 * buffers are consumed during compaction, so they are freed back to their pool
 * as soon as the batch is compacted (only the data buffers must outlive the
 * producer, since DensePack reads them via run(b)).
 *
 * @tparam T  Element type stored in seq (flag_seq stores bool).
 */
template <typename T>
chunk_seq pack(const chunk_seq& seq, const std::string& result_prefix,
               const chunk_seq& flag_seq) {
  const size_t n_in = seq.chunks.size();
  if (n_in == 0) return {};
  // flag_seq must be chunk-parallel to seq (one flag chunk per data chunk).
  assert(flag_seq.chunks.size() == n_in &&
         "flag_seq must have the same chunk count as seq");

  return DensePack<T>(n_in, result_prefix, [&](size_t base, size_t batch_n) {
    // Read this batch's data slice and its parallel flag slice, each
    // with its own reader so completions can only belong to this batch.
    chunk_seq data_sub, flag_sub;
    data_sub.chunks.assign(seq.chunks.begin() + base,
                           seq.chunks.begin() + base + batch_n);
    flag_sub.chunks.assign(flag_seq.chunks.begin() + base,
                           flag_seq.chunks.begin() + base + batch_n);

    auto reader = std::make_unique<ChunkSequenceReader<T>>();
    reader->PrepChunks(data_sub);
    reader->Start(5, 32, 16);

    // The flag reader is drained within this producer, so it may be a
    // local that is destroyed at scope exit (after its buffers are
    // freed), unlike the data reader which the batch must keep alive.
    ChunkSequenceReader<bool> flag_reader;
    flag_reader.PrepChunks(flag_sub);
    flag_reader.Start(5, 32, 16);

    struct FC {
      T* buf;
      size_t n;
      size_t idx;
    };
    std::vector<FC> fc(batch_n);
    for (size_t i = 0; i < batch_n; i++) {
      auto [ptr, n, cidx] = reader->Poll();
      fc[i] = {ptr, n, cidx};
    }
    // Restore logical order before packing.
    std::sort(fc.begin(), fc.end(),
              [](const FC& a, const FC& b) { return a.idx < b.idx; });

    struct BC {
      bool* buf;
      size_t n;
      size_t idx;
    };
    std::vector<BC> bc(batch_n);
    for (size_t i = 0; i < batch_n; i++) {
      auto [ptr, n, cidx] = flag_reader.Poll();
      bc[i] = {ptr, n, cidx};
    }
    std::sort(bc.begin(), bc.end(),
              [](const BC& a, const BC& b) { return a.idx < b.idx; });

    // Compact survivors to the front of each data buffer, in parallel,
    // gated by the same-index flag buffer.  Both slices cover exactly
    // the index range [base, base+batch_n), so after sorting each by
    // index, position b lines up in both.
    detail::PackBatch<T> batch;
    batch.reader = std::move(reader);
    batch.bufs.resize(batch_n);
    batch.counts.resize(batch_n);
    parlay::parallel_for(
        0, batch_n,
        [&](size_t b) {
          assert(fc[b].idx == bc[b].idx &&
                 "flag chunk not aligned to data chunk");
          assert(bc[b].n >= fc[b].n && "flag chunk shorter than data chunk");
          T* buf = fc[b].buf;
          const bool* flag = bc[b].buf;
          const size_t n = fc[b].n;
          size_t s = 0;
          for (size_t j = 0; j < n; j++)
            if (flag[j]) buf[s++] = buf[j];
          batch.bufs[b] = buf;
          batch.counts[b] = s;
        },
        /*granularity=*/1);

    // Flag buffers are fully consumed; return them to their pool before
    // flag_reader is destroyed at scope exit.
    for (const auto& e : bc) flag_reader.allocator.Free(e.buf);

    return batch;
  });
}

/**
 * Like the chunk-parallel pack overload, but the gate is a predicate evaluated
 * against a chunk-parallel selector of arbitrary type U rather than a
 * precomputed bool chunk_seq.  keep_seq must be chunk-parallel to seq (one
 * selector chunk per data chunk, identical per-chunk element counts, index
 * aligned); element g of seq survives iff pred(keep_seq[g]) is true.
 *
 * This exists to fuse "map to a flag, force the flag to disk, then pack by it"
 * into a single pass: callers that already hold a chunk-parallel selector on
 * SSD (e.g. bucket ids from ChunkMap) can pack directly on `pred(id) == ...`
 * without materializing an intermediate bool chunk_seq — saving a full read of
 * the selector, a full write of the flags, and a full read of the flags.
 *
 * @tparam T     Element type stored in seq.
 * @tparam U     Element type stored in keep_seq (the selector).
 * @tparam Pred  Callable U -> bool.
 */
template <typename T, typename U, typename Pred>
chunk_seq pack_if(const chunk_seq& seq, const std::string& result_prefix,
                  const chunk_seq& keep_seq, Pred pred) {
  const size_t n_in = seq.chunks.size();
  if (n_in == 0) return {};
  // keep_seq must be chunk-parallel to seq (one selector chunk per data chunk).
  assert(keep_seq.chunks.size() == n_in &&
         "keep_seq must have the same chunk count as seq");

  return DensePack<T>(n_in, result_prefix, [&](size_t base, size_t batch_n) {
    // Read this batch's data slice and its parallel selector slice, each
    // with its own reader so completions can only belong to this batch.
    chunk_seq data_sub, keep_sub;
    data_sub.chunks.assign(seq.chunks.begin() + base,
                           seq.chunks.begin() + base + batch_n);
    keep_sub.chunks.assign(keep_seq.chunks.begin() + base,
                           keep_seq.chunks.begin() + base + batch_n);

    auto reader = std::make_unique<ChunkSequenceReader<T>>();
    reader->PrepChunks(data_sub);
    reader->Start(5, 32, 16);

    // The selector reader is drained within this producer, so it may be
    // a local destroyed at scope exit (after its buffers are freed),
    // unlike the data reader which the batch must keep alive.
    ChunkSequenceReader<U> keep_reader;
    keep_reader.PrepChunks(keep_sub);
    keep_reader.Start(5, 32, 16);

    struct FC {
      T* buf;
      size_t n;
      size_t idx;
    };
    std::vector<FC> fc(batch_n);
    for (size_t i = 0; i < batch_n; i++) {
      auto [ptr, n, cidx] = reader->Poll();
      fc[i] = {ptr, n, cidx};
    }
    // Restore logical order before packing.
    std::sort(fc.begin(), fc.end(),
              [](const FC& a, const FC& b) { return a.idx < b.idx; });

    struct KC {
      U* buf;
      size_t n;
      size_t idx;
    };
    std::vector<KC> kc(batch_n);
    for (size_t i = 0; i < batch_n; i++) {
      auto [ptr, n, cidx] = keep_reader.Poll();
      kc[i] = {ptr, n, cidx};
    }
    std::sort(kc.begin(), kc.end(),
              [](const KC& a, const KC& b) { return a.idx < b.idx; });

    // Compact survivors to the front of each data buffer, in parallel,
    // gated by pred() over the same-index selector buffer.  Both slices
    // cover the index range [base, base+batch_n), so after sorting each
    // by index, position b lines up in both.
    detail::PackBatch<T> batch;
    batch.reader = std::move(reader);
    batch.bufs.resize(batch_n);
    batch.counts.resize(batch_n);
    parlay::parallel_for(
        0, batch_n,
        [&](size_t b) {
          assert(fc[b].idx == kc[b].idx &&
                 "selector chunk not aligned to data chunk");
          assert(kc[b].n >= fc[b].n &&
                 "selector chunk shorter than data chunk");
          T* buf = fc[b].buf;
          const U* keep = kc[b].buf;
          const size_t n = fc[b].n;
          size_t s = 0;
          for (size_t j = 0; j < n; j++)
            if (pred(keep[j])) buf[s++] = buf[j];
          batch.bufs[b] = buf;
          batch.counts[b] = s;
        },
        /*granularity=*/1);

    // Selector buffers are fully consumed; return them to their pool
    // before keep_reader is destroyed at scope exit.
    for (const auto& e : kc) keep_reader.allocator.Free(e.buf);

    return batch;
  });
}

/**
 * Value-predicate pack: keep every element g of seq for which pred(seq[g]) is
 * true, writing survivors tightly packed and index-ordered (element order
 * preserved).  Unlike the other overloads there is no selector at all --
 * neither a DRAM boolean nor a chunk-parallel keep_seq on disk -- the predicate
 * is evaluated directly against each data element.
 *
 * This is the scalable "pack by a property of the value" primitive: callers
 * that would otherwise map values to a flag/id sequence, force it to disk, then
 * pack by it can instead pack in a single read pass, recomputing the (cheap)
 * property inline.  It reads only seq -- no selector read, no flag write, and
 * no n-element boolean in DRAM.
 *
 * A thin producer on top of DensePack, like the other overloads.
 *
 * @tparam T     Element type stored in seq.
 * @tparam Pred  Callable T -> bool.
 */
template <typename T, typename Pred>
chunk_seq pack_value(const chunk_seq& seq, const std::string& result_prefix,
                     Pred pred) {
  const size_t n_in = seq.chunks.size();
  if (n_in == 0) return {};

  return DensePack<T>(n_in, result_prefix, [&](size_t base, size_t batch_n) {
    // Read this batch's contiguous slice with its own reader, so
    // completions can only belong to this batch.
    chunk_seq sub;
    sub.chunks.assign(seq.chunks.begin() + base,
                      seq.chunks.begin() + base + batch_n);
    auto reader = std::make_unique<ChunkSequenceReader<T>>();
    reader->PrepChunks(sub);
    reader->Start(5, 32, 16);

    struct FC {
      T* buf;
      size_t n;
      size_t idx;
    };
    std::vector<FC> fc(batch_n);
    for (size_t i = 0; i < batch_n; i++) {
      auto [ptr, n, cidx] = reader->Poll();
      fc[i] = {ptr, n, cidx};
    }
    // Restore logical order before packing.
    std::sort(fc.begin(), fc.end(),
              [](const FC& a, const FC& b) { return a.idx < b.idx; });

    // Compact survivors to the front of each buffer, in parallel, gated
    // by pred() over the element itself.
    detail::PackBatch<T> batch;
    batch.reader = std::move(reader);
    batch.bufs.resize(batch_n);
    batch.counts.resize(batch_n);
    parlay::parallel_for(
        0, batch_n,
        [&](size_t b) {
          T* buf = fc[b].buf;
          const size_t n = fc[b].n;
          size_t s = 0;
          for (size_t j = 0; j < n; j++)
            if (pred(buf[j])) buf[s++] = buf[j];
          batch.bufs[b] = buf;
          batch.counts[b] = s;
        },
        /*granularity=*/1);

    return batch;
  });
}

}  // namespace plaid

// ChunkHistogramByIndex / ChunkHistogramByKey
//
// (was ChunkSequence/Primitives/histogram_by_index.h)
// ============================================================================

#define NUM_SSDS 30
#ifndef PRACTICAL_SSDS
#define PRACTICAL_SSDS 8
#endif

// template<typename T>
// parlay::sequence<size_t> ExternalHistogramByIndexExternalSeq(const chunk_seq&
// seq, const std::vector<std::string> &new_filenames, size_t num_unique){

//     //one assumption we make is that we have a dense case, otherwise this
//     counting sort idea doesn't really work
//     //because it wastes so much memory
//     constexpr size_t buffer_size_bytes = 4 << 20, buffer_size =
//     buffer_size_bytes / sizeof(T); size_t num_chunks = (num_unique +
//     buffer_size - 1) / buffer_size; //is this right? I think this is left
//     over from the iota logic where num_unique = n size_t read_count = 0;
//     std::vector<size_t>* store(num_unique) = calloc(num_unique *
//     sizeof(size_t*));
//       auto& chunk_headers = seq.ordered_underlying_sequence;
//       size_t expected_reads;
//     // size_t expected_reads = (chunk_headers.size() + NUM_SSDS - 1) /
//     NUM_SSDS; chunk_headers.size() % PRACTICAL_SSDS == 0 ? expected_reads =
//     (chunk_headers.size() / NUM_SSDS) : expected_reads = chunk_headers.size()
//     / PRACTICAL_SSDS + 1; UnorderedChunkReader<T, 4 << 20> reader;
//     reader.PrepFiles(chunk_headers); //prepfiles needs to be changed to
//     accomodate chunk headers reader.Start(); std::vector<T*>
//     store_local(PRACTICAL_SSDS);

//     while(read_count < expected_read_count){
//         //instead of calculating the expected number of read batches, maybe
//         the best way to do this is to check the reader.poll?
//         //but this seems much simpler
//         parlay::parallel_for(0, PRACTICAL_SSDS, [&](size_t i){
//             auto [ptr, size, _, index, which_chunk, filename] =
//             reader.Poll();
//             //no reason to use aligned alloc since we don't need to write
//             this store_local[i] = (size_t*)calloc(num_unique); size_t
//             buffer_index = 0; for(size_t k = 0; k < size; k++){
//                     store_local[i][ptr[k]]++;
//             }
//             });
//     //we now need to add the respective buffers back to the in-memory
//     sequence
//     //perhaps this can be done with a parlay tabulate or map or something
//     clever but a sequential add isn't too bad

//         for(int i =0; i < PRACTICAL_SSDS; i++){
//             // for(int k = 0; k < num_unique; k++){

//             //     store[k]+=store_local[i][k];

//             // }
//             store = parlay::map(store, [&](size_t j){
//                 return store[j] + store_local[i][j]; //maybe wasteful if too
//                 many 0 elements or the sequence is not large enough
//                 //to get much benefit from parallelism
//             });
//             // auto tab = parlay::tabulate(store, [&](size_t j)){
//             //     return store[j] + store_local[i][j];
//             // }
//             free(store_local[i]);
//         }
//         read_count++;
//     }

//     return store;

//     // std::sort(seq.begin(), seq.end(), [&](const chunk_header& i, const
//     chunk_header& j){
//     //     return i.index < j.index;
//     // });
//     // return seq;
// }

namespace plaid {

template <typename T>
parlay::sequence<size_t> ChunkHistogramByIndex(const chunk_seq& seq,
                                               size_t num_unique) {
  // as I understand it::
  // removeworker is a function template that starts the reader's io_uring
  // producer threads then generates one worker tas per hardware worker polling
  // a single reader an arbitrary worker takes the next chunk to enforce load
  // balancing poll blocks if the queue is empty but filling, will return nulptr
  // once all readers have finished and the queue is empty this stops the
  // workers
  auto remove_from_queue = plaid::RemoveWorker<T>(
      seq, /*reader_threads=*/10, [&](ChunkSequenceReader<T>& reader) {
        // create parlay seq init to 0 with num_unique values
        parlay::sequence<size_t> remove(num_unique, 0);
        while (true) {
          // poll once; this thread will continue and keep polling until it
          // blocks, which means there's nothing left in the queue
          auto [ptr, size, index] = reader.Poll();

          if (ptr == nullptr)
            break;  // the null should apply to all threads and the poll itself
                    // is threadsafe
          for (size_t k = 0; k < size; k++) {
            remove[ptr[k]]++;  // logical increment
          }
          reader.allocator.Free(
              ptr);  // need to free ptr to allow more reads to be polled
        }
        return remove;
      });
  parlay::sequence<size_t> total(num_unique, 0);  // final counts sequence
  for (auto& remove : remove_from_queue) {
    for (size_t j = 0; j < num_unique; j++) {
      total[j] += remove[j];
    }
  }
  return total;
}

// Keyed variant: instead of counting a chunk_seq whose elements are already the
// bucket ids, count seq's *values* by key_fn(value) -> bucket index.  This lets
// callers get per-bucket counts straight from the value sequence without first
// materializing an id chunk_seq to disk (no ChunkMap write pass, no read-back
// of the ids) -- the bucket key is derived on the fly during the single read
// pass.
//
// @tparam T       Element type stored in seq.
// @tparam KeyFn   Callable T -> integral bucket index in [0, num_buckets).
template <typename T, typename KeyFn>
parlay::sequence<size_t> ChunkHistogramByKey(const chunk_seq& seq,
                                             size_t num_buckets, KeyFn key_fn) {
  auto locals = plaid::RemoveWorker<T>(
      seq, /*reader_threads=*/10, [&](ChunkSequenceReader<T>& reader) {
        parlay::sequence<size_t> h(num_buckets, 0);
        while (true) {
          auto [ptr, size, index] = reader.Poll();
          if (ptr == nullptr) break;
          for (size_t k = 0; k < size; k++) h[(size_t)key_fn(ptr[k])]++;
          reader.allocator.Free(ptr);
        }
        return h;
      });
  parlay::sequence<size_t> total(num_buckets, 0);
  for (auto& h : locals)
    for (size_t j = 0; j < num_buckets; j++) total[j] += h[j];
  return total;
}

}  // namespace plaid

// ChunkFindIf
//
// (was ChunkSequence/Primitives/find_if.h)
// ============================================================================

namespace plaid {

/**
 * Return the logical index of the first element (smallest position) across all
 * chunks of seq for which pred is true, or the total element count n if none
 * match (matching parlay::find_if's "not found" convention).
 *
 * Ported from the Parlay_Primitives_for_MultiSSD external find_if.  Each parlay
 * worker (via RemoveWorker) scans the chunks it is handed, tracking the
 * smallest matching global index it sees; the per-worker minima are then
 * combined.  A chunk's elements start at global position idx * (CHUNK_SIZE /
 * sizeof(T)) because the index-ordered dense invariant means every chunk but
 * the last is full.  Note CHUNK_SIZE / sizeof(T) (not chunk_seq.h's uint64
 * ELEMS_PER_CHUNK), so non-uint64 element types are indexed correctly.
 *
 * @tparam T  Element type stored in the chunks.
 * @tparam F  Predicate type; must be callable as bool(T).
 */
template <typename T, typename F>
size_t ChunkFindIf(const chunk_seq& seq, F pred) {
  const size_t epct = CHUNK_SIZE / sizeof(T);

  // "Not found" sentinel = total element count.  seq.chunks.size() is the
  // *chunk* count and must not be used here, or every real match collapses to
  // that tiny value.
  size_t n = 0;
  for (const auto& c : seq.chunks) n += c.used / sizeof(T);
  if (n == 0) return 0;

  auto locals = RemoveWorker<T>(
      seq, /*reader_threads=*/10, [&, n, epct](ChunkSequenceReader<T>& reader) {
        size_t best = n;
        while (true) {
          auto [ptr, m, idx] = reader.Poll();
          if (ptr == nullptr) break;
          for (size_t j = 0; j < m; j++) {
            if (pred(ptr[j])) {
              best = std::min(best, idx * epct + j);
              break;  // first match in this chunk is its smallest index
            }
          }
          reader.allocator.Free(ptr);
        }
        return best;
      });

  // Combine per-worker minima (num_workers entries — a trivial sequential min).
  size_t result = n;
  for (size_t v : locals) result = std::min(result, v);
  return result;
}

}  // namespace plaid

// ChunkPartition -- k-way split
//
// (was ChunkSequence/Primitives/partition.h)
// ============================================================================

namespace plaid {

// key_fn may return this to DROP an element (route it to no bucket), so callers
// that only keep some elements (e.g. a filter, or quickhull discarding points
// inside the peak triangle) never pay to copy or write them.
inline constexpr size_t PARTITION_DROP = SIZE_MAX;

/**
 * Split `seq` into `num_buckets` output chunk_seqs in a SINGLE streaming read
 * pass: each element is routed to bucket `key_fn(elem)` (or dropped if that is
 * PARTITION_DROP).  This is the k-way generalization of ChunkFilter — one
 * filter is `num_buckets == 1` with a keep/drop key_fn — done with **one**
 * long-lived reader and **one** writer instead of k separate filter passes (k
 * reads + k reader/writer setups).  Routing runs across all parlay workers
 * (each polls the shared reader to exhaustion, mirroring
 * ChunkReduce/RemoveWorker).
 *
 * INVARIANT: each returned bucket is a valid library chunk_seq — index-ordered
 * and **dense-except-last** (every chunk but the bucket's final one holds
 * exactly CHUNK_SIZE/sizeof(T) elements), exactly like a ChunkFilter output.  A
 * bucket buffer is flushed only when full; its one short chunk is the last,
 * highest- index one.  Buckets are returned SEPARATELY on purpose: do NOT
 * concatenate them into one sequence — that would drop each bucket's trailing
 * partial chunk into the middle of the sequence, breaking the delayed layer's
 * ELEMS_PER_CHUNK grid (zip alignment) and eager plaid::size.  A
 * caller needing one fused sequence must re-densify (a repack pass /
 * from_chunks), not glue chunk lists.
 *
 * Ordering within a bucket is completion order (the reader is unordered), NOT
 * the input order — callers needing sorted/index order must not rely on it.
 *
 * All buckets share one file per drive under `result_prefix` (each chunk
 * records its own file + offset), so removing the prefix's files frees every
 * bucket.
 *
 * @tparam T       Element type stored in seq.
 * @tparam KeyFn   Callable T -> size_t bucket in [0, num_buckets) or
 * PARTITION_DROP.
 */
template <typename T, typename KeyFn>
std::vector<chunk_seq> ChunkPartition(const chunk_seq& seq, size_t num_buckets,
                                      const std::string& result_prefix,
                                      KeyFn key_fn) {
  static_assert(CHUNK_SIZE % sizeof(T) == 0,
                "sizeof(T) must divide CHUNK_SIZE for O_DIRECT alignment");
  CHECK(num_buckets > 0) << "ChunkPartition: num_buckets must be > 0";

  std::vector<chunk_seq> out(num_buckets);
  if (seq.chunks.empty()) return out;

  const size_t ept = CHUNK_SIZE / sizeof(T);
  const size_t num_drives = GetSSDList().size();

  // One output file per drive (shared by all buckets), truncated to clear any
  // stale data from a prior run (the writer opens O_CREAT but not O_TRUNC).
  std::vector<std::string> filenames(num_drives);
  for (size_t d = 0; d < num_drives; d++) {
    filenames[d] = GetFileName(result_prefix, d);
    int fd = open(filenames[d].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    SYSCALL(fd);
    SYSCALL(close(fd));
  }

  UnorderedWriterConfig wcfg;
  wcfg.num_threads = num_drives;
  wcfg.io_uring_size = 32;
  wcfg.queue_size = 64;
  wcfg.num_files = num_drives;
  UnorderedFileWriter<T> writer;
  writer.Start(filenames, wcfg);

  // One reader for the whole pass; every parlay worker polls it to exhaustion.
  // Its process-wide buffer pool also backs the OUTPUT assembly buffers below,
  // so an emitted chunk is a recycled pool buffer (returned by the writer's
  // deleter after the write lands), never a fresh aligned_alloc.  A fresh
  // aligned_alloc(CHUNK_SIZE) per emitted chunk goes through mmap (>128 KB) and
  // takes the process-wide mmap_sem, which serializes all workers — the same
  // trap to_vector hit — so we must reuse pool buffers instead.
  ChunkSequenceReader<T> reader;
  reader.PrepChunks(seq);
  reader.Start(/*num_threads=*/10, /*queue_depth=*/32, /*max_requests=*/16);
  auto* const alloc = &reader.allocator;

  // Drive/offset assignment + per-bucket chunk-list append.  This is the ONLY
  // lock on the hot path and it is touched just once per FULL chunk (once per
  // ept elements a worker routes to a bucket), so it is uncontended in
  // practice. writer.Push runs OUTSIDE it: a full writer queue then stalls only
  // the one emitting worker, never a shared bucket.
  std::mutex place_mu;
  size_t slot = 0;
  std::vector<size_t> drive_off(num_drives, 0);
  auto emit_chunk = [&](size_t b, T* buf, size_t used) {
    size_t d, base;
    {
      std::lock_guard<std::mutex> lk(place_mu);
      d = slot++ % num_drives;
      base = drive_off[d];
      drive_off[d] += CHUNK_SIZE;
      out[b].chunks.push_back(
          chunk{filenames[d], base, used * sizeof(T), out[b].chunks.size()});
    }
    if (used < ept) memset(buf + used, 0, (ept - used) * sizeof(T));
    // Recycle the buffer back into the reader's pool once the write lands.
    writer.Push(std::shared_ptr<T>(buf, [alloc](T* p) { alloc->Free(p); }), ept,
                d, base);
  };

  // Per-(worker,bucket) PRIVATE current assembly buffer: no shared assembly and
  // no per-bucket lock on the scatter path, so routing scales across all
  // workers like ChunkReduce (whose folds are also fully private) rather than
  // serializing ~all workers onto 1-2 bucket mutexes.  A worker only ever
  // touches its own worker_id() slots, and a parlay task runs uninterrupted on
  // one worker, so these need no synchronization.  Buffers are drawn lazily
  // from the pool.
  const size_t W = std::max<size_t>(1, parlay::num_workers());
  std::vector<T*> cur((size_t)W * num_buckets, nullptr);
  std::vector<size_t> cnt((size_t)W * num_buckets, 0);

  parlay::parallel_for(
      0, W,
      [&](size_t) {
        const size_t w = parlay::worker_id();
        while (true) {
          auto [ptr, n, idx] = reader.Poll();
          (void)idx;
          if (ptr == nullptr) break;  // sequence exhausted
          for (size_t k = 0; k < n; k++) {
            const size_t j = key_fn(ptr[k]);
            if (j == PARTITION_DROP) continue;
            CHECK(j < num_buckets)
                << "ChunkPartition: bucket id " << j
                << " out of range (num_buckets=" << num_buckets << ")";
            const size_t si = w * num_buckets + j;
            if (cur[si] == nullptr) {
              cur[si] = alloc->Alloc();
              cnt[si] = 0;
            }
            cur[si][cnt[si]++] = ptr[k];
            if (cnt[si] == ept) {  // private buffer full -> emit
              emit_chunk(j, cur[si], ept);
              cur[si] = nullptr;
              cnt[si] = 0;
            }
          }
          alloc->Free(ptr);
        }
      },
      /*granularity=*/1);

  // Consolidate the per-worker partial tails (up to W per bucket) into densely
  // packed chunks so each bucket stays dense-except-last: emit full chunks as
  // an accumulator fills and exactly one final partial.  Runs after the
  // parallel phase, so these emits get the highest (last) indices; total tail
  // data is < W*num_buckets*CHUNK_SIZE, negligible next to the streamed input.
  for (size_t b = 0; b < num_buckets; b++) {
    T* acc = nullptr;
    size_t acc_cnt = 0;
    for (size_t w = 0; w < W; w++) {
      const size_t si = w * num_buckets + b;
      T* src = cur[si];
      const size_t sc = cnt[si];
      if (src == nullptr) continue;
      size_t off = 0;
      while (off < sc) {
        if (acc == nullptr) {
          acc = alloc->Alloc();
          acc_cnt = 0;
        }
        const size_t take = std::min(ept - acc_cnt, sc - off);
        memcpy(acc + acc_cnt, src + off, take * sizeof(T));
        acc_cnt += take;
        off += take;
        if (acc_cnt == ept) {
          emit_chunk(b, acc, ept);
          acc = nullptr;
          acc_cnt = 0;
        }
      }
      alloc->Free(src);
    }
    if (acc_cnt > 0)
      emit_chunk(b, acc, acc_cnt);
    else if (acc)
      alloc->Free(acc);
  }

  writer.Wait();
  return out;
}

}  // namespace plaid

// flatten
//
// (was ChunkSequence/Primitives/flatten.h)
// ============================================================================

namespace plaid {

inline chunk_seq flatten(const std::vector<chunk_seq>& chunk_sequences) {
  size_t count = 0;
  for (size_t i = 0; i < chunk_sequences.size(); i++) {
    count += chunk_sequences[i].chunks.size();
  }
  // size_t count = 0;

  // for(int i = 0; i < chunk_sequences.size(); i++){
  //     count+= chunk_sequences[i].size();
  // }
  // chunk_seq chunker(count);
  // // for(int i = 1; i < chunk_sequences.size(); i++){
  // //     chunk_sequences[0]chunk_seq.app

  // // }
  chunk_seq chunker;
  chunker.chunks.resize(count);
  size_t overall_counter = 0;
  for (size_t i = 0; i < chunk_sequences.size(); i++) {
    for (size_t k = 0; k < chunk_sequences[i].chunks.size(); k++) {
      auto c = chunk_sequences[i].chunks[k];
      c.index = overall_counter;
      chunker.chunks[overall_counter] = c;
      overall_counter++;
    }
  }
  return chunker;
}
inline chunk_seq flatten(const parlay::sequence<chunk_seq>& chunk_sequences) {
  size_t count = 0;
  for (size_t i = 0; i < chunk_sequences.size(); i++) {
    count += chunk_sequences[i].chunks.size();
  }
  // size_t count = 0;

  // for(int i = 0; i < chunk_sequences.size(); i++){
  //     count+= chunk_sequences[i].size();
  // }
  // chunk_seq chunker(count);
  // // for(int i = 1; i < chunk_sequences.size(); i++){
  // //     chunk_sequences[0]chunk_seq.app

  // // }
  chunk_seq chunker;
  chunker.chunks.resize(count);
  size_t overall_counter = 0;
  for (size_t i = 0; i < chunk_sequences.size(); i++) {
    for (size_t k = 0; k < chunk_sequences[i].chunks.size(); k++) {
      auto c = chunk_sequences[i].chunks[k];
      c.index = overall_counter;
      chunker.chunks[overall_counter] = c;
      overall_counter++;
    }
  }
  return chunker;
}

}  // namespace plaid

// materialize
//
// (was ChunkSequence/Primitives/materialize.h)
// ============================================================================

namespace plaid {

// Read an entire out-of-core chunk_seq into a single in-DRAM parlay::sequence,
// preserving logical (index) order.  The caller is responsible for ensuring the
// whole sequence fits in DRAM -- materialize is the base case of algorithms
// like kth_smallest, which only call it once the residual set is small.
template <typename T>
parlay::sequence<T> materialize(const chunk_seq& seq,
                                size_t reader_threads = 10) {
  const size_t n_chunks = seq.chunks.size();

  // Per-chunk element offsets in index order (chunk.used is a byte count).
  // elem_offset[i] = number of elements in chunks[0 .. i-1].
  std::vector<size_t> elem_offset(n_chunks + 1, 0);
  for (size_t i = 0; i < n_chunks; i++)
    elem_offset[i + 1] = elem_offset[i] + seq.chunks[i].used / sizeof(T);
  const size_t n = elem_offset[n_chunks];

  parlay::sequence<T> out(n);

  // Workers poll chunks in arbitrary completion order; each scatters its chunk
  // into out[] at the offset dictated by the chunk's index, so the result is
  // ordered regardless of I/O completion order (the same scatter-by-index
  // pattern ChunkScan's pass 1 uses).  There is no accumulator to combine, so
  // the per-worker return value is an unused placeholder.
  RemoveWorker<T>(seq, /*reader_threads=*/std::max<size_t>(1, reader_threads),
                  [&](ChunkSequenceReader<T>& reader) {
                    while (true) {
                      auto [ptr, cnt, chunk_idx] = reader.Poll();
                      if (ptr == nullptr) break;
                      std::memcpy(out.data() + elem_offset[chunk_idx], ptr,
                                  cnt * sizeof(T));
                      reader.allocator.Free(ptr);
                    }
                    return 0;
                  });

  return out;
}

// Sequential materialize: same result as materialize(seq), read with blocking
// O_DIRECT preads on the calling thread instead of a ChunkSequenceReader.
//
// For use *inside* a parlay::parallel_for whose iterations each materialize one
// DRAM-sized piece (the per-bucket base case of random_shuffle / sample_sort).
// The eager materialize would there spin up a reader -- reader_threads io_uring
// rings and a nested parlay parallel region -- per call, so B concurrent calls
// mean B * reader_threads rings (the RLIMIT_MEMLOCK churn) and a per-call setup
// cost that a ~128 MB bucket never amortizes.  The outer loop already supplies
// the parallelism, so each bucket is cheapest read straight through.
template <typename T>
parlay::sequence<T> sequential_materialize(const chunk_seq& seq) {
  const size_t n_chunks = seq.chunks.size();
  if (n_chunks == 0) return {};

  // Read in logical (index) order regardless of vector order, so the result is
  // ordered even if the caller handed us headers that are not index-sorted.
  std::vector<const chunk*> ordered;
  ordered.reserve(n_chunks);
  for (const chunk& c : seq.chunks) ordered.push_back(&c);
  std::sort(ordered.begin(), ordered.end(),
            [](const chunk* a, const chunk* b) { return a->index < b->index; });

  std::vector<size_t> elem_offset(n_chunks + 1, 0);
  for (size_t i = 0; i < n_chunks; i++)
    elem_offset[i + 1] = elem_offset[i] + ordered[i]->used / sizeof(T);

  parlay::sequence<T> out(elem_offset[n_chunks]);

  // One bounce buffer for the whole pass: out.data() is not O_DIRECT-aligned,
  // and the read length must be rounded up past `used`, so each chunk lands in
  // the aligned buffer and is copied to its slice of the result.
  T* buf = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
  CHECK(buf != nullptr) << "sequential_materialize: buffer allocation failed";

  // The chunks of one bucket usually share a handful of drive files; cache the
  // fds so a bucket costs a few opens rather than one per chunk.
  std::map<std::string, int> fd_cache;
  for (size_t i = 0; i < n_chunks; i++) {
    const chunk* c = ordered[i];
    if (c->used == 0) continue;
    auto [it, inserted] = fd_cache.emplace(c->filename, -1);
    if (inserted) {
      it->second = open(c->filename.c_str(), O_RDONLY | O_DIRECT);
      SYSCALL(it->second);
    }
    SYSCALL(pread(it->second, buf, AlignUp(c->used), (off_t)c->begin_addr));
    std::memcpy(out.data() + elem_offset[i], buf, c->used);
  }

  for (auto& [name, fd] : fd_cache) close(fd);
  free(buf);
  return out;
}

// Delayed-source materialize: read a *fused* sequence straight into DRAM
// without forcing it to disk first.  d::force(chain) + materialize(seq) would
// move the intermediate out and back (n writes + n reads); this runs the chain
// during the one streaming read pass of its sources and lands the elements
// directly in the result, so the only I/O is the sources' reads.
//
// Most nodes size logical chunk i at ELEMS_PER_CHUNK (uint64_t-based), but that
// is a per-node convention, not a universal one -- e.g. cut_source<T> grids on
// CHUNK_SIZE/sizeof(T) so a re-windowed slice of a non-8-byte element type (a
// weighted_edge cut, say) still spans physical reads correctly.  So the offset
// of chunk ci in `out` is computed from a prefix sum of chunk_len(), not
// assumed to be ci * ELEMS_PER_CHUNK -- that assumption previously overflowed
// `out` (and its neighbors) for any node whose grid disagreed with it.
//
// SFINAE'd on D::value_type (which chunk_seq does not have) so a chunk_seq
// still selects the eager overload above.  Unlike force/filter there is no <=8B
// element limit: nothing goes to the on-disk chunk grid, so zip's std::pair
// elements can be materialized as-is.
template <class D, class = typename D::value_type>
parlay::sequence<typename D::value_type> materialize(const D& d) {
  using R = typename D::value_type;
  const size_t nc = d.num_chunks();
  std::vector<size_t> offset(nc + 1, 0);
  for (size_t i = 0; i < nc; i++) offset[i + 1] = offset[i] + d.chunk_len(i);

  parlay::sequence<R> out(offset[nc]);
  delayed::for_each_chunk(d, [&](size_t ci, size_t n, auto it) {
    R* dst = out.data() + offset[ci];
    for (size_t k = 0; k < n; k++) {
      dst[k] = *it;
      ++it;
    }
  });
  return out;
}

// Sequential delayed-source materialize: same result as materialize(d), but
// driven by delayed::sequential_for_each_chunk (blocking, calling-thread
// O_DIRECT preads) instead of delayed::for_each_chunk (its own
// ChunkSequenceReader + dispatcher thread).  Use this, not materialize(d),
// from *inside* an already-parallel outer loop over many small delayed
// ranges -- e.g. Bellman-Ford's per-vertex delayed::cut of its edge
// chunk_seq -- for the same reason sequential_materialize(chunk_seq) above
// exists.
template <class D, class = typename D::value_type>
parlay::sequence<typename D::value_type> sequential_materialize(const D& d) {
  delayed::SequentialReadContext ctx;
  return sequential_materialize(d, ctx);
}

// Context overload: reuses a caller-supplied SequentialReadContext (fd cache +
// buffer pool) across many calls instead of opening/allocating fresh state
// every time.  For a caller like Bellman-Ford that does one small
// delayed::cut materialize per vertex per round from inside an already
// parallel outer loop, one context per parlay::worker_id() (see
// external_bellman_ford.h) amortizes the opens/allocations across the whole
// algorithm instead of paying them O(rounds*n) times.
template <class D, class = typename D::value_type>
parlay::sequence<typename D::value_type> sequential_materialize(
    const D& d, delayed::SequentialReadContext& ctx) {
  using R = typename D::value_type;
  const size_t nc = d.num_chunks();
  std::vector<size_t> offset(nc + 1, 0);
  for (size_t i = 0; i < nc; i++) offset[i + 1] = offset[i] + d.chunk_len(i);

  parlay::sequence<R> out(offset[nc]);
  delayed::sequential_for_each_chunk(d, ctx, [&](size_t ci, size_t n, auto it) {
    R* dst = out.data() + offset[ci];
    for (size_t k = 0; k < n; k++) {
      dst[k] = *it;
      ++it;
    }
  });
  return out;
}

namespace delayed {

// Re-exposed under `delayed::` so callers that build a delayed chain (e.g.
// `delayed::cut`) can materialize it without stepping back out to the
// enclosing namespace.  Both overloads just forward to the definitions above:
// the existing generic materialize(D) already SFINAEs on D::value_type, so it
// accepts any fused delayed node (including cut_source) as-is; the chunk_seq
// overload exists only because a plain chunk_seq has no ::value_type and so
// can't reach that overload via qualified `delayed::materialize` lookup (which
// skips ADL into the enclosing namespace).
template <typename T>
parlay::sequence<T> materialize(const chunk_seq& seq,
                                size_t reader_threads = 10) {
  return plaid::materialize<T>(seq, reader_threads);
}

template <class D, class = typename D::value_type>
parlay::sequence<typename D::value_type> materialize(const D& d) {
  return plaid::materialize(d);
}

template <class D, class = typename D::value_type>
parlay::sequence<typename D::value_type> sequential_materialize(const D& d) {
  return plaid::sequential_materialize(d);
}

template <class D, class = typename D::value_type>
parlay::sequence<typename D::value_type> sequential_materialize(
    const D& d, SequentialReadContext& ctx) {
  return plaid::sequential_materialize(d, ctx);
}

}  // namespace delayed

}  // namespace plaid

// cut -- slice / shift / guard-limb
//
// (was ChunkSequence/Primitives/cut.h)
// ============================================================================

#define NUM_SSDS 30
#ifndef PRACTICAL_SSDS
#define PRACTICAL_SSDS 8
#define BUFFER_SIZE 512
#endif

// logic of the chunk cut method:

// we take an external sequence and two indices. we return an external sequence
// that is a copy of the requested range of the original sequence. key to this
// is that there are two points of difficulty:
// 1. The start of the requested chunk may not be aligned, in which case we'll
// read that chunk and create a new chunk header for it that is aligned
// 2. The end of the requested chunk may not be aligned, in which case we'll
// again read that chunk and create a new chunk header for it

namespace plaid {

inline chunk_seq cut_by_chunk(const chunk_seq& seq, size_t chunk_begin,
                              size_t chunk_end) {
  chunk_seq out;
  if (chunk_begin >= chunk_end || chunk_begin >= seq.chunks.size()) return out;
  chunk_end = std::min(chunk_end, seq.chunks.size());
  out.chunks.reserve(chunk_end - chunk_begin);
  for (size_t i = chunk_begin; i < chunk_end; i++) {
    chunk c = seq.chunks[i];
    c.index = out.chunks.size();
    out.chunks.push_back(c);
  }
  return out;
}

// ── metadata shift / guard-limb append (zero-chunk aliasing) ─────────────────
// A shift (multiply a big integer by base^(k*ELEMS_PER_CHUNK)) and a guard-limb
// append both need chunks that *read as zeros* without moving any data.  We
// keep one zero-filled CHUNK_SIZE block per drive, written once, and alias it
// as many times as needed — safe because these chunks are only ever read, never
// written.
inline const std::string& zero_chunk_prefix() {
  static const std::string p = "bimul_zero";
  return p;
}

// Write one zero-filled CHUNK_SIZE block per drive (idempotent for the run).
inline void ensure_zero_chunks() {
  static std::once_flag once;
  std::call_once(once, [] {
    const size_t nd = GetSSDList().size();
    void* buf = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
    CHECK(buf != nullptr) << "ensure_zero_chunks: buffer allocation failed";
    memset(buf, 0, CHUNK_SIZE);
    for (size_t d = 0; d < nd; d++) {
      const std::string fn = GetFileName(zero_chunk_prefix(), d);
      int fd = open(fn.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
      SYSCALL(fd);
      SYSCALL(pwrite(fd, buf, CHUNK_SIZE, (off_t)0));
      close(fd);
    }
    free(buf);
  });
}

// One chunk header aliasing drive d's shared zero block, exposing `used` bytes.
inline chunk zero_chunk_header(size_t d, size_t used, size_t index) {
  ensure_zero_chunks();
  const size_t nd = GetSSDList().size();
  return {GetFileName(zero_chunk_prefix(), d % nd), 0, used, index};
}

// seq * base^(k * ELEMS_PER_CHUNK), as pure metadata: prepend k full zero
// chunks (spread across drives) and reindex seq's chunks by +k.  Preserves the
// dense-except-last invariant (the prepended chunks are full; seq's own tail
// stays the tail) and leaves the top limb untouched (a canonical non-negative
// operand stays canonical).  No data is written.
inline chunk_seq prepend_zero_chunks(const chunk_seq& seq, size_t k) {
  if (k == 0) return seq;
  chunk_seq out;
  out.chunks.reserve(k + seq.chunks.size());
  for (size_t j = 0; j < k; j++)
    out.chunks.push_back(zero_chunk_header(j, CHUNK_SIZE, j));
  for (const chunk& c : seq.chunks) {
    chunk nc = c;
    nc.index = out.chunks.size();
    out.chunks.push_back(nc);
  }
  return out;
}

// Append one chunk holding `used` zero bytes (aliased, no write).  The caller
// must guarantee seq's current last chunk is full, so the result stays
// dense-except-last.  Used to attach a zero guard limb (used == sizeof(limb))
// to a chunk-aligned cut half whose top limb has its sign bit set.
inline chunk_seq append_zero_chunk(const chunk_seq& seq, size_t used) {
  chunk_seq out = seq;
  out.chunks.push_back(
      zero_chunk_header(out.chunks.size(), used, out.chunks.size()));
  return out;
}

// maybe we'll want to make this callable directly on an external sequence in
// the future

// a couple of notes: cut does not modify the original sequence
// even if the read locations are already aligned, we currently rewrite them to
// not share data. slice methods typically return a new sequence anyway, so I
// think this is not an issue currently we rewrite to a new buffer for each
// start/end read because we don't know whether they'll be aligned properly this
// is not the case for the middle reads because we know that our index runs
// through them, but we might start/end in the middle of a chunk but we know
// that we're going to copy everything anyway, so perhaps we can shave off the
// write to a new alignment
template <typename T>
chunk_seq sequential_cut_no_compression(const chunk_seq& seq,
                                        size_t start_index, size_t end_index) {
  // if(end_index > (plaid::size(seq) * CHUNK_SIZE / sizeof(T)) ||
  // start_index >= end_index){ //start_index is size_t unsigned
  if (end_index > (size<T>(seq)) ||
      start_index >= end_index) {  // start_index is size_t unsigned
    return {};
  }
  // we could allocate this perfectly if we computed a scan over the chunk
  // headers, but I don't think it's worth it
  parlay::sequence<chunk> chunk_headers;
  size_t tracker = start_index;
  size_t counter = 0;
  size_t index_counter = 0;
  while (index_counter < seq.chunks.size() &&
         counter + (seq.chunks[index_counter].used / sizeof(T)) < start_index) {
    // this could be more simply implemented with a single tracker
    counter += seq.chunks[index_counter].used / sizeof(T);
    tracker -= seq.chunks[index_counter].used / sizeof(T);
    index_counter++;
  }
  // we have now found the correct chunk for the start
  T* buff = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
  int fd1 =
      open(seq.chunks[index_counter].filename.c_str(), O_RDONLY | O_DIRECT);
  // A failed open() (e.g. EMFILE under fd pressure) must not silently flow into
  // the pread below as fd == -1 -- SYSCALL only logs, it isn't fatal, so that
  // used to corrupt this chunk's size/offset bookkeeping instead of failing
  // loudly here.
  CHECK(fd1 >= 0) << "sequential_cut_no_compression: open failed for "
                  << seq.chunks[index_counter].filename << ": "
                  << std::strerror(errno);
  // seq.chunks[index_counter].used/sizeof(T) - tracker is the #of elements used
  // in the block - the expected start position of the first index we want to
  // see, which means we need to read size of the difference between the two to
  // get all the data from start to end
  SYSCALL(pread(fd1, buff, AlignUp(seq.chunks[index_counter].used),
                (off_t)seq.chunks[index_counter].begin_addr));
  memmove(buff, buff + tracker,
          (seq.chunks[index_counter].used / sizeof(T) - tracker) * sizeof(T));
  // Distinct suffix from the end seam below: if the start and end chunks land
  // on the same drive their base filenames are identical, so a shared "_cut"
  // suffix would make both seams write offset 0 of the same file and clobber
  // each other.
  std::string start_cut = seq.chunks[index_counter].filename + "_cut_start";

  int fd1_filename1 =
      open(start_cut.c_str(), O_WRONLY | O_DIRECT | O_CREAT, 0644);
  // we know that the O_DIRECT alignment below us is already full of the
  // original data, so we're trying the one above
  //  SYSCALL(pwrite(fd1_filename,buff,
  //  (seq.chunks[index_counter].used/sizeof(T)-tracker) * sizeof(T),
  //  (off_t)(AlignUp(seq.chunks[index_counter].begin_addr +
  //  tracker*sizeof(T)))));
  SYSCALL(
      pwrite(fd1_filename1, buff,
             AlignUp((seq.chunks[index_counter].used / sizeof(T) - tracker) *
                     sizeof(T)),
             (off_t)0));

  close(fd1);
  close(fd1_filename1);
  chunk start_chunk;
  // start_chunk.filename = seq.chunks[index_counter].filename;
  // start_chunk.begin_addr =(AlignUp(seq.chunks[index_counter].begin_addr +
  // tracker*sizeof(T)));
  start_chunk.filename = start_cut;
  start_chunk.begin_addr = (0);
  start_chunk.used =
      (seq.chunks[index_counter].used / sizeof(T) - tracker) * sizeof(T);
  start_chunk.index = 0;

  counter += seq.chunks[index_counter].used / sizeof(T);
  index_counter++;
  // tracker-=seq.chunks[index_counter].used/sizeof(T);

  tracker = end_index - counter;
  chunk_headers.push_back(start_chunk);
  while (index_counter < seq.chunks.size() &&
         counter + (seq.chunks[index_counter].used / sizeof(T)) < end_index) {
    // this could be more simply implemented with a single tracker
    counter += seq.chunks[index_counter].used / sizeof(T);
    tracker -= seq.chunks[index_counter].used / sizeof(T);
    chunk_headers.push_back(seq.chunks[index_counter]);
    index_counter++;
  }
  // we have now reached the end chunk, so we need potentially just the first
  // part of it.

  // we're allocating a new buffer because eventually we want these start/end
  // chunks things to be in a parallel do
  T* buf = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
  int fd =
      open(seq.chunks[index_counter].filename.c_str(), O_RDONLY | O_DIRECT);
  CHECK(fd >= 0) << "sequential_cut_no_compression: open failed for "
                 << seq.chunks[index_counter].filename << ": "
                 << std::strerror(errno);

  // SYSCALL(pread(fd, buf, AlignUp(seq.chunks[index_counter].used/sizeof(T) -
  // tracker) * sizeof(T), (off_t) (seq.chunks[index_counter].begin_addr +
  // tracker*sizeof(T)))); SYSCALL(pread(fd, buf,
  // AlignUp(seq.chunks[index_counter].used/sizeof(T) - tracker) * sizeof(T),
  // (off_t) AlignDown((seq.chunks[index_counter].begin_addr +
  // tracker*sizeof(T)))));
  SYSCALL(pread(fd, buf, AlignUp(tracker * sizeof(T)),
                (off_t)seq.chunks[index_counter].begin_addr));

  std::string end_cut = seq.chunks[index_counter].filename + "_cut_end";
  int fd_filename = open(end_cut.c_str(), O_WRONLY | O_DIRECT | O_CREAT, 0644);
  // SYSCALL(pwrite(fd_filename,buf,
  // (seq.chunks[index_counter].used/sizeof(T)-tracker) * sizeof(T),
  // (off_t)(AlignUp(seq.chunks[index_counter].begin_addr +
  // tracker*sizeof(T)))));
  SYSCALL(pwrite(fd_filename, buf, AlignUp(tracker * sizeof(T)), (off_t)0));
  close(fd);
  close(fd_filename);
  chunk end_chunk;
  end_chunk.filename = end_cut;
  end_chunk.begin_addr = 0;
  end_chunk.used = tracker * sizeof(T);
  end_chunk.index = chunk_headers.size();

  chunk_headers.push_back(end_chunk);

  // maybe don't return a local variable
  chunk_seq sequence =
      from_chunks(chunk_headers);  // use the constructor for the chunk
                                   // sequence, doesn't exist yet
  free(buf);
  free(buff);
  return sequence;
}

}  // namespace plaid

// scan_find
//
// (was ChunkSequence/Primitives/scan_find.h)
// ============================================================================

// the goal of this method is to compute a scan over the number of elements
// actually used in a sequence, which allows us to more efficiently search the
// external sequence this is just sequential for now but I'll parallelize it
// later on
namespace plaid {
template <typename T>
void scan_size(const chunk_seq& seq, parlay::sequence<size_t>& pseq) {
  size_t acc = 0;
  for (size_t i = 0; i < seq.chunks.size(); i++) {
    pseq[i] = acc;
    acc += seq.chunks[i].used / sizeof(T);
  }
}

template <typename T>
T scan_find(const chunk_seq& seq, const parlay::sequence<size_t>& pseq,
            size_t g) {
  const size_t res =
      std::upper_bound(pseq.begin(), pseq.end(), g) - pseq.begin() - 1;
  const auto& c = seq.chunks[res];
  const size_t local = g - pseq[res];

  // We want one element, so read only the O_DIRECT-aligned block that holds it
  // rather than the whole CHUNK_SIZE chunk.  This cuts the per-pivot I/O by
  // ~CHUNK_SIZE/O_DIRECT_MULTIPLE (e.g. 4 MiB -> 4 KiB); with 31*8 pivot probes
  // that is the difference between ~GiB and ~MiB of reads for the sample phase.
  const size_t byte_off = c.begin_addr + local * sizeof(T);
  const size_t aligned_off = AlignDown(byte_off);
  const size_t delta = byte_off - aligned_off;  // element's offset in block
  const size_t read_len = AlignUp(delta + sizeof(T));

  char* buf = (char*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, read_len);
  CHECK(buf != nullptr) << "allocation wrong";
  int fd = open(c.filename.c_str(), O_DIRECT | O_RDONLY);
  SYSCALL(fd);
  SYSCALL(pread(fd, buf, read_len, (off_t)aligned_off));
  close(fd);
  T val;
  std::memcpy(&val, buf + delta, sizeof(T));
  free(buf);
  return val;
}

}  // namespace plaid

// linear_find
//
// (was ChunkSequence/Primitives/linear_find.h)
// ============================================================================

// template<typename T>
// T plaid::LinearFind(chunk_seq& seq, size_t index){
// size_t top = seq.chunks.size();
// for(int i = 0; i < top; i++){
//     if(index < seq.chunks[i].used){
//         int filedes = open(seq.chunks[i].filename.c_str(), O_DIRECT |
//         O_RDONLY); T* buffer = calloc(seq.chunks[i].used * sizeof(T));
//         lseek(filedes, seq.chunks[i].begin_addr, SEEK_SET);
//         read(filedes, buffer, seq.chunks[i].used);
//         T val = buffer[index];
//         free(buffer);
//         return val;
//     }
//     else{
//         index -= seq.chunks[i].used;
//     }}}

template <typename T>
T find(const chunk_seq& seq, size_t g) {
  for (const auto& c : seq.chunks) {
    const size_t cnt = c.used / sizeof(T);
    if (g < cnt) {
      T* buf = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
      CHECK(buf != nullptr) << "allocation wrong";
      int fd = open(c.filename.c_str(), O_DIRECT | O_RDONLY);
      SYSCALL(fd);
      SYSCALL(pread(fd, buf, AlignUp(c.used), (off_t)c.begin_addr));
      close(fd);
      T val = buf[g];
      free(buf);
      return val;
    }
    g -= cnt;
  }
  CHECK(false) << "out of range";
  return T{};
}

#endif  // PLAID_SECONDARY_PRIMITIVES_H
