#ifndef CHUNK_GROUP_BY_H
#define CHUNK_GROUP_BY_H

#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/Primitives/chunk_seq_reader.h"
#include "ChunkSequence/Primitives/bucketed_file_writer.h"
#include "absl/log/check.h"
#include "configs.h"
#include "parlay/parallel.h"
#include "utils/file_utils.h"
#include "utils/unordered_file_writer.h"

namespace plaid {

/**
 * Split `seq` into `num_buckets` output chunk_seqs in a SINGLE streaming read
 * pass: every element is routed to bucket `key_fn(elem)`.
 *
 * This is the bucketed-writer version of the grouping, sharing its buffering
 * substrate with count_sort's disk_span-aware overload
 * (Primitives/count_sort.h): each (worker, bucket) pair gets a small
 * SAMPLE_SORT_BUCKET_SIZE (4 KiB) staging buffer, via bucket_allocator, that
 * drains through a BucketWriter (Primitives/bucketed_file_writer.h) once
 * full. Peak DRAM is therefore ~num_workers * num_buckets *
 * SAMPLE_SORT_BUCKET_SIZE, independent of CHUNK_SIZE -- so this stays
 * feasible even when num_buckets grows with n (e.g. sample_sort's
 * ~n/2^27 buckets). For a small, FIXED bucket count (a handful, e.g. a
 * quickhull-style 2-3-way split), group_by_index_partition_small below
 * trades that scaling away for a simpler, lock-free full-CHUNK_SIZE buffer
 * per (worker, bucket) -- cheaper per element, but its memory is
 * num_workers * num_buckets * CHUNK_SIZE, which only stays small when
 * num_buckets does.
 *
 * Each bucket gets one file, placed by GetFileName(prefix, bucket_index) --
 * i.e. bucket_index % num_drives, so buckets spread across every drive
 * whenever num_buckets > num_drives (the common case here).
 *
 * INVARIANT: each returned bucket is index-ordered and dense-except-last,
 * exactly like ChunkPartition's / ChunkFilter's output. Buckets are returned
 * SEPARATELY on purpose: do NOT concatenate them into one sequence -- that
 * would drop each bucket's trailing partial chunk into the middle of the
 * sequence, breaking the delayed layer's ELEMS_PER_CHUNK grid (zip alignment)
 * and eager plaid::size. A caller needing one fused sequence must re-densify
 * (a repack pass / from_chunks), not glue chunk lists.
 *
 * Ordering within a bucket is completion order (the reader is unordered and
 * BucketWriter accumulates opportunistically), NOT the input order --
 * callers needing sorted/index order must not rely on it.
 *
 * All buckets' `key_fn` must return a value in [0, num_buckets); a value
 * outside that range is a CHECK failure, not a silent drop, matching
 * ChunkPartition's no-PARTITION_DROP-style-sentinel contract (this is pure
 * grouping, not filtering).
 *
 * @tparam T     Element type stored in seq.
 * @tparam KeyFn Callable T -> size_t bucket in [0, num_buckets).
 */
template <typename T, typename KeyFn>
std::vector<chunk_seq> group_by_index(const chunk_seq& seq, size_t num_buckets,
                                       const std::string& result_prefix,
                                       KeyFn key_fn) {
  static_assert(CHUNK_SIZE % sizeof(T) == 0,
                "sizeof(T) must divide CHUNK_SIZE for O_DIRECT alignment");
  CHECK(num_buckets > 0) << "group_by_index: num_buckets must be > 0";

  std::vector<chunk_seq> out(num_buckets);
  if (seq.chunks.empty()) return out;

  constexpr size_t kBufElems = SAMPLE_SORT_BUCKET_SIZE / sizeof(T);
  constexpr size_t kWriterIoThreads = 2;

  BucketWriter<T> writer(result_prefix, num_buckets);

  // INVARIANT: every thread that touches bucket_allocator must be a parlay
  // worker.  bucket_allocator is backed by parlay::internal::block_allocator,
  // whose free lists are keyed by parlay::worker_id() -- a thread_local that
  // silently returns 0 on any thread the scheduler never adopted.  So the
  // writer's I/O threads run as parlay tasks (never plain std::threads), and
  // the scatter is capped to P - kWriterIoThreads workers below to avoid
  // oversubscribing the shared pool (see count_sort.h's identical note).
  CHECK(parlay::num_workers() > kWriterIoThreads)
      << "group_by_index: need > " << kWriterIoThreads << " parlay workers";
  const size_t scatter_workers = parlay::num_workers() - kWriterIoThreads;

  ChunkSequenceReader<T> reader;
  reader.PrepChunks(seq);
  reader.Start(/*num_threads=*/10, /*queue_depth=*/32, /*max_requests=*/16);

  // One live bucket_allocator buffer per (worker, bucket), persisting across
  // the scatter loop's chunk-by-chunk reader polls -- same shape as
  // count_sort's `buf`/`fill`, indexed by parlay::worker_id().
  const size_t W = std::max<size_t>(1, parlay::num_workers());
  std::vector<T*> buf(W * num_buckets, nullptr);
  std::vector<size_t> fill(W * num_buckets, 0);
  for (size_t i = 0; i < W * num_buckets; i++)
    buf[i] = (T*)bucket_allocator::alloc();

  // results is assigned inside the scatter branch below: writer.ReapResult()
  // closes the pending queue, which is what lets the I/O threads' RunIoThread
  // loops (the other par_do branch) terminate.
  std::vector<typename BucketWriter<T>::Result> results;
  parlay::par_do(
      [&] {
        parlay::parallel_for(
            0, kWriterIoThreads, [&](size_t) { writer.RunIoThread(); },
            /*granularity=*/1);
      },
      [&] {
        parlay::parallel_for(
            0, scatter_workers,
            [&](size_t) {
              const size_t w = parlay::worker_id();
              while (true) {
                auto [ptr, n, idx] = reader.Poll();
                (void)idx;
                if (ptr == nullptr) break;  // sequence exhausted
                for (size_t k = 0; k < n; k++) {
                  const size_t b = (size_t)key_fn(ptr[k]);
                  CHECK(b < num_buckets)
                      << "group_by_index: bucket id " << b
                      << " out of range (num_buckets=" << num_buckets << ")";
                  const size_t si = w * num_buckets + b;
                  buf[si][fill[si]++] = ptr[k];
                  if (fill[si] == kBufElems) {
                    writer.Write(b, buf[si], kBufElems);
                    buf[si] = (T*)bucket_allocator::alloc();
                    fill[si] = 0;
                  }
                }
                reader.allocator.Free(ptr);
              }
            },
            /*granularity=*/1);

        // Flush every worker's residual per-bucket buffer.
        for (size_t i = 0; i < W * num_buckets; i++) {
          const size_t b = i % num_buckets;
          if (fill[i] > 0)
            writer.Write(b, buf[i], fill[i]);
          else
            bucket_allocator::free((BucketData*)buf[i]);
        }

        results = writer.ReapResult();
      });

  writer.CloseFiles();
  bucket_allocator::finish();

  // Carve each bucket's file into CHUNK_SIZE slices -- the same "no repack
  // pass" carving count_sort uses.
  for (size_t b = 0; b < num_buckets; b++) {
    size_t idx = 0;
    const auto& r = results[b];
    for (size_t off = 0; off < r.true_bytes; off += CHUNK_SIZE)
      out[b].chunks.push_back(
          {r.filename, off, std::min<size_t>(CHUNK_SIZE, r.true_bytes - off),
           idx++});
  }

  return out;
}

/**
 * Same contract as group_by_index above (single streaming read pass, k-way
 * grouping via key_fn, no drop, dense-except-last, completion order), but
 * with ChunkPartition's (Primitives/partition.h) ORIGINAL buffering shape:
 * one shared ChunkSequenceReader<T> polled by every parlay worker, and a
 * PRIVATE, full-CHUNK_SIZE assembly buffer per (worker, bucket) pair -- no
 * per-bucket lock on the scatter path, so routing scales across all workers
 * like ChunkReduce (whose folds are also fully private) rather than
 * serializing onto a few bucket mutexes.
 *
 * This is only safe for a SMALL, FIXED bucket count: memory is
 * num_workers * num_buckets * CHUNK_SIZE, with no budget check and no cap on
 * the backing ChunkSequenceReader::Allocator pool.  It is the right choice
 * for something like a quickhull-style 2-3-way split (few buckets, and the
 * simplicity/no-locking wins); it is NOT safe for a bucket count that grows
 * with n (e.g. sample_sort's ~n/2^27 buckets) -- use group_by_index above for
 * that, whose two-level buffering keeps memory bounded regardless of
 * num_buckets.
 *
 * All buckets share one file per drive under `result_prefix` (each chunk
 * records its own file + offset, round-robined across every bucket
 * combined), so removing the prefix's files frees every bucket, and every
 * bucket's data is automatically spread across every drive (no per-bucket
 * "one file = one drive" placement).
 *
 * @tparam T     Element type stored in seq.
 * @tparam KeyFn Callable T -> size_t bucket in [0, num_buckets).
 */
template <typename T, typename KeyFn>
std::vector<chunk_seq> group_by_index_partition_small(
    const chunk_seq& seq, size_t num_buckets, const std::string& result_prefix,
    KeyFn key_fn) {
  static_assert(CHUNK_SIZE % sizeof(T) == 0,
                "sizeof(T) must divide CHUNK_SIZE for O_DIRECT alignment");
  CHECK(num_buckets > 0)
      << "group_by_index_partition_small: num_buckets must be > 0";

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
  // takes the process-wide mmap_sem, which serializes all workers, so we must
  // reuse pool buffers instead.
  ChunkSequenceReader<T> reader;
  reader.PrepChunks(seq);
  reader.Start(/*num_threads=*/10, /*queue_depth=*/32, /*max_requests=*/16);
  auto* const alloc = &reader.allocator;

  // Drive/offset assignment + per-bucket chunk-list append.  This is the ONLY
  // lock on the hot path and it is touched just once per FULL chunk (once per
  // ept elements a worker routes to a bucket), so it is uncontended in
  // practice.  writer.Push runs OUTSIDE it: a full writer queue then stalls
  // only the one emitting worker, never a shared bucket.
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
            CHECK(j < num_buckets)
                << "group_by_index_partition_small: bucket id " << j
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

/**
 * Group `seq` by an arbitrary key `key_of(elem)`, approximated via
 * `bucket = hash(key_of(elem)) % num_buckets` — the same hash-bucket
 * convention count_sort_by_key/unique.h already use, NOT parlaylib's
 * group_by_key (which returns exactly one entry per distinct key via a
 * hash-based collect_reduce_sparse).  Two distinct keys MAY collide into the
 * same output bucket here; callers that need every distinct key isolated
 * must pick num_buckets large enough to make collisions acceptable for their
 * use, or post-process a bucket further (e.g. unique.h's own in-memory
 * std::unique finishing pass per bucket).  Implemented by composing the
 * hash-mod key function and forwarding straight to group_by_index (see its
 * doc for the ordering/no-drop/no-concatenate invariants, which apply
 * unchanged here).
 *
 * @tparam T      Element type stored in seq.
 * @tparam KeyOf  Callable T -> K, the key to group by.
 * @tparam Hash   Callable K -> size_t; defaults to std::hash<K>.
 */
template <typename T, typename KeyOf,
          typename Hash = std::hash<std::invoke_result_t<KeyOf, T>>>
std::vector<chunk_seq> group_by_key(const chunk_seq& seq, size_t num_buckets,
                                     const std::string& result_prefix,
                                     KeyOf key_of, Hash hash = {}) {
  CHECK(num_buckets > 0) << "group_by_key: num_buckets must be > 0";
  return group_by_index<T>(
      seq, num_buckets, result_prefix,
      [key_of, hash, num_buckets](const T& e) {
        return (size_t)(hash(key_of(e)) % num_buckets);
      });
}

}  // namespace plaid

#endif  // CHUNK_GROUP_BY_H
