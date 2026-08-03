#ifndef NESTED_SEQ_H
#define NESTED_SEQ_H

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "ChunkSequence/chunk_seq.h"
#include "absl/log/check.h"
#include "configs.h"
#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "parlay/utilities.h"  // parlay::hash64
#include "utils/file_utils.h"
#include "utils/unordered_file_writer.h"

namespace ChunkSequenceOps {

/**
 * nested_seq<T> — a prototype out-of-core *sequence of sequences* of T.
 *
 * Lives in parallel to chunk_seq.  Where a chunk_seq cuts its logical sequence
 * at raw CHUNK_SIZE byte boundaries (so one io_uring read = one chunk), a
 * nested_seq packs *whole inner sequences* into each chunk, balanced in DRAM to
 * ~CHUNK_SIZE.  A worker therefore reads one chunk and gets back a bundle of
 * complete inner sequences, processing whole sequences with NO cross-chunk carry
 * and NO boundary merge (contrast ChunkSegmentedReduce).
 *
 * Layout is CSR-like: raw T values live contiguously on disk, and a single DRAM
 * array `seq_len_scan` (the exclusive prefix sum of inner-sequence lengths, the
 * degree_scan analog) plus per-chunk (first_seq, num_seqs) fully describes the
 * structure.  Within a chunk the inner sequences are packed contiguously in seq
 * order with no interior gaps, so for inner sequence j living in a chunk whose
 * first inner sequence is s:
 *     len(j)       = seq_len_scan[j+1] - seq_len_scan[j]
 *     local_off(j) = seq_len_scan[j]   - seq_len_scan[s]   (elements into the buffer)
 * Unlike chunk_csr, the global element index is NOT the disk position (each chunk
 * pads its tail out to CHUNK_SIZE), but the *local* offset within a chunk is
 * recoverable from the scan because packing preserves seq order and contiguity.
 *
 * Two invariants make "one bundle = one io_uring read":
 *   (1) every inner sequence fits in one chunk (asserted at construction) — the
 *       feasibility precondition that lets the greedy packer always place the
 *       next sequence in a fresh chunk;
 *   (2) the packer closes a chunk before its summed lengths would exceed
 *       elems_per_chunk, so raw.used <= CHUNK_SIZE and a chunk is exactly one
 *       aligned read.  (2) is the actual no-straddle guarantee; (1) makes (2)
 *       always satisfiable.  Trade-off: because we pack whole sequences, a
 *       chunk's raw.used is usually strictly < CHUNK_SIZE (internal tail padding
 *       that is read-but-ignored) — space/bandwidth traded for zero boundary
 *       logic.
 */

struct nested_chunk {
  chunk raw;         // one CHUNK_SIZE io_uring read; raw.index == position in chunks
  size_t first_seq;  // global index of the first inner sequence in this chunk
  size_t num_seqs;   // count of whole inner sequences packed here (>= 1)
};

template <typename T = uint64_t>
struct nested_seq {
  // Index-ordered: chunks[i].raw.index == i.
  std::vector<nested_chunk> chunks;
  // Exclusive prefix sum of inner-sequence lengths; size == total_seqs + 1.
  parlay::sequence<size_t> seq_len_scan;

  static constexpr size_t elems_per_chunk = CHUNK_SIZE / sizeof(T);

  size_t total_seqs() const {
    return seq_len_scan.empty() ? 0 : seq_len_scan.size() - 1;
  }
  size_t len(size_t i) const { return seq_len_scan[i + 1] - seq_len_scan[i]; }

  // A flat chunk_seq view over the raw chunk headers (index-ordered), for
  // feeding the existing ChunkSequenceReader / ExternalTransform / RemoveWorker.
  chunk_seq raw_view() const {
    chunk_seq cs;
    cs.chunks.reserve(chunks.size());
    for (const auto& nc : chunks) cs.chunks.push_back(nc.raw);
    return cs;
  }

  // The chunk holding inner sequence i.  Chunks partition [0, total_seqs)
  // contiguously, so this is the last chunk whose first_seq <= i.
  size_t which_chunk(size_t i) const {
    CHECK(!chunks.empty());
    size_t lo = 0, hi = chunks.size();
    while (lo + 1 < hi) {
      size_t mid = (lo + hi) / 2;
      if (chunks[mid].first_seq <= i)
        lo = mid;
      else
        hi = mid;
    }
    return lo;
  }

  // Materialize inner sequence i into DRAM.  Reads the single chunk that holds
  // it (one blocking O_DIRECT read) and slices out its local range.  Models
  // chunk_csr::get_adjacent, but never spans chunks (invariant (1)).
  parlay::sequence<T> get(size_t i) const {
    CHECK(i < total_seqs());
    const nested_chunk& nc = chunks[which_chunk(i)];
    const size_t local_off = seq_len_scan[i] - seq_len_scan[nc.first_seq];
    const size_t l = len(i);
    parlay::sequence<T> out(l);
    if (l == 0) return out;

    T* buf = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
    CHECK(buf != nullptr) << "nested_seq::get: buffer allocation failed";
    int fd = open(nc.raw.filename.c_str(), O_RDONLY | O_DIRECT);
    SYSCALL(fd);
    SYSCALL(pread(fd, buf, AlignUp(nc.raw.used), (off_t)nc.raw.begin_addr));
    close(fd);
    for (size_t k = 0; k < l; k++) out[k] = buf[local_off + k];
    free(buf);
    return out;
  }

  // Write every inner sequence's values contiguously (no padding) to a local
  // file, in seq order — i.e. the flat concatenation of all inner sequences.
  // Models chunk_seq::consolidate.
  void consolidate_flat(const std::string& output_path) const {
    int out_fd = open(output_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    SYSCALL(out_fd);

    std::vector<const nested_chunk*> ordered;
    ordered.reserve(chunks.size());
    for (const auto& nc : chunks) ordered.push_back(&nc);
    std::sort(ordered.begin(), ordered.end(),
              [](const nested_chunk* a, const nested_chunk* b) {
                return a->raw.index < b->raw.index;
              });

    void* buf = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
    CHECK(buf != nullptr) << "consolidate_flat: buffer allocation failed";
    std::map<std::string, int> fd_cache;
    for (const nested_chunk* nc : ordered) {
      if (nc->raw.used == 0) continue;
      auto [it, inserted] = fd_cache.emplace(nc->raw.filename, -1);
      if (inserted) {
        it->second = open(nc->raw.filename.c_str(), O_DIRECT | O_RDONLY);
        SYSCALL(it->second);
      }
      SYSCALL(pread(it->second, buf, AlignUp(nc->raw.used),
                    (off_t)nc->raw.begin_addr));
      SYSCALL(write(out_fd, buf, nc->raw.used));
    }

    free(buf);
    for (auto& [name, fd] : fd_cache) close(fd);
    close(out_fd);
  }

  // Read the whole nested_seq into DRAM as a vector of inner sequences.
  // Convenience for tests; assumes it fits in memory.
  std::vector<parlay::sequence<T>> to_nested_vector() const {
    std::vector<parlay::sequence<T>> out(total_seqs());
    T* buf = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
    CHECK(buf != nullptr) << "to_nested_vector: buffer allocation failed";
    std::map<std::string, int> fd_cache;
    for (const auto& nc : chunks) {
      if (nc.raw.used > 0) {
        auto [it, inserted] = fd_cache.emplace(nc.raw.filename, -1);
        if (inserted) {
          it->second = open(nc.raw.filename.c_str(), O_DIRECT | O_RDONLY);
          SYSCALL(it->second);
        }
        SYSCALL(pread(it->second, buf, AlignUp(nc.raw.used),
                      (off_t)nc.raw.begin_addr));
      }
      const size_t base = seq_len_scan[nc.first_seq];
      for (size_t k = 0; k < nc.num_seqs; k++) {
        const size_t gseq = nc.first_seq + k;
        const size_t local_off = seq_len_scan[gseq] - base;
        const size_t l = len(gseq);
        parlay::sequence<T> s(l);
        for (size_t j = 0; j < l; j++) s[j] = buf[local_off + j];
        out[gseq] = std::move(s);
      }
    }
    free(buf);
    for (auto& [name, fd] : fd_cache) close(fd);
    return out;
  }
};

/**
 * Build a nested_seq<T> of `num_seqs` inner sequences by applying `f` to each
 * index: f(i) -> parlay::sequence<T>.  Every inner sequence must fit in one
 * chunk (asserted).
 *
 * Batched greedy bin-packing (structurally like DensePack's batch loop, but
 * NON-splitting — a run is never split across a chunk boundary): each batch of
 * `batch_seqs` indices is generated in parallel, then a sequential greedy pass
 * threads the current partial chunk (carry) across batches, flushing to the
 * io_uring writer whenever the next whole sequence would overflow the chunk.
 */
template <typename T = uint64_t, typename F>
nested_seq<T> NestedTabulate(size_t num_seqs, const std::string& result_prefix,
                             F f, size_t batch_seqs = 256) {
  static_assert(CHUNK_SIZE % sizeof(T) == 0,
                "sizeof(T) must divide CHUNK_SIZE for O_DIRECT alignment");
  constexpr size_t epc = CHUNK_SIZE / sizeof(T);
  const size_t num_drives = GetSSDList().size();
  CHECK(batch_seqs > 0);

  // Create/truncate one output file per drive.  The writer opens files with
  // O_CREAT but not O_TRUNC, so stale data from a prior run is cleared here;
  // streaming writes then grow each file by explicit offset (no fallocate,
  // since the final size is unknown until packing completes).
  std::vector<std::string> filenames(num_drives);
  parlay::parallel_for(
      0, num_drives,
      [&](size_t d) {
        filenames[d] = GetFileName(result_prefix, d);
        int fd = open(filenames[d].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        SYSCALL(fd);
        SYSCALL(close(fd));
      },
      /*granularity=*/1);

  UnorderedWriterConfig wcfg;
  wcfg.num_threads = num_drives;
  wcfg.io_uring_size = 32;
  wcfg.queue_size = 64;
  wcfg.num_files = num_drives;
  UnorderedFileWriter<T> writer;
  writer.Start(filenames, wcfg);

  nested_seq<T> result;
  result.seq_len_scan.resize(num_seqs + 1);
  result.seq_len_scan[0] = 0;

  // Per-drive next write offset; balls-in-bins drive placement via hash64 of
  // the emission slot (deterministic, matches the eager engine's emitter).
  std::vector<size_t> next_off(num_drives, 0);
  size_t slot = 0;

  // Current (carry) chunk being filled.
  T* cur = nullptr;
  size_t cur_fill = 0;    // elements packed so far
  size_t cur_first = 0;   // first_seq of the current chunk
  size_t cur_count = 0;   // inner sequences packed so far

  auto flush = [&]() {
    if (cur == nullptr) return;
    memset(cur + cur_fill, 0, (epc - cur_fill) * sizeof(T));  // zero the tail
    const size_t d = parlay::hash64(slot) % num_drives;
    const size_t off = next_off[d];
    next_off[d] += CHUNK_SIZE;
    slot++;
    nested_chunk nc;
    nc.raw = chunk{filenames[d], off, cur_fill * sizeof(T), result.chunks.size()};
    nc.first_seq = cur_first;
    nc.num_seqs = cur_count;
    result.chunks.push_back(nc);
    writer.Push(std::shared_ptr<T>(cur, free), CHUNK_SIZE / sizeof(T), d, off);
    cur = nullptr;
    cur_fill = 0;
    cur_count = 0;
  };

  for (size_t base = 0; base < num_seqs; base += batch_seqs) {
    const size_t bn = std::min(batch_seqs, num_seqs - base);
    std::vector<parlay::sequence<T>> results(bn);
    parlay::parallel_for(0, bn, [&](size_t b) {
      results[b] = f(base + b);
      CHECK(results[b].size() <= epc)
          << "NestedTabulate: inner sequence " << (base + b) << " has length "
          << results[b].size() << " > elems_per_chunk (" << epc
          << "); inner sequences must fit in one chunk";
    });

    for (size_t b = 0; b < bn; b++) {
      const size_t i = base + b;
      const size_t l = results[b].size();
      result.seq_len_scan[i + 1] = result.seq_len_scan[i] + l;

      // Invariant (2): close the current chunk before it would overflow.
      if (cur != nullptr && cur_fill + l > epc) flush();
      if (cur == nullptr) {
        cur = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
        CHECK(cur != nullptr) << "NestedTabulate: buffer allocation failed";
        cur_fill = 0;
        cur_first = i;
        cur_count = 0;
      }
      if (l > 0) memcpy(cur + cur_fill, results[b].data(), l * sizeof(T));
      cur_fill += l;
      cur_count++;
    }
  }
  flush();

  writer.Wait();
  return result;
}

}  // namespace ChunkSequenceOps

#endif  // NESTED_SEQ_H
