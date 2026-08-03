#ifndef NESTED_GATHER_H
#define NESTED_GATHER_H

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ChunkSequence/dense_pack.h"
#include "ChunkSequence/nested_seq.h"
#include "absl/log/check.h"
#include "configs.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "utils/file_utils.h"

namespace ChunkSequenceOps {

// Direction of the gather's read plan (Beamer-style direction optimization).
//   kPull  — read whole graph chunks (bandwidth-efficient; large frontiers)
//   kPush  — read only each selected row's bytes (work-efficient; small ones)
//   kAuto  — decide per call from the frontier's edge count vs |E|
enum class GatherDir { kAuto, kPush, kPull };

namespace detail {

// One open fd per distinct graph file, opened once and SHARED read-only across
// all workers (O_DIRECT preads carry an explicit offset, so a single fd is safe
// to share).  This keeps the open-fd count at O(#files) instead of
// O(#workers * #files) — the latter exhausts RLIMIT_NOFILE on a many-core,
// many-drive box, after which open() fails, the bad fd gets used, and every
// pread reports EBADF ("Bad file descriptor").  Mirrors ChunkSequenceReader.
template <typename T>
std::unordered_map<std::string, int> open_graph_fds(const nested_seq<T>& g) {
  std::unordered_map<std::string, int> fds;
  for (const auto& nc : g.chunks) {
    if (fds.find(nc.raw.filename) == fds.end()) {
      int fd = open(nc.raw.filename.c_str(), O_RDONLY | O_DIRECT);
      SYSCALL(fd);
      fds.emplace(nc.raw.filename, fd);
    }
  }
  return fds;
}
inline void close_graph_fds(std::unordered_map<std::string, int>& fds) {
  for (auto& [name, fd] : fds) close(fd);
  fds.clear();
}

// Per-worker row reader over a nested_seq graph: fetches inner sequence `u`.
// One instance PER parlay worker (indexed by parlay::worker_id()), so the
// scratch buffer is never shared across threads.  A single CHUNK_SIZE buffer
// serves both read strategies (a row always lies within one chunk-sized
// O_DIRECT window, so a byte-range read of one row also fits in CHUNK_SIZE).
// The fd table is shared (owned by the caller), not per-reader.
template <typename T>
struct RowReader {
  T* buf = nullptr;
  size_t cached_chunk = (size_t)-1;  // input chunk currently whole in buf (pull)
  const std::unordered_map<std::string, int>* fds = nullptr;  // shared, read-only

  RowReader() = default;
  RowReader(const RowReader&) = delete;
  RowReader& operator=(const RowReader&) = delete;
  RowReader(RowReader&& o) noexcept
      : buf(o.buf), cached_chunk(o.cached_chunk), fds(o.fds) {
    o.buf = nullptr;
  }
  ~RowReader() {
    if (buf) free(buf);  // fds are shared and closed by the caller, not here
  }

  int fd_for(const std::string& name) const { return fds->at(name); }

  // Returns a pointer to inner sequence u's data (valid only until the next
  // call on this reader) and writes its length to *len; nullptr for an empty
  // row.  The push path reads only the row's byte window; the pull path reads
  // (and caches) the whole chunk so consecutive same-chunk rows reuse it.
  const T* read_ptr(const nested_seq<T>& g, uint64_t u, bool push, size_t* len) {
    const size_t ci = g.which_chunk(u);
    const nested_chunk& nc = g.chunks[ci];
    const size_t local_off = g.seq_len_scan[u] - g.seq_len_scan[nc.first_seq];
    const size_t l = g.seq_len_scan[u + 1] - g.seq_len_scan[u];
    *len = l;
    if (l == 0) return nullptr;
    if (!buf) {
      buf = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
      CHECK(buf != nullptr) << "RowReader: buffer allocation failed";
    }
    if (push) {
      // Byte-range read of just this row, aligned down to O_DIRECT_MULTIPLE.
      const size_t row_b0 = nc.raw.begin_addr + local_off * sizeof(T);
      const size_t astart = AlignDown(row_b0);
      const size_t alen = AlignUp(row_b0 + l * sizeof(T)) - astart;
      SYSCALL(pread(fd_for(nc.raw.filename), buf, alen, (off_t)astart));
      cached_chunk = (size_t)-1;  // buf no longer holds a whole chunk
      return (const T*)((const char*)buf + (row_b0 - astart));
    }
    if (ci != cached_chunk) {
      SYSCALL(pread(fd_for(nc.raw.filename), buf, AlignUp(nc.raw.used),
                    (off_t)nc.raw.begin_addr));
      cached_chunk = ci;
    }
    return buf + local_off;
  }

  parlay::sequence<T> read(const nested_seq<T>& g, uint64_t u, bool push) {
    size_t l;
    const T* p = read_ptr(g, u, push, &l);
    parlay::sequence<T> out(l);
    for (size_t j = 0; j < l; j++) out[j] = p[j];
    return out;
  }

  // Copy row u's data into dst; returns its length (no per-row allocation).
  size_t read_into(const nested_seq<T>& g, uint64_t u, bool push, T* dst) {
    size_t l;
    const T* p = read_ptr(g, u, push, &l);
    for (size_t j = 0; j < l; j++) dst[j] = p[j];
    return l;
  }
};

// A DensePack producer batch of gathered rows: run(b) is row b's neighbour ids,
// which DensePack concatenates densely (splitting across chunk boundaries is
// fine — the flattened output needs no row boundaries).
template <typename T>
struct GatherFlatBatch {
  std::vector<parlay::sequence<T>> rows;
  size_t size() const { return rows.size(); }
  DensePackRun<T> run(size_t b) const { return {rows[b].data(), rows[b].size()}; }
};

// Shared front half of both gathers: distinct, id-sorted selection + the
// push/pull direction decision.  Returns `sel`; writes the chosen direction to
// `push_out`.
template <typename T>
std::vector<uint64_t> select_and_decide(
    const nested_seq<T>& g, const parlay::sequence<uint64_t>& frontier_ids,
    GatherDir dir, bool& push_out) {
  // Sort + dedup in PARALLEL: on a many-core box a single-threaded std::sort of
  // a large frontier (100k-1M ids) pins one core while the rest idle — one of
  // the "1-2 cores busy" culprits in the push path.  Ascending-id order also
  // gives the whole-chunk (pull) read strategy its locality.
  parlay::sequence<uint64_t> su = parlay::sort(frontier_ids);
  su = parlay::unique(su);
  std::vector<uint64_t> sel(su.begin(), su.end());
  parlay::parallel_for(0, sel.size(), [&](size_t k) {
    CHECK(sel[k] < g.total_seqs()) << "NestedGather: id " << sel[k] << " >= |V|";
  });

  if (dir == GatherDir::kPush)
    push_out = true;
  else if (dir == GatherDir::kPull)
    push_out = false;
  else {
    size_t frontier_edges = 0;
    for (uint64_t u : sel)
      frontier_edges += g.seq_len_scan[u + 1] - g.seq_len_scan[u];
    const size_t m = g.total_seqs() == 0 ? 0 : g.seq_len_scan.back();
    push_out = frontier_edges * 20 < m;  // push when < ~5% of edges touched
  }
  return sel;
}

}  // namespace detail

/**
 * NestedGather — the `map(frontier, u -> G[u])` step, materialized as a
 * nested_seq: output row k is graph `g`'s adjacency row for the k-th *distinct*
 * selected vertex, in ascending vertex-id order (order-insensitive for BFS,
 * which only flattens + filters the result).
 *
 * `frontier_ids` is a DRAM sequence of vertex ids (<= |V|, same budget as the
 * BFS distance array; the graph is the only out-of-core operand).  The output
 * layout is planned entirely from `g.seq_len_scan` (the selected rows' lengths
 * are known with no I/O) and produced by reusing NestedTabulate's greedy
 * whole-row packer, with the generator reading each row from `g`.
 *
 * Direction optimization: `dir` picks the read strategy; kAuto reads only the
 * selected rows (push) when the frontier touches a small fraction of the edges,
 * else sweeps whole chunks (pull).  Push and pull differ ONLY in that read
 * strategy — the DRAM plan, packer and output are identical.  If
 * `chose_push_out` is non-null it receives the direction actually used.
 */
template <typename T>
nested_seq<T> NestedGather(const nested_seq<T>& g,
                           const parlay::sequence<uint64_t>& frontier_ids,
                           const std::string& result_prefix,
                           GatherDir dir = GatherDir::kAuto,
                           bool* chose_push_out = nullptr) {
  bool push;
  std::vector<uint64_t> sel =
      detail::select_and_decide(g, frontier_ids, dir, push);
  if (chose_push_out) *chose_push_out = push;
  if (sel.empty()) return {};

  auto fds = std::make_shared<std::unordered_map<std::string, int>>(
      detail::open_graph_fds(g));
  auto readers =
      std::make_shared<std::vector<detail::RowReader<T>>>(parlay::num_workers());
  for (auto& r : *readers) r.fds = fds.get();

  nested_seq<T> result = NestedTabulate<T>(
      sel.size(), result_prefix, [&g, sel, readers, push](size_t i) {
        return (*readers)[parlay::worker_id()].read(g, sel[i], push);
      });
  detail::close_graph_fds(*fds);
  return result;
}

/**
 * NestedGatherFlatten — the FUSED `flatten(map(frontier, u -> G[u]))`: reads the
 * frontier's rows and packs their concatenated neighbour ids straight into a
 * dense flat chunk_seq, WITHOUT ever materializing the intermediate nested_seq.
 * Same result as NestedFlatten(NestedGather(...)), but one write pass instead of
 * writing the nested_seq and reading it back.  Built on DensePack (one run per
 * selected row); direction choice is identical to NestedGather.
 */
template <typename T>
chunk_seq NestedGatherFlatten(const nested_seq<T>& g,
                              const parlay::sequence<uint64_t>& frontier_ids,
                              const std::string& result_prefix,
                              GatherDir dir = GatherDir::kAuto,
                              bool* chose_push_out = nullptr) {
  constexpr size_t epct = CHUNK_SIZE / sizeof(T);
  bool push;
  auto sel = std::make_shared<std::vector<uint64_t>>(
      detail::select_and_decide(g, frontier_ids, dir, push));
  if (chose_push_out) *chose_push_out = push;
  if (sel->empty()) return {};

  // Group selected rows into runs, one DensePack virtual chunk each.  Two forces
  // set the target run size:
  //   * too SMALL (e.g. one row) drowns the pass in per-run overhead — the
  //     original bug that made fused push 5x slower;
  //   * too LARGE (a full output chunk, epct) yields too FEW runs — a frontier
  //     of fe edges makes only ceil(fe/epct) runs, so a sparse push level has
  //     1-2 runs and uses 1-2 cores no matter how many the box has (the "1-2
  //     cores pinned" symptom on the 64-core machine).
  // So target ~oversample*num_workers runs (parallelism), floored so each run
  // still batches many rows (amortizes overhead AND keeps whole-chunk read
  // locality: a run is a contiguous ascending-id slice of sel, so its rows
  // mostly share one input chunk).  Capped at epct so a run never exceeds one
  // output chunk.  All from seq_len_scan — no I/O.
  size_t total_sel_len = 0;
  for (uint64_t u : *sel) total_sel_len += g.seq_len_scan[u + 1] - g.seq_len_scan[u];
  const size_t P = parlay::num_workers() ? parlay::num_workers() : 1;
  size_t target = epct;
  {
    const size_t want = total_sel_len / (P * 4) + 1;  // ~4 runs/worker
    target = std::max<size_t>(4096, std::min(epct, want));  // >= 32 KiB of ids
  }
  auto starts = std::make_shared<std::vector<size_t>>();  // group bounds into sel
  starts->push_back(0);
  {
    size_t cur = 0;
    for (size_t k = 0; k < sel->size(); k++) {
      const uint64_t u = (*sel)[k];
      const size_t l = g.seq_len_scan[u + 1] - g.seq_len_scan[u];
      if (cur > 0 && cur + l > target) {
        starts->push_back(k);
        cur = 0;
      }
      cur += l;
    }
    starts->push_back(sel->size());
  }
  const size_t num_groups = starts->size() - 1;

  auto fds = std::make_shared<std::unordered_map<std::string, int>>(
      detail::open_graph_fds(g));
  auto readers =
      std::make_shared<std::vector<detail::RowReader<T>>>(parlay::num_workers());
  for (auto& r : *readers) r.fds = fds.get();

  chunk_seq result = DensePack<T>(
      num_groups, result_prefix,
      [&g, sel, starts, readers, push](size_t base, size_t batch_n) {
        detail::GatherFlatBatch<T> batch;
        batch.rows.resize(batch_n);
        parlay::parallel_for(
            0, batch_n,
            [&](size_t b) {
              const size_t gi = base + b;
              const size_t lo = (*starts)[gi], hi = (*starts)[gi + 1];
              size_t total = 0;
              for (size_t k = lo; k < hi; k++) {
                const uint64_t u = (*sel)[k];
                total += g.seq_len_scan[u + 1] - g.seq_len_scan[u];
              }
              parlay::sequence<T> buf(total);
              auto& rr = (*readers)[parlay::worker_id()];
              size_t off = 0;
              for (size_t k = lo; k < hi; k++)
                off += rr.read_into(g, (*sel)[k], push, buf.data() + off);
              batch.rows[b] = std::move(buf);
            },
            1);
        return batch;
      });
  detail::close_graph_fds(*fds);
  return result;
}

}  // namespace ChunkSequenceOps

#endif  // NESTED_GATHER_H
