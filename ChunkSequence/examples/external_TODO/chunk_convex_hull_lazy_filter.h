#ifndef CHUNK_CONVEX_HULL_LAZY_FILTER_H
#define CHUNK_CONVEX_HULL_LAZY_FILTER_H

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "ChunkSequence/Primitives/delayed.h"
#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/examples/external/chunk_convex_hull.h"
#include "absl/log/check.h"
#include "configs.h"
#include "parlay/primitives.h"
#include "utils/file_utils.h"
#include "utils/unordered_file_writer.h"

// Variant of UpperHull (chunk_convex_hull.h) that uses delayed::lazy_filter in
// place of ChunkPartition for every narrowing step.  Same algorithm shape, same
// DRAM base case (qh_mem), same farthest-point ChunkReduce -- only the
// partitioning primitive changes.  ChunkPartition does the top-level "keep
// points above the base line" gate and each recursion level's left/right split
// in ONE read+write pass (see chunk_convex_hull.h:212-217, 276-277 for why:
// its buckets are provably disjoint, so one k-bucket pass replaces what would
// otherwise be k separate filters).  lazy_filter is single-predicate, so the
// faithful swap is one lazy_filter per bucket ChunkPartition used to produce,
// each forced back to a chunk_seq before the recursion continues -- expected to
// cost more I/O than ChunkPartition's single combined pass (each lazy_filter
// pays its own full counting pass over the input, and materializing pays
// another read of the matched chunks). That added cost is what the companion
// benchmark (convex_hull_lazy_filter.cpp) measures.
//
// delayed::force can't be used to materialize these lazy_filter results: it
// static_asserts sizeof(R) <= 8 because every delayed node's OWN abstract
// chunking is expressed in units of the global, 8-byte-based ELEMS_PER_CHUNK,
// so force's one-physical-CHUNK_SIZE-buffer-per-logical-chunk assumption only
// holds when sizeof(R) <= 8. hpoint is 32 bytes (chunk_convex_hull.h:39-42
// already flags "the delayed layer is limited to <=8 B"). Since hpoint's own
// physical epc = CHUNK_SIZE/sizeof(hpoint) divides ELEMS_PER_CHUNK exactly
// (4x, because 32/8 == 4), materialize_wide below generalizes force's
// precompute-then-stream shape to split each logical chunk into
// ceil(chunk_len(ci)/epc) physical output pieces instead of assuming exactly
// one -- kept local to this example rather than changing delayed::force itself.

namespace plaid {

namespace cd = plaid::delayed;

namespace detail {

// Generalized delayed::force for element types wider than 8 bytes -- see the
// file banner above. Modeled directly on force's own body (chunk_delayed.h).
template <class D>
chunk_seq materialize_wide(const D& d, const std::string& result_prefix) {
  using R = typename D::value_type;
  static_assert(CHUNK_SIZE % sizeof(R) == 0,
                "sizeof(R) must divide CHUNK_SIZE for O_DIRECT alignment");
  constexpr size_t epc =
      CHUNK_SIZE / sizeof(R);  // physical elems/chunk for THIS R

  const size_t nc = d.num_chunks();
  if (nc == 0) return {};
  const size_t num_drives = GetSSDList().size();

  // Precompute, per logical chunk ci, how many physical pieces it splits into
  // (every non-last logical chunk has exactly ELEMS_PER_CHUNK elements, so
  // exactly ELEMS_PER_CHUNK/epc pieces -- 4 for hpoint -- all full; only the
  // very last piece of the very last logical chunk is ever partial), and each
  // piece's flat output index (prefix sum over pieces(ci)).
  std::vector<size_t> pieces(nc), piece_off(nc + 1, 0);
  for (size_t ci = 0; ci < nc; ci++) {
    pieces[ci] = (d.chunk_len(ci) + epc - 1) / epc;
    piece_off[ci + 1] = piece_off[ci] + pieces[ci];
  }
  const size_t total_pieces = piece_off[nc];
  if (total_pieces == 0) return {};

  // Randomly assign each output piece to a drive; insertion order within a
  // drive gives its CHUNK_SIZE-aligned slot -- same balls-in-bins precompute
  // as force(), just keyed by flat piece index instead of logical chunk index.
  std::vector<size_t> drive_of(total_pieces);
  {
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, num_drives - 1);
    for (size_t i = 0; i < total_pieces; i++) drive_of[i] = dist(rng);
  }
  std::vector<std::vector<size_t>> drive_pieces(num_drives);
  for (size_t i = 0; i < total_pieces; i++)
    drive_pieces[drive_of[i]].push_back(i);
  std::vector<size_t> slot_of(total_pieces);
  for (size_t dr = 0; dr < num_drives; dr++)
    for (size_t s = 0; s < drive_pieces[dr].size(); s++)
      slot_of[drive_pieces[dr][s]] = s;

  // Pre-fallocate each drive file to its exact final size.
  std::vector<std::string> filenames(num_drives);
  parlay::parallel_for(
      0, num_drives,
      [&](size_t dr) {
        filenames[dr] = GetFileName(result_prefix, dr);
        const size_t file_size = drive_pieces[dr].size() * CHUNK_SIZE;
        if (file_size == 0) return;
        int fd =
            open(filenames[dr].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        SYSCALL(fd);
        if (fallocate(fd, 0, 0, (off_t)file_size) != 0)
          SYSCALL(ftruncate(fd, (off_t)file_size));
        SYSCALL(close(fd));
      },
      /*granularity=*/1);

  // Output chunk descriptors are fully determined up front (index-ordered).
  std::vector<chunk> out_chunks(total_pieces);
  for (size_t ci = 0; ci < nc; ci++) {
    size_t remaining = d.chunk_len(ci);
    for (size_t k = 0; k < pieces[ci]; k++) {
      const size_t idx = piece_off[ci] + k;
      const size_t take = std::min(remaining, epc);
      out_chunks[idx] = {filenames[drive_of[idx]], slot_of[idx] * CHUNK_SIZE,
                         take * sizeof(R), idx};
      remaining -= take;
    }
  }

  UnorderedWriterConfig wcfg;
  wcfg.num_threads = num_drives;
  wcfg.io_uring_size = 32;
  wcfg.queue_size = 64;
  wcfg.num_files = num_drives;
  UnorderedFileWriter<R> writer;
  writer.Start(filenames, wcfg);

  cd::for_each_chunk(d, [&](size_t ci, size_t n, auto it) {
    size_t remaining = n;
    for (size_t k = 0; k < pieces[ci]; k++) {
      const size_t idx = piece_off[ci] + k;
      const size_t take = std::min(remaining, epc);
      R* out = (R*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
      CHECK(out != nullptr) << "materialize_wide: allocation failed";
      for (size_t i = 0; i < take; i++) {
        out[i] = *it;
        ++it;
      }
      if (take * sizeof(R) < CHUNK_SIZE)
        memset((char*)out + take * sizeof(R), 0, CHUNK_SIZE - take * sizeof(R));
      writer.Push(std::shared_ptr<R>(out, free), CHUNK_SIZE / sizeof(R),
                  drive_of[idx], slot_of[idx] * CHUNK_SIZE);
      remaining -= take;
    }
  });

  writer.Wait();
  return {out_chunks};
}

// Same shape as quickhull_ext (chunk_convex_hull.h) -- same base case via
// qh_mem, same FarMonoid reduce for mid -- except the single 2-bucket
// ChunkPartition call becomes two separate lazy_filter + materialize_wide
// round trips, one per sub-region. Left and right can't share a scratch prefix
// the way ChunkPartition's two buckets do (materialize_wide writes to
// result_prefix directly, one call per side), so they get distinct ones.
inline std::vector<uint64_t> quickhull_ext_lazy(const chunk_seq& pts, hpoint l,
                                                hpoint r, size_t budget_elems,
                                                const std::string& scratch) {
  const size_t n = plaid::size<hpoint>(pts);
  if (n == 0) return {};

  // Base case: identical to quickhull_ext's -- materialize and finish in DRAM.
  if (n <= budget_elems) {
    parlay::sequence<hpoint> S;
    {
      std::vector<hpoint> v = pts.to_vector<hpoint>();
      S = parlay::tabulate(v.size(), [&](size_t i) { return v[i]; });
    }  // free the std::vector before the in-DRAM quickhull runs
    auto res = qh_mem(std::move(S), l, r);
    return std::vector<uint64_t>(res.begin(), res.end());
  }

  ext_split_counter().fetch_add(1, std::memory_order_relaxed);
  const hpoint mid =
      ChunkReduce<hpoint, FarMonoid::Acc>(pts, FarMonoid{l, r}).pt;

  const std::string sp =
      scratch + "p" + std::to_string(prefix_counter().fetch_add(1));

  auto d = cd::delay<hpoint>(pts);
  auto left_lazy = cd::lazy_filter(
      d, [l, mid](const hpoint& p) { return area(l, mid, p) > 0; });
  auto right_lazy = cd::lazy_filter(
      d, [mid, r](const hpoint& p) { return area(mid, r, p) > 0; });
  chunk_seq left_seq = materialize_wide(left_lazy, sp + "L");
  chunk_seq right_seq = materialize_wide(right_lazy, sp + "R");

  // Left and right have their own scratch; recurse into each (sequentially,
  // so both are still readable) before removing them.
  std::vector<uint64_t> LR =
      quickhull_ext_lazy(left_seq, l, mid, budget_elems, sp);
  std::vector<uint64_t> RR =
      quickhull_ext_lazy(right_seq, mid, r, budget_elems, sp);
  cleanup_prefix(sp + "L");
  cleanup_prefix(sp + "R");

  std::vector<uint64_t> out;
  out.reserve(LR.size() + 1 + RR.size());
  out.insert(out.end(), LR.begin(), LR.end());
  out.push_back(mid.idx);
  out.insert(out.end(), RR.begin(), RR.end());
  return out;
}

}  // namespace detail

// Same contract as UpperHull (chunk_convex_hull.h) -- see its doc comment --
// except every ChunkPartition call is replaced by delayed::lazy_filter +
// materialize_wide. Default scratch_prefix differs from UpperHull's
// ("ch_scratch") purely so the two leave distinguishable files when run
// back-to-back in the same process (harmless either way: prefix_counter() is a
// single shared monotonic counter, so the two never actually collide).
inline std::vector<uint64_t> UpperHullLazyFilter(
    const chunk_seq& points, size_t dram_budget_bytes,
    const std::string& scratch_prefix = "ch_lazy_scratch") {
  detail::ext_split_counter().store(0, std::memory_order_relaxed);
  if (plaid::size<hpoint>(points) == 0) return {};

  const size_t budget_elems =
      std::max<size_t>(1, dram_budget_bytes / sizeof(hpoint));

  const detail::MinMaxMonoid::Acc mm =
      ChunkReduce<hpoint, detail::MinMaxMonoid::Acc>(points,
                                                     detail::MinMaxMonoid{});
  const hpoint minp = mm.mn, maxp = mm.mx;

  const std::string ap = scratch_prefix + "a" +
                         std::to_string(detail::prefix_counter().fetch_add(1));
  auto d = cd::delay<hpoint>(points);
  auto above_lazy = cd::lazy_filter(
      d, [minp, maxp](const hpoint& p) { return area(minp, maxp, p) > 0; });
  chunk_seq above = detail::materialize_wide(above_lazy, ap);

  std::vector<uint64_t> between = detail::quickhull_ext_lazy(
      above, minp, maxp, budget_elems, scratch_prefix);
  detail::cleanup_prefix(ap);

  std::vector<uint64_t> out;
  out.reserve(between.size() + 2);
  out.push_back(minp.idx);
  out.insert(out.end(), between.begin(), between.end());
  out.push_back(maxp.idx);
  return out;
}

}  // namespace plaid

#endif  // CHUNK_CONVEX_HULL_LAZY_FILTER_H
