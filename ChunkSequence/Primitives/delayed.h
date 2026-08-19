#ifndef CHUNK_DELAYED_H
#define CHUNK_DELAYED_H

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ChunkSequence/Primitives/chunk_seq.h"
#include "absl/log/check.h"
#include "configs.h"
#include "parlay/primitives.h"
#include "utils/file_utils.h"

namespace plaid {
// Retained name for the input-chunk batch size (== the DensePack batch size).
// Defined here rather than alongside ChunkFilter so that delayed.h depends on
// nothing but the substrate: primitives.h includes THIS header (materialize
// runs over delayed sources), so a dependency the other way would be a cycle.
static constexpr size_t FILTER_BATCH_SIZE = DENSE_PACK_BATCH_SIZE;
}  // namespace plaid

// ─────────────────────────────────────────────────────────────────────────────
// Block-delayed sequences over chunk_seq  (recursive read-plan model).
//
// A port of parlaylib's "block iterable delayed" design to the out-of-core
// (SSD) setting.  Where the eager primitives round-trip every intermediate
// through the SSDs, a delayed sequence fuses an operation chain so
// intermediates never touch disk: `map` is lazy, `reduce`/`force` consume in
// one read pass, and `scan` is partially delayed (one read pass for block
// offsets, then a new lazy sequence).
//
// A "block" is one CHUNK_SIZE chunk: parlay workers process chunks in parallel;
// iteration is sequential within a chunk.
//
// ── Representation: a delayed sequence is a *tree of nodes* ──────────────────
// Every node (leaf_source, leaf_index, map_node, scan_node, zip_node) exposes a
// small uniform interface so one generic driver can execute *any* composition —
// including nested/N-ary zips and zips of maps/scans:
//
//   size_t length()      const;             // logical element count
//   size_t num_chunks()  const;             // number of logical chunks
//   size_t chunk_start(i) const;            // global element offset of chunk i
//   size_t chunk_len(i)  const;             // elements in logical chunk i
//
// Together these four describe the node's *own* partition of [0, length).  It
// is NOT necessarily the global ELEMS_PER_CHUNK grid: leaf_source<T> and
// cut_source<T> partition on their own epc = CHUNK_SIZE/sizeof(T), so a chain
// over 4-byte or 32-byte elements has chunks of a different element count than
// one over uint64_t.  A node that combines children (zip_node) must therefore
// reconcile two partitions rather than assume they agree.
//
//   // (1) READ PLAN: register the physical reads this node needs for logical
//   //     chunk i via Planner::need (keyed by source chunk_seq*, so a source
//   //     appearing in several leaves of the chunk collapses to one read).
//   //     Internal nodes forward to children *left-to-right*; a node past its
//   //     own range (a padded child) registers nothing.
//   template<class Planner> void plan(size_t i, Planner& p) const;
//
//   // (2) BUILD: construct the fused forward-iterator for logical chunk i,
//   //     already advanced `skip` elements past that chunk's start, pulling
//   //     each in-range leaf's buffer from Resolver::next.  build() MUST
//   //     visit children in the SAME left-to-right order as plan(), and must
//   //     consume exactly one next() per need() plan() registered -- including
//   //     buffers that `skip` steps past -- so the positional resolver lines
//   //     up.  For i beyond this node's range it returns a dummy iterator
//   //     (consuming no buffer); such an iterator is always wrapped by an
//   //     enclosing pad_iter with remaining==0, so it is never dereferenced.
//   //
//   //     `skip` exists so a parent may re-grid onto a partition finer than
//   //     this node's own and still address a sub-range of one of its chunks.
//   //     Every node absorbs it in O(1) except scan_node, which must fold the
//   //     skipped elements to reach the right accumulator (O(skip)).
//   template<class Resolver> auto build(size_t i, size_t skip, Resolver& r)
//       const;
//
// One logical chunk of a *leaf* is one physical read (a chunk_seq stores each
// logical chunk as one contiguous region on one drive).  The "one logical chunk
// -> several physical reads" case is the zip_node union; identical reads from a
// shared source (e.g. A,B in both zip(A,B) and a scan of zip(A,B)) are deduped,
// so C=f(A,B) reuses A,B's buffers instead of re-reading them.
//
// Three drivers execute a tree.  for_each_chunk streams: one read pass whose
// dispatcher releases each chunk to a parlay worker the instant that chunk's
// own reads land, so reads and compute overlap (used by reduce/scan/force).
// for_each_window collects a FILTER_BATCH_SIZE window before computing, needed
// by filter's sequential cross-chunk carry.  sequential_for_each_chunk reads
// via blocking pread on the calling thread instead of a ChunkSequenceReader,
// for small ranges consumed from inside an already-parallel outer loop.
// Everything is templated (no std::function) so the fused chain inlines.
//
// LIFETIME: a leaf_source holds a pointer to its chunk_seq; every source in the
// tree must outlive every terminal call.  force writes one output chunk per
// logical chunk, so it requires only that a logical chunk's elements fit one
// CHUNK_SIZE chunk (checked at run time) — zip's std::pair elements are still
// meant to stay transient and be map-ed to a scalar before force, but a
// *narrower* R than the grid it was sized on is fine: the output is then
// ragged, and delay<R>() reads it back correctly.
// ─────────────────────────────────────────────────────────────────────────────

namespace plaid {
namespace delayed {

// ── lazy forward iterators (sequential within a chunk) ───────────────────────

// Generates f(cur), f(cur+1), …  Base iterator for `tabulate` (no buffer).
template <class F>
struct counting_value_iter {
  size_t cur;
  F f;
  auto operator*() const { return f(cur); }
  counting_value_iter& operator++() {
    ++cur;
    return *this;
  }
};
template <class F>
counting_value_iter<F> make_counting(size_t cur, F f) {
  return {cur, f};
}

// Lazily applies g to the element under `it`.  Composes a `map` onto any chain.
template <class It, class G>
struct map_iter {
  It it;
  G g;
  auto operator*() const { return g(*it); }
  map_iter& operator++() {
    ++it;
    return *this;
  }
};
template <class It, class G>
map_iter<It, G> make_map_iter(It it, G g) {
  return {it, g};
}

// Exclusive prefix scan iterator: *iter is the running accumulator (seeded per
// chunk with that chunk's offset); ++ folds the underlying element in.
template <class It, class F, class V>
struct scan_iter {
  It it;
  F f;
  V acc;
  V operator*() const { return acc; }
  scan_iter& operator++() {
    acc = f(acc, *it);
    ++it;
    return *this;
  }
};
template <class It, class F, class V>
scan_iter<It, F, V> make_scan_iter(It it, F f, V acc) {
  return {it, f, acc};
}

// Yields the first `remaining` elements of `it`, then `pad` forever.  Used by
// zip to fill a shorter operand's tail (remaining==0 for a chunk it never
// reaches — its inner iterator is then never dereferenced, so a null/dummy
// inner is safe).
template <class It, class V>
struct pad_iter {
  It it;
  size_t remaining;
  V pad;
  V operator*() const { return remaining ? (V)(*it) : pad; }
  pad_iter& operator++() {
    if (remaining) {
      ++it;
      --remaining;
    }
    return *this;
  }
};
template <class It, class V>
pad_iter<It, V> make_pad_iter(It it, size_t remaining, V pad) {
  return {it, remaining, pad};
}

// Zips two (already pad-wrapped) iterators into a sequence of std::pair.
template <class ItA, class ItB>
struct zip_iter {
  ItA a;
  ItB b;
  auto operator*() const { return std::pair{*a, *b}; }
  zip_iter& operator++() {
    ++a;
    ++b;
    return *this;
  }
};
template <class ItA, class ItB>
zip_iter<ItA, ItB> make_zip_iter(ItA a, ItB b) {
  return {a, b};
}

// Raw walk over a heap-materialized buffer, kept alive via shared_ptr for the
// iterator's lifetime.  Used by filter_node::build, which re-filters the source
// chunk(s) a logical chunk needs into a small compacted buffer (see
// lazy_filter).
template <class T>
struct materialized_iter {
  std::shared_ptr<std::vector<T>> buf;
  size_t pos = 0;
  const T& operator*() const { return (*buf)[pos]; }
  materialized_iter& operator++() {
    ++pos;
    return *this;
  }
};

// ── read planning: dedup + resolve ───────────────────────────────────────────
//
// A node's plan() calls Planner::need once per in-range leaf, keyed by
// (source chunk_seq*, physical chunk index), so a source that appears in
// several leaves of one logical chunk *at the same physical index* (e.g. A and
// B in both zip(A,B) and a scan of zip(A,B)) collapses to a single physical
// read.  The index is part of the key -- not just the source pointer -- because
// a node may also need several *different* physical indices from the same
// source within one logical chunk (e.g. cut_source spanning two chunks, or
// filter_node's predecessor search spanning a run of source chunks); those
// must NOT collapse into each other.  build() then calls Resolver::next once
// per in-range leaf, in the same left-to-right order, to get that leaf's
// resolved buffer.
struct Planner {
  std::vector<chunk> unique_reads;  // this chunk's deduped reads (slot order)
  std::vector<uint32_t>
      leaf_slots;  // one local slot per in-range leaf occurrence
  std::vector<const chunk_seq*>
      src_of;  // dedup key per unique read (parallel array)

  void need(const chunk_seq* src, const chunk& c) {
    for (uint32_t s = 0; s < src_of.size(); s++)  // fanout is tiny: linear scan
      if (src_of[s] == src && unique_reads[s].index == c.index) {
        leaf_slots.push_back(s);
        return;
      }
    leaf_slots.push_back((uint32_t)unique_reads.size());
    src_of.push_back(src);
    unique_reads.push_back(c);
  }
};
struct Resolver {
  const std::vector<char*>* bufs;  // this chunk's buffers, by local slot
  const std::vector<uint32_t>*
      leaf_slots;  // the list Planner produced (same order)
  size_t cursor = 0;
  char* next() { return (*bufs)[(*leaf_slots)[cursor++]]; }
};

// ── nodes ────────────────────────────────────────────────────────────────────
// Shared helper: elements in logical chunk i of a sequence of `len` elements.
inline size_t grid_chunk_len(size_t len, size_t i) {
  const size_t base = i * ELEMS_PER_CHUNK;
  return base >= len ? 0 : std::min(ELEMS_PER_CHUNK, len - base);
}
inline size_t grid_num_chunks(size_t len) {
  return (len + ELEMS_PER_CHUNK - 1) / ELEMS_PER_CHUNK;
}
inline size_t grid_chunk_start(size_t i) { return i * ELEMS_PER_CHUNK; }

// Leaf backed by a materialized chunk_seq on SSD (from `delay`).  One logical
// chunk == one physical read (chunks[i]); a chunk_seq stores each logical chunk
// contiguously on one drive.  T is the stored element type.
//
// The partition follows the source's actual per-chunk `used`, so a *ragged*
// chunk_seq -- one with a partial chunk anywhere but the end, as produced by
// plaid::flatten, plaid::reverse, or a narrowing ChunkMap -- is a valid source.
// `starts` is null for the overwhelmingly common dense-except-last case, where
// the partition is the closed form i*epc and no table is built; see `delay`.
template <class T>
struct leaf_source {
  using value_type = T;
  const chunk_seq* src;
  size_t len;  // total element count
  size_t nc;   // number of chunks (== src->chunks.size())
  // null  => dense: chunk i starts at i*epc.
  // set   => ragged: exclusive prefix of per-chunk element counts, size nc+1.
  std::shared_ptr<const std::vector<size_t>> starts;

  // Elements a full physical CHUNK_SIZE chunk holds for this T -- see
  // cut_source::epc; using the global (uint64_t-sized) ELEMS_PER_CHUNK here
  // instead only matches physical layout when sizeof(T) == 8.
  static constexpr size_t epc = CHUNK_SIZE / sizeof(T);

  size_t length() const { return len; }
  size_t num_chunks() const { return nc; }
  size_t chunk_start(size_t i) const {
    return starts ? (*starts)[std::min(i, nc)] : i * epc;
  }
  // See the uniform_grid contract in the header comment: a dense source is the
  // closed form i*epc, a ragged one is not.
  bool uniform_grid(size_t* epc_out) const {
    *epc_out = epc;
    return starts == nullptr;
  }
  size_t chunk_len(size_t i) const {
    if (i >= nc) return 0;
    if (starts) return (*starts)[i + 1] - (*starts)[i];
    const size_t base = i * epc;
    return base >= len ? 0 : std::min(epc, len - base);
  }

  template <class Planner>
  void plan(size_t i, Planner& p) const {
    // A ragged source may carry an empty chunk; it contributes no elements, so
    // skip its read entirely.  build() tests the same condition, keeping the
    // positional resolver in step.
    if (i < nc && chunk_len(i) > 0)  // index-ordered: chunks[i].index == i
      p.need(src, src->chunks[i]);   // one read (shared if the src repeats)
  }  // else padded/out-of-range/empty: no read
  template <class Resolver>
  const T* build(size_t i, size_t skip, Resolver& r) const {
    if (i >= nc || chunk_len(i) == 0)
      return nullptr;  // dummy; enclosing pad never derefs it
    return reinterpret_cast<const T*>(r.next()) + skip;
  }
};

// Walks a re-windowed slice of a source: the first `lo_remaining` elements come
// from `lo` (already offset into the source's physical chunk), then the rest
// from `hi` (the start of the next physical chunk).  `hi` is never dereferenced
// when a chunk's slice lands fully inside one physical chunk (lo_remaining
// never hits 0 before the caller stops iterating).
template <class T>
struct cut_iter {
  const T* lo;
  size_t lo_remaining;
  const T* hi;
  // Advance `s` elements without touching memory: consume the `lo` segment
  // first, then spill into `hi` (the build-time `skip` a re-gridding parent
  // asks for).
  cut_iter& skip_ahead(size_t s) {
    const size_t from_lo = std::min(s, lo_remaining);
    lo += from_lo;
    lo_remaining -= from_lo;
    hi += s - from_lo;
    return *this;
  }
  T operator*() const { return lo_remaining ? *lo : *hi; }
  cut_iter& operator++() {
    if (lo_remaining) {
      ++lo;
      --lo_remaining;
    } else {
      ++hi;
    }
    return *this;
  }
};

// Leaf over an arbitrary [start_index, end_index) slice of a source chunk_seq
// (from `cut`), re-indexed to its own CHUNK_SIZE/sizeof(T)-element grid (`epc`
// below) starting at 0.  Because the slice's grid origin is offset from the
// source's own physical chunk boundaries by `start_index % epc`, every output
// logical chunk lands at the same offset into a physical chunk and so spans at
// most two physical reads (the tail of one, the head of the next) -- never
// more, since one output chunk holds <= epc elements and physical chunks
// (besides the source's last) are exactly epc elements.
template <class T>
struct cut_source {
  using value_type = T;
  const chunk_seq* src;
  size_t start_index;  // offset into src, in elements
  size_t len;          // slice length

  // Elements a full physical CHUNK_SIZE chunk holds for this T -- NOT the
  // global (uint64_t-sized) ELEMS_PER_CHUNK, which only matches physical
  // layout when sizeof(T) == 8.  For any other element size (e.g. the
  // 32-byte weighted_edge), using ELEMS_PER_CHUNK here picks the wrong
  // physical chunk / offset once the cut range passes the true per-chunk
  // element count and reads out of bounds.
  static constexpr size_t epc = CHUNK_SIZE / sizeof(T);

  size_t length() const { return len; }
  size_t num_chunks() const { return (len + epc - 1) / epc; }
  size_t chunk_start(size_t i) const { return i * epc; }
  bool uniform_grid(size_t* epc_out) const {
    *epc_out = epc;
    return true;
  }
  size_t chunk_len(size_t i) const {
    const size_t base = i * epc;
    return base >= len ? 0 : std::min(epc, len - base);
  }

  // Physical layout of output chunk i: up to two (chunk-index, take-count)
  // segments.  phys_hi == (size_t)-1 means the chunk fits in one physical read.
  struct Seg {
    size_t phys_lo, offset_lo, take_lo, phys_hi, take_hi;
  };
  Seg segments(size_t i) const {
    Seg s{};
    const size_t cl = chunk_len(i);
    s.phys_hi = (size_t)-1;
    if (cl == 0) return s;
    const size_t g0 = start_index + i * epc;
    s.phys_lo = g0 / epc;
    s.offset_lo = g0 % epc;
    const size_t avail_lo =
        src->chunks[s.phys_lo].used / sizeof(T) - s.offset_lo;
    s.take_lo = std::min(cl, avail_lo);
    const size_t rem = cl - s.take_lo;
    if (rem > 0) {
      s.phys_hi = s.phys_lo + 1;
      s.take_hi = rem;
    }
    return s;
  }

  template <class Planner>
  void plan(size_t i, Planner& p) const {
    if (i >= num_chunks()) return;
    Seg s = segments(i);
    // Keyed by the physical chunk's own address (not `src`), since one
    // output chunk can need two distinct physical reads from the same
    // source -- src-level dedup (as leaf_source uses) would collapse them.
    p.need(reinterpret_cast<const chunk_seq*>(&src->chunks[s.phys_lo]),
           src->chunks[s.phys_lo]);
    if (s.phys_hi != (size_t)-1)
      p.need(reinterpret_cast<const chunk_seq*>(&src->chunks[s.phys_hi]),
             src->chunks[s.phys_hi]);
  }
  template <class Resolver>
  cut_iter<T> build(size_t i, size_t skip, Resolver& r) const {
    if (i >= num_chunks()) return {nullptr, 0, nullptr};
    Seg s = segments(i);
    const T* lo = reinterpret_cast<const T*>(r.next()) + s.offset_lo;
    const T* hi = (s.phys_hi != (size_t)-1)
                      ? reinterpret_cast<const T*>(r.next())
                      : nullptr;
    // Both reads are consumed regardless of `skip`, since plan() registered
    // both -- the resolver is positional, so build must not skip a next().
    return cut_iter<T>{lo, s.take_lo, hi}.skip_ahead(skip);
  }
};

// Leaf that generates element i as f(i), with no source files (from
// `tabulate`).
template <class F>
struct leaf_index {
  using value_type = std::decay_t<std::invoke_result_t<F, size_t>>;
  size_t n;
  F f;

  size_t length() const { return n; }
  size_t num_chunks() const { return grid_num_chunks(n); }
  size_t chunk_start(size_t i) const { return grid_chunk_start(i); }
  bool uniform_grid(size_t* epc_out) const {
    *epc_out = ELEMS_PER_CHUNK;
    return true;
  }
  size_t chunk_len(size_t i) const { return grid_chunk_len(n, i); }

  template <class Planner>
  void plan(size_t, Planner&) const {}  // no reads
  template <class Resolver>
  auto build(size_t i, size_t skip, Resolver&) const {
    // Skipping is free here: just start counting further in.
    return make_counting(chunk_start(i) + skip, f);  // padded if out-of-range
  }
};

// Lazily map g over every element of child D.
template <class D, class G>
struct map_node {
  using value_type =
      std::decay_t<std::invoke_result_t<G, typename D::value_type>>;
  D d;
  G g;

  size_t length() const { return d.length(); }
  size_t num_chunks() const { return d.num_chunks(); }
  size_t chunk_start(size_t i) const { return d.chunk_start(i); }
  bool uniform_grid(size_t* epc_out) const { return d.uniform_grid(epc_out); }
  size_t chunk_len(size_t i) const { return d.chunk_len(i); }

  template <class Planner>
  void plan(size_t i, Planner& p) const {
    d.plan(i, p);
  }
  template <class Resolver>
  auto build(size_t i, size_t skip, Resolver& r) const {
    return make_map_iter(d.build(i, skip, r),
                         g);  // skip is elementwise: forward
  }
};

// Exclusive prefix scan of child D under monoid M.  Per-chunk offsets are
// precomputed at construction (see `scan`); build seeds scan_iter per chunk.
template <class D, class M>
struct scan_node {
  using value_type = typename D::value_type;
  D d;
  M m;
  std::shared_ptr<std::vector<value_type>> offsets;

  size_t length() const { return d.length(); }
  size_t num_chunks() const { return d.num_chunks(); }
  size_t chunk_start(size_t i) const { return d.chunk_start(i); }
  bool uniform_grid(size_t* epc_out) const { return d.uniform_grid(epc_out); }
  size_t chunk_len(size_t i) const { return d.chunk_len(i); }

  template <class Planner>
  void plan(size_t i, Planner& p) const {
    d.plan(i, p);
  }
  template <class Resolver>
  auto build(size_t i, size_t skip, Resolver& r) const {
    value_type seed = (i < offsets->size()) ? (*offsets)[i] : m.identity;
    auto it = make_scan_iter(d.build(i, 0, r), m, seed);
    // A scan cannot skip cheaply: the accumulator at chunk-relative position
    // `skip` is the fold of the preceding `skip` elements, so it must walk
    // them.  O(skip) -- the one real cost of a parent re-gridding onto a
    // finer partition than this node's own (see zip_node).
    for (size_t j = 0; j < skip; j++) ++it;
    return it;
  }
};

// One logical chunk of a re-gridding zip: a maximal run of elements lying
// inside a single chunk of A *and* a single chunk of B.  `ai`/`bi` are the
// child chunk indices (== that child's num_chunks() when the run is past the
// child's end, i.e. pure padding), `skipA`/`skipB` the element offsets into
// them.
struct zip_piece {
  size_t start, len;
  size_t ai, skipA;
  size_t bi, skipB;
};

// Partition [0, len) at the union of A's and B's chunk boundaries -- the
// coarsest partition that still lands every piece inside one chunk of each
// child.  Past a child's end that child contributes no boundaries, so the
// padded tail is cut by the surviving side alone.
template <class DA, class DB>
inline std::vector<zip_piece> build_zip_pieces(const DA& a, const DB& b,
                                               size_t len) {
  std::vector<zip_piece> out;
  const size_t nA = a.num_chunks(), nB = b.num_chunks();
  size_t ai = 0, bi = 0, pos = 0;
  while (pos < len) {
    // Advance past chunks ending at or before pos.  This also steps over empty
    // chunks, which a ragged source may contain.
    while (ai < nA && a.chunk_start(ai) + a.chunk_len(ai) <= pos) ai++;
    while (bi < nB && b.chunk_start(bi) + b.chunk_len(bi) <= pos) bi++;
    size_t end = len;
    if (ai < nA) end = std::min(end, a.chunk_start(ai) + a.chunk_len(ai));
    if (bi < nB) end = std::min(end, b.chunk_start(bi) + b.chunk_len(bi));
    CHECK(end > pos) << "zip: child partition does not tile [0, length)";
    zip_piece pc;
    pc.start = pos;
    pc.len = end - pos;
    // A child already exhausted is addressed out of range, so it plans no read
    // and the enclosing pad_iter supplies its value instead.
    pc.ai = (ai < nA && a.chunk_start(ai) <= pos) ? ai : nA;
    pc.skipA = (pc.ai < nA) ? pos - a.chunk_start(pc.ai) : 0;
    pc.bi = (bi < nB && b.chunk_start(bi) <= pos) ? bi : nB;
    pc.skipB = (pc.bi < nB) ? pos - b.chunk_start(pc.bi) : 0;
    out.push_back(pc);
    pos = end;
  }
  return out;
}

// Element-wise pairing of two child nodes; element i = {A[i], B[i]}.  The
// shorter child is padded with its pad value up to len = max(lenA, lenB).
// Nesting (zip(zip(A,B), C)) and delayed operands (zip(map(A), scan(...))) work
// because plan/build simply recurse into the children.
//
// The two children need not agree on a chunk partition -- they disagree
// whenever one is ragged, or when they store different element widths (a
// uint32_t leaf holds twice as many elements per CHUNK_SIZE chunk as a
// uint64_t one).  When they disagree this node re-grids onto the union of
// their boundaries (`pieces`), addressing each child by that child's own chunk
// index plus an element offset.  A child chunk straddling two pieces is then
// referenced twice; the drivers dedup those references so it is still read
// once (see detail::plan_chunks).
//
// When both children already partition on the same closed-form grid -- every
// all-8-byte, dense chain -- `pieces` is null and this node behaves exactly as
// it did before re-gridding existed: no table is built (it would cost tens of
// MB on a multi-TB sequence) and no lookup enters the path.
template <class DA, class DB>
struct zip_node {
  using value_type =
      std::pair<typename DA::value_type, typename DB::value_type>;
  DA a;
  DB b;
  typename DA::value_type padA;
  typename DB::value_type padB;
  size_t lenA, lenB, len;
  size_t uepc;  // the shared grid divisor, used when pieces == nullptr
  std::shared_ptr<const std::vector<zip_piece>> pieces;

  size_t length() const { return len; }
  size_t num_chunks() const {
    return pieces ? pieces->size() : (len + uepc - 1) / uepc;
  }
  size_t chunk_start(size_t i) const {
    if (!pieces) return i * uepc;
    return i < pieces->size() ? (*pieces)[i].start : len;
  }
  bool uniform_grid(size_t* epc_out) const {
    *epc_out = uepc;
    return pieces == nullptr;
  }
  size_t chunk_len(size_t i) const {
    if (pieces) return i < pieces->size() ? (*pieces)[i].len : 0;
    const size_t base = i * uepc;
    return base >= len ? 0 : std::min(uepc, len - base);
  }

  template <class Planner>
  void plan(size_t i, Planner& p) const {
    if (i >= num_chunks()) return;  // padded/out-of-range: no reads
    if (pieces) {
      const zip_piece& pc = (*pieces)[i];
      a.plan(pc.ai, p);  // union of children's reads,
      b.plan(pc.bi, p);  // left-to-right (matches build)
      return;
    }
    a.plan(i, p);
    b.plan(i, p);
  }
  template <class Resolver>
  auto build(size_t i, size_t skip, Resolver& r) const {
    size_t ai = i, bi = i, sa = skip, sb = skip;
    const size_t cl = chunk_len(i);
    const size_t eb = chunk_start(i) + skip;
    const size_t n = cl - std::min(skip, cl);
    if (pieces && i < pieces->size()) {
      const zip_piece& pc = (*pieces)[i];
      ai = pc.ai;
      bi = pc.bi;
      sa = pc.skipA + skip;
      sb = pc.skipB + skip;
    }
    const size_t rA =
        eb >= lenA ? 0 : std::min(n, lenA - eb);  // A's real count
    const size_t rB =
        eb >= lenB ? 0 : std::min(n, lenB - eb);  // B's real count
    // Sequence the two child builds explicitly: both advance the resolver,
    // and C++ leaves function-argument evaluation order unspecified.
    auto ia = a.build(ai, sa, r);
    auto ib = b.build(bi, sb, r);
    return make_zip_iter(make_pad_iter(ia, rA, padA),
                         make_pad_iter(ib, rB, padB));
  }
};

// A delayed, non-writing filter: chunk i's survivors are computed by re-reading
// and re-running `pred` over whichever physical chunk(s) of `d` a predecessor
// search over a precomputed survivor-count prefix sum resolves to.  Never
// allocates a chunk_seq or writes to disk -- see lazy_filter (in the terminals
// section below) for how `offsets`/`total` are computed.
template <class D, class Pred>
struct filter_node {
  using value_type = typename D::value_type;
  D d;
  Pred pred;
  std::shared_ptr<std::vector<size_t>>
      offsets;   // size d.num_chunks()+1; exclusive prefix
  size_t total;  // sum of per-source-chunk survivor counts

  size_t length() const { return total; }
  size_t num_chunks() const { return grid_num_chunks(total); }
  size_t chunk_start(size_t i) const { return grid_chunk_start(i); }
  bool uniform_grid(size_t* epc_out) const {
    *epc_out = ELEMS_PER_CHUNK;
    return true;
  }
  size_t chunk_len(size_t i) const { return grid_chunk_len(total, i); }

  // Predecessor search: last physical (source) chunk index k with
  // offsets[k] <= g, i.e. the source chunk containing filtered index g.
  size_t locate(size_t g) const {
    auto it = std::upper_bound(offsets->begin(), offsets->end(), g);
    return (size_t)(it - offsets->begin()) - 1;
  }

  template <class Planner>
  void plan(size_t i, Planner& p) const {
    const size_t n = chunk_len(i);
    if (n == 0) return;
    const size_t g_lo = chunk_start(i);
    const size_t src_lo = locate(g_lo);
    const size_t src_hi = locate(g_lo + n - 1);
    for (size_t k = src_lo; k <= src_hi; k++) d.plan(k, p);
  }

  template <class Resolver>
  auto build(size_t i, size_t skip, Resolver& r) const {
    const size_t n = chunk_len(i);
    auto buf = std::make_shared<std::vector<value_type>>();
    if (n > 0) {
      buf->reserve(n);
      const size_t g_lo = chunk_start(i);
      const size_t src_lo = locate(g_lo);
      // Mirrors plan()'s [src_lo, locate(g_lo+n-1)] range exactly: since
      // `offsets` is the exact survivor-count prefix sum, this loop always
      // stops with k == locate(g_lo+n-1), never overshooting past it.
      for (size_t k = src_lo; buf->size() < n; k++) {
        auto it = d.build(k, 0, r);
        const size_t src_n = d.chunk_len(k);
        size_t skip = (k == src_lo) ? (g_lo - (*offsets)[k]) : 0;
        for (size_t j = 0; j < src_n && buf->size() < n; j++, ++it) {
          if (!pred(*it)) continue;
          if (skip > 0) {
            --skip;
            continue;
          }
          buf->push_back(*it);
        }
      }
    }
    // The whole chunk is materialized regardless (plan() registered its full
    // read set); `skip` is just the start offset into that buffer.
    return materialized_iter<value_type>{buf, skip};
  }
};

// Public total element count of any delayed sequence.
template <class D>
size_t size(const D& d) {
  return d.length();
}

// ── constructors / combinators (all lazy, no I/O) ────────────────────────────

// Wrap an on-SSD chunk_seq as a delayed sequence (identity transform).
template <class T = uint64_t>
auto delay(const chunk_seq& seq) {
  const size_t nc = seq.chunks.size();
  constexpr size_t epc = CHUNK_SIZE / sizeof(T);
  if (nc == 0) return leaf_source<T>{&seq, 0, 0, nullptr};

  // Fast path: dense-except-last, so the partition is the closed form and no
  // per-chunk table is needed.  Worth detecting -- a multi-TB sequence has
  // millions of chunks, and a table would cost tens of MB per leaf.
  bool dense = true;
  for (size_t i = 0; i + 1 < nc; i++)
    if (seq.chunks[i].used != epc * sizeof(T)) {
      dense = false;
      break;
    }
  if (dense)
    return leaf_source<T>{&seq,
                          (nc - 1) * epc + seq.chunks[nc - 1].used / sizeof(T),
                          nc, nullptr};

  // Ragged: take the partition from each chunk's own `used`.  This is what
  // makes flatten/reverse output usable as a delayed source, and it is also
  // what makes delay<R> correct over a sequence written on a *different*
  // element grid -- a narrowing ChunkMap (u64 -> u32) or a force() whose node
  // grid divisor is not sizeof(R) leaves every chunk uniformly partly full,
  // which the closed form would over-count.
  auto starts = std::make_shared<std::vector<size_t>>(nc + 1, 0);
  for (size_t i = 0; i < nc; i++)
    (*starts)[i + 1] = (*starts)[i] + seq.chunks[i].used / sizeof(T);
  const size_t len = (*starts)[nc];
  return leaf_source<T>{&seq, len, nc, std::move(starts)};
}

// A delayed [start_index, end_index) slice of an on-SSD chunk_seq, re-indexed
// to start at 0.  Unlike `delay`, which is an identity view (chunk i of the
// view IS chunk i of the source), this re-windows the source, so consuming it
// (e.g. via `reduce`, `force`, or ExternalPrimitives' delayed-source
// `materialize`) never writes the slice back to disk the way
// sequential_cut_no_compression does.
template <class T = uint64_t>
auto cut(const chunk_seq& seq, size_t start_index, size_t end_index) {
  CHECK(start_index <= end_index)
      << "cut: start_index " << start_index << " > end_index " << end_index;
  const size_t nc = seq.chunks.size();
  // Physical chunks (besides the last) hold CHUNK_SIZE/sizeof(T) elements of
  // T, not the global (uint64_t-sized) ELEMS_PER_CHUNK -- see cut_source::epc.
  const size_t total = nc == 0 ? 0
                               : (nc - 1) * (CHUNK_SIZE / sizeof(T)) +
                                     seq.chunks[nc - 1].used / sizeof(T);
  CHECK(end_index <= total)
      << "cut: end_index " << end_index << " exceeds source length " << total;
  return cut_source<T>{&seq, start_index, end_index - start_index};
}

// A delayed sequence whose element i is f(i), with no source files.
template <class F>
auto tabulate(size_t n, F f) {
  return leaf_index<F>{n, f};
}

// Lazily map g over every element (no temp buffer, no I/O).
template <class D, class G>
auto map(D d, G g) {
  return map_node<D, G>{d, g};
}

// Shared by every zip overload: take the closed-form fast path when both
// children already partition on one grid, else build the re-gridding table.
template <class DA, class DB>
auto make_zip_node(DA a, DB b, typename DA::value_type padA,
                   typename DB::value_type padB) {
  const size_t lenA = a.length(), lenB = b.length();
  const size_t len = std::max(lenA, lenB);
  size_t ea = ELEMS_PER_CHUNK, eb = ELEMS_PER_CHUNK;
  const bool ua = a.uniform_grid(&ea);
  const bool ub = b.uniform_grid(&eb);
  if (ua && ub && ea == eb)
    return zip_node<DA, DB>{a, b, padA, padB, lenA, lenB, len, ea, nullptr};
  return zip_node<DA, DB>{a,
                          b,
                          padA,
                          padB,
                          lenA,
                          lenB,
                          len,
                          ea,
                          std::make_shared<const std::vector<zip_piece>>(
                              build_zip_pieces(a, b, len))};
}

// Strict zip: element i = {a[i], b[i]}.  Both operands must have equal length.
template <class DA, class DB>
auto zip(DA a, DB b) {
  const size_t lenA = a.length(), lenB = b.length();
  CHECK(lenA == lenB) << "zip: length mismatch " << lenA << " vs " << lenB
                      << " (use zip(a, b, pad) to pad the shorter side)";
  return make_zip_node(a, b, typename DA::value_type{},
                       typename DB::value_type{});
}

// Padded zip: if operands differ in length the shorter is padded with `pad` up
// to max(lenA, lenB).  A single pad value requires a shared element type.
template <class DA, class DB, class Pad>
auto zip(DA a, DB b, Pad pad) {
  using VA = typename DA::value_type;
  using VB = typename DB::value_type;
  static_assert(std::is_same_v<VA, VB>,
                "zip(a, b, pad): a single pad value requires both operands to "
                "share a value_type");
  return make_zip_node(a, b, (VA)pad, (VB)pad);
}

// Padded zip with a per-operand pad value, for operands whose element types
// differ -- e.g. zipping a uint32_t-limb sequence against a uint64_t-limb one,
// where no single pad value has a type common to both.
template <class DA, class DB, class PadA, class PadB>
auto zip(DA a, DB b, PadA padA, PadB padB) {
  return make_zip_node(a, b, (typename DA::value_type)padA,
                       (typename DB::value_type)padB);
}

namespace detail {

// Presents chunks [base, base+w) of `d` as a 0..w-1 sequence, so
// for_each_window can plan a window through the same plan_chunks that
// for_each_chunk uses (plan_chunks only ever calls plan()).
template <class D>
struct WindowView {
  const D* d;
  size_t base, w;
  template <class Planner>
  void plan(size_t i, Planner& p) const {
    d->plan(base + i, p);
  }
};

// Shared by for_each_chunk's overloads and PersistentReadContext's
// constructor: plan every chunk of `d` up front (metadata only, no I/O) into
// the deduped-reads + per-chunk buffer/slot bookkeeping for_each_chunk needs.
// A (chunk, slot) pair that consumes one physical read.
using ReadConsumer = std::pair<uint32_t, uint32_t>;

template <class D>
struct PlannedChunks {
  std::vector<chunk> refs;  // deduped reads; .index = global read-id
  // read-id -> every (chunk, slot) that consumes it.  Usually one entry; more
  // when several logical chunks live inside one physical chunk, which is what
  // a re-gridding zip produces.  Recording them all is what turns "referenced
  // k times" into "read once, held until the k-th consumer is done".
  std::vector<std::vector<ReadConsumer>> consumers;
  std::vector<size_t> refcnt;                 // read-id -> #consuming chunks
  std::vector<size_t> remaining;              // chunk -> reads not yet landed
  std::vector<std::vector<char*>> cbufs;      // per-chunk buffers (slot order)
  std::vector<std::vector<uint32_t>> crids;   // per-chunk slot -> read-id
  std::vector<std::vector<uint32_t>> cslots;  // per-chunk leaf_slots
};

// Plan every chunk up front (metadata only, no I/O), deduping identical
// physical reads *across* chunks -- not just within one, as Planner does.
//
// Dedup is capped at a `lookback` window of logical chunks: a read is only
// merged into an existing one when the chunk that last referenced it is at
// most `lookback` chunks back.  Sharing beyond that would keep a buffer alive
// arbitrarily long, so the cap is what bounds live buffers.  It never binds in
// practice: the partitions that share a physical chunk (a re-gridding zip, a
// cut_source seam, filter_node's predecessor run) only ever share it between
// *consecutive* logical chunks.
// Dedup key: the physical chunk, identified as (source, chunk index) -- the
// same key Planner uses within a chunk.
using ReadKey = std::pair<const void*, size_t>;
struct ReadKeyHash {
  size_t operator()(const ReadKey& k) const {
    const size_t h = std::hash<const void*>{}(k.first);
    return h ^ (k.second * 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
  }
};
// value: {read-id, logical chunk that last referenced it}
using ReadSeen =
    std::unordered_map<ReadKey, std::pair<uint32_t, size_t>, ReadKeyHash>;

template <class D>
PlannedChunks<D> plan_chunks(const D& d, size_t nc,
                             size_t lookback = FILTER_BATCH_SIZE) {
  PlannedChunks<D> pc;
  pc.remaining.resize(nc);
  pc.cbufs.resize(nc);
  pc.crids.resize(nc);
  pc.cslots.resize(nc);

  // Two generations of the dedup map, rotated every `lookback` chunks and the
  // stale one dropped.  Without this the map would grow to one entry per
  // physical chunk in the whole sequence (millions on a multi-TB run); with
  // it, it holds only what the live window can still match against.
  ReadSeen cur, prev;
  size_t gen_start = 0;

  for (size_t ci = 0; ci < nc; ci++) {
    if (ci - gen_start >= lookback) {
      prev.swap(cur);
      cur.clear();
      gen_start = ci;
    }
    Planner pl;
    d.plan(ci, pl);
    const size_t k = pl.unique_reads.size();
    pc.remaining[ci] = k;
    pc.cbufs[ci].assign(k, nullptr);
    pc.crids[ci].resize(k);
    pc.cslots[ci] = std::move(pl.leaf_slots);
    for (size_t sl = 0; sl < k; sl++) {
      chunk c = pl.unique_reads[sl];
      const ReadKey key{(const void*)pl.src_of[sl], c.index};

      uint32_t rid = (uint32_t)-1;
      auto hit = cur.find(key);
      if (hit == cur.end()) {
        auto ph = prev.find(key);
        // Promote a live hit out of the older generation so it stays matchable.
        if (ph != prev.end() && ci - ph->second.second <= lookback)
          hit = cur.emplace(key, ph->second).first;
      }
      if (hit != cur.end() && ci - hit->second.second <= lookback) {
        rid = hit->second.first;
        hit->second.second = ci;
        pc.refcnt[rid]++;
      }

      if (rid == (uint32_t)-1) {
        rid = (uint32_t)pc.refs.size();
        cur[key] = {rid, ci};
        c.index = rid;  // the reader keys completions by this
        pc.refs.push_back(std::move(c));
        pc.refcnt.push_back(1);
        pc.consumers.emplace_back();
      }
      pc.crids[ci][sl] = rid;
      pc.consumers[rid].push_back({(uint32_t)ci, (uint32_t)sl});
    }
  }
  return pc;
}
}  // namespace detail

// ── drivers ──────────────────────────────────────────────────────────────────
//
// for_each_window and for_each_chunk both plan each logical chunk with a
// Planner (deduping shared reads), issue the reads through the async
// ChunkSequenceReader, and — once a chunk's buffers are resident — build its
// fused iterator with a Resolver.  They differ only in *scheduling*:
// for_each_window collects a whole window before computing (needed by
// filter's sequential carry); for_each_chunk streams — it releases a chunk to
// a worker the instant that chunk's own reads land, so reads and compute
// overlap continuously with no window barrier.  sequential_for_each_chunk
// (below, after for_each_chunk) plans and builds each chunk the same way but
// reads via blocking pread on the calling thread instead of a
// ChunkSequenceReader — no reader/thread/io_uring setup at all — for small
// ranges consumed from inside an already-parallel outer loop.

// Windowed: collect a FILTER_BATCH_SIZE window, then hand it to `wbody`, which
// may build any chunk via build_chunk(local_b).  Buffers are freed after wbody
// returns, so a consumer must copy anything it keeps.  Used by filter (whose
// dense-packing carry threads sequentially across chunks in index order).
template <class D, class WindowBody>
void for_each_window(const D& d, WindowBody&& wbody,
                     size_t reader_threads = 8) {
  const size_t nc = d.num_chunks();
  for (size_t base = 0; base < nc; base += FILTER_BATCH_SIZE) {
    const size_t w = std::min(FILTER_BATCH_SIZE, nc - base);

    // Plan each chunk of the window into one flat read list, deduped both
    // within a chunk (by Planner) and across the window's chunks (by
    // plan_chunks), so a physical chunk several logical chunks share is read
    // once and its buffer held for all of them.
    auto wp = detail::plan_chunks(detail::WindowView<D>{&d, base, w}, w);
    std::vector<chunk>& refs = wp.refs;
    std::vector<std::vector<char*>>& cbufs = wp.cbufs;
    std::vector<std::vector<uint32_t>>& cslots = wp.cslots;
    const size_t total = refs.size();

    auto build_chunk = [&](size_t b) {
      Resolver r{&cbufs[b], &cslots[b], 0};
      return d.build(base + b, 0, r);
    };

    if (total == 0) {
      wbody(base, w, build_chunk);
      continue;
    }  // pure index: no I/O

    chunk_seq rs;
    rs.chunks = std::move(refs);
    ChunkSequenceReader<char> reader;
    reader.PrepChunks(rs);
    reader.Start(reader_threads, 32, 16);
    std::vector<char*> bufs(total, nullptr);
    for (size_t k = 0; k < total; k++) {  // completions arrive out of order
      auto [buf, n, rid] = reader.Poll();
      (void)n;
      CHECK(buf != nullptr) << "delayed: short read";
      bufs[rid] = buf;
      for (const detail::ReadConsumer& u : wp.consumers[rid])
        cbufs[u.first][u.second] = buf;
    }

    wbody(base, w, build_chunk);

    // Free once per read, not once per reference -- a shared buffer appears in
    // several chunks' slot lists.
    for (char* p : bufs)
      if (p) reader.allocator.Free(p);
  }
}

// Streaming: one read pass over the whole sequence with per-chunk async
// release. A dispatcher thread assembles chunks from the reader's out-of-order
// completions and hands each finished chunk to a parlay worker, so body runs
// while later chunks are still being read (no window barrier).  body must be
// chunk-disjoint and order-independent — true for reduce (writes sums[ci]) and
// force (writes chunk ci's precomputed drive/slot).  reader_threads defaults to
// 10 to match the eager ChunkReduce reader (the config that reaches device-read
// speed); one reader serves the whole pass, so there is no per-window setup
// cost.
template <class D, class Body>
void for_each_chunk(const D& d, Body&& body, size_t reader_threads = 10,
                    size_t compute_workers = 0) {
  const size_t nc = d.num_chunks();
  if (nc == 0) return;
  // 0 = "use the whole pool"; callers that share the pool with other tasks
  // (e.g. count_sort's writer I/O threads) pass P - kWriterIoThreads so the
  // scatter fork-join matches the available workers instead of oversubscribing.
  if (compute_workers == 0) compute_workers = parlay::num_workers();

  // Plan every chunk up front (metadata only): deduped reads + per-chunk state.
  auto pc = detail::plan_chunks(d, nc);
  std::vector<chunk>& refs = pc.refs;
  std::vector<size_t>& remaining = pc.remaining;
  std::vector<std::vector<char*>>& cbufs = pc.cbufs;
  std::vector<std::vector<uint32_t>>& cslots = pc.cslots;
  const size_t total = refs.size();

  auto run_chunk = [&](size_t ci) {
    Resolver r{&cbufs[ci], &cslots[ci], 0};
    auto it = d.build(ci, 0, r);
    body(ci, d.chunk_len(ci), it);
  };

  if (total == 0) {  // pure index: no I/O, no reader
    parlay::parallel_for(0, nc, [&](size_t ci) { run_chunk(ci); }, 1);
    return;
  }

  chunk_seq rs;
  rs.chunks = std::move(refs);
  ChunkSequenceReader<char> reader;
  reader.PrepChunks(rs);
  reader.Start(reader_threads, 32, 16, /*buf_queue_sz=*/128);

  SimpleQueue<size_t> ready;  // ready chunk ids (bounded backlog)
  ready.SetSizeLimit(FILTER_BATCH_SIZE);

  // A read shared by several logical chunks is issued once; the buffer lives
  // until the last of them is done with it, so workers release by refcount
  // rather than unconditionally.
  std::unique_ptr<std::atomic<size_t>[]> refcnt(new std::atomic<size_t>[total]);
  for (size_t k = 0; k < total; k++)
    refcnt[k].store(pc.refcnt[k], std::memory_order_relaxed);
  auto release = [&](size_t ci) {
    for (size_t sl = 0; sl < pc.crids[ci].size(); sl++) {
      char* p = cbufs[ci][sl];
      if (p == nullptr) continue;
      if (refcnt[pc.crids[ci][sl]].fetch_sub(1, std::memory_order_acq_rel) == 1)
        reader.allocator.Free(p);
    }
  };

  // Dispatcher: assemble chunks from out-of-order completions; release each the
  // moment its last read lands.  Single-threaded assembly ⇒ no atomics on
  // `remaining`; the ready queue's push/poll gives workers the happens-before
  // on cbufs[ci].  When `ready` is full it blocks here, which back-pressures
  // the reader, so live buffers stay bounded (no window, but a budget).
  std::thread dispatcher([&] {
    for (size_t ci = 0; ci < nc; ci++)  // chunks needing no reads (e.g. a
      if (remaining[ci] == 0)
        ready.Push(ci);  // padded tail) are ready immediately
    for (size_t done = 0; done < total; done++) {
      auto [buf, n, rid] = reader.Poll();
      (void)n;
      CHECK(buf != nullptr) << "delayed: short read";
      for (const detail::ReadConsumer& u : pc.consumers[rid]) {
        cbufs[u.first][u.second] = buf;
        if (--remaining[u.first] == 0) ready.Push(u.first);
      }
    }
    ready.Close();
  });

  // Workers: build + compute each ready chunk, then release its buffers.
  parlay::parallel_for(
      0, compute_workers,
      [&](size_t) {
        while (true) {
          auto [ci, code] = ready.Poll((size_t)0);
          if (code == QueueCode::FINISH) break;
          run_chunk(ci);
          release(ci);
        }
      },
      1);

  dispatcher.join();
}

// Reusable read state for sequential_for_each_chunk / sequential_materialize: a
// caller-owned fd cache + a pool of persistent CHUNK_SIZE O_DIRECT buffers, so
// opens and CHUNK_SIZE allocations happen once across MANY calls instead of
// once per call.  Meant to be constructed once (e.g. one per
// parlay::worker_id() slot, held for an algorithm's whole lifetime -- see
// chunk_partition.h's / count_sort.h's per-worker-slot idiom, valid because a
// parlay task runs uninterrupted on one worker) and threaded through every
// sequential_for_each_chunk / sequential_materialize call that would otherwise
// pay open()+aligned_alloc() per call (e.g. Bellman-Ford's per-vertex
// delayed::cut, called O(rounds*n) times).
//
// The buffer pool grows on demand -- slot s covers "unique read slot s" across
// calls -- and never shrinks; a cut_source needs at most 2 slots (one physical
// chunk, or a chunk-boundary-straddling two), but nothing here hardcodes that,
// so a wider-fanout node just grows the pool further.  Non-copyable (owns fds +
// raw buffers).
//
// fd_cache is bounded (LRU, MAX_CACHED_FDS entries) rather than growing
// forever: an algorithm like Bellman-Ford holds one context per parlay worker
// for its whole run and touches one fd per unique *physical chunk file* across
// O(rounds*n) calls, so an unbounded per-worker cache can accumulate enough
// open fds (across all workers) to blow past the process's RLIMIT_NOFILE --
// observed as open() failing with EMFILE. Rounds revisit the same vertices, so
// LRU still captures most of the reuse an unbounded cache would.
struct SequentialReadContext {
  static constexpr size_t MAX_CACHED_FDS = 256;

  std::list<std::string> lru;  // front = most recently used
  std::unordered_map<std::string,
                     std::pair<int, std::list<std::string>::iterator>>
      fd_cache;
  std::vector<char*>
      buf_pool;  // persistent CHUNK_SIZE aligned buffers, grow-on-demand

  SequentialReadContext() = default;
  SequentialReadContext(const SequentialReadContext&) = delete;
  SequentialReadContext& operator=(const SequentialReadContext&) = delete;

  // Returns an open fd for `filename`, opening (and evicting the LRU entry if
  // the cache is full) on a miss. Unlike the old direct fd_cache.emplace, a
  // failed open() is fatal rather than being silently cached as -1 and fed to
  // a later pread -- that used to fail with EBADF on every subsequent access
  // to the same filename, silently leaving the caller's buffer untouched.
  int get_fd(const std::string& filename) {
    auto it = fd_cache.find(filename);
    if (it != fd_cache.end()) {
      lru.splice(lru.begin(), lru, it->second.second);
      return it->second.first;
    }
    if (fd_cache.size() >= MAX_CACHED_FDS) {
      const std::string& victim = lru.back();
      close(fd_cache.at(victim).first);
      fd_cache.erase(victim);
      lru.pop_back();
    }
    int fd = open(filename.c_str(), O_RDONLY | O_DIRECT);
    CHECK(fd >= 0) << "SequentialReadContext: open failed for " << filename
                   << ": " << std::strerror(errno);
    lru.push_front(filename);
    fd_cache.emplace(filename, std::make_pair(fd, lru.begin()));
    return fd;
  }

  ~SequentialReadContext() {
    for (char* p : buf_pool) free(p);
    for (auto& [name, entry] : fd_cache) close(entry.first);
  }
};

// Sequential streaming: one blocking, O_DIRECT pread per physical chunk, on
// the calling thread -- no ChunkSequenceReader, no io_uring rings, no
// dispatcher thread.  For use when D covers only a handful of logical chunks
// (e.g. one vertex's adjacency slice via delayed::cut) from *inside* an
// already-parallel outer loop: the async drivers above pay reader_threads
// io_uring rings + a dispatcher thread per call, a cost a few-hundred-byte
// payload never amortizes when paid once per outer-loop iteration.  Modeled
// on ExternalPrimitives/materialize.h's sequential_materialize(chunk_seq), but
// generalized to any delayed node D via the existing generic Planner/Resolver
// interface -- not specific to cut_source.  Unlike for_each_chunk there is no
// upfront across-chunk planning; each logical chunk is planned, read, and
// consumed one at a time, which is fine for the small-range case this is for.
//
// The SequentialReadContext overload reuses the caller-supplied fd cache and
// buffer pool across calls instead of opening/allocating fresh state every
// time -- fd/buffer lifetime is then caller-controlled (freed when ctx is
// destroyed, not at the end of this call).  The no-context overload keeps the
// original one-shot cost profile (its own private context, torn down here).
template <class D, class Body>
void sequential_for_each_chunk(const D& d, SequentialReadContext& ctx,
                               Body&& body) {
  const size_t nc = d.num_chunks();
  if (nc == 0) return;

  for (size_t ci = 0; ci < nc; ci++) {
    Planner pl;
    d.plan(ci, pl);
    const size_t nr = pl.unique_reads.size();

    if (ctx.buf_pool.size() < nr) {  // grow-on-demand, never shrink
      const size_t old = ctx.buf_pool.size();
      ctx.buf_pool.resize(nr, nullptr);
      for (size_t s = old; s < nr; s++) {
        ctx.buf_pool[s] =
            (char*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
        CHECK(ctx.buf_pool[s] != nullptr)
            << "sequential_for_each_chunk: allocation failed";
      }
    }

    for (size_t s = 0; s < nr; s++) {
      const chunk& c = pl.unique_reads[s];
      char* buf = ctx.buf_pool[s];
      if (c.used == 0) continue;
      int fd = ctx.get_fd(c.filename);
      SYSCALL(pread(fd, buf, AlignUp(c.used), (off_t)c.begin_addr));
    }

    Resolver r{&ctx.buf_pool, &pl.leaf_slots, 0};
    auto it = d.build(ci, 0, r);
    body(ci, d.chunk_len(ci), it);  // buffers are ctx-owned: reused by the next
  }  // chunk/call, not freed here.
}

template <class D, class Body>
void sequential_for_each_chunk(const D& d, Body&& body) {
  SequentialReadContext ctx;
  sequential_for_each_chunk(d, ctx, std::forward<Body>(body));
}

// Reusable execution context for repeatedly running for_each_chunk /
// segmented_reduce over the SAME physical read plan across many calls -- the
// async-driver analogue of SequentialReadContext above, but for
// for_each_chunk's ChunkSequenceReader/dispatcher-thread path instead of
// sequential_for_each_chunk's blocking-pread path.
//
// Meant to be constructed ONCE outside a fixed-point iteration's round loop
// (e.g. external_bellman_ford_fast, ChunkSequence/examples/
// chunk_bellman_ford.h) and passed into every round's
// for_each_chunk/segmented_reduce call, so the expensive part -- io_uring
// rings, reader OS threads, and fd opens (see PersistentChunkSequenceReader,
// chunk_seq_reader.h) -- is paid ONCE for the whole run instead of once per
// round.
//
// CONTRACT (same-shape reads): every call sharing one PersistentReadContext<D>
// must plan to the exact same physical reads, in the same order, as the `d`
// passed to the constructor -- same source chunk_seq(s), same chunk
// count/order.  The context plans ONCE (at construction) and its
// PersistentChunkSequenceReader replays that fixed plan every round; only
// `d`'s runtime VALUES (e.g. a captured distance array) may differ between
// calls, not its STRUCTURE.  This holds for Bellman-Ford: `graph.edges` /
// `graph.degree_scan` never change across rounds, only the map lambda's
// captured `d` does, so per_edge's plan (chunks read from graph.edges) is
// identical every round.  A caller whose read plan genuinely differs between
// calls must NOT share a context -- use the plain (non-context) overloads.
// The context-aware for_each_chunk overload below CHECKs this contract on
// every call (fail loud rather than silently misroute completions).
//
// CONCURRENCY: at most one call into a shared context may be in flight at a
// time (mirrors PersistentChunkSequenceReader's single-round-in-flight
// requirement).  A fixed-point round loop is inherently sequential -- each
// round's call fully returns before the next round starts -- so this holds
// trivially and needs no extra synchronization on the caller side.
//
// LIFETIME: must outlive every for_each_chunk/segmented_reduce call that uses
// it.  Non-copyable (owns threads/fds via its reader).
template <class D>
class PersistentReadContext {
 public:
  explicit PersistentReadContext(const D& d, size_t reader_threads = 10,
                                 size_t queue_depth = 32,
                                 size_t max_requests = 16,
                                 size_t buf_queue_sz = 128) {
    const size_t nc = d.num_chunks();
    auto pc = detail::plan_chunks(d, nc);
    expected_reads = std::move(pc.refs);
    chunk_seq rs;
    rs.chunks = expected_reads;
    if (!expected_reads.empty())
      reader.Start(rs, reader_threads, queue_depth, max_requests, buf_queue_sz);
  }

  PersistentReadContext(const PersistentReadContext&) = delete;
  PersistentReadContext& operator=(const PersistentReadContext&) = delete;
  ~PersistentReadContext() =
      default;  // reader's dtor joins threads, closes fds

  PersistentChunkSequenceReader<char> reader;
  std::vector<chunk>
      expected_reads;  // captured at construction; validated every call
};

// Persistent-context overload: reuses ctx's already-running io_uring rings +
// reader worker threads (see PersistentReadContext /
// PersistentChunkSequenceReader) instead of building a fresh
// ChunkSequenceReader/reader-thread set for this call.  `d` must plan to the
// SAME physical reads every call sharing `ctx` (see PersistentReadContext's
// class doc above) -- validated via CHECK below. The dispatcher thread and
// compute-worker pool are still spawned fresh each call (cheap: no io_uring/fd
// setup); only ctx's reader persists across calls. `compute_workers` has the
// same meaning as the plain overload.
template <class D, class Body>
void for_each_chunk(const D& d, Body&& body, PersistentReadContext<D>& ctx,
                    size_t compute_workers = 0) {
  const size_t nc = d.num_chunks();
  if (nc == 0) return;
  if (compute_workers == 0) compute_workers = parlay::num_workers();

  auto pc = detail::plan_chunks(d, nc);
  std::vector<chunk>& refs = pc.refs;
  std::vector<size_t>& remaining = pc.remaining;
  std::vector<std::vector<char*>>& cbufs = pc.cbufs;
  std::vector<std::vector<uint32_t>>& cslots = pc.cslots;
  const size_t total = refs.size();

  CHECK(total == ctx.expected_reads.size())
      << "PersistentReadContext: read plan changed between calls (expected "
      << ctx.expected_reads.size() << " reads, got " << total << ")";
  for (size_t k = 0; k < total; k++)
    CHECK(refs[k].filename == ctx.expected_reads[k].filename &&
          refs[k].begin_addr == ctx.expected_reads[k].begin_addr &&
          refs[k].used == ctx.expected_reads[k].used)
        << "PersistentReadContext: read plan diverged at read " << k
        << " -- shared contexts require an identical physical read plan "
           "on every call (see PersistentReadContext's class doc)";

  auto run_chunk = [&](size_t ci) {
    Resolver r{&cbufs[ci], &cslots[ci], 0};
    auto it = d.build(ci, 0, r);
    body(ci, d.chunk_len(ci), it);
  };

  if (total == 0) {  // pure index: no I/O
    parlay::parallel_for(0, nc, [&](size_t ci) { run_chunk(ci); }, 1);
    return;
  }

  ctx.reader.StartRound();  // cheap: no ring/thread/fd churn

  SimpleQueue<size_t> ready;
  ready.SetSizeLimit(FILTER_BATCH_SIZE);

  // Same shared-read refcounting as the plain overload.
  std::unique_ptr<std::atomic<size_t>[]> refcnt(new std::atomic<size_t>[total]);
  for (size_t k = 0; k < total; k++)
    refcnt[k].store(pc.refcnt[k], std::memory_order_relaxed);
  auto release = [&](size_t ci) {
    for (size_t sl = 0; sl < pc.crids[ci].size(); sl++) {
      char* p = cbufs[ci][sl];
      if (p == nullptr) continue;
      if (refcnt[pc.crids[ci][sl]].fetch_sub(1, std::memory_order_acq_rel) == 1)
        ctx.reader.allocator.Free(p);
    }
  };

  std::thread dispatcher([&] {
    for (size_t ci = 0; ci < nc; ci++)
      if (remaining[ci] == 0) ready.Push(ci);
    for (size_t done = 0; done < total; done++) {
      auto [buf, n, rid] = ctx.reader.Poll();
      (void)n;
      CHECK(buf != nullptr) << "delayed: short read";
      for (const detail::ReadConsumer& u : pc.consumers[rid]) {
        cbufs[u.first][u.second] = buf;
        if (--remaining[u.first] == 0) ready.Push(u.first);
      }
    }
    ready.Close();
  });

  parlay::parallel_for(
      0, compute_workers,
      [&](size_t) {
        while (true) {
          auto [ci, code] = ready.Poll((size_t)0);
          if (code == QueueCode::FINISH) break;
          run_chunk(ci);
          release(ci);
        }
      },
      1);

  dispatcher.join();
}

// ── terminals ────────────────────────────────────────────────────────────────

// Per-chunk monoid reduction: sums[i] = reduction of chunk i.  Shared by reduce
// and scan's first pass (c == num_chunks accumulators fit in RAM).
template <class D, class Monoid>
std::vector<typename D::value_type> per_chunk_reduce(const D& d, Monoid m) {
  using R = typename D::value_type;
  std::vector<R> sums(d.num_chunks());
  for_each_chunk(d, [&](size_t ci, size_t n, auto it) {
    R s = m.identity;
    for (size_t i = 0; i < n; i++) {
      s = m(s, *it);
      ++it;
    }
    sums[ci] = s;
  });
  return sums;
}

// reduce: fold the whole sequence under the monoid (one read pass).
template <class D, class Monoid>
typename D::value_type reduce(const D& d, Monoid m) {
  using R = typename D::value_type;
  std::vector<R> sums = per_chunk_reduce(d, m);
  R acc = m.identity;  // c is small: sequential combine
  for (const R& s : sums) acc = m(acc, s);
  return acc;
}

// scan: exclusive prefix scan (parlay convention), partially delayed.
//   Pass 1 (one read pass): per-chunk reductions -> block offsets + total.
//   Pass 2 (lazy): a scan_node that, when consumed, re-reads the source and
//   runs the seeded within-chunk scan.  Returns {scan_node, total}.
template <class D, class Monoid>
auto scan(const D& d, Monoid m) {
  using R = typename D::value_type;
  std::vector<R> sums = per_chunk_reduce(d, m);
  const size_t nc = sums.size();

  auto offsets = std::make_shared<std::vector<R>>(nc);
  R run = m.identity;
  for (size_t i = 0; i < nc; i++) {
    (*offsets)[i] = run;
    run = m(run, sums[i]);
  }
  const R total = run;

  return std::pair{scan_node<D, Monoid>{d, m, offsets}, total};
}

// lazy_filter: a delayed filter that never writes to disk and never creates a
// chunk_seq (contrast with the eager, disk-writing `filter` above, which
// densely repacks survivors into a fresh chunk_seq via for_each_window).
//   Pass 1 (one streaming read pass, no allocation beyond the counts):
//   per-chunk survivor counts under `pred`.  Then a sequential prefix sum over
//   those counts.  Consuming the returned node (via for_each_chunk, reduce,
//   scan, map, force, sequential_for_each_chunk, another lazy_filter, ...)
//   re-reads only the physical chunks a given logical output chunk's
//   predecessor search resolves to, and re-applies `pred` locally -- see
//   filter_node::plan/build above.
//
// Trade-off: a low-selectivity `pred` makes one logical output chunk's source
// span many physical chunks (bounded only by how sparse `pred` is), so a
// single chunk_len(i)-worth of survivors can cost reading most of `d`.  This is
// inherent to not maintaining a separate index of match positions, and is the
// same shape of trade-off cut_source already accepts for boundary-straddling
// slices.

// I had claude write this up, but let's try to actually understand what's going
// on here to make sure it implemented the logic I designed
template <class D, class Pred>
auto lazy_filter(const D& d, Pred pred) {
  const size_t nc = d.num_chunks();
  std::vector<size_t> counts(
      nc);  // this is the vector that will store the scans for later access
  for_each_chunk(d, [&](size_t ci, size_t n, auto it) {
    size_t c = 0;
    for (size_t j = 0; j < n; j++, ++it)
      if (pred(*it)) c++;
    counts[ci] = c;
  });  // for each chunk, get a counter that represents the number of surviving
       // elements for that chunk

  auto offsets = std::make_shared<std::vector<size_t>>(nc + 1);
  size_t run = 0;
  for (size_t i = 0; i < nc; i++) {
    (*offsets)[i] = run;
    run += counts[i];
  }
  (*offsets)[nc] = run;
  // this is the scan array

  // and we just return that. Huh.
  return filter_node<D, Pred>{d, pred, offsets, run};
}

namespace detail {
// segmented_reduce's shared body: `bounds` (size num_segments+1, exclusive
// prefix over D's own element indices, bounds[0]==0, bounds.back()==d.length())
// partitions D into contiguous segments; returns one R per segment,
// monoid-reduced over every element in that segment.  One streaming pass
// regardless of how many segments there are or how many chunks a segment
// spans: each chunk classifies every segment it touches as fully owned (no
// other chunk can touch it -> written directly) or boundary (touches the
// chunk's first or last element -> stashed per chunk index for an
// O(n_chunks) sequential merge afterward, chaining through segments spanning
// many consecutive chunks).  Same mechanism as ChunkSegmentedReduce,
// generalized from a raw chunk_seq<T> to any composed delayed node (so a
// preceding map/zip/etc. fuses into this one pass instead of paying a
// separate read).  `run_pass(body)` is either the plain-reader or the
// persistent-context for_each_chunk call -- factored out so both
// segmented_reduce overloads share this boundary-merge logic verbatim.
template <class D, class Monoid, class RunPass>
parlay::sequence<typename D::value_type> segmented_reduce_generic(
    const D& d, const parlay::sequence<size_t>& bounds, Monoid m,
    RunPass&& run_pass) {
  using R = typename D::value_type;
  const size_t nc = d.num_chunks();
  const size_t num_segments = bounds.size() - 1;

  std::vector<size_t> chunk_start(nc + 1,
                                  0);  // global element offset of chunk i
  for (size_t i = 0; i < nc; i++)
    chunk_start[i + 1] = chunk_start[i] + d.chunk_len(i);

  parlay::sequence<R> out(num_segments, m.identity);
  std::vector<std::vector<std::pair<size_t, R>>> boundary(nc);

  run_pass([&](size_t ci, size_t n, auto it) {
    if (n == 0) return;
    const size_t global_start = chunk_start[ci];
    const size_t global_end = global_start + n;

    const size_t v_lo =
        (size_t)(std::upper_bound(bounds.begin(), bounds.end(), global_start) -
                 bounds.begin()) -
        1;
    const size_t v_hi = (size_t)(std::upper_bound(bounds.begin(), bounds.end(),
                                                  global_end - 1) -
                                 bounds.begin()) -
                        1;

    auto finalize = [&](size_t v, R val) {
      const bool is_boundary = (v == v_lo && bounds[v_lo] < global_start) ||
                               (v == v_hi && bounds[v_hi + 1] > global_end);
      if (is_boundary)
        boundary[ci].push_back({v, val});
      else
        out[v] = val;
    };

    size_t cur_v = v_lo;
    R cur_val = m.identity;
    for (size_t i = 0; i < n; i++) {
      const size_t g = global_start + i;
      while (g >= bounds[cur_v + 1]) {
        finalize(cur_v, cur_val);
        cur_v++;
        cur_val = m.identity;
      }
      cur_val = m(cur_val, *it);
      ++it;
    }
    finalize(cur_v, cur_val);
  });

  bool have_open = false;
  size_t open_v = 0;
  R open_val = m.identity;
  for (size_t c = 0; c < nc; c++) {
    for (auto& [v, val] : boundary[c]) {
      if (have_open && v == open_v) {
        open_val = m(open_val, val);
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
}  // namespace detail

template <class D, class Monoid>
parlay::sequence<typename D::value_type> segmented_reduce(
    const D& d, const parlay::sequence<size_t>& bounds, Monoid m,
    size_t reader_threads = 10) {
  return detail::segmented_reduce_generic(d, bounds, m, [&](auto&& body) {
    for_each_chunk(d, std::forward<decltype(body)>(body), reader_threads);
  });
}

// Persistent-context overload: see PersistentReadContext's class doc and the
// context-aware for_each_chunk overload above for the reuse mechanism and its
// same-read-plan contract.
template <class D, class Monoid>
parlay::sequence<typename D::value_type> segmented_reduce(
    const D& d, const parlay::sequence<size_t>& bounds, Monoid m,
    PersistentReadContext<D>& ctx) {
  return detail::segmented_reduce_generic(d, bounds, m, [&](auto&& body) {
    for_each_chunk(d, std::forward<decltype(body)>(body), ctx);
  });
}

// force: materialize a delayed sequence to a real chunk_seq on SSD (one file
// per drive, balls-in-bins).  Returns an index-ordered chunk_seq.  Allocates a
// fresh output buffer per chunk (a scan chain reads the source element on ++
// *after* the accumulator is emitted, so in-place reuse would corrupt it).
template <class D>
chunk_seq force(const D& d, const std::string& result_prefix) {
  using R = typename D::value_type;
  static_assert(CHUNK_SIZE % sizeof(R) == 0,
                "sizeof(R) must divide CHUNK_SIZE for O_DIRECT alignment");

  const size_t nc = d.num_chunks();
  if (nc == 0) return {};
  const size_t num_drives = GetSSDList().size();

  // Randomly assign each output chunk to a drive; insertion order within a
  // drive gives its CHUNK_SIZE-aligned slot.
  std::vector<size_t> drive_of(nc);
  {
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, num_drives - 1);
    for (size_t i = 0; i < nc; i++) drive_of[i] = dist(rng);
  }
  std::vector<std::vector<size_t>> drive_chunks(num_drives);
  for (size_t i = 0; i < nc; i++) drive_chunks[drive_of[i]].push_back(i);
  std::vector<size_t> slot_of(nc);
  for (size_t dr = 0; dr < num_drives; dr++)
    for (size_t s = 0; s < drive_chunks[dr].size(); s++)
      slot_of[drive_chunks[dr][s]] = s;

  // Pre-fallocate each drive file to its exact final size.
  std::vector<std::string> filenames(num_drives);
  parlay::parallel_for(
      0, num_drives,
      [&](size_t dr) {
        filenames[dr] = GetFileName(result_prefix, dr);
        const size_t file_size = drive_chunks[dr].size() * CHUNK_SIZE;
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
  // One output chunk per logical chunk of `d`, holding chunk_len(i) elements of
  // R -- so `used` is only CHUNK_SIZE when d's partition happens to be R's own
  // grid.  Otherwise the output is *ragged*, which delay<R>() reads back
  // correctly (it takes the partition from each chunk's `used`); it just costs
  // disk.  The only hard requirement is that a logical chunk's elements fit one
  // physical chunk -- which is what used to be spelled sizeof(R) <= 8, and is
  // still what stops a caller forcing zip's 16-byte std::pair off an 8-byte
  // grid.
  std::vector<chunk> out_chunks(nc);
  for (size_t i = 0; i < nc; i++) {
    const size_t nbytes = d.chunk_len(i) * sizeof(R);
    CHECK(nbytes <= CHUNK_SIZE)
        << "force: logical chunk " << i << " holds " << d.chunk_len(i)
        << " elements of " << sizeof(R) << " bytes (" << nbytes
        << "), which exceeds CHUNK_SIZE " << CHUNK_SIZE
        << " -- map wider values (e.g. zip's std::pair) down to a narrower "
           "scalar before force";
    out_chunks[i] = {filenames[drive_of[i]], slot_of[i] * CHUNK_SIZE, nbytes,
                     i};
  }

  UnorderedWriterConfig wcfg;
  wcfg.num_threads = num_drives;
  wcfg.io_uring_size = 32;
  wcfg.queue_size = 64;
  wcfg.num_files = num_drives;
  UnorderedFileWriter<R> writer;
  writer.Start(filenames, wcfg);

  for_each_chunk(d, [&](size_t ci, size_t n, auto it) {
    R* out = (R*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
    CHECK(out != nullptr) << "delayed::force: allocation failed";
    for (size_t i = 0; i < n; i++) {
      out[i] = *it;
      ++it;
    }
    memset((char*)out + n * sizeof(R), 0, CHUNK_SIZE - n * sizeof(R));
    writer.Push(std::shared_ptr<R>(out, free), CHUNK_SIZE / sizeof(R),
                drive_of[ci], slot_of[ci] * CHUNK_SIZE);
  });

  writer.Wait();
  return {out_chunks};
}

// filter: pack survivors (pred over the fused elements) into a dense chunk_seq.
// Modeled on ChunkFilter — index-contiguous windows, per-chunk survivor
// compaction, prefix sums, parallel scatter, dense CHUNK_SIZE packing with a
// cross-window carry — but each chunk's elements come from walking the fused
// node iterator, so preceding maps/zips fuse into this read pass.  Returns an
// index-ordered chunk_seq.
template <class D, class Pred>
chunk_seq filter(const D& d, const std::string& result_prefix, Pred pred) {
  using R = typename D::value_type;
  static_assert(CHUNK_SIZE % sizeof(R) == 0,
                "sizeof(R) must divide CHUNK_SIZE for O_DIRECT alignment");
  static_assert(sizeof(R) <= sizeof(uint64_t),
                "filter: the on-disk chunk grid assumes <=8B elements");

  if (d.num_chunks() == 0) return {};
  const size_t num_drives = GetSSDList().size();
  const size_t epct = CHUNK_SIZE / sizeof(R);  // elements per output chunk

  // Create/truncate one output file per drive (writer opens with O_CREAT only).
  std::vector<std::string> filenames(num_drives);
  parlay::parallel_for(
      0, num_drives,
      [&](size_t dr) {
        filenames[dr] = GetFileName(result_prefix, dr);
        int fd =
            open(filenames[dr].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        SYSCALL(fd);
        SYSCALL(close(fd));
      },
      1);

  std::vector<size_t> next_slot(num_drives, 0);
  std::mt19937_64 rng(std::random_device{}());
  std::uniform_int_distribution<size_t> drive_dist(0, num_drives - 1);

  std::vector<R> carry;  // survivors not yet filling a full chunk (< epct)
  carry.reserve(epct);
  std::vector<chunk> out_chunks;
  size_t out_idx = 0;

  UnorderedWriterConfig wcfg;
  wcfg.num_threads = num_drives;
  wcfg.io_uring_size = 32;
  wcfg.queue_size = 64;
  wcfg.num_files = num_drives;
  UnorderedFileWriter<R> writer;
  writer.Start(filenames, wcfg);

  // One window == one FILTER_BATCH_SIZE batch.  for_each_window runs windows
  // sequentially, so the cross-window `carry` threads correctly.
  for_each_window(d, [&](size_t base, size_t w, auto build_chunk) {
    std::vector<R*> surv(w, nullptr);  // compacted survivors of chunk base+b
    std::vector<size_t> scount(w, 0);

    parlay::parallel_for(
        0, w,
        [&](size_t b) {
          auto it = build_chunk(b);
          const size_t n = d.chunk_len(base + b);
          R* sb = (R*)malloc(std::max<size_t>(1, n) * sizeof(R));
          CHECK(sb != nullptr) << "delayed::filter: allocation failed";
          size_t s = 0;
          for (size_t j = 0; j < n; j++) {
            R v = *it;
            ++it;
            if (pred(v)) sb[s++] = v;
          }
          surv[b] = sb;
          scount[b] = s;
        },
        1);

    // Prefix sums: offset[b] = absolute position of chunk b's first survivor.
    std::vector<size_t> offset(w + 1);
    offset[0] = carry.size();
    for (size_t b = 0; b < w; b++) offset[b + 1] = offset[b] + scount[b];
    const size_t total = offset[w];
    const size_t num_out = total / epct;
    const size_t new_carry_cnt = total % epct;

    // Allocate output buffers (full chunks + 1 overflow for the new carry).
    const size_t num_alloc = num_out + (new_carry_cnt > 0 ? 1 : 0);
    std::vector<R*> obuf(num_alloc, nullptr);
    for (size_t k = 0; k < num_alloc; k++) {
      obuf[k] = (R*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
      CHECK(obuf[k] != nullptr) << "delayed::filter: output allocation failed";
      memset(obuf[k], 0, CHUNK_SIZE);
    }
    if (!carry.empty() && num_alloc > 0)
      memcpy(obuf[0], carry.data(), carry.size() * sizeof(R));

    // Parallel scatter (non-overlapping ranges by construction).
    parlay::parallel_for(
        0, w,
        [&](size_t b) {
          if (scount[b] == 0) return;
          const R* src = surv[b];
          size_t pos = offset[b], rem = scount[b], src_o = 0;
          while (rem > 0) {
            const size_t k = pos / epct;
            const size_t off = pos % epct;
            const size_t can = std::min(rem, epct - off);
            memcpy(obuf[k] + off, src + src_o, can * sizeof(R));
            pos += can;
            src_o += can;
            rem -= can;
          }
        },
        1);

    for (size_t b = 0; b < w; b++) free(surv[b]);

    // Push full output chunks with balls-in-bins drive assignment.
    for (size_t k = 0; k < num_out; k++) {
      const size_t dr = drive_dist(rng);
      const size_t slot = next_slot[dr]++;
      writer.Push(std::shared_ptr<R>(obuf[k], free), CHUNK_SIZE / sizeof(R), dr,
                  slot * CHUNK_SIZE);
      out_chunks.push_back(
          {filenames[dr], slot * CHUNK_SIZE, CHUNK_SIZE, out_idx++});
    }

    carry.resize(new_carry_cnt);
    if (new_carry_cnt > 0) {
      memcpy(carry.data(), obuf[num_out], new_carry_cnt * sizeof(R));
      free(obuf[num_out]);
    }
  });

  // Flush the final partial chunk.
  if (!carry.empty()) {
    R* buf = (R*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
    CHECK(buf != nullptr) << "delayed::filter: final allocation failed";
    memset(buf, 0, CHUNK_SIZE);
    memcpy(buf, carry.data(), carry.size() * sizeof(R));
    const size_t dr = drive_dist(rng);
    const size_t slot = next_slot[dr]++;
    writer.Push(std::shared_ptr<R>(buf, free), CHUNK_SIZE / sizeof(R), dr,
                slot * CHUNK_SIZE);
    out_chunks.push_back({filenames[dr], slot * CHUNK_SIZE,
                          carry.size() * sizeof(R), out_idx++});
  }

  writer.Wait();
  return {out_chunks};
}

}  // namespace delayed
}  // namespace plaid

#endif  // CHUNK_DELAYED_H
