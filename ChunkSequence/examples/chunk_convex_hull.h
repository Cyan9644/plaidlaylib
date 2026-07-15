#ifndef CHUNK_CONVEX_HULL_H
#define CHUNK_CONVEX_HULL_H

#include <atomic>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

#include "absl/log/check.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_reduce.h"
#include "ChunkSequence/chunk_partition.h"
#include "utils/file_utils.h"
#include "configs.h"

// Out-of-core 2D upper convex hull via the quickhull divide-and-conquer.
//
// The out-of-core analogue of parlaylib's examples/quickhull.h.  A point cloud
// too large for DRAM lives as a chunk_seq of `hpoint`s; the hull is found by
// recursively (a) reducing to the point farthest from the current dividing line
// and (b) filtering the two sub-regions above the two new lines, exactly as
// upstream does over an in-DRAM sequence -- but the reduce and the two filters
// are the eager out-of-core primitives (ChunkReduce / ChunkFilter), so the
// working set streams off the SSDs and never has to fit in DRAM.
//
// Recursion has a DRAM base case: once a sub-region fits a caller-supplied byte
// budget it is materialized (to_vector) and finished with an in-DRAM quickhull
// (qh_mem) -- so only the top few levels touch the drives.  This is the same
// "shrink the working set until it fits, then go in-memory" pattern the external
// sort uses; near the root the input is huge and out-of-core, near the leaves it
// is small and in-DRAM.
//
// Element layout: `hpoint` is 32 bytes (two coords + original index + pad).
// 32 divides CHUNK_SIZE, so every chunk stays O_DIRECT-aligned; the eager
// engine and DensePack are generic in the element size (only the *delayed* layer
// is limited to <=8 B), so nothing here is special-cased for the fat element.
// `idx` carries the point's original position so the result is a list of hull
// vertex indices, matching upstream `upper_hull`'s output (and its tie-break:
// among equally-far points the one with the larger original index wins, which is
// exactly upstream's `maximum<pair<double,int>>` on (area, index)).
//
// This finds the UPPER hull (leftmost to rightmost across the top); the lower
// hull is symmetric (negate y, or run the same recursion below the min--max
// line) and a full hull is upper ++ lower with the shared endpoints dropped.

namespace ChunkSequenceOps {

struct hpoint {
    double   x;
    double   y;
    uint64_t idx;   // original position in the input sequence
    uint64_t pad;   // pad to 32 B so sizeof(hpoint) divides CHUNK_SIZE
};
static_assert(sizeof(hpoint) == 32, "hpoint must be 32 bytes");
static_assert(CHUNK_SIZE % sizeof(hpoint) == 0,
              "sizeof(hpoint) must divide CHUNK_SIZE for O_DIRECT alignment");

// Twice the signed area of triangle (a,b,c); > 0 iff c is left of the directed
// line a->b (counter-clockwise).  Identical to upstream's `area`.
inline double area(const hpoint& a, const hpoint& b, const hpoint& c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

namespace detail {

// Count of out-of-core split levels taken by the most recent UpperHull call, so
// drivers/tests can confirm the recursion actually left DRAM (rather than
// materializing the whole input immediately).  Reset at the top of UpperHull.
inline std::atomic<size_t>& ext_split_counter() {
    static std::atomic<size_t> c{0};
    return c;
}

// Monotonic id source for unique per-node scratch prefixes (short, unlike a
// path string that would grow with recursion depth and blow the filename limit).
inline std::atomic<size_t>& prefix_counter() {
    static std::atomic<size_t> c{0};
    return c;
}

// Remove the one-file-per-drive scratch left by a ChunkFilter under `prefix`.
inline void cleanup_prefix(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++)
        unlink(GetFileName(prefix, d).c_str());
}

// Max-by-(area, idx) reduction returning the winning point (identity loses via
// key = -inf).  Ties by the point's ORIGINAL idx, so it is order-independent:
// qh_mem works on the completion-ordered output of ChunkPartition, not just an
// increasing-idx sequence, and picks the same mid as the out-of-core FarMonoid
// and upstream (which ties on the element of Idxs = the original index).
struct FarPoint { double key; hpoint pt; };
struct FarPointMonoid {
    FarPoint identity{-std::numeric_limits<double>::infinity(), hpoint{}};
    FarPoint operator()(const FarPoint& a, const FarPoint& b) const {
        if (b.key > a.key || (b.key == a.key && b.pt.idx > a.pt.idx)) return b;
        return a;
    }
};

// In-DRAM upper-hull quickhull on the points strictly above the line l--r (in any
// order).  A faithful port of upstream quickhull.h that carries the points
// themselves (so no back-reference into a global array) and returns original
// indices.  The farthest point is chosen by (area, idx) (FarPointMonoid), so the
// result is identical regardless of the order S arrives in.
//
// `S` is taken by value and freed before recursing (upstream's `Idxs.clear()` +
// `std::move` discipline): the two children partition S's points, so without
// freeing the parent the parallel recursion would hold O(n log n) live points at
// once and blow up DRAM on large base cases.  With it, peak stays O(n).
inline parlay::sequence<uint64_t> qh_mem(parlay::sequence<hpoint> S,
                                         hpoint l, hpoint r) {
    const long n = (long)S.size();
    if (n == 0) return parlay::sequence<uint64_t>();
    if (n == 1) return parlay::sequence<uint64_t>(1, S[0].idx);

    // Farthest point from l--r; delayed keys, so the pairs are never materialized.
    const hpoint mid = parlay::reduce(
        parlay::delayed_tabulate(n, [&](long i) {
            return FarPoint{area(l, r, S[i]), S[i]}; }),
        FarPointMonoid{}).pt;

    auto left  = parlay::filter(S, [&](const hpoint& p) {
        return area(l, mid, p) > 0; });
    auto right = parlay::filter(S, [&](const hpoint& p) {
        return area(mid, r, p) > 0; });
    S = parlay::sequence<hpoint>();   // free the parent's points before recursing

    parlay::sequence<uint64_t> LR, RR;
    parlay::par_do_if(n > 1000,
        [&] { LR = qh_mem(std::move(left),  l,   mid); },
        [&] { RR = qh_mem(std::move(right), mid, r);   });

    parlay::sequence<parlay::sequence<uint64_t>> parts = {
        std::move(LR), parlay::sequence<uint64_t>(1, mid.idx), std::move(RR)};
    return parlay::flatten(parts);
}

// Farthest-point argmax as a ChunkReduce monoid: fold each hpoint into the
// running best (largest (area(l,r,p), p.idx)), and combine two per-worker bests.
struct FarMonoid {
    hpoint l, r;
    struct Acc { double key; hpoint pt; bool has; };
    Acc identity{0.0, hpoint{}, false};

    static bool better(const Acc& a, const Acc& b) {  // is b >= a's winner?
        if (!a.has) return true;
        if (!b.has) return false;
        return (b.key > a.key) || (b.key == a.key && b.pt.idx > a.pt.idx);
    }
    Acc operator()(const Acc& a, const hpoint& p) const {
        Acc b{area(l, r, p), p, true};
        return better(a, b) ? b : a;
    }
    Acc operator()(const Acc& a, const Acc& b) const {
        return better(a, b) ? b : a;
    }
};

// Combined min/max-by-(x,y) as a ChunkReduce monoid (upstream's `minmax_element`
// with the same (x, then y) comparator).  Endpoints of the top-level hull.
struct MinMaxMonoid {
    struct Acc { hpoint mn, mx; bool has; };
    Acc identity{hpoint{}, hpoint{}, false};

    static bool less(const hpoint& a, const hpoint& b) {
        return (a.x < b.x) || (a.x == b.x && a.y < b.y);
    }
    Acc operator()(Acc a, const hpoint& p) const {
        if (!a.has) return {p, p, true};
        if (less(p, a.mn)) a.mn = p;
        if (less(a.mx, p)) a.mx = p;
        return a;
    }
    Acc operator()(Acc a, const Acc& b) const {
        if (!a.has) return b;
        if (!b.has) return a;
        if (less(b.mn, a.mn)) a.mn = b.mn;
        if (less(a.mx, b.mx)) a.mx = b.mx;
        return a;
    }
};

// Recursive out-of-core quickhull over the points strictly above l--r (`pts`, any
// order).  Returns the hull vertices strictly between l and r, left to right, as
// original indices.  Frees its own scratch.
inline std::vector<uint64_t> quickhull_ext(const chunk_seq& pts,
                                           hpoint l, hpoint r,
                                           size_t budget_elems,
                                           const std::string& scratch) {
    const size_t n = ChunkSequenceOps::size<hpoint>(pts);
    if (n == 0) return {};

    // Base case: the sub-region fits DRAM -> materialize and finish in memory.
    if (n <= budget_elems) {
        parlay::sequence<hpoint> S;
        {
            std::vector<hpoint> v = pts.to_vector<hpoint>();
            S = parlay::tabulate(v.size(), [&](size_t i) { return v[i]; });
        }  // free the std::vector before the in-DRAM quickhull runs
        auto res = qh_mem(std::move(S), l, r);
        return std::vector<uint64_t>(res.begin(), res.end());
    }

    // Out-of-core split: the farthest point (one read pass), then ONE partition
    // pass that routes each point to the left sub-region (above l--mid), the right
    // sub-region (above mid--r), or drops it (inside triangle l,mid,r).  The two
    // sub-regions are disjoint (mid is the apex, so no point above l--r is above
    // both new lines), so a single ChunkPartition replaces the two ChunkFilters:
    // one read pass and one reader/writer instead of two of each.
    ext_split_counter().fetch_add(1, std::memory_order_relaxed);
    const hpoint mid =
        ChunkReduce<hpoint, FarMonoid::Acc>(pts, FarMonoid{l, r}).pt;

    const std::string sp =
        scratch + "p" + std::to_string(prefix_counter().fetch_add(1));
    std::vector<chunk_seq> parts = ChunkPartition<hpoint>(pts, /*num_buckets=*/2, sp,
        [l, mid, r](const hpoint& p) -> size_t {
            if (area(l, mid, p) > 0) return 0;   // left sub-region
            if (area(mid, r, p) > 0) return 1;   // right sub-region
            return PARTITION_DROP;               // inside the peak triangle
        });

    // Left and right buckets share `sp`'s files; recurse into each (sequentially,
    // so both are still readable) before removing them.
    std::vector<uint64_t> LR = quickhull_ext(parts[0], l,   mid, budget_elems, sp);
    std::vector<uint64_t> RR = quickhull_ext(parts[1], mid, r,   budget_elems, sp);
    cleanup_prefix(sp);

    std::vector<uint64_t> out;
    out.reserve(LR.size() + 1 + RR.size());
    out.insert(out.end(), LR.begin(), LR.end());
    out.push_back(mid.idx);
    out.insert(out.end(), RR.begin(), RR.end());
    return out;
}

} // namespace detail

/**
 * Out-of-core 2D upper convex hull of the point cloud `points` (a chunk_seq of
 * hpoint).  Returns the hull vertices as original indices, left to right
 * (leftmost point, upper-hull interior vertices, rightmost point) -- the same
 * output as parlaylib's upper_hull on the equivalent in-DRAM sequence.
 *
 * @param points           input points, any order (idx = original position).
 * @param dram_budget_bytes a sub-region at or below this many bytes is finished
 *                          in DRAM; above it, the region is split with the
 *                          out-of-core reduce+partition primitives.  Clamped so at
 *                          least one element is always a valid base case.
 * @param scratch_prefix   file-name prefix for the recursion's intermediate
 *                          chunk_seqs (all cleaned up before returning).
 */
inline std::vector<uint64_t> UpperHull(const chunk_seq& points,
                                       size_t dram_budget_bytes,
                                       const std::string& scratch_prefix = "ch_scratch") {
    detail::ext_split_counter().store(0, std::memory_order_relaxed);
    if (ChunkSequenceOps::size<hpoint>(points) == 0) return {};

    const size_t budget_elems =
        std::max<size_t>(1, dram_budget_bytes / sizeof(hpoint));

    // Endpoints: min and max by (x, y).
    const detail::MinMaxMonoid::Acc mm =
        ChunkReduce<hpoint, detail::MinMaxMonoid::Acc>(points, detail::MinMaxMonoid{});
    const hpoint minp = mm.mn, maxp = mm.mx;

    // Points strictly above the min--max line: the upper-hull interior candidates.
    // A 1-bucket ChunkPartition (keep-or-drop) so this too is a single-reader pass
    // -- quickhull never touches the per-batch windowed ChunkFilter.
    const std::string ap =
        scratch_prefix + "a" + std::to_string(detail::prefix_counter().fetch_add(1));
    chunk_seq above = ChunkPartition<hpoint>(points, /*num_buckets=*/1, ap,
        [minp, maxp](const hpoint& p) -> size_t {
            return area(minp, maxp, p) > 0 ? 0 : PARTITION_DROP;
        })[0];

    std::vector<uint64_t> between =
        detail::quickhull_ext(above, minp, maxp, budget_elems, ap);
    detail::cleanup_prefix(ap);

    std::vector<uint64_t> out;
    out.reserve(between.size() + 2);
    out.push_back(minp.idx);
    out.insert(out.end(), between.begin(), between.end());
    out.push_back(maxp.idx);
    return out;
}

// Number of out-of-core split levels the last UpperHull call performed (0 means
// the whole input fit the DRAM budget and was finished in memory).
inline size_t last_ext_splits() {
    return detail::ext_split_counter().load(std::memory_order_relaxed);
}

} // namespace ChunkSequenceOps

#endif // CHUNK_CONVEX_HULL_H
