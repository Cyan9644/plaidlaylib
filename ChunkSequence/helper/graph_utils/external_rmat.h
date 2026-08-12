// external_rmat.h — out-of-core RMAT graph generation into a chunk_csr.
//
// The DRAM path (graph_utils<size_t>::rmat_symmetric_graph + add_weights) needs
// roughly 2-2.5x the *on-disk* edge bytes resident while it builds: the
// generated edge list, symmetrize's filter/transpose copies, two
// group_by_index outer arrays at 16 bytes per vertex each, and finally the
// weighted graph itself at 32 bytes per directed edge -- with the unweighted
// graph still alive alongside it.  At n = 2^32 / avg_degree 2 that is past
// 500 GiB, which is what OOM-killed bellman_fordExample at the 256 GiB sweep
// point.
//
// The observation that makes this avoidable: **an RMAT edge is a pure function
// of its index**.  rmat_edges_ draws edge i from gen[i], and add_weights draws
// (u,v)'s weight from gen[min(u,v)][max(u,v)] -- neither depends on any other
// edge.  So the whole edge list can be tabulated straight onto the drives with
// nothing resident, and CSR order is then just a sort.  Four passes:
//
//   1. tabulate 2*m_gen src_edge{src,dst,w} -- indices [0,m_gen) are the
//      forward edges, [m_gen,2*m_gen) the reversed ones (that *is* the
//      symmetrization).  Reads nothing; DRAM is the writer's buffer pool.
//   2. direct_sample_sort by (src,dst).
//   3. one DensePackStream pass with a 1-element forward halo that drops
//      self-loops, drops all but the last of each duplicate run, projects to
//      weighted_edge (dropping src -- that is exactly CSR row layout), and
//      run-length counts degrees into degree_scan with one atomic per run.
//   4. prefix-sum degree_scan in place.
//
// The result is the *same graph* the DRAM path builds: same generator
// expressions, and upstream's pre-symmetrize remove_duplicates is subsumed by
// step 3's adjacent dedup (both end at "the set of distinct non-self (u,v),
// symmetrized").  The only difference is that each vertex's neighbor list comes
// out sorted rather than hash-ordered, which Bellman-Ford's min-relax cannot
// see.  external_rmat_test checks that parity element-wise.
//
// Residual DRAM is degree_scan alone: 8(n+1) bytes, 32 GiB at n = 2^32, which
// chunk_csr requires in DRAM by construction.  Transient disk is ~3-4x the
// final edge bytes (step 1's output plus the sort's buckets), all swept before
// returning.
//
// NOTE ON SORT CHOICE: this uses direct_sample_sort, not sample_sort.  The
// latter pads its heap_tree pivots with std::numeric_limits<T>::max()
// (external_samplesort.h:145), which for a program-defined T with no
// numeric_limits specialization is a value-initialized T -- the *minimum*, not
// the maximum -- so it would silently misroute.  direct_sample_sort is the one
// already exercised on struct element types (chunk_dc3.h, chunk_suffix_array.h)
// and needs only operator< / operator==.

#ifndef EXTERNAL_RMAT_H
#define EXTERNAL_RMAT_H

#include <parlay/primitives.h>
#include <parlay/random.h>
#include <parlay/sequence.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <utility>

#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/Primitives/dense_pack.h"
#include "ChunkSequence/examples/external_TODO/chunk_sa_common.h"  // sa_detail::sweep
#include "ChunkSequence/examples/external_TODO/direct_samplesort.h"
#include "absl/log/check.h"
#include "configs.h"

// rmat_edge / the weight distribution are reused verbatim from the in-memory
// reference so the two graphs match.  This too must precede the macro-defining
// header below -- `vertex` is graph_utils' template parameter.
#include "ChunkSequence/examples/in_memory/graph/graph_utils/graph_utils.h"

// Parsed last: this #defines bare `vertex`/`weight` macros with no #undef, so
// everything above must already be preprocessed.  Nothing below may use those
// two words as identifiers.
#include "ChunkSequence/helper/external_compressed_sparse_row.h"

namespace ExternalGraphUtils {

// The on-disk element of the intermediate (pre-CSR) edge list.  8 + 8 + 16 = 32
// bytes exactly -- long double aligns to 16, so there is no tail padding and 32
// divides CHUNK_SIZE, keeping every chunk O_DIRECT-aligned.
struct src_edge {
  size_t src;
  size_t dst;
  long double w;

  // Sort key is (src, dst): src groups the CSR rows, dst makes duplicates
  // adjacent so step 3 can dedup with a 1-element lookahead.  The weight is
  // payload, never a key -- a duplicate (u,v) always carries the same weight
  // anyway, since add_weights derives it from (min(u,v), max(u,v)).
  bool operator<(const src_edge& o) const {
    if (src != o.src) return src < o.src;
    return dst < o.dst;
  }
  bool operator==(const src_edge& o) const {
    return src == o.src && dst == o.dst;
  }
};
static_assert(sizeof(src_edge) == 32, "src_edge must be exactly 32 bytes");
static_assert(CHUNK_SIZE % sizeof(src_edge) == 0,
              "sizeof(src_edge) must divide CHUNK_SIZE for O_DIRECT alignment");
static_assert(
    CHUNK_SIZE % sizeof(weighted_edge) == 0,
    "sizeof(weighted_edge) must divide CHUNK_SIZE for O_DIRECT alignment");

namespace detail {

// Edge `i` of rmat_edges_(logn, m, a, b, c), reproduced exactly: that function
// tabulates over gen[i] with a shared uniform_real_distribution<double>(0,1),
// and rmat_edge threads a single draw stream through the recursion (it takes
// the generator by value but the lambda holds `r` by reference).  The
// distribution is stateless, so constructing it per call -- which keeps this
// callable from many workers without a data race -- yields identical values.
inline std::pair<size_t, size_t> rmat_edge_at(size_t i, int logn, double a,
                                              double b, double c) {
  parlay::random_generator gen;
  std::uniform_real_distribution<double> dis(0.0, 1.0);
  auto r = gen[i];
  return graph_utils<size_t>::rmat_edge(logn, a, b, c, 0.0,
                                        [&]() { return dis(r); });
}

// add_weights<long double>(G, minw, maxw)'s weight for edge (u,v), reproduced
// exactly -- keyed on the unordered pair so (u,v) and (v,u) agree, which is
// what makes the symmetrized graph consistently weighted.
inline long double weight_of(size_t u, size_t v, long double minw,
                             long double maxw) {
  parlay::random_generator gen;
  std::uniform_real_distribution<long double> dis(minw, maxw);
  auto r = gen[std::min(u, v)][std::max(u, v)];
  return dis(r);
}

}  // namespace detail

/**
 * Out-of-core equivalent of
 *   add_weights<long double>(rmat_symmetric_graph(n_req, m_req), minw, maxw)
 * flattened into CSR row order.
 *
 * n_req is rounded to a power of two exactly as upstream does (n = 1 <<
 * round(log2(n_req))), and m_req is halved before generation to match
 * rmat_symmetric_graph's own rmat_edges_(logn, m/2) call, so passing the same
 * (n_req, m_req) to both produces the same graph.
 *
 * `prefix` names the returned edge files; the intermediates live under
 * `prefix + "_gen"` / `prefix + "_srt"` and are swept before returning.
 */
inline chunk_csr external_rmat_symmetric_graph(size_t n_req, size_t m_req,
                                               const std::string& prefix,
                                               long double minw,
                                               long double maxw, double a = .5,
                                               double b = .15, double c = .15) {
  const int logn = (int)std::round(std::log2((double)n_req));
  const size_t n = size_t{1} << logn;
  const size_t m_gen = m_req / 2;

  chunk_csr graph;
  graph.degree_scan = parlay::sequence<size_t>(n + 1, 0);
  if (m_gen == 0) return graph;  // no edges: degree_scan is already all zero

  // ── 1. generate forward + reverse edges straight to disk ────────────────
  // Indices [m_gen, 2*m_gen) re-run the RMAT recursion for an edge the first
  // half already computed, so this does 2x upstream's generation work.  That
  // is the cheaper trade: the alternative (generate m_gen edges, then a map
  // pass emitting the reversed copy) costs a full read + write pass over the
  // edge list -- hundreds of GB of I/O at benchmark scale -- to save some
  // logn-deep recursions that the drives are waiting on anyway.
  const std::string gen_pfx = prefix + "_gen";
  chunk_seq raw =
      plaid::tabulate<src_edge>(2 * m_gen, gen_pfx, [&](size_t i) {
        const bool reversed = (i >= m_gen);
        auto [u, v] =
            detail::rmat_edge_at(reversed ? i - m_gen : i, logn, a, b, c);
        if (reversed) std::swap(u, v);
        return src_edge{u, v, detail::weight_of(u, v, minw, maxw)};
      });

  // ── 2. CSR order ─────────────────────────────────────────────────────────
  const std::string srt_pfx = prefix + "_srt";
  chunk_seq sorted = plaid::direct_sample_sort<src_edge>(
      raw, std::less<>{}, srt_pfx);
  plaid::sa_detail::sweep(gen_pfx);

  // The forward halo below is only a correct "logical successor" if no chunk
  // is empty.  direct_sample_sort never emits one (an empty bucket
  // contributes zero chunks, and every slice it does emit has used > 0), but
  // the dedup silently over-counts a degree if that ever changes.
  for (const chunk& ch : sorted.chunks)
    CHECK(ch.used > 0) << "external_rmat: empty chunk in sorted edge list";

  // ── 3+4. dedup, drop self-loops, project to CSR rows, count degrees ─────
  graph.edges = plaid::DensePackStream<src_edge, weighted_edge>(
      sorted, prefix, /*halo=*/1,
      [&](const src_edge* in, size_t cnt, uint64_t /*gpos*/,
          const src_edge* halo_buf, size_t halo_n) {
        parlay::sequence<weighted_edge> out;
        out.reserve(cnt);

        // Degrees are accumulated per *run* of equal src rather than per
        // edge: the input is sorted, so a hub vertex's whole row costs one
        // atomic instead of millions.
        size_t run_src = 0, run_len = 0;
        auto flush_run = [&] {
          if (run_len == 0) return;
          __atomic_fetch_add(&graph.degree_scan[run_src + 1], run_len,
                             __ATOMIC_RELAXED);
          run_len = 0;
        };

        for (size_t j = 0; j < cnt; j++) {
          const src_edge& e = in[j];
          if (e.src == e.dst) continue;  // symmetrize drops self edges
          // Keep the LAST element of each duplicate run: that is the
          // formulation a *forward* halo can express, and it selects the
          // same set as keeping the first.
          const src_edge* next =
              (j + 1 < cnt) ? &in[j + 1] : (halo_n > 0 ? halo_buf : nullptr);
          if (next != nullptr && *next == e) continue;

          if (run_len > 0 && e.src != run_src) flush_run();
          run_src = e.src;
          run_len++;
          out.push_back(weighted_edge{e.dst, e.w});
        }
        flush_run();
        return out;
      });
  plaid::sa_detail::sweep(srt_pfx);

  // degree_scan[v+1] holds deg(v); an inclusive scan over [1, n] turns it
  // into the exclusive prefix sum chunk_csr expects (row v is
  // [degree_scan[v], degree_scan[v+1])), with degree_scan[0] left at 0.
  parlay::scan_inclusive_inplace(parlay::make_slice(
      graph.degree_scan.begin() + 1, graph.degree_scan.end()));

  return graph;
}

}  // namespace ExternalGraphUtils

#endif  // EXTERNAL_RMAT_H
