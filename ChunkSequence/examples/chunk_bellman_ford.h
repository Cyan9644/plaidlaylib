#include <parlay/io.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>

#include <limits>
#include <optional>

#include "ChunkSequence/Primitives/delayed.h"
#include "ChunkSequence/helper/external_compressed_sparse_row.h"

// **************************************************************
// Parallel Bellman Ford
// Returns an optional, which is empty if there is a negative weight
// cycle, or otherwise returns the distance to each vertex.
// **************************************************************

// Note on parallel bellman-ford in general:
// we accept transpose graphs, so the edges are just reversed in order.
// this is to do pull-based bellman ford, which does not have write conflicts
// because the reads are on the shared data rather than the writes
//
// The basic process of bellman ford is to relax all edges number-of-vertices
// times; the check is based on the edges of the graph rather than the
// vertices, so for each edge we check whether d[other vertex] +
// weight(other_vertex, target_vertex) < d[target_vertex]. Potentially n-1
// iterations since the graph could be a line, so relaxation may need to
// propagate to the far end.
//
// One key point: because our (and most) csr is inherently read-only, we use a
// delayed cut to avoid instantiating the per-vertex adjacency subsequences on
// disk.

// this function accepts a transposed graph
parlay::sequence<weight> external_bellman_ford(chunk_csr& graph, vertex start,
                                               size_t* rounds_out = nullptr) {
  auto N = graph.degree_scan;
  // you can't materialize the edge list, which is the whole point
  chunk_seq& F = graph.edges;

  size_t n = graph.degree_scan.size() - 1;
  auto max_size = std::numeric_limits<long double>::max();
  parlay::sequence<long double> d(n, max_size);

  d[start] = 0;

  std::vector<plaid::delayed::SequentialReadContext> ctxs(
      std::max<size_t>(1, parlay::num_workers()));

  for (size_t i = 0; i < n; i++) {
    auto pass = parlay::tabulate(
        n,
        [&](size_t v) {
          // this is not actually a complete delay operation, it just prevents
          // the cut from writing back intermediates
          auto& ctx = ctxs[parlay::worker_id()];
          auto adjacent = plaid::delayed::sequential_materialize(
              plaid::delayed::cut<weighted_edge>(F, N[v], N[v + 1]),
              ctx);  // get the adjacency list for this vertex
          return parlay::reduce(parlay::delayed_tabulate(
                                    adjacent.size(),
                                    [&](size_t e) {
                                      return d[adjacent[e].connecting_vertex] +
                                             adjacent[e].edge_weight;
                                    }),
                                parlay::minimum<long double>());
        });

    pass[start] = 0;

    if (pass == d) {
      if (rounds_out) *rounds_out = i + 1;
      return d;
    }
    d = std::move(pass);
  }

  if (rounds_out) *rounds_out = n;
  return parlay::sequence<weight>();
}

// the performance gain in the fast_external_bellman_ford is that it doesn't
// issue individual preads per vertex instead we just do a long pass over the
// full edge sequence and adds the cost (if lower than existing) to the running
// minimum for the "to" part of the edge (the vertex it's going to) this is
// considerably better in all cases but especially the dense case, where edges
// in a single chunk are likely to belong to one specific vertex
//
// Quick point on how this actually works: looking up each edge to find where it
// fits in the degree scan is not efficient so we locate the vertex the chunk's
// first and last element fall in in a dense case these are probably equivalent,
// but supposing that they're different, we know that any vertex's edges between
// those are ALL of that vertex's edges since the edge list is sorted by vertex
// ID -- this means that for that vertex, all updates have been found and we can
// compute the distance directly. for the vertices on either end of the chunk,
// we don't know whether we have all of their contributions yet we CAN directly
// check whether we've moved into the territory of a specific vertex with the
// degree scan, though, so if we do so for an end vertex we need to stash the
// partial edge data for those vertices this boundary stash is sorted by index,
// so at the very end of the algorithm, we just need to combine them and then
// calculate the final distances for that pass based on the combined data we may
// wonder whether this boundary stash can actually fit in memory, but it
// definitely should be able to: there are at most 2 vertices that spill over
// per chunk, so the size of the total boundary array is at most ~2 * #chunks
// which means that it's O(#chunks) which is smaller than the edge sequence size
// by a factor of elements per chunk, which is ~large maybe like 32000 times
// Every round used to build its own ChunkSequenceReader (io_uring rings +
// reader threads + fd opens) from scratch via segmented_reduce/for_each_chunk,
// tearing it all down again at the end of the round -- expensive in general,
// and on WSL2 specifically it can hit io_uring's asynchronous RLIMIT_MEMLOCK
// reclaim (InitIoUringWithRetry, utils/file_utils.h) on every round's ring
// creation, burning real wall-clock time with neither disk nor CPU active.
// graph.edges/graph.degree_scan never change across rounds (only `d`'s
// captured values do), so per_edge's physical read plan is identical every
// round -- build ONE PersistentReadContext before the loop and reuse it for
// every round's segmented_reduce call instead.
parlay::sequence<weight> external_bellman_ford_fast(
    chunk_csr& graph, vertex start, size_t* rounds_out = nullptr) {
  size_t n = graph.degree_scan.size() - 1;
  auto max_size = std::numeric_limits<long double>::max();
  parlay::sequence<long double> d(n, max_size);

  d[start] = 0;

  struct MinDistMonoid {
    long double identity = std::numeric_limits<long double>::max();
    long double operator()(long double a, long double b) const {
      return std::min(a, b);
    }
  };

  // `d` is captured BY REFERENCE, so per_edge itself needs to be built only
  // ONCE: reassigning the *variable* `d` below (d = std::move(pass)) is visible
  // through the reference on every subsequent round without rebuilding the
  // node.  (Building a fresh per_edge each round would also give it a distinct
  // closure type from round to round's -- lambda types are unique per
  // lexical lambda-expression -- which would defeat reusing one `ctx`.)
  auto per_edge = plaid::delayed::map(
      plaid::delayed::delay<weighted_edge>(graph.edges),
      [&](weighted_edge e) { return d[e.connecting_vertex] + e.edge_weight; });
  plaid::delayed::PersistentReadContext<decltype(per_edge)> ctx(per_edge);

  size_t i = 0;
  for (; i < n; i++) {
    auto pass = plaid::delayed::segmented_reduce(
        per_edge, graph.degree_scan, MinDistMonoid{}, ctx);

    pass[start] = 0;

    if (pass == d) {
      if (rounds_out) *rounds_out = i + 1;
      return d;
    }
    d = std::move(pass);
  }

  if (rounds_out) *rounds_out = i;
  return parlay::sequence<weight>();
}
