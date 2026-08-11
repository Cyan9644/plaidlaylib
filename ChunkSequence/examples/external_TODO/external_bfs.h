#include <parlay/primitives.h>
#include <parlay/sequence.h>

#include <atomic>
#include <limits>
#include <string>

#include "ChunkSequence/helper/external_compressed_sparse_row.h"
#include "ChunkSequence/Primitives/delayed.h"
#include "ChunkSequence/Primitives/flat_map.h"
#include "ChunkSequence/Primitives/chunk_seq.h"

// this is the external version of the parallel bfs algorithm

// Ok, now that we've implemented a simple csr version below, let's make a
// simple external version.
//  template <typename V>
//  auto BFS_simple_delay(V start, const chunk_csr& G){

//   //our assumption
//   //make sequence: if i is the start, mark it as visited
//   auto visited = parlay::tabulate<std::atomic<bool>>(G.degree_scan.size()-1,
//   [&] (long i){
//     return (i==start) ? true : false;});

//     //the current frontier is just the first vertex
//     //spinning up a whole tabulate is wasteful here
//   chunk_seq frontier = ChunkSequenceOps::tabulate<size_t>(1, "bfs_frontier0",
//   [&](short i){
//     return start;
//   });
//   parlay::sequence<chunk_seq> frontiers;
//   size_t round=0;

// //need to figure out how this works
//   std::vector<ChunkSequenceOps::delayed::SequentialReadContext> ctxs(
//       std::max<size_t>(1, parlay::num_workers()));

//   while (!frontier.chunks.empty()){
//     //add the current frontier to the frontiers list

//     frontiers.push_back(frontier);
//     auto out = ChunkSequenceOps::ChunkFlatMap((frontier), [&](vertex u){

//       return ChunkSequenceOps::delayed::lazy_filter(G.delay_get_adjacent(u),
//       [&](weighted_edge e){
//       // visited[e.connecting_vertex] ? 0 : 1;
//       bool expected = false;
//       return (!visited[e.connecting_vertex]) &&
//       visited[e.connecting_vertex].compare_exchange_strong(expected,
//       true);});

//       });
//   frontier = ChunkSequenceOps::delayed::force(out, "filter_prefix");
//     }

//     // frontier = ChunkSequenceOps::delayed::force(out);
//     return frontiers;
// }

// let's implement a quick csr bfs to get the hang of this again since it's been
// a while since I wrote a graph algorithm with csr

// we're assuming that the weights are all set to something uniform so it may as
// well be an unweighted graph another assumption is that the vertex list can
// live in memory, but the edge list cannot
template <typename V>
auto BFS_simple(V start, const chunk_csr& G) {
  // our assumption
  // make sequence: if i is the start, mark it as visited
  auto visited = parlay::tabulate<std::atomic<bool>>(
      G.degree_scan.size() - 1,
      [&](long i) { return (i == start) ? true : false; });

  // the current frontier is just the first vertex
  // spinning up a whole tabulate is wasteful here
  chunk_seq frontier = ChunkSequenceOps::tabulate<size_t>(
      1, "bfs_frontier0", [&](short i) { return start; });
  parlay::sequence<chunk_seq> frontiers;
  size_t round = 0;

  // need to figure out how this works
  std::vector<ChunkSequenceOps::delayed::SequentialReadContext> ctxs(
      std::max<size_t>(1, parlay::num_workers()));

  while (!frontier.chunks.empty()) {
    // add the current frontier to the frontiers list
    frontiers.push_back(frontier);
    frontier = ChunkSequenceOps::ChunkFlatMap<size_t, size_t>(
        frontier, "bfs_frontier" + std::to_string(++round), [&](size_t u) {
          auto& ctx = ctxs[parlay::worker_id()];
          parlay::sequence<weighted_edge> adjacent =
              ChunkSequenceOps::delayed::sequential_materialize(
                  G.delay_get_adjacent(u), ctx);
          parlay::sequence<size_t> out;
          for (auto&& e : adjacent) {
            size_t v = e.connecting_vertex;
            bool expected = false;
            // prevent others from updating status while we're checking it here
            // yeah this is going to be VERY slow if we need to access an atomic
            // for each push back. a potentially much faster way to do this is
            // to allocate some small space for each block so I think in order
            // to speed this up we'll eventually need to make this loop parallel
            if (!visited[v].load() &&
                visited[v].compare_exchange_strong(expected, true))
              out.push_back(v);
          }
          return out;
        });
  }

  // this should basically work, because each iteration essentially runs at
  // filter speed. the main problem is essentially that when we read from the
  // delayed cut, we need to actually issue a read for each individual edge of
  // course this is slow because it's not buffered and we need to re-read from
  // the same areas multiple times -- see the

  return frontiers;
}

// external_bfs: the streaming out-of-core BFS. Same level-by-level frontier
// expansion as BFS_simple, but each round does ONE streaming pass over ALL
// edges (a boolean segmented-reduce keyed by chunk_csr's degree_scan bounds)
// instead of a per-vertex delay_get_adjacent + sequential_materialize read.
// That alone removes BFS_simple's small/duplicate reads -- but the streaming
// pass itself used to rebuild a fresh reader (io_uring rings, reader
// threads, fd opens) EVERY round, via segmented_reduce_over_edges's eager
// RemoveWorker. G.edges/G.degree_scan never change across rounds -- only
// `dist`'s captured values do -- so, exactly like external_bellman_ford_fast
// (external_bellman_ford.h) does for the same reason, build ONE
// PersistentReadContext before the loop and reuse it for every round's
// delayed::segmented_reduce call instead.
template <typename V>
auto external_bfs(V start, chunk_csr& G, size_t* rounds_out = nullptr) {
  size_t n = G.degree_scan.size() - 1;

  // dist[v] == the round v was first reached; "max" sentinel == unvisited.
  // Doubles as BFS_simple's `visited` check (dist[v] != max) and, read at
  // round r-1, as "is v in the current frontier" for the streaming pass.
  auto unvisited = std::numeric_limits<size_t>::max();
  parlay::sequence<size_t> dist(n, unvisited);
  dist[start] = 0;

  // the current frontier is just the first vertex
  // spinning up a whole tabulate is wasteful here
  // NOTE: "bfs_frontier_fast" (not BFS_simple's "bfs_frontier") -- the two
  // implementations run back-to-back in the same process in bfs.cpp, and
  // sharing a prefix would let this loop's writes clobber BFS_simple's
  // still-live frontier chunk_seqs out from under it. Still starts with
  // "bfs_frontier", so the existing "bfs_frontier*" cleanup globs
  // (run_benches.py's data_globs, scripts/clean_bfs.sh) still catch it.
  chunk_seq frontier = ChunkSequenceOps::tabulate<size_t>(
      1, "bfs_frontier_fast0", [&](short i) { return start; });
  parlay::sequence<chunk_seq> frontiers;
  size_t round = 0;

  struct OrMonoid {
    bool identity = false;
    bool operator()(bool a, bool b) const { return a || b; }
  };

  // `dist` is captured BY REFERENCE, so per_edge itself needs to be built
  // only ONCE: reassigning `cur_round` below on every iteration is visible
  // through the reference without rebuilding the node (same reasoning as
  // external_bellman_ford_fast's `per_edge` -- see its comment).
  size_t cur_round = 0;
  auto per_edge = ChunkSequenceOps::delayed::map(
      ChunkSequenceOps::delayed::delay<weighted_edge>(G.edges),
      [&](weighted_edge e) { return dist[e.connecting_vertex] == cur_round; });
  ChunkSequenceOps::delayed::PersistentReadContext<decltype(per_edge)> ctx(
      per_edge);

  while (!frontier.chunks.empty()) {
    // add the current frontier to the frontiers list
    frontiers.push_back(frontier);

    // one streaming pass over ALL edges: for each vertex v, OR over its
    // incident edges whether the neighbor is exactly at `round` (i.e. in the
    // frontier just pushed above) -- replaces BFS_simple's per-vertex
    // delay_get_adjacent + sequential_materialize reads, and reuses `ctx`'s
    // already-running io_uring rings/reader threads instead of paying setup
    // cost again this round.
    cur_round = round;
    parlay::sequence<bool> reached =
        ChunkSequenceOps::delayed::segmented_reduce(per_edge, G.degree_scan,
                                                    OrMonoid{}, ctx);

    ++round;
    parlay::sequence<bool> newly_reached = parlay::tabulate(n, [&](size_t v) {
      if (reached[v] && dist[v] == unvisited) {
        dist[v] = round;
        return true;
      }
      return false;
    });
    parlay::sequence<size_t> next_ids =
        parlay::pack_index<size_t>(newly_reached);

    frontier = ChunkSequenceOps::tabulate<size_t>(
        next_ids.size(), "bfs_frontier_fast" + std::to_string(round),
        [&](size_t i) { return next_ids[i]; });
  }

  if (rounds_out) *rounds_out = round;
  return frontiers;
}

// let's implement a quick csr bfs to get the hang of this again since it's been
// a while since I wrote a graph algorithm with csr
//
//  NOTE: kept as reference pseudocode only, not wired up -- `n` below was never
//  declared anywhere (not a typo fixable by inspection: no member/local of that
//  name exists on any plausible `graph`), and since it doesn't depend on either
//  template parameter, two-phase lookup diagnoses it at *definition* time,
//  which broke compilation of this whole header for any includer regardless of
//  whether BFS_csr is ever instantiated. Disabled via #if 0 rather than deleted
//  so the sketch stays visible.
//  #if 0
//  template <typename vertex, typename graph>
//  auto BFS_csr(vertex start, const graph& G) {
//    using nested_seq = parlay::sequence<parlay::sequence<vertex>>;

//   //make sequence: if i is the start, mark it as visited, otherwise, false.
//   auto visited = parlay::tabulate<std::atomic<bool>>(G.edges.size(), [&]
//   (long i) {
//     return (i==start) ? true : false; });

//     //the current frontier is just the first vertex
//   parlay::sequence<vertex> frontier(1,start);
//   nested_seq frontiers;
//   while (frontier.size() > 0) {
//     //add the current frontier to the frontiers list
//     frontiers.push_back(frontier);

//     //for each vertex u in the frontier, return its adjacency list.
//     Flattening this gives us the full next frontier list
//     //of nodes that could be visited
//     auto out = flatten(map(frontier, [&] (vertex u) {return G.f.cut(n[u],
//     n[u+1]);}));

//     //filter the frontier based on whether those nodes have yet been visited
//     frontier = filter(out, [&] (auto&& v) {
//       bool expected = false;
//       //prevent others from updating v's status while we're checking it here
//       return (!visited[v]) && visited[v].compare_exchange_strong(expected,
//       true);});
//   }

//   return frontiers;
// }
// #endif

// The in-memory reference BFS this file's out-of-core BFS_simple is compared
// against lives at examples/in_memory/graph/bfs.h (byte-identical to
// deps/parlaylib-examples/BFS.h) -- kept out of this header (rather than
// duplicated here as it briefly was) since both this header and that one
// declare a global-scope `BFS(vertex, graph)` template, and a translation
// unit that includes both (bfs.cpp) would otherwise get a redefinition error.