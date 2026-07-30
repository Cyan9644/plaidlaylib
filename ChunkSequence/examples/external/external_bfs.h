#include <atomic>
#include <string>

#include <parlay/primitives.h>
#include <parlay/sequence.h>

#include "ChunkSequence/ExternalGraph/external_compressed_sparse_row.h"
#include "ChunkSequence/chunk_delayed.h"
#include "ChunkSequence/chunk_flat_map.h"
#include "ChunkSequence/chunk_seq.h"

//this is the external version of the parallel bfs algorithm

//Ok, now that we've implemented a simple csr version below, let's make a simple external version.



//let's implement a quick csr bfs to get the hang of this again since it's been a while since I wrote a graph algorithm
//with csr

//we're assuming that the weights are all set to something uniform so it may as well be an unweighted graph
//another assumption is that the vertex list can live in memory, but the edge list cannot
//
// NOTE: the template parameter is named V, not "vertex" -- external_compressed_sparse_row.h
// #defines `vertex` to size_t (and `weight` to long double) as pseudo-typedefs, with no #undef.
// A template parameter literally named `vertex` gets silently rewritten by the preprocessor to
// `size_t`, which shadows the real ::size_t for the rest of the function body.
template <typename V>
auto BFS_simple(V start, const chunk_csr& G) {

  //our assumption
  //make sequence: if i is the start, mark it as visited, otherwise, false.
  auto visited = parlay::tabulate<std::atomic<bool>>(G.degree_scan.size()-1, [&] (long i){
    return (i==start) ? true : false;});


    //the current frontier is just the first vertex
    //spinning up a whole tabulate is wasteful here
  chunk_seq frontier = ChunkSequenceOps::tabulate<size_t>(1, "bfs_frontier0", [&](short i){
    return start;
  });
  parlay::sequence<chunk_seq> frontiers;
  size_t round=0;

//need to figure out how this works
  std::vector<ChunkSequenceOps::delayed::SequentialReadContext> ctxs(
      std::max<size_t>(1, parlay::num_workers()));

  while (!frontier.chunks.empty()){
    //add the current frontier to the frontiers list
    frontiers.push_back(frontier);

    //for each vertex u in the frontier, return its adjacency list. Flattening this gives us the full next frontier list
    //of nodes that could be visited, filtered inline via the visited CAS below (so no separate filter pass is needed)

    //this is different from the rabin-karp chunk_flatmap because it maps over elements rather than chunks --
    //ChunkFlatMap's elementwise overload (chunk_flat_map.h) takes f: T -> parlay::sequence<R>
    frontier = ChunkSequenceOps::ChunkFlatMap<size_t, size_t>(frontier, "bfs_frontier" + std::to_string(++round),[&] (size_t u) {

    auto& ctx = ctxs[parlay::worker_id()];
    parlay::sequence<weighted_edge> adjacent =ChunkSequenceOps::delayed::sequential_materialize(G.delay_get_adjacent(u), ctx);
    parlay::sequence<size_t> out;
    for (auto&&e : adjacent){
      size_t v = e.connecting_vertex;
      bool expected = false;
        //prevent others from updating status while we're checking it here
        //yeah this is going to be VERY slow if we need to access an atomic for each push back.
        //a potentially much faster way to do this is to allocate some small space for each block
        //so I think in order to speed this up we'll eventually need to make this loop parallel
       if (!visited[v].load() && visited[v].compare_exchange_strong(expected, true))
          out.push_back(v);
        }
        return out;
        });
    }

  //this should basically work, because each iteration essentially runs at filter speed.
  //the main problem is essentially that when we read from the delayed cut, we need to actually issue a read for each individual edge
  //of course this is slow because it's not buffered and we need to re-read from the same areas multiple times -- see the

  return frontiers;
}





//let's implement a quick csr bfs to get the hang of this again since it's been a while since I wrote a graph algorithm
//with csr
//
// NOTE: kept as reference pseudocode only, not wired up -- `n` below was never
// declared anywhere (not a typo fixable by inspection: no member/local of that
// name exists on any plausible `graph`), and since it doesn't depend on either
// template parameter, two-phase lookup diagnoses it at *definition* time, which
// broke compilation of this whole header for any includer regardless of
// whether BFS_csr is ever instantiated. Disabled via #if 0 rather than deleted
// so the sketch stays visible.
// #if 0
// template <typename vertex, typename graph>
// auto BFS_csr(vertex start, const graph& G) {
//   using nested_seq = parlay::sequence<parlay::sequence<vertex>>;

//   //make sequence: if i is the start, mark it as visited, otherwise, false.
//   auto visited = parlay::tabulate<std::atomic<bool>>(G.edges.size(), [&] (long i) {
//     return (i==start) ? true : false; });


//     //the current frontier is just the first vertex
//   parlay::sequence<vertex> frontier(1,start);
//   nested_seq frontiers;
//   while (frontier.size() > 0) {
//     //add the current frontier to the frontiers list
//     frontiers.push_back(frontier);

//     //for each vertex u in the frontier, return its adjacency list. Flattening this gives us the full next frontier list
//     //of nodes that could be visited
//     auto out = flatten(map(frontier, [&] (vertex u) {return G.f.cut(n[u], n[u+1]);}));

//     //filter the frontier based on whether those nodes have yet been visited
//     frontier = filter(out, [&] (auto&& v) {
//       bool expected = false;
//       //prevent others from updating v's status while we're checking it here
//       return (!visited[v]) && visited[v].compare_exchange_strong(expected, true);});
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