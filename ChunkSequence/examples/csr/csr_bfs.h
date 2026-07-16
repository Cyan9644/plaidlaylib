#include <atomic>

#include <parlay/primitives.h>
#include <parlay/sequence.h>

//this csr implementation of bfs is the template for the external version (which uses an adapted csr structure)
//it assumes a struct csr that does not actually exist, but just pretend it does for this example so we can abstract the external parts for the moment

auto BFS(vertex start, const csr& graph) {
  //this will create a boolean sequence of all 0s except for the index of the start vertex, which will be 1
  //this is definitely not the most efficient way to implement this externally since we would very much prefer not to write back to disk with tabulate
  //so I think the best thing to do will probably be to compute an in-memory tabulate or, if needed, an external delayed tabulate 
  //but since the vertices should be able to fit in memory, maybe it's okay to keep this

  auto num_vertices = graph.degree_scan.size();
  auto visited = parlay::tabulate<std::atomic<bool>>(num_vertices, [&] (long i) {
    return (i==start) ? true : false; });

  parlay::sequence<vertex> frontier(1,start);
  parlay::sequence<parlay::sequence<vertex>>; frontiers;
  while (frontier.size() > 0) {
    frontiers.push_back(frontier);

    // get out edges of the frontier and flatten
    auto out = flatten(map(frontier, [&] (vertex u) {return G[u];}));

    // keep the v that succeed in setting the visited array
    frontier = filter(out, [&] (auto&& v) {
      bool expected = false;
      return (!visited[v]) && visited[v].compare_exchange_strong(expected, true);});
  }

  return frontiers;
}