#include <limits>
#include <optional>

#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/io.h>

#include "ChunkSequence/ExternalGraph/external_compressed_sparse_row.h"
#include "ChunkSequence/chunk_segmented_reduce.h"


 //using weighted_vertices = parlay::sequence<std::pair<vertex,wtype>>;
// **************************************************************
// Parallel Bellman Ford
// Returns an optional, which is empty if there is a negative weight
// cycle, or otherwise returns the distance to each vertex.
// **************************************************************


//Note on parallel bellman-ford in general:
//we accept transpose graphs, so the edges are just reversed in order.
//this is to do pull-based bellman ford, which does not have write conflicts because the reads are on the shared data rather than the writes





// //let's take a look at how bellman-ford actually works



// //the basic process of bellman ford is to relax all edges number of vertices times
// //our check is actually based on the edges of the graph rather than the vertices
// //so we check for each edge whether d[other vertex] + d[other_vertex, target_vertex] < d[target_vertex]
// //potentially n-1 iterations since the graph could just be a line, so you'd need to relax to the end point


// //ok, so let's think about how we want to do this in csr.
// //first let's make a parlay implementation that works with csr


// //the class csr_graph doesn't actually exist, it's just to show how this might be implemneted in parlay
// parlay::sequence<weight> csr_bellman_ford_paired(csr_graph graph, vertex start){

// auto N = graph.degree_scan;
// parlay::sequence<std::pair<size_t, weight>> F = graph.edges;

// size_t n = graph.num_vertices;
// size_t max_size = (size_t)-1;
// parlay::sequence<long double> d(n, max_size);

// d[start] = 0;
// auto iterate = parlay::iota(n);
// for(int i = 0; i < n; i++){


// auto pass = parlay::map(iterate, [&](size_t v){ //iterate over the csr_graph by its vertex "adjacency list," which here is really just a size_t
//   //since there's no physical adjacency list

// //yeah this should probably be a pair in the external sequence proper if we need to cut like this

// auto adjacent = parlay::cut(F, N[v], N[v+1]); //get the adjacency list for this vertex
// // auto corresponding_edge_weights = parlay::cut(k, N[v], N[v+1]);
// return parlay::reduce(parlay::delayed_tabulate(adjacent.size(), [&](size_t e){

//   return d[adjacent[e].first] + adjacent[e].second;

// }), parlay::minimum<long double>());

// });

// pass[start] = 0;


// if(pass == d){
//   return d;
// }
// d= std::move(pass);
// }

// return parlay::sequence<weight>();


// }



//ok, now that we have a semi-working parlay method, let's think about how we can actually port this to external memory, starting with this same outline:

//1st optimization is to remove the iota call, since we don't want to instantiate it on disk.
//so instead of chunk_map, we'll use chunk_tabulate

//one thing that's also important is that we have a write-free algorithm -- because our (and most) csr is inherently read-only,
//we should use a delayed cut to avoid instantiating the subsequences on disk

parlay::sequence<weight> external_bellman_ford(chunk_csr& graph, vertex start){

auto N = graph.degree_scan;
//you can't materialize the edge list, which is the whole point
// parlay::sequence<std::pair<size_t, weight>> F =ChunkSequenceOps::materialize<std::pair<size_t, weight>>(graph.edges);

chunk_seq& F = graph.edges; //one point of note here is that chunk_sequence doesn't accept a particular type.
//we actually want these edges to be pairs of size_t, weight to indicate their destination and weighting

size_t n = graph.degree_scan.size() - 1;
// size_t max_size = (size_t)-1;
auto max_size = std::numeric_limits<long double>::max();
parlay::sequence<long double> d(n, max_size);

d[start] = 0;

// One reusable read context per parlay worker, allocated ONCE for the whole
// algorithm's lifetime (outside the rounds loop, not per round/vertex) so the
// edge file's fd + CHUNK_SIZE buffers are opened/allocated once and reused
// for every vertex of every round instead of once per delayed-cut call (see
// SequentialReadContext in chunk_delayed.h for why that matters). Zero
// synchronization needed: a parlay task runs uninterrupted on one worker,
// the same per-worker-slot idiom chunk_partition.h / chunk_count_sort.h use.
std::vector<ChunkSequenceOps::delayed::SequentialReadContext> ctxs(
    std::max<size_t>(1, parlay::num_workers()));

// auto iterate = parlay::iota(n);
for(size_t i = 0; i < n; i++){


auto pass = parlay::tabulate(n, [&](size_t v){ //iterate over the csr_graph by its vertex "adjacency list," which here is really just a size_t
  //since there's no physical adjacency list

//yeah this should probably be a pair in the external sequence proper if we need to cut like this

//ChunkSequenceOps::delayed::materialize should instantiate the delayed sequence
//one key point is that we need to have a sequential materialize here since we're already calling it in tabulate
//it might also help to use sequential versions of the other methods here

//this is not actually a complete delay operation, it just prevents the cut from writing back intermediates
auto& ctx = ctxs[parlay::worker_id()];
auto adjacent = ChunkSequenceOps::delayed::sequential_materialize(ChunkSequenceOps::delayed::cut<weighted_edge>(F, N[v], N[v+1]), ctx); //get the adjacency list for this vertex
// auto corresponding_edge_weights = parlay::cut(k, N[v], N[v+1]);
return parlay::reduce(parlay::delayed_tabulate(adjacent.size(), [&](size_t e){

  return d[adjacent[e].connecting_vertex] + adjacent[e].edge_weight;

}), parlay::minimum<long double>());

});

//ChunkSequeneOps::tabulate returns a chunk_sequence disk that lives on disk, sadly
//however, our assumption is that the EDGES cannot live in memory, not that the vertices cannot;
//this means we
//  actually need a version of tabulate that returns the result in-memory, which is why we use parlay tabulate instead of chunk_tabulate

pass[start] = 0;


if(pass == d){
  return d;
}
d= std::move(pass);
}

return parlay::sequence<weight>();


}


// **************************************************************
// Fast variant: identical algorithm/convergence check as
// external_bellman_ford above, but relaxes all vertices for a round with a
// single streaming, io_uring-parallel pass over the edge chunk_seq
// (ChunkSegmentedReduce) rather than one blocking pread per vertex.
// external_bellman_ford (above) no longer pays reader-setup cost per vertex
// either -- it shares a SequentialReadContext (fd cache + buffer pool) across
// all vertices/rounds via one context per parlay worker -- but each vertex
// there is still a single blocking pread on whichever worker picks it up, so
// this variant's real advantage is genuine per-round I/O parallelism (queue
// depth across reader_threads io_uring rings) rather than avoiding reader
// setup. Which wins depends on the graph: many small blocking preads spread
// across workers vs. one wide streaming pass with deep read pipelining.
// **************************************************************
parlay::sequence<weight> external_bellman_ford_fast(chunk_csr& graph, vertex start){

auto N = graph.degree_scan;
chunk_seq& F = graph.edges;

size_t n = graph.degree_scan.size() - 1;
auto max_size = std::numeric_limits<long double>::max();
parlay::sequence<long double> d(n, max_size);

d[start] = 0;

struct MinDistMonoid {
    long double identity = std::numeric_limits<long double>::max();
    long double operator()(long double a, long double b) const { return std::min(a, b); }
};

for(size_t i = 0; i < n; i++){

auto pass = ChunkSequenceOps::ChunkSegmentedReduce<weighted_edge, long double>(
    F, N,
    [&](const weighted_edge& e) { return d[e.connecting_vertex] + e.edge_weight; },
    MinDistMonoid{});

pass[start] = 0;

if(pass == d){
  return d;
}
d = std::move(pass);
}

return parlay::sequence<weight>();

}


















// //the class csr_graph doesn't actually exist, it's just to show how this might be implemneted in parlay
// parlay::sequence<weight> csr_bellman_ford_disjoint(csr_graph graph, vertex start){

// auto N = graph.degree_scan;
// auto F = graph.adjacent;
// auto k = graph.edge_weights;

// size_t n = graph.num_vertices;
// size_t max_size = (size_t)-1;
// parlay::sequence<long double> d(n, max_size);

// d[start] = 0;
// auto iterate = parlay::iota(n);
// for(int i = 0; i < n; i++){


// auto pass = parlay::map(iterate, [&](size_t v){ //iterate over the csr_graph by its vertex "adjacency list," which here is really just a size_t
//   //since there's no physical adjacency list

// //yeah this should probably be a pair in the external sequence proper if we need to cut like this

// auto adjacent = parlay::cut(F, N[v], N[v+1]); //get the adjacency list for this vertex
// auto corresponding_edge_weights = parlay::cut(k, N[v], N[v+1]);
// return parlay::reduce(parlay::delayed_tabulate(adjacent.size(), [&](size_t e){

//   return d[adjacent[e]] + corresponding_edge_weights[e];

// }), parlay::minimum<long double>());

// });

// pass[start] = 0;


// if(pass == d){
//   return d;
// }
// d= std::move(pass);
// }

// return parlay::sequence<weight>();


// }










// // **************************************************************
// // An "Easter Egg"
// // Here is a variant of the Bellman-Ford algorithm that only updates
// // vertices when one of their in neighbors has changed.
// // It uses the standard technique if enough vertices have changed.
// // It can be significantly faster.  For example, on a graph with
// // unit-weight edges it does O(m) total work, i.e. no more than BFS.
// // Needs the transpose graph.
// // Yes, it is more complicated than trivial basic version.
// // **************************************************************
// template <typename wtype, typename vertex, typename weighted_graph>
// auto bellman_ford_lazy(vertex start, const weighted_graph& G,
//                        const weighted_graph& GT) {
//   using edge = std::pair<vertex, wtype>;
//   long n = GT.size();
//   static constexpr wtype max_d = std::numeric_limits<wtype>::max();

//   // used to detect if first to update a distance in a lazy round
//   auto visited = parlay::tabulate<std::atomic<bool>>(n, [&] (long i) {
//     return (i==start) ? true : false; });

//   // distances used when lazy (need to be atomic)
//   auto da = parlay::tabulate<std::atomic<wtype>>(n, [&] (long i) {
//     return (i==start) ? 0.0 : max_d; });

//   // distances used whey greedy
//   parlay::sequence<wtype> d(n, max_d);
//   d[start] = 0.0;

//   // initially just the start is active
//   parlay::sequence<vertex> active(1, start);
//   long cnt = 0;
//   long total = 0;
//   bool lazy = true;
//   int num_active = 1;

//   while (active.size() > 0 && ++cnt <= n) {
//     // different threshold if growing or shrinking
//     bool do_lazy = ((active.size() < num_active && active.size() * 2 < n)
//                     || active.size() * 8 < n);
//     num_active = active.size();

//     // *** use lazy version if number of active vertices is small enough
//     if (do_lazy) {
//       // if previous round was greedy, then need to copy from d to da
//       if (!lazy)
//         parlay::parallel_for(0, n, [&] (long i) {da[i] = d[i];});
//       lazy = true;

//       // get out edges of the active set along with the distance to the far side
//       auto edges = delayed::flatten(parlay::map(active, [&] (vertex u) {
//         return delayed::map(G[u], [&,u] (auto e) {
//           return std::pair{e.first, da[u] + e.second};});}));

//       // add to active for next round if first to reduced distance to v
//       active = delayed::to_sequence(delayed::map_maybe(edges, [&] (auto p) {
//         auto [v, de] = p;
//         bool expected = false;
//         return ((parlay::write_min(&da[v], de, std::less<float>())
//                  && !visited[v]
//                  && visited[v].compare_exchange_strong(expected, true))
//                 ? std::optional<vertex>{v}
//                 : std::nullopt);}));

//       // clear visited for next round
//       parlay::for_each(active, [&] (vertex v) { visited[v] = false;});
//     }

//       // *** use greedy if number of active vertices is large enough
//     else {

//       // if previous round was lazy copy from da to d
//       if (lazy)
//         parlay::parallel_for(0, n, [&] (long i) {d[i] = da[i];});
//       lazy = false;

//       // same as standard Bellman Ford
//       auto dn = parlay::map(GT, [&] (auto& ngh) {
//         return parlay::reduce(parlay::delayed_map(ngh, [&] (auto e) {
//           return d[e.first] + e.second;}), parlay::minimum<wtype>());});
//       dn[start] = 0.0;

//       // gather those that have changed into active
//       active = parlay::pack_index<vertex>(parlay::delayed_tabulate(n, [&] (long i) {
//         return dn[i] != d[i];}));
//       d = std::move(dn);
//     }
//   }

//   // if last round was lazy, then copy from da to d
//   if (lazy)
//     parlay::parallel_for(0, n, [&] (long i) { d[i] = da[i];});
//   return (cnt <= n) ? std::optional(std::move(d)) : std::nullopt;
// }













































// //original weighted_graph: using weighted_graph = parlay::sequence<weighted_vertices<wtype>>;
// //using weighted_vertices = parlay::sequence<std::pair<vertex,wtype>>;
// auto bellman_ford(size_t start_ID, const weighted_chunk_csr csr) {
//   long n = ChunkSequenceOps::size(csr); //total number of edges in the graph
//   parlay::sequence<long double> d(n, std::numeric_limits<long double>::max());
//   d[start] = 0.0;

//   for (int i=0; i < n; i++) {
//     auto dn = ChunkSequenceOps::map(csr.edge_weights, [&] (auto& ngh) {
//       return parlay::reduce(parlay::delayed_map(ngh, [&] (auto e) {
//         return d[e.first] + e.second;}), parlay::minimum<long double>());});
//     dn[start] = 0.0;
//     if (dn == d) return std::optional(d);
//     d = std::move(dn);
//   }
//   return std::optional<parlay::sequence<long double>>();
// }


















// //from parlaylib, will implement later
// // **************************************************************
// // An "Easter Egg"
// // Here is a variant of the Bellman-Ford algorithm that only updates
// // vertices when one of their in neighbors has changed.
// // It uses the standard technique if enough vertices have changed.
// // It can be significantly faster.  For example, on a graph with
// // unit-weight edges it does O(m) total work, i.e. no more than BFS.
// // Needs the transpose graph.
// // Yes, it is more complicated than trivial basic version.
// // **************************************************************
// template <typename wtype, typename vertex, typename weighted_graph>
// auto bellman_ford_lazy(vertex start, const weighted_graph& G,
//                        const weighted_graph& GT) {
//   using edge = std::pair<vertex, wtype>;
//   long n = GT.size();
//   static constexpr wtype max_d = std::numeric_limits<wtype>::max();

//   // used to detect if first to update a distance in a lazy round
//   auto visited = parlay::tabulate<std::atomic<bool>>(n, [&] (long i) {
//     return (i==start) ? true : false; });

//   // distances used when lazy (need to be atomic)
//   auto da = parlay::tabulate<std::atomic<wtype>>(n, [&] (long i) {
//     return (i==start) ? 0.0 : max_d; });

//   // distances used whey greedy
//   parlay::sequence<wtype> d(n, max_d);
//   d[start] = 0.0;

//   // initially just the start is active
//   parlay::sequence<vertex> active(1, start);
//   long cnt = 0;
//   long total = 0;
//   bool lazy = true;
//   int num_active = 1;

//   while (active.size() > 0 && ++cnt <= n) {
//     // different threshold if growing or shrinking
//     bool do_lazy = ((active.size() < num_active && active.size() * 2 < n)
//                     || active.size() * 8 < n);
//     num_active = active.size();

//     // *** use lazy version if number of active vertices is small enough
//     if (do_lazy) {
//       // if previous round was greedy, then need to copy from d to da
//       if (!lazy)
//         parlay::parallel_for(0, n, [&] (long i) {da[i] = d[i];});
//       lazy = true;

//       // get out edges of the active set along with the distance to the far side
//       auto edges = delayed::flatten(parlay::map(active, [&] (vertex u) {
//         return delayed::map(G[u], [&,u] (auto e) {
//           return std::pair{e.first, da[u] + e.second};});}));

//       // add to active for next round if first to reduced distance to v
//       active = delayed::to_sequence(delayed::map_maybe(edges, [&] (auto p) {
//         auto [v, de] = p;
//         bool expected = false;
//         return ((parlay::write_min(&da[v], de, std::less<float>())
//                  && !visited[v]
//                  && visited[v].compare_exchange_strong(expected, true))
//                 ? std::optional<vertex>{v}
//                 : std::nullopt);}));

//       // clear visited for next round
//       parlay::for_each(active, [&] (vertex v) { visited[v] = false;});
//     }

//       // *** use greedy if number of active vertices is large enough
//     else {

//       // if previous round was lazy copy from da to d
//       if (lazy)
//         parlay::parallel_for(0, n, [&] (long i) {d[i] = da[i];});
//       lazy = false;

//       // same as standard Bellman Ford
//       auto dn = parlay::map(GT, [&] (auto& ngh) {
//         return parlay::reduce(parlay::delayed_map(ngh, [&] (auto e) {
//           return d[e.first] + e.second;}), parlay::minimum<wtype>());});
//       dn[start] = 0.0;

//       // gather those that have changed into active
//       active = parlay::pack_index<vertex>(parlay::delayed_tabulate(n, [&] (long i) {
//         return dn[i] != d[i];}));
//       d = std::move(dn);
//     }
//   }

//   // if last round was lazy, then copy from da to d
//   if (lazy)
//     parlay::parallel_for(0, n, [&] (long i) { d[i] = da[i];});
//   return (cnt <= n) ? std::optional(std::move(d)) : std::nullopt;
// }