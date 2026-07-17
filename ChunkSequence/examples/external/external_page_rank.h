#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/delayed.h>

// **************************************************************
// The pagerank algorithm on a sparse graph
// **************************************************************

//   using row = parlay::sequence<element>;
// using element = std::pair<int,float>;
//   using sparse_matrix = parlay::sequence<row>;

//one thing we need to consider here is how we actually want to represent the sparse matrix 
// parlay just represnets it using a normal sequence of rows, so i think that's best for an accurate comparison

//in this case, how do we actually want to reperesnet the in-memory portion of the matrix?
//I think the most effective way is probably to use a parlay sequence of external sequences, with each external sequence representing the elements of a row

//remember that a sparse matrix is essentially a sparse adjacency matrix, so we can represent the matrix itself with compressed sparse row

//the basic approach we're looking for is a streaming one -- we need to read all chunks at the beginning and find out 
//where they actually belong later on to not read multiple times from the same chunk
//to actually do this, we're going to want a residuals bucket that contains elements from those rows which have not yet had all of their elements streamed
//so that when they do come in, they can complete while new rows are being processed and I/O is outgoing




using matrix = chunk_csr;



using vector = parlay::sequence<double>;




//our assumption here is that the actual sparse matrix itself cannot fit in memory, but the vectors can
//one note is that this is obviously going to be slow, since it runs purely at I/O speed since to do any computation, you need to materialize rows
// sparse matrix vector multiplication
//the only real work in this algorithm is sparse matrix vector multiplication
//the original pagerank algorithm uses unweighted edges, so we'll assume that the "edge weights" here are actually the values of the elements in the matrix
template <typename matrix>
auto vector_multiply(matrix const& graph, vector const& vec){

return parlay::map(graph, [&](auto const& r){//we want to map over each "vertex," which is actually a row in the matrix representation


auto per_edge = ChunkSequenceOps::delayed::map(
    ChunkSequenceOps::delayed::delay<weighted_edge>(graph.edges),
    [&](weighted_edge e) {return e.edge_weight;}); //gives us a map to the value of each of the actual elements in the matrix 

    //we want this to compute the reduce of the elements of this row i, i.e. those "edges" corresponding to graph.edges[graph.degree_scan[i] to graph.degree_scan[i+1]]

auto pass = ChunkSequenceOps::delayed::segmented_reduce(per_edge, graph.degree_scan, [&](weighted_edge i, vector& vec, size_t degree){ //degree here is the position of the element with respect to how it would be represented in a sparse matrix, that is its position in a row which is just the offset of the degrees in the csr representation + the number of passed vertices that our current vertex did not have an edge to

    return i.edge_weight * vec[degree];

//pass should now have the streamed reduction of all the elements computed by multiplying the elements of the "row", which is really just a section of the csr corresponding to a given vertex, and the vector, with any non-found values in the csr representation being treated as 0 and skipped for the computation
return pass;

});



});



//   return parlay::map(mat, [&] (auto const& r){
//     if (r.size() < 100) {//
//         parlay::sequence seq = ChunkSequenceOps::materialize(r);
//       double result = 0.0;
//       for(auto e : seq) result += vec[e.first] * e.second;
//       return result;
//     }
//     return parlay::reduce(parlay::delayed::map(ChunkSequenceOps::materialize(r), [&] (auto e) {
//       return vec[e.first] * e.second;}));},100);
}


template <typename matrix>
vector pagerank(matrix const& mat, int iters) {
  double d = .85; // damping factor
  long n = mat.size();
  vector v(n,1.0/n);
  for (int i=0; i < iters; i++) {
    auto a = vector_multiply(mat, v);
    v = parlay::map(a, [&] (double a) {return (1-d)/n + d * a;});
  }
  return v;
}