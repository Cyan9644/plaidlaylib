// Correctness test for the out-of-core RMAT generator
// (examples/external/graph_utils/external_rmat.h).
//
// external_rmat_symmetric_graph exists so bellman_fordExample can build a graph
// larger than DRAM; its whole justification is that it produces the SAME graph
// that graph_utils' rmat_symmetric_graph + add_weights builds in memory, just
// without ever holding it.  That claim is what this test pins down: for each of
// the three density regimes bellman_ford.cpp sweeps (avg_degree 2, 8, and n/2),
// build the graph both ways and compare
//
//   - the vertex count and the total edge count,
//   - every vertex's degree (via degree_scan), and
//   - every edge, element-wise: destination vertex AND exact long double
//   weight,
//     after sorting the in-memory row (the out-of-core rows come out sorted by
//     destination; the in-memory ones are hash-ordered by remove_duplicates,
//     which Bellman-Ford's min-relax cannot distinguish).
//
// It also checks the structural invariants the rest of the stack depends on:
// the returned edge chunk_seq must be index-ordered and dense-except-last (the
// delayed layer's chunk grid assumes it), and degree_scan must be a genuine
// exclusive prefix sum of length n+1.
//
//   usage: externalRmatTest [global --flags] [n]     (default 65536)
//
// n is deliberately large enough by default that the edge list spans several
// chunks, so the DensePackStream dedup pass is exercised across chunk seams --
// the one place a 1-element halo can go wrong.

#include "ChunkSequence/helper/graph_utils/external_rmat.h"

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "ChunkSequence/Primitives/materialize.h"
#include "ChunkSequence/Primitives/chunk_seq.h"
#include "absl/log/check.h"
#include "parlay/primitives.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"

namespace {

using vertex_utils = graph_utils<size_t>;

void cleanup_prefix(const std::string& prefix) {
  for (const std::string& dir : GetSSDList()) {
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
      const std::string name = e.path().filename().string();
      if (name.rfind(prefix, 0) == 0) std::filesystem::remove(e.path(), ec);
    }
  }
}

// Every chunk but the last must be full, and indices must be 0..k-1 in order:
// delayed::delay / segmented_reduce lay their chunk grid out on that basis.
bool check_dense(const chunk_seq& seq, size_t elem_size,
                 size_t expected_elems) {
  const size_t nc = seq.chunks.size();
  size_t total = 0;
  for (size_t i = 0; i < nc; i++) {
    const chunk& c = seq.chunks[i];
    if (c.index != i) {
      std::cout << "  FAIL: chunk " << i << " has index " << c.index << "\n";
      return false;
    }
    if (i + 1 < nc && c.used != CHUNK_SIZE) {
      std::cout << "  FAIL: interior chunk " << i << " holds " << c.used
                << " bytes, not a full " << CHUNK_SIZE << "\n";
      return false;
    }
    if (c.used % elem_size != 0) {
      std::cout << "  FAIL: chunk " << i << " holds " << c.used
                << " bytes, not a whole number of elements\n";
      return false;
    }
    total += c.used / elem_size;
  }
  if (total != expected_elems) {
    std::cout << "  FAIL: edge chunks hold " << total
              << " elements, degree_scan says " << expected_elems << "\n";
    return false;
  }
  return true;
}

bool run_case(size_t n_req, size_t avg_degree, const std::string& label) {
  const size_t m_req = avg_degree * n_req;
  const size_t n = size_t{1} << (int)std::round(std::log2((double)n_req));
  const std::string prefix = "rmat_test_" + label;

  std::cout << "\n=== " << label << ": n_req=" << n_req << " (n=" << n
            << ") avg_degree=" << avg_degree << " ===\n";

  chunk_csr graph = ExternalGraphUtils::external_rmat_symmetric_graph(
      n_req, m_req, prefix, /*minw=*/1.0L, /*maxw=*/20.0L);

  bool ok = true;
  if (graph.degree_scan.size() != n + 1) {
    std::cout << "  FAIL: degree_scan has " << graph.degree_scan.size()
              << " entries, expected " << (n + 1) << "\n";
    cleanup_prefix(prefix);
    return false;
  }
  if (graph.degree_scan[0] != 0) {
    std::cout << "  FAIL: degree_scan[0] = " << graph.degree_scan[0]
              << ", expected 0\n";
    ok = false;
  }
  for (size_t v = 0; v < n; v++) {
    if (graph.degree_scan[v] > graph.degree_scan[v + 1]) {
      std::cout << "  FAIL: degree_scan not monotone at vertex " << v << "\n";
      ok = false;
      break;
    }
  }
  const size_t m = graph.degree_scan[n];
  std::cout << "  built " << m << " edges over " << graph.edges.chunks.size()
            << " chunks\n";

  ok &= check_dense(graph.edges, sizeof(weighted_edge), m);

  // Reference graph, in DRAM.
  auto G = vertex_utils::rmat_symmetric_graph((long)n_req, (long)m_req);
  auto WG = vertex_utils::add_weights<long double>(G, 1.0L, 20.0L);
  G = decltype(G){};

  if (WG.size() != n) {
    std::cout << "  FAIL: in-memory graph has " << WG.size()
              << " vertices, expected " << n << "\n";
    ok = false;
  } else {
    size_t m_ref = 0;
    for (size_t v = 0; v < n; v++) m_ref += WG[v].size();
    if (m_ref != m) {
      std::cout << "  FAIL: in-memory graph has " << m_ref
                << " edges, out-of-core has " << m << "\n";
      ok = false;
    }
  }

  if (ok) {
    auto edges = ChunkSequenceOps::materialize<weighted_edge>(graph.edges);
    CHECK(edges.size() >= m)
        << "materialize returned " << edges.size() << " < " << m << " edges";
    for (size_t v = 0; v < n && ok; v++) {
      std::vector<std::pair<size_t, long double>> ref(WG[v].begin(),
                                                      WG[v].end());
      std::sort(ref.begin(), ref.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
      const size_t deg = graph.degree_scan[v + 1] - graph.degree_scan[v];
      if (deg != ref.size()) {
        std::cout << "  FAIL: vertex " << v << " degree " << deg
                  << " out-of-core vs " << ref.size() << " in-memory\n";
        ok = false;
        break;
      }
      for (size_t k = 0; k < deg; k++) {
        const weighted_edge& e = edges[graph.degree_scan[v] + k];
        if (e.connecting_vertex != ref[k].first ||
            e.edge_weight != ref[k].second) {
          std::cout << "  FAIL: vertex " << v << " edge " << k
                    << ": out-of-core (" << e.connecting_vertex << ", "
                    << (double)e.edge_weight << ") vs in-memory ("
                    << ref[k].first << ", " << (double)ref[k].second << ")\n";
          ok = false;
          break;
        }
      }
    }
  }

  // The generator must leave no intermediates behind: only the edge files.
  for (const std::string& dir : GetSSDList()) {
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
      const std::string name = e.path().filename().string();
      if (name.rfind(prefix + "_gen", 0) == 0 ||
          name.rfind(prefix + "_srt", 0) == 0) {
        std::cout << "  FAIL: generator left intermediate " << name << "\n";
        ok = false;
      }
    }
  }

  std::cout << "  " << (ok ? "PASS" : "FAIL") << "\n";
  cleanup_prefix(prefix);
  return ok;
}

}  // namespace

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  // direct_sample_sort opens many files per worker; the same fd-limit lift
  // every external example applies.
  RaiseFdLimit();
  // Both sides of every comparison build a DRAM reference graph, so this test
  // caps its own sizes rather than taking `make test TEST_ARGS=<n>` at face
  // value: the dense case's avg_degree is n/2, so its edge count grows as
  // n^2/2 and a few million vertices would be tens of billions of edges.
  const size_t requested = (argc > 1) ? std::stoull(argv[1]) : 65536;
  const size_t n_req = std::min<size_t>(requested, 1u << 20);
  const size_t n_dense = std::min<size_t>(requested, 4096);
  if (n_req != requested || n_dense != requested)
    std::cout << "(capped: sparse/balanced at n_req=" << n_req
              << ", dense at n_req=" << n_dense
              << " -- both sides need a DRAM reference graph)\n";

  bool all_pass = true;
  all_pass &= run_case(n_req, 2, "sparse");
  all_pass &= run_case(n_req, 8, "balanced");
  all_pass &= run_case(n_dense, std::max<size_t>(1, n_dense / 2), "dense");

  // A single-chunk graph too: exercises the paths where the whole edge list
  // fits one chunk, so the dedup pass never sees a seam and the sort takes
  // its small-n base case.
  all_pass &= run_case(256, 2, "tiny");

  std::cout << "\n" << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
  return all_pass ? 0 : 1;
}
