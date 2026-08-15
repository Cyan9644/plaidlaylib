// Correctness test for external_bellman_ford / external_bellman_ford_fast
// (chunk_bellman_ford.h).
//
// Builds a few small, hand-constructed CSR graphs (not RMAT-generated, so this
// test exercises the solvers in isolation from external_rmat.h /
// direct_sample_sort -- see external_rmat_test.cpp for the generator's own
// correctness check) and cross-checks both solvers against the upstream
// parlaylib bellman_ford reference (deps/parlaylib-examples/bellman_ford.h) on
// the same graph, built independently as a DRAM weighted_graph.  Both take a
// TRANSPOSE (in-edge) graph and relax pull-style, so the same adjacency lists
// feed both sides directly.
//
// Cases: a simple directed path, a graph with a cheaper two-hop shortcut (so a
// naive single-relax-per-round bug would under-count rounds), and a
// disconnected graph (checks "unreached" sentinel handling on both solvers).
//
// Exits 0 iff every case passes for both solvers.

#include <unistd.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "parlay/primitives.h"

#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/examples/chunk_bellman_ford.h"
#include "parlaylib-examples/bellman_ford.h"
#include "utils/file_utils.h"

namespace {

// Anything past this is "unreached" on either solver's sentinel encoding
// (upstream fills with numeric_limits<long double>::max(); both out-of-core
// solvers do the same) -- matches bellman_ford.cpp's own threshold.
bool unreached(long double d) { return d > 1e15L; }

void cleanup_prefix(const std::string& prefix) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    unlink(GetFileName(prefix, d).c_str());
}

// One test case: `in_edges[v]` lists v's (source, weight) in-edges -- the
// transpose adjacency both solvers pull from.  `expected[v]` is the distance
// from vertex 0, or -1 to mean "unreached".
struct Case {
  std::string label;
  std::vector<std::vector<std::pair<size_t, long double>>> in_edges;
  std::vector<long double> expected;  // < 0 means unreached
};

// Builds the transpose chunk_csr for `c.in_edges` (CSR row v = v's in-edges,
// matching chunk_bellman_ford.h's expected layout) and the equivalent
// upstream weighted_graph (same transpose, in-DRAM), so both sides solve the
// identical problem.
bool run_case(const Case& c) {
  const size_t n = c.in_edges.size();
  const std::string prefix = "bft_edges_" + c.label;

  std::cout << "\n=== " << c.label << " (n=" << n << ") ===\n";

  parlay::sequence<size_t> degree_scan(n + 1, 0);
  for (size_t v = 0; v < n; v++)
    degree_scan[v + 1] = degree_scan[v] + c.in_edges[v].size();

  std::vector<weighted_edge> flat;
  flat.reserve(degree_scan[n]);
  for (size_t v = 0; v < n; v++)
    for (const auto& [src, w] : c.in_edges[v])
      flat.push_back(weighted_edge{src, w});

  chunk_csr graph;
  graph.degree_scan = degree_scan;
  graph.edges =
      plaid::to_chunk_seq(parlay::to_sequence(flat), prefix);

  // Same transpose adjacency, in DRAM, for the upstream reference.
  auto WG = parlay::tabulate(n, [&](size_t v) {
    return parlay::to_sequence(c.in_edges[v]);
  });

  const size_t start = 0;
  bool ok = true;

  size_t rounds = 0;
  auto d_slow = external_bellman_ford(graph, start, &rounds);
  size_t fast_rounds = 0;
  auto d_fast = external_bellman_ford_fast(graph, start, &fast_rounds);
  auto d_ref_opt = bellman_ford<long double>(start, WG);

  CHECK(d_ref_opt.has_value())
      << "reference bellman_ford did not converge for case " << c.label;
  const auto& d_ref = *d_ref_opt;

  auto check_against_ref = [&](const std::string& name,
                               const parlay::sequence<long double>& d) {
    if (d.size() != n) {
      std::cout << "  FAIL (" << name << "): did not converge (got " << d.size()
                << " distances, expected " << n << ")\n";
      ok = false;
      return;
    }
    for (size_t v = 0; v < n; v++) {
      const bool ref_u = unreached(d_ref[v]);
      const bool got_u = unreached(d[v]);
      if (ref_u != got_u || (!ref_u && d[v] != d_ref[v])) {
        std::cout << "  FAIL (" << name << ") at vertex " << v << ": got "
                  << (double)d[v] << " (unreached=" << got_u << "), expected "
                  << (double)d_ref[v] << " (unreached=" << ref_u << ")\n";
        ok = false;
      }
    }
  };

  check_against_ref("slow", d_slow);
  check_against_ref("fast", d_fast);

  // Also check against the case's own hand-computed expectations, as an
  // independent statement of intent (not derived from either solver).
  for (size_t v = 0; v < n && ok; v++) {
    const bool exp_u = c.expected[v] < 0;
    if (!exp_u && (d_slow.size() != n || d_slow[v] != c.expected[v])) {
      std::cout << "  FAIL (slow vs hand-computed) at vertex " << v << ": got "
                << (d_slow.size() == n ? (double)d_slow[v] : -1.0)
                << ", expected " << (double)c.expected[v] << "\n";
      ok = false;
    }
    if (exp_u && d_slow.size() == n && !unreached(d_slow[v])) {
      std::cout << "  FAIL (slow vs hand-computed) at vertex " << v
                << ": expected unreached, got " << (double)d_slow[v] << "\n";
      ok = false;
    }
  }

  std::cout << "  " << (ok ? "PASS" : "FAIL") << "  (slow " << rounds
            << " rounds, fast " << fast_rounds << " rounds)\n";
  cleanup_prefix(prefix);
  return ok;
}

}  // namespace

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  RaiseFdLimit();

  std::vector<Case> cases;

  // Simple path 0 -> 1 -> 2 -> 3, weights 1, 2, 3.
  cases.push_back(
      {"path",
       {/*v0*/ {},
        /*v1*/ {{0, 1.0L}},
        /*v2*/ {{1, 2.0L}},
        /*v3*/ {{2, 3.0L}}},
       {0.0L, 1.0L, 3.0L, 6.0L}});

  // 0 -> 1 direct (w=5) vs 0 -> 2 -> 1 (w=1+1=2): the shortcut must win,
  // which requires at least 2 relaxation rounds.
  cases.push_back({"shortcut",
                   {/*v0*/ {},
                    /*v1*/ {{0, 5.0L}, {2, 1.0L}},
                    /*v2*/ {{0, 1.0L}}},
                   {0.0L, 2.0L, 1.0L}});

  // 0 -> 1 (w=1); vertex 2 isolated (no in-edges, no out-edges) -> unreached.
  cases.push_back({"disconnected",
                   {/*v0*/ {},
                    /*v1*/ {{0, 1.0L}},
                    /*v2*/ {}},
                   {0.0L, 1.0L, -1.0L}});

  bool all_pass = true;
  for (const auto& c : cases) all_pass &= run_case(c);

  std::cout << "\n" << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
  return all_pass ? 0 : 1;
}
