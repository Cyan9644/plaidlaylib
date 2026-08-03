// nested_bfsExample — out-of-core BFS on the nested_seq substrate (the FUSED
// direction-optimizing variant, NestedBFSDirOpt(..., fused=true)) vs the
// in-memory parlaylib reference (deps/parlaylib-examples/BFS.h) on the same
// graph, from the same start vertex, with the distances cross-checked.
//
// The graph is an undirected random graph (a connected backbone path plus
// random shortcuts) built in DRAM as an adjacency list, then tabulated onto the
// drives as a nested_seq (row v = v's neighbours).  BFS itself runs out-of-core;
// the in-memory baseline is parlaylib's BFS() on the DRAM adjacency.
//
// Only the FUSED out-of-core BFS is benchmarked here (per request); the other
// nested BFS variants live in ChunkSequence/nested_bfs.h and are exercised by
// nestedBfsTest / bin/nestedBfsCompare.
//
// NOTE: graph CONSTRUCTION is DRAM-bounded (the adjacency is materialized in
// memory before being written out), so the example is gated by a RAM budget and
// cannot sweep past DRAM — the out-of-core value it demonstrates is in the BFS
// pass, not the build.  (An out-of-core symmetric graph generator, like
// external_rmat.h for the CSR bfsExample, would lift that — future work.)
//
//   usage: nested_bfsExample [global --flags] [n] [avg_degree]
//     n           number of vertices        (default 200000)
//     avg_degree  average (undirected) degree (default 8)
//
//   CSV,n,m,build_s,bfs_s,inmem_bfs_s,reached,throughput_gb_s
//     m = directed edge count; throughput = edge bytes (m*8) / bfs_s (dataset
//     size / op time, the convention the other examples use, not literal touched
//     bytes).  inmem_bfs_s is blank when the baseline is skipped past budget.

#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "ChunkSequence/nested_bfs.h"
#include "ChunkSequence/nested_seq.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "parlaylib-examples/BFS.h"  // upstream in-memory baseline (global BFS<>)
#include "utils/command_line.h"
#include "utils/file_utils.h"

using Clock = std::chrono::steady_clock;
static constexpr uint64_t INF = std::numeric_limits<uint64_t>::max();

static double elapsed(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}
static double to_gb(size_t bytes) { return (double)bytes / (1024.0 * 1024.0 * 1024.0); }
static std::string f9(double v) {
  std::ostringstream o;
  o << std::setprecision(9) << v;
  return o.str();
}
static void cleanup_prefix(const std::string& prefix) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    unlink(GetFileName(prefix, d).c_str());
}

// Undirected random graph in DRAM: connected backbone + random shortcuts.
static parlay::sequence<parlay::sequence<uint64_t>> build_graph(size_t V,
                                                                size_t avg_degree) {
  std::vector<std::vector<uint64_t>> adj(V);
  auto add = [&](uint64_t a, uint64_t b) {
    if (a != b) {
      adj[a].push_back(b);
      adj[b].push_back(a);
    }
  };
  for (uint64_t i = 0; i + 1 < V; i++) add(i, i + 1);
  std::mt19937_64 rng(2024);
  std::uniform_int_distribution<uint64_t> pick(0, V ? V - 1 : 0);
  const size_t extra = V * avg_degree / 2;
  for (size_t e = 0; e < extra; e++) add(pick(rng), pick(rng));

  parlay::sequence<parlay::sequence<uint64_t>> G(V);
  for (size_t v = 0; v < V; v++)
    G[v] = parlay::sequence<uint64_t>(adj[v].begin(), adj[v].end());
  return G;
}

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  RaiseFdLimit();

  const size_t V = (argc > 1) ? std::stoull(argv[1]) : 200000;
  const size_t avg_degree = (argc > 2) ? std::stoull(argv[2]) : 8;

  // RAM budget gate, evaluated before building anything: the whole example is
  // DRAM-bounded (adjacency + baseline + the transient std::vector graph).
  const size_t m_est = V * avg_degree;  // ~ directed edges
  const size_t est = V * 64 + m_est * 24;
  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys / 2;
  if (const char* e = getenv("NESTED_BFS_BUDGET_BYTES")) budget = std::stoull(e);
  if (budget != 0 && est > budget) {
    std::cout << "SKIPPED: n=" << V << " needs ~" << std::setprecision(3)
              << to_gb(est) << " GB DRAM (adjacency + baseline), past the "
              << to_gb(budget) << " GB budget (NESTED_BFS_BUDGET_BYTES)\n";
    return 0;  // no CSV line: run_benches drops the point and keeps sweeping
  }

  std::cout << "Building " << V << "-vertex graph (avg degree " << avg_degree
            << ") in DRAM..." << std::flush;
  auto G = build_graph(V, avg_degree);
  std::cout << " done\n";

  const std::string gp = "nbfs_ex_g";
  std::cout << "Tabulating graph onto drives (nested_seq)..." << std::flush;
  auto t0 = Clock::now();
  auto g = ChunkSequenceOps::NestedTabulate<uint64_t>(
      V, gp, [&](size_t v) { return G[v]; });
  const double build_s = elapsed(t0);
  const size_t m = g.total_seqs() == 0 ? 0 : g.seq_len_scan.back();
  std::cout << " done (" << m << " directed edges, " << std::fixed
            << std::setprecision(4) << build_s << "s)\n";

  const size_t source = 0;

  std::cout << "Running out-of-core BFS (fused NestedBFSDirOpt)..." << std::flush;
  t0 = Clock::now();
  parlay::sequence<uint64_t> dist =
      ChunkSequenceOps::NestedBFSDirOpt(g, source, /*fused=*/true);
  const double bfs_s = elapsed(t0);
  size_t reached = 0, maxd = 0;
  for (uint64_t d : dist)
    if (d != INF) {
      reached++;
      if (d > maxd) maxd = d;
    }
  const double gb_s = to_gb(m * sizeof(uint64_t)) / bfs_s;
  std::cout << " done (" << reached << "/" << V << " reached, max dist " << maxd
            << ", " << std::setprecision(4) << bfs_s << "s, "
            << std::setprecision(2) << gb_s << " GB/s)\n";

  std::cout << "Running in-memory BFS (parlaylib)..." << std::flush;
  t0 = Clock::now();
  auto frontiers = BFS(source, G);  // upstream: sequence of per-level frontiers
  const double inmem_bfs_s = elapsed(t0);
  std::cout << " done (" << std::setprecision(4) << inmem_bfs_s << "s)\n";

  // Cross-check: derive per-vertex distance from the baseline's frontiers.
  std::vector<uint64_t> distm(V, INF);
  for (size_t lvl = 0; lvl < frontiers.size(); lvl++)
    for (uint64_t v : frontiers[lvl]) distm[v] = lvl;
  bool agree = true;
  for (size_t v = 0; v < V && agree; v++)
    if (dist[v] != distm[v]) {
      std::cout << "*** MISMATCH at vertex " << v << ": out-of-core " << dist[v]
                << " != in-mem " << distm[v] << " ***\n";
      agree = false;
    }
  if (agree) std::cout << "cross-check OK: distances agree\n";

  std::cout << "CSV,n,m,build_s,bfs_s,inmem_bfs_s,reached,throughput_gb_s\n"
            << "CSV," << V << ',' << m << ',' << f9(build_s) << ',' << f9(bfs_s)
            << ',' << f9(inmem_bfs_s) << ',' << reached << ',' << f9(gb_s)
            << '\n';

  cleanup_prefix(gp);
  return agree ? 0 : 1;
}
