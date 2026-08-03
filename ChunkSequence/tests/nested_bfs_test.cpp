#include <unistd.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <vector>

#include "ChunkSequence/nested_bfs.h"
#include "ChunkSequence/nested_seq.h"
#include "parlay/sequence.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"

static constexpr uint64_t INF = std::numeric_limits<uint64_t>::max();

static void cleanup_prefix(const std::string& prefix) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    unlink(GetFileName(prefix, d).c_str());
}

// Build a deterministic undirected graph in DRAM as an adjacency list.
//   - a backbone path 0-1-...-(V-2) guarantees a reachable component with real
//     depth (so BFS runs several rounds);
//   - random extra undirected edges add shortcuts;
//   - vertex V-1 is left ISOLATED (empty adjacency) to exercise unreachable/INF.
static std::vector<std::vector<uint64_t>> build_graph(size_t V) {
  std::vector<std::vector<uint64_t>> adj(V);
  auto add_undirected = [&](uint64_t a, uint64_t b) {
    if (a == b) return;
    adj[a].push_back(b);
    adj[b].push_back(a);
  };
  if (V >= 2)
    for (uint64_t i = 0; i + 2 < V; i++) add_undirected(i, i + 1);  // backbone

  std::mt19937_64 rng(12345);
  const size_t connected = (V >= 1) ? V - 1 : 0;  // last vertex stays isolated
  if (connected > 1) {
    std::uniform_int_distribution<uint64_t> pick(0, connected - 1);
    const size_t extra = connected * 3;  // avg degree ~6
    for (size_t e = 0; e < extra; e++) add_undirected(pick(rng), pick(rng));
  }
  return adj;
}

// Reference BFS in DRAM (queue), unreachable = INF.
static std::vector<uint64_t> golden_bfs(
    const std::vector<std::vector<uint64_t>>& adj, uint64_t src) {
  const size_t V = adj.size();
  std::vector<uint64_t> dist(V, INF);
  if (V == 0) return dist;
  dist[src] = 0;
  std::queue<uint64_t> q;
  q.push(src);
  while (!q.empty()) {
    uint64_t u = q.front();
    q.pop();
    for (uint64_t v : adj[u])
      if (dist[v] == INF) {
        dist[v] = dist[u] + 1;
        q.push(v);
      }
  }
  return dist;
}

static bool cmp_dist(const char* who, const parlay::sequence<uint64_t>& got,
                     const std::vector<uint64_t>& exp) {
  if (got.size() != exp.size()) {
    std::cout << "FAIL " << who << " size " << got.size() << " != " << exp.size()
              << "\n";
    return false;
  }
  for (size_t v = 0; v < exp.size(); v++)
    if (got[v] != exp[v]) {
      std::cout << "FAIL " << who << " dist[" << v << "] got=" << got[v]
                << " expected=" << exp[v] << "\n";
      return false;
    }
  return true;
}

// Run pull NestedBFS AND push NestedBFSPush from src; both must equal the DRAM
// golden (and therefore each other — a differential check of the two variants).
static bool run_source(const ChunkSequenceOps::nested_seq<uint64_t>& g,
                       const std::vector<std::vector<uint64_t>>& adj,
                       uint64_t src) {
  std::cout << "  source=" << src << ": " << std::flush;
  std::vector<uint64_t> exp = golden_bfs(adj, src);

  parlay::sequence<uint64_t> pull = ChunkSequenceOps::NestedBFS(g, src);
  parlay::sequence<uint64_t> pull_unfused =
      ChunkSequenceOps::NestedBFSPullUnfused(g, src);
  std::vector<char> dirs;  // per-level: 1=push gather, 0=pull gather
  parlay::sequence<uint64_t> push = ChunkSequenceOps::NestedBFSPush(
      g, src, ChunkSequenceOps::GatherDir::kAuto, &dirs);
  parlay::sequence<uint64_t> push_fused = ChunkSequenceOps::NestedBFSPushFused(
      g, src, ChunkSequenceOps::GatherDir::kAuto);
  parlay::sequence<uint64_t> diropt_unfused =
      ChunkSequenceOps::NestedBFSDirOpt(g, src, /*fused=*/false);
  parlay::sequence<uint64_t> diropt_fused =
      ChunkSequenceOps::NestedBFSDirOpt(g, src, /*fused=*/true);

  bool ok = cmp_dist("pull-fused", pull, exp) &
            cmp_dist("pull-unfused", pull_unfused, exp) &
            cmp_dist("push-materialized", push, exp) &
            cmp_dist("push-fused", push_fused, exp) &
            cmp_dist("diropt-unfused", diropt_unfused, exp) &
            cmp_dist("diropt-fused", diropt_fused, exp);
  if (!ok) return false;

  uint64_t max_finite = 0, reached = 0;
  for (uint64_t d : exp)
    if (d != INF) {
      reached++;
      if (d > max_finite) max_finite = d;
    }
  size_t n_push = 0, n_pull = 0;
  for (char c : dirs) (c ? n_push : n_pull)++;
  std::cout << "all 6 impls == golden  (reached " << reached << "/" << exp.size()
            << ", max dist " << max_finite << "; gather dirs: " << n_push
            << " push / " << n_pull << " pull)\n";
  return true;
}

// Force each gather direction and confirm both still match the golden — a
// direct check that the push (per-row) and pull (whole-chunk) read paths agree.
static bool run_forced_dirs(const ChunkSequenceOps::nested_seq<uint64_t>& g,
                            const std::vector<std::vector<uint64_t>>& adj,
                            uint64_t src) {
  std::cout << "  forced dirs from source=" << src << ": " << std::flush;
  std::vector<uint64_t> exp = golden_bfs(adj, src);
  auto forced_push = ChunkSequenceOps::NestedBFSPush(
      g, src, ChunkSequenceOps::GatherDir::kPush);
  auto forced_pull = ChunkSequenceOps::NestedBFSPush(
      g, src, ChunkSequenceOps::GatherDir::kPull);
  bool ok = cmp_dist("forced-push", forced_push, exp) &
            cmp_dist("forced-pull", forced_pull, exp);
  if (ok) std::cout << "both match golden\n";
  return ok;
}

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);

  size_t V = 5000;
  if (argc > 1) V = std::stoull(argv[1]);

  std::cout << "nested BFS test  |V|=" << V << "\n" << std::flush;

  auto adj = build_graph(V);
  const std::string prefix = "nested_bfs_graph";
  // inner sequence v = neighbour list of v -> a nested_seq graph.
  auto g = ChunkSequenceOps::NestedTabulate<uint64_t>(
      V, prefix, [&](size_t v) {
        return parlay::sequence<uint64_t>(adj[v].begin(), adj[v].end());
      });

  bool all_pass = true;
  all_pass &= run_source(g, adj, 0);
  if (V > 1) all_pass &= run_source(g, adj, V / 2);
  if (V > 1) all_pass &= run_source(g, adj, V - 1);  // isolated source
  all_pass &= run_forced_dirs(g, adj, 0);

  // With a graph big enough to span several chunks, a BFS from a low-degree
  // start should exercise BOTH gather directions automatically: tiny early/late
  // frontiers pick push, the large middle frontier picks pull.
  if (V >= 200000) {
    std::vector<char> dirs;
    ChunkSequenceOps::NestedBFSPush(g, 0, ChunkSequenceOps::GatherDir::kAuto,
                                    &dirs);
    bool has_push = false, has_pull = false;
    for (char c : dirs) (c ? has_push : has_pull) = true;
    std::cout << "  auto direction mix: ";
    if (has_push && has_pull) {
      std::cout << "both push and pull occurred  OK\n";
    } else {
      std::cout << "FAIL: expected both directions, got "
                << (has_push ? "push " : "") << (has_pull ? "pull" : "") << "\n";
      all_pass = false;
    }
  }

  cleanup_prefix(prefix);
  std::cout << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
  return all_pass ? 0 : 1;
}
