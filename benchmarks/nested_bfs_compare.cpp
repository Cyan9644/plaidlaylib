// nested_bfs_compare — time two direction-optimizing BFS impls head-to-head.
// Both switch PUSH/PULL per level by frontier size (NestedBFSDirOpt); they
// differ only in whether the push and pull steps use their fused primitives:
//
//   unfused : push = NestedGather + NestedFlatten,  pull = NestedMap + NestedReduce
//   fused   : push = NestedGatherFlatten,           pull = NestedMapReduce
//
// Both compute the same BFS distances (verified).  Manual-only:
//     rm -f bin/nestedBfsCompare && make bin/nestedBfsCompare
//     bin/nestedBfsCompare [num_vertices] [avg_degree]

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ChunkSequence/nested_bfs.h"
#include "ChunkSequence/nested_seq.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"

static double secs(std::chrono::steady_clock::time_point a,
                   std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// Random undirected graph (backbone path + random shortcuts) as adjacency lists.
static std::vector<std::vector<uint64_t>> build_graph(size_t V, size_t avg_deg) {
  std::vector<std::vector<uint64_t>> adj(V);
  auto add = [&](uint64_t a, uint64_t b) {
    if (a != b) {
      adj[a].push_back(b);
      adj[b].push_back(a);
    }
  };
  for (uint64_t i = 0; i + 1 < V; i++) add(i, i + 1);  // connected backbone
  std::mt19937_64 rng(2024);
  std::uniform_int_distribution<uint64_t> pick(0, V ? V - 1 : 0);
  const size_t extra = V * avg_deg / 2;
  for (size_t e = 0; e < extra; e++) add(pick(rng), pick(rng));
  return adj;
}

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);

  size_t V = 2000000;      // vertices
  size_t avg_deg = 8;      // average degree
  if (argc > 1) V = std::stoull(argv[1]);
  if (argc > 2) avg_deg = std::stoull(argv[2]);

  auto adj = build_graph(V, avg_deg);
  const std::string src_prefix = "nbfsc_graph";
  auto g = ChunkSequenceOps::NestedTabulate<uint64_t>(
      V, src_prefix, [&](size_t v) {
        return parlay::sequence<uint64_t>(adj[v].begin(), adj[v].end());
      });
  const size_t edges = g.total_seqs() == 0 ? 0 : g.seq_len_scan.back();
  std::cout << "nested BFS: direction-optimizing, unfused vs fused\n"
            << "  |V|=" << V << ", |E(dir)|=" << edges << " ("
            << edges * sizeof(uint64_t) / 1e6 << " MB), " << g.chunks.size()
            << " chunks, parlay workers=" << parlay::num_workers() << "\n"
            << std::flush;

  const uint64_t source = 0;

  std::vector<char> dirs;  // per-level 1=push / 0=pull (same schedule for both)
  auto a0 = std::chrono::steady_clock::now();
  auto unfused = ChunkSequenceOps::NestedBFSDirOpt(g, source, /*fused=*/false, &dirs);
  auto a1 = std::chrono::steady_clock::now();
  auto fused = ChunkSequenceOps::NestedBFSDirOpt(g, source, /*fused=*/true);
  auto a2 = std::chrono::steady_clock::now();

  bool agree = unfused.size() == fused.size();
  for (size_t i = 0; i < unfused.size() && agree; i++)
    if (unfused[i] != fused[i]) agree = false;

  size_t n_push = 0, n_pull = 0;
  for (char c : dirs) (c ? n_push : n_pull)++;
  const double un_s = secs(a0, a1), fu_s = secs(a1, a2);

  std::cout << "  schedule: " << dirs.size() << " levels (" << n_push
            << " push / " << n_pull << " pull)\n"
            << "  unfused (Gather+Flatten / Map+Reduce) : " << un_s << " s\n"
            << "  fused   (GatherFlatten / MapReduce)   : " << fu_s << " s\n"
            << "  speedup (unfused / fused)             : " << (un_s / fu_s)
            << "x\n"
            << "  agree: " << (agree ? "yes" : "NO") << "\n";
  std::cout << "CSV,V,edges,workers,levels,pushes,pulls,unfused_s,fused_s,speedup,agree\n"
            << "CSV," << V << "," << edges << "," << parlay::num_workers() << ","
            << dirs.size() << "," << n_push << "," << n_pull << "," << un_s << ","
            << fu_s << "," << (un_s / fu_s) << "," << (agree ? 1 : 0) << "\n";

  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    unlink(GetFileName(src_prefix, d).c_str());
  return agree ? 0 : 1;
}
