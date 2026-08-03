#ifndef NESTED_BFS_H
#define NESTED_BFS_H

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "ChunkSequence/ExternalPrimitives/materialize.h"
#include "ChunkSequence/chunk_filter.h"
#include "ChunkSequence/nested_gather.h"
#include "ChunkSequence/nested_ops.h"
#include "ChunkSequence/nested_seq.h"
#include "absl/log/check.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "utils/file_utils.h"

namespace ChunkSequenceOps {

// min monoid over vertex distances; identity is "unreachable".
struct NestedBFSMin {
  uint64_t identity = std::numeric_limits<uint64_t>::max();
  uint64_t operator()(uint64_t a, uint64_t b) const { return a < b ? a : b; }
};

// max (a.k.a. logical-OR over {0,1}) monoid, for the pull step's "does vertex v
// have any neighbour in the current frontier?" reduction.
struct NestedBFSMax {
  uint64_t identity = 0;
  uint64_t operator()(uint64_t a, uint64_t b) const { return a > b ? a : b; }
};

/**
 * Breadth-first search over a graph stored as a nested_seq<uint64_t>: inner
 * sequence v holds the *in-neighbour* ids of vertex v (for a symmetric /
 * undirected graph this is simply v's neighbour list).  Returns a DRAM distance
 * array; unreachable vertices are UINT64_MAX.
 *
 * This is the PULL / topology-driven formulation (Bellman-Ford with unit
 * weights): every round is one full streaming sweep of the whole graph via
 *
 *     cand[v] = min over u in N(v) of (dist[u] + 1)         // NestedReduce, min monoid
 *     dist[v] = min(dist[v], cand[v])                        // DRAM relax
 *
 * repeated until no distance changes.  It converges after eccentricity(source)+1
 * rounds and re-reads the whole graph each round — the standard streaming
 * out-of-core BFS trade-off (cf. external_bellman_ford_fast).  It needs NO
 * per-vertex scatter, only the per-vertex reduce that nested_seq's whole-
 * sequence-per-chunk layout makes lock-free.  The distance array is DRAM-
 * resident (8 B / vertex), consistent with the seq_len_scan assumption.
 *
 * A vertex whose id appears as a neighbour is used to index `dist`, so all
 * stored neighbour ids must be < g.total_seqs().
 */
inline parlay::sequence<uint64_t> NestedBFS(const nested_seq<uint64_t>& g,
                                            uint64_t source,
                                            size_t reader_threads = 10) {
  constexpr uint64_t INF = std::numeric_limits<uint64_t>::max();
  const size_t V = g.total_seqs();
  parlay::sequence<uint64_t> dist(V, INF);
  if (V == 0) return dist;
  CHECK(source < V) << "NestedBFS: source " << source << " >= |V|=" << V;
  dist[source] = 0;

  bool changed = true;
  while (changed) {
    // One full sweep: for each vertex v, the cheapest (dist[u]+1) over its
    // in-neighbours u.  This is a fused inner map+reduce — map u -> dist[u]+1,
    // reduce by min — so nothing intermediate is written.  dist is read-only
    // for the duration of the sweep.
    parlay::sequence<uint64_t> cand = NestedMapReduce<uint64_t, uint64_t>(
        g,
        [&dist, INF](uint64_t u) -> uint64_t {
          const uint64_t d = dist[u];
          return d == INF ? INF : d + 1;  // guard against overflow
        },
        NestedBFSMin{}, reader_threads);

    // Relax in DRAM; each v writes only dist[v] (disjoint), count the changes.
    const size_t nchanged = parlay::reduce(parlay::tabulate(V, [&](size_t v) {
      if (cand[v] < dist[v]) {
        dist[v] = cand[v];
        return (size_t)1;
      }
      return (size_t)0;
    }));
    changed = nchanged > 0;
  }
  return dist;
}

namespace detail {
inline void unlink_prefix(const std::string& prefix) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++) unlink(GetFileName(prefix, d).c_str());
}
}  // namespace detail

/**
 * UNFUSED pull BFS — the same topology-driven BFS as NestedBFS, but written
 * purely from the core nested_seq library with NO fusion primitive: each round
 * MATERIALIZES the mapped graph with NestedMap (every neighbour id u -> dist[u]+1,
 * written to disk as a nested_seq), then reduces it per row with NestedReduce
 * (min).  NestedBFS instead fuses these two into a single NestedMapReduce pass.
 *
 * So this is the "naive compositional" baseline: read g + write the mapped
 * nested_seq + read it back each round (~3x the I/O of the fused NestedBFS's
 * single read pass).  It uses only NestedMap + NestedReduce (+ a DRAM relax);
 * no NestedMapReduce, no NestedGatherFlatten, no flat-chunk_seq filter.
 */
inline parlay::sequence<uint64_t> NestedBFSPullUnfused(
    const nested_seq<uint64_t>& g, uint64_t source, size_t reader_threads = 10) {
  constexpr uint64_t INF = std::numeric_limits<uint64_t>::max();
  const size_t V = g.total_seqs();
  parlay::sequence<uint64_t> dist(V, INF);
  if (V == 0) return dist;
  CHECK(source < V) << "NestedBFSPullUnfused: source " << source
                    << " >= |V|=" << V;
  dist[source] = 0;

  const std::string mp = "nbfs_pull_map";
  bool changed = true;
  while (changed) {
    // MAP (materialized): map each neighbour u -> dist[u]+1, same row shape.
    nested_seq<uint64_t> mapped = NestedMap<uint64_t, uint64_t>(
        g, mp, [&dist, INF](const uint64_t* row, size_t len) {
          parlay::sequence<uint64_t> r(len);
          for (size_t i = 0; i < len; i++) {
            const uint64_t d = dist[row[i]];
            r[i] = (d == INF) ? INF : d + 1;
          }
          return r;
        });
    // REDUCE: per-row min over the materialized nested_seq (identity element
    // map — a plain reduce, no fused map).
    parlay::sequence<uint64_t> cand = NestedReduce<uint64_t, uint64_t>(
        mapped, [](uint64_t x) { return x; }, NestedBFSMin{}, reader_threads);

    const size_t nchanged = parlay::reduce(parlay::tabulate(V, [&](size_t v) {
      if (cand[v] < dist[v]) {
        dist[v] = cand[v];
        return (size_t)1;
      }
      return (size_t)0;
    }));
    changed = nchanged > 0;
  }
  detail::unlink_prefix(mp);
  return dist;
}

/**
 * PUSH / frontier BFS over a graph stored as a nested_seq<uint64_t> (inner
 * sequence v = out-neighbours of v).  Returns a DRAM distance array (UINT64_MAX
 * = unreachable) — the same output as the pull NestedBFS, so the two can be
 * diffed against each other.
 *
 * This is the out-of-core rendering of the canonical parlay frontier BFS
 * (deps/parlaylib-examples/BFS.h): each level
 *
 *     ns   = NestedGather(g, frontier)     // map(frontier, u -> G[u]) -> nested_seq
 *     out  = NestedFlatten(ns)             // flatten -> flat chunk_seq of candidates
 *     next = ChunkFilter(out, claim)       // atomic-claim filter -> next frontier
 *     frontier = materialize(next)         // survivors -> DRAM ids
 *
 * The map's nested-sequence intermediate is materialized on disk (the whole
 * point).  `claim` is an atomic CAS on `visited` (V entries, same budget as
 * dist): duplicates in `out` (a vertex reached from several frontier vertices)
 * are deduped by the CAS — the "lock" — so each vertex is claimed once and each
 * edge relaxed once (work-efficient).  `dir` is forwarded to NestedGather (kAuto
 * = direction-optimized per level); if `level_push` is non-null it receives one
 * entry per level: 1 if that level's gather used the push read, 0 if pull.
 *
 * `fused` selects how the push step's `flatten(map(...))` is done:
 *   - false (default): the map materializes a nested_seq on disk (NestedGather),
 *     which NestedFlatten then reads back — the compositional, materialized
 *     intermediate.
 *   - true: NestedGatherFlatten fuses the two into a single dense-pack pass, so
 *     the nested intermediate is never written.
 * Both give identical results; see NestedBFSPushFused and the bfs compare bench.
 */
inline parlay::sequence<uint64_t> NestedBFSPush(
    const nested_seq<uint64_t>& g, uint64_t source,
    GatherDir dir = GatherDir::kAuto, std::vector<char>* level_push = nullptr,
    bool fused = false) {
  constexpr uint64_t INF = std::numeric_limits<uint64_t>::max();
  const size_t V = g.total_seqs();
  parlay::sequence<uint64_t> dist(V, INF);
  if (V == 0) return dist;
  CHECK(source < V) << "NestedBFSPush: source " << source << " >= |V|=" << V;

  // Atomic claim array (zero-initialized), source pre-claimed.
  std::vector<std::atomic<char>> visited(V);
  parlay::parallel_for(0, V, [&](size_t i) {
    visited[i].store(0, std::memory_order_relaxed);
  });
  visited[source].store(1, std::memory_order_relaxed);
  dist[source] = 0;

  const std::string gp = "nbfs_gather", fp = "nbfs_flat", np = "nbfs_next";

  parlay::sequence<uint64_t> frontier(1, source);
  uint64_t level = 0;
  while (frontier.size() > 0) {
    level++;
    // flatten(map(frontier, neighbors)) -> flat candidate chunk_seq `out`.
    bool chose_push = false;
    chunk_seq out;
    if (fused) {
      // Fused: dense-pack the frontier rows straight to the flat list; the
      // nested intermediate is never written.
      out = NestedGatherFlatten<uint64_t>(g, frontier, gp, dir, &chose_push);
    } else {
      // Materialized: write the mapped nested_seq, then read it back to flatten.
      nested_seq<uint64_t> ns =
          NestedGather<uint64_t>(g, frontier, gp, dir, &chose_push);
      out = NestedFlatten<uint64_t>(ns, fp);
    }
    if (level_push) level_push->push_back(chose_push ? 1 : 0);
    if (out.chunks.empty()) break;  // no candidates -> done

    // filter(out, claim): keep the v that win the CAS; record their distance.
    const uint64_t lv = level;
    chunk_seq next = ChunkFilter<uint64_t>(out, np, [&, lv](uint64_t v) {
      char expected = 0;
      if (visited[v].load(std::memory_order_relaxed) == 0 &&
          visited[v].compare_exchange_strong(expected, 1)) {
        dist[v] = lv;  // unique winner writes dist[v] -> no race
        return true;
      }
      return false;
    });

    frontier = materialize<uint64_t>(next);
  }

  detail::unlink_prefix(gp);
  detail::unlink_prefix(fp);
  detail::unlink_prefix(np);
  return dist;
}

// Fused push BFS: identical to NestedBFSPush but the push step's flatten(map(...))
// is a single dense-pack pass (NestedGatherFlatten) that never materializes the
// nested intermediate.  Same result as NestedBFSPush; fewer writes per level.
inline parlay::sequence<uint64_t> NestedBFSPushFused(
    const nested_seq<uint64_t>& g, uint64_t source,
    GatherDir dir = GatherDir::kAuto, std::vector<char>* level_push = nullptr) {
  return NestedBFSPush(g, source, dir, level_push, /*fused=*/true);
}

/**
 * DIRECTION-OPTIMIZING BFS — per level, choose a PUSH step (expand the frontier's
 * neighbours) or a PULL step (for every unvisited vertex, ask whether any
 * neighbour is in the frontier), based on the frontier's edge count vs |E|.
 *
 * `fused` toggles whether BOTH steps use their fused primitive:
 *   - fused=false: push = NestedGather + NestedFlatten; pull = NestedMap + NestedReduce.
 *   - fused=true : push = NestedGatherFlatten;          pull = NestedMapReduce.
 * So the compare below times "neither fused" vs "both fused" while each runs the
 * same push/pull schedule.  If `level_dirs` is non-null it gets 1=push / 0=pull
 * per level.
 *
 * Note on the crossover: unlike in-memory Beamer, the streaming pull step cannot
 * early-terminate (its NestedMapReduce always reads the whole graph), so push
 * (reads only frontier edges) wins for small frontiers, and pull (reads all
 * edges once but writes only O(V) flags rather than O(frontier-edges)
 * candidates) wins once the frontier is dense.
 */
inline parlay::sequence<uint64_t> NestedBFSDirOpt(
    const nested_seq<uint64_t>& g, uint64_t source, bool fused,
    std::vector<char>* level_dirs = nullptr) {
  constexpr uint64_t INF = std::numeric_limits<uint64_t>::max();
  const size_t V = g.total_seqs();
  parlay::sequence<uint64_t> dist(V, INF);
  if (V == 0) return dist;
  CHECK(source < V) << "NestedBFSDirOpt: source " << source << " >= |V|=" << V;

  std::vector<std::atomic<char>> visited(V);
  parlay::parallel_for(0, V, [&](size_t i) {
    visited[i].store(0, std::memory_order_relaxed);
  });
  visited[source].store(1, std::memory_order_relaxed);
  dist[source] = 0;

  const size_t m = g.seq_len_scan.empty() ? 0 : g.seq_len_scan.back();
  const std::string gp = "nbfs_do_g", fp = "nbfs_do_f", np = "nbfs_do_n",
                    mp = "nbfs_do_m";

  const bool trace = getenv("PLAID_BFS_TRACE") != nullptr;
  // Push→pull crossover.  A push level does NOT just read its `fe` frontier
  // edges: it writes them out (flatten), reads them back and writes the
  // survivors (filter), then reads the survivors (materialize) — roughly
  // c≈3-4× fe bytes moved, all through the single-threaded UnorderedFileWriter.
  // A pull level reads the whole graph once (m bytes) and writes NOTHING to
  // disk (NestedMapReduce returns a DRAM flag array).  So push wins only while
  // c·fe < m, i.e. fe < m/c ≈ m/4 — not the naive fe < m/2 (which counts push
  // as if it moved just fe bytes).  Measured: at fe≈m/3.8 a push level cost
  // ~1.5× the equivalent pull sweep on the dev box.  `PLAID_BFS_PUSH_DENOM`
  // overrides the denominator to retune for a different I/O substrate.
  size_t push_denom = 4;
  if (const char* e = getenv("PLAID_BFS_PUSH_DENOM")) push_denom = std::stoull(e);
  parlay::sequence<uint64_t> frontier(1, source);
  uint64_t level = 0;
  while (frontier.size() > 0) {
    level++;
    // Edges incident to the frontier (cheap: from seq_len_scan, in DRAM).
    const size_t fe = parlay::reduce(parlay::map(frontier, [&](uint64_t u) {
      return g.seq_len_scan[u + 1] - g.seq_len_scan[u];
    }));
    const bool do_push = fe * push_denom < m;  // push while the frontier is sparse
    if (level_dirs) level_dirs->push_back(do_push ? 1 : 0);
    const size_t front_sz = frontier.size();
    const auto t_lvl = std::chrono::steady_clock::now();

    if (do_push) {
      // PUSH: flatten(map(frontier, neighbours)) then atomic-claim.
      chunk_seq out;
      if (fused) {
        out = NestedGatherFlatten<uint64_t>(g, frontier, gp, GatherDir::kAuto);
      } else {
        nested_seq<uint64_t> ns =
            NestedGather<uint64_t>(g, frontier, gp, GatherDir::kAuto);
        out = NestedFlatten<uint64_t>(ns, fp);
      }
      if (out.chunks.empty()) break;
      const uint64_t lv = level;
      chunk_seq next = ChunkFilter<uint64_t>(out, np, [&, lv](uint64_t v) {
        char e = 0;
        if (visited[v].load(std::memory_order_relaxed) == 0 &&
            visited[v].compare_exchange_strong(e, 1)) {
          dist[v] = lv;
          return true;
        }
        return false;
      });
      frontier = materialize<uint64_t>(next);
    } else {
      // PULL: flag[v] = does v have any neighbour in the level-(L-1) frontier?
      const uint64_t prev = level - 1;
      parlay::sequence<uint64_t> flag;
      if (fused) {
        flag = NestedMapReduce<uint64_t, uint64_t>(
            g, [&dist, prev](uint64_t u) { return dist[u] == prev ? 1 : 0; },
            NestedBFSMax{});
      } else {
        nested_seq<uint64_t> mapped = NestedMap<uint64_t, uint64_t>(
            g, mp, [&dist, prev](const uint64_t* row, size_t len) {
              parlay::sequence<uint64_t> r(len);
              for (size_t i = 0; i < len; i++) r[i] = dist[row[i]] == prev ? 1 : 0;
              return r;
            });
        flag = NestedReduce<uint64_t, uint64_t>(
            mapped, [](uint64_t x) { return x; }, NestedBFSMax{});
        detail::unlink_prefix(mp);
      }
      // Claim every unvisited vertex flagged this round (disjoint -> no atomics).
      parlay::sequence<uint64_t> claimed =
          parlay::filter(parlay::iota<uint64_t>(V), [&](uint64_t v) {
            return dist[v] == INF && flag[v] != 0;
          });
      parlay::parallel_for(0, claimed.size(), [&](size_t i) {
        const uint64_t v = claimed[i];
        dist[v] = level;
        visited[v].store(1, std::memory_order_relaxed);
      });
      frontier = std::move(claimed);
    }
    if (trace) {
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t_lvl)
                            .count();
      fprintf(stderr,
              "  [bfs] level %2llu  %s  |front|=%-9zu fe=%-10zu next=%-9zu "
              "%.2f ms\n",
              (unsigned long long)level, do_push ? "PUSH" : "pull", front_sz,
              fe, frontier.size(), ms);
    }
  }

  detail::unlink_prefix(gp);
  detail::unlink_prefix(fp);
  detail::unlink_prefix(np);
  detail::unlink_prefix(mp);
  return dist;
}

}  // namespace ChunkSequenceOps

#endif  // NESTED_BFS_H
