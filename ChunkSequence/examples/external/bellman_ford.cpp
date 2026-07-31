// bellman_fordExample — out-of-core Bellman-Ford vs the in-memory parlaylib
// reference, swept over sparse/balanced/dense RMAT graphs at the same n.
//
// external_bellman_ford (external_bellman_ford.h) is a port of the "pull"
// variant of parlaylib's bellman_ford (examples/in_memory/graph/bellman_ford.h,
// byte-identical to deps/parlaylib-examples/bellman_ford.h): each round,
// relax every vertex's distance from its in-neighbors' chunk_csr row.
// external_bellman_ford_fast (same header) is a drop-in alternative that does
// the same thing with one streaming pass per round instead of a per-vertex
// reader setup — see the header for why. This benchmark builds three graphs
// from the same vertex count n but different average degree
// (sparse/balanced/dense), runs both out-of-core variants and the in-memory
// algorithm from the same start vertex on each, and cross-checks the
// resulting distances; it exits non-zero if any case mismatches or fails to
// converge.
//
// "Dense" here means high average degree relative to n (m scales with n),
// not a near-complete graph: RMAT concentrates degree into a few hub
// vertices rather than spreading it evenly, so even the dense case stays a
// skewed power-law graph, just a much denser one. That is consistent with
// every other case using the same generator (rmat_symmetric_graph) and lets
// the sweep isolate the effect of edge count alone.
//
// Graph construction IS out-of-core (external_rmat.h): an RMAT edge is a pure
// function of its index, so the edge list is tabulated straight onto the
// drives, put into CSR order with direct_sample_sort, and
// deduped/projected/counted in one DensePackStream pass.  Nothing graph-sized
// is DRAM-resident except chunk_csr's degree_scan (8 bytes/vertex, which the
// data structure requires). Transient disk is ~3-4x the final edge bytes and is
// swept before the build returns.  The graph is symmetric (undirected), so its
// own adjacency doubles as its transpose GT, which both algorithms pull from.
//
// The in-memory baseline still needs a DRAM graph, so when it is enabled the
// binary builds a SECOND copy via graph_utils' rmat_symmetric_graph +
// add_weights.  Both generators are deterministic functions of (n_req, m_req)
// and use the same draws, so they produce the same graph (up to neighbor order
// within a row, which min-relax cannot see) -- which makes the distance
// cross-check below double as an end-to-end check that they agree.  That DRAM
// copy is what used to OOM-kill this binary at benchmark scale; it is now gated
// by a budget (below) and simply not built when it doesn't fit.
//
// Defaults are deliberately tiny.  external_bellman_ford relaxes a vertex by
// calling sequential_materialize on a fresh delayed cut of the edge chunk_seq
// PER VERTEX PER ROUND (a blocking pread per chunk touched, via a per-worker
// SequentialReadContext -- see external_bellman_ford.h), so its cost scales
// as O(rounds * n) reader setups, not O(rounds * m) bytes read.  That is
// expected to make it dramatically slower than the in-memory baseline, and
// dramatically slower than external_bellman_ford_fast (which does one
// streaming pass per round), even on small graphs; this benchmark exists to
// measure exactly that gap, not to hide it.  Because of that, run_case()
// skips the per-vertex method once a case's edge bytes exceed
// BELLMAN_FORD_PER_VERTEX_MAX_BYTES (default 512 KiB, env-overridable) --
// past that it isn't useful benchmark data, just wall-clock cost -- and
// leaves its CSV fields blank for that point.  external_bellman_ford_fast is
// never gated and always runs at every size; the in-memory baseline is
// skipped past BELLMAN_FORD_INMEM_MAX_N vertices (default 2^30,
// env-overridable) -- past that point RMAT generation + bellman_ford's O(n)
// `long double` distance arrays (and, for the "dense" case, its n^2/2-scaling
// edge count) stop being a useful DRAM baseline and are just wall-clock/memory
// cost.  It runs regardless of PLAID_TRACE (raise the env var on a box with
// enough DRAM to still want the comparison at larger n; see run_case()).
//
// A second, size-derived gate guards DRAM directly (BELLMAN_FORD_BUILD_BUDGET_
// BYTES, default half of physical RAM, 0 disables).  run_case() estimates both
// the unavoidable out-of-core footprint (degree_scan + the two distance arrays)
// and the in-memory baseline's graph BEFORE allocating anything: if only the
// baseline is over budget it is dropped and the out-of-core methods still run
// and report; if even the out-of-core footprint is over budget the case is
// skipped with no CSV line, which run_benches.py drops with a warning.  This
// is what an OOM kill used to look like -- exit -9, no CSV, and a trace with a
// build_start mark and no build_end.
//
//   usage: bellman_fordExample [global --flags] [n] [balanced_avg_degree]
//   [case]
//     n                    requested vertex count, rounded up to a power of
//                          two by the RMAT generator (default 200); shared by
//                          all three cases
//     balanced_avg_degree  avg_degree for the "balanced" case only (default
//                          8, matching parlaylib's own bellman_ford.cpp
//                          driver); the "sparse" case always uses avg_degree
//                          2, and "dense" always uses n/2
//     case                 "all" (default), or "sparse"/"balanced"/"dense" to
//                          run and print a CSV line for just that one case
//                          instead of all three -- used by
//                          benchmarks/run_benches.py's EXAMPLES registry /
//                          io_trace.py, which need exactly one CSV line per
//                          invocation (they keep only the last line seen)
//
// One CSV line per case:
//   CSV,case,n,m,build_s,op_s,inmem_op_s,reachable,throughput_gb_s,fast_op_s,fast_reachable,fast_throughput_gb_s
//   throughput = edge bytes (m * sizeof(weighted_edge)) / op_s (fast_* are
//   the same fields for external_bellman_ford_fast).  op_s/reachable/
//   throughput_gb_s are blank when the per-vertex method is skipped past its
//   byte budget; inmem_op_s is blank when the in-memory baseline is skipped
//   (n past BELLMAN_FORD_INMEM_MAX_N).

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>

#include "absl/log/check.h"
#include "parlay/primitives.h"

// Upstream-shaped in-memory reference: RMAT graph generation/weighting and the
// plain (non-lazy) pull-based Bellman-Ford, byte-identical to
// deps/parlaylib-examples/{helper/graph_utils.h,bellman_ford.h}.
//
// Must be parsed BEFORE external_compressed_sparse_row.h below: that header
// #defines bare `vertex`/`weight` macros (size_t / long double) with no
// #undef, which would otherwise rewrite these headers' `vertex` template
// parameter (and every use of it) out from under them while they're being
// preprocessed.
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/examples/external/bench_drives.h"
#include "ChunkSequence/examples/in_memory/graph/bellman_ford.h"
#include "ChunkSequence/examples/in_memory/graph/graph_utils/graph_utils.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "utils/trace_marker.h"
// Before external_bellman_ford.h: this pulls in dense_pack/direct_samplesort,
// which must be parsed before external_compressed_sparse_row.h's bare macros.
#include "ChunkSequence/examples/external/external_bellman_ford.h"
#include "ChunkSequence/examples/external/graph_utils/external_rmat.h"

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}
static double to_gb(size_t bytes) {
  return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static void cleanup_prefix(const std::string& prefix) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    unlink(GetFileName(prefix, d).c_str());
}

// The two implementations use different sentinel encodings for "unreached":
// external_bellman_ford fills with (long double)(size_t)-1 (~1.8e19), while
// the in-memory version fills with std::numeric_limits<long double>::max()
// (~1.2e4932). Treat anything past a generous finite-distance ceiling (edge
// weights are in [1,20], so any real distance stays far below this) as
// "unreached" on both sides rather than comparing sentinel magnitudes.
static bool unreached(long double d) { return d > 1e15L; }

using vertex_utils = graph_utils<size_t>;

// Builds one RMAT graph (n_req vertices, avg_degree * n_req edges before
// rounding/symmetrizing), runs external_bellman_ford and the in-memory
// baseline from the same start vertex, cross-checks them, prints a summary
// and a CSV line, and cleans up the edge files it wrote. Returns true iff
// both sides converged and agreed.
static bool run_case(const std::string& label, size_t n_req,
                     size_t avg_degree) {
  const std::string edge_prefix = "bf_edges_" + label;

  std::cout << "\n=== case: " << label << " (avg_degree " << avg_degree
            << ") ===\n";

  // n is a pure function of n_req (RMAT rounds up to a power of two), so
  // every budget decision below can be made BEFORE anything is allocated.
  const size_t n = size_t{1} << (int)std::round(std::log2((double)n_req));
  const size_t m_req = avg_degree * n_req;  // directed edges, pre-dedup

  // Two DRAM gates, both evaluated up front.
  //
  // ext_bytes is what the out-of-core path itself cannot avoid: chunk_csr's
  // degree_scan (8 bytes/vertex, DRAM-resident by construction --
  // external_compressed_sparse_row.h) plus external_bellman_ford_fast's two
  // `long double` distance arrays (16 bytes/vertex each).
  //
  // inmem_bytes is the in-memory baseline's graph, which is the expensive
  // one: ~48 bytes/vertex of parlay::sequence outer arrays (G, symmetrize's
  // transpose, WG) plus ~64 bytes per requested edge (8 in G, 32 in WG, and
  // symmetrize's 16-byte edge-list copies) -- see external_rmat.h's header
  // for why that peak is what OOM-killed this binary at n = 2^32.
  const size_t ext_bytes = 8 * (n + 1) + 2 * 16 * n;
  const size_t inmem_bytes = 48 * n + 64 * m_req;

  size_t inmem_max_n = 1ull << 30;
  if (const char* e = getenv("BELLMAN_FORD_INMEM_MAX_N"))
    inmem_max_n = std::stoull(e);
  bool inmem_ok = n <= inmem_max_n;

  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys / 2;
  if (const char* e = getenv("BELLMAN_FORD_BUILD_BUDGET_BYTES"))
    budget = std::stoull(e);

  if (budget != 0 && ext_bytes > budget) {
    // Nothing here is runnable. Print no CSV line at all and return
    // success: run_benches.py drops a CSV-less point with a warning and
    // keeps sweeping, which is what we want instead of an OOM kill.
    std::cout << "SKIPPED: n=" << n << " needs " << std::setprecision(3)
              << to_gb(ext_bytes) << " GB of DRAM for degree_scan + distance "
              << "arrays alone, past the " << to_gb(budget) << " GB budget "
              << "(BELLMAN_FORD_BUILD_BUDGET_BYTES)\n";
    return true;
  }
  if (inmem_ok && budget != 0 && ext_bytes + inmem_bytes > budget) {
    // Only the baseline doesn't fit -- drop it and still run (and report)
    // the out-of-core methods rather than losing the whole point.
    std::cout << "in-memory baseline disabled: its DRAM graph needs ~"
              << std::setprecision(3) << to_gb(inmem_bytes) << " GB, past the "
              << to_gb(budget) << " GB budget\n";
    inmem_ok = false;
  }

  std::cout << "Generating " << n_req << "-vertex RMAT graph (avg degree "
            << avg_degree << ") out-of-core..." << std::flush;
  trace_mark(("build_start_" + label).c_str());
  auto t0 = Clock::now();

  // Symmetric (undirected) weighted graph built straight onto the drives --
  // also its own transpose, so it feeds the "pull" Bellman-Ford directly on
  // both sides of the compare.  Nothing graph-sized is ever DRAM-resident
  // here except degree_scan; see external_rmat.h.
  chunk_csr graph = ExternalGraphUtils::external_rmat_symmetric_graph(
      n_req, m_req, edge_prefix, /*minw=*/1.0L, /*maxw=*/20.0L);
  const size_t m = graph.degree_scan[n];

  const double build_s = elapsed(t0);
  trace_mark(("build_end_" + label).c_str());
  std::cout << " done (" << n << " vertices, " << m << " edges, " << std::fixed
            << std::setprecision(4) << build_s << "s)\n";

  // The baseline's own copy of the same graph, in DRAM.  Built independently
  // rather than shared with the out-of-core path: both generators are
  // deterministic functions of (n_req, m_req), so building twice turns the
  // distance cross-check below into an end-to-end check that they agree.
  // Not counted in build_s, which times the out-of-core build only.
  vertex_utils::weighted_graph<long double> WG;
  if (inmem_ok) {
    std::cout << "Generating the in-memory baseline's copy..." << std::flush;
    auto G = vertex_utils::rmat_symmetric_graph((long)n_req, (long)m_req);
    WG = vertex_utils::add_weights<long double>(G, 1.0L, 20.0L);
    G = decltype(G){};  // fully consumed; drop before the baseline runs
    std::cout << " done\n";
  }

  const size_t start = 0;

  // external_bellman_ford (per-vertex) does O(rounds*n) reader setups (see
  // the file header), so it's dramatically slower than everything else here
  // even at modest edge counts -- not useful benchmark data past a small
  // budget, and impractically slow to just leave running. Skip it past a
  // byte budget (edge bytes, matching every other example's
  // EXAMPLE_INMEM_BUDGET_BYTES-style RAM-cliff gate), leaving its CSV
  // fields blank so the plotted "out-of-core, per-vertex" line simply stops
  // there (benchmarks/run_benches.py's _series already drops blanks).
  size_t per_vertex_budget = 512 * 1024;  // 512 KiB of edges
  if (const char* e = getenv("BELLMAN_FORD_PER_VERTEX_MAX_BYTES"))
    per_vertex_budget = std::stoull(e);
  const size_t edge_bytes = m * sizeof(weighted_edge);
  const bool per_vertex_ok = edge_bytes <= per_vertex_budget;

  bench_drives::settle_drives();  // isolate the first timed read from the
                                  // build's writeback

  bool ext_converged = false;
  size_t reachable = 0;
  size_t rounds = 0;
  double op_s = 0, gb_s = 0;
  parlay::sequence<long double> d_ext;
  if (per_vertex_ok) {
    std::cout << "Running out-of-core Bellman-Ford..." << std::flush;
    trace_mark(("op_start_" + label).c_str());
    t0 = Clock::now();
    d_ext = external_bellman_ford(graph, start, &rounds);
    op_s = elapsed(t0);
    trace_mark(("op_end_" + label).c_str());
    std::cout << " done\n";

    // external_bellman_ford returns an empty sequence (rather than an
    // optional) if it doesn't converge within n rounds.
    ext_converged = d_ext.size() == n;
    if (ext_converged)
      reachable =
          parlay::reduce(parlay::map(d_ext, [](long double dv) -> size_t {
            return unreached(dv) ? 0 : 1;
          }));

    // edge_bytes is ONE round's worth; the algorithm relaxes every edge
    // once per round, so the bytes actually moved off disk are
    // edge_bytes * rounds, not edge_bytes alone.
    gb_s = to_gb(edge_bytes) * (double)rounds / op_s;
    std::cout << reachable << "/" << n << " vertices reachable   "
              << std::setprecision(4) << op_s << "s   " << std::setprecision(2)
              << gb_s << " GB/s (edges read)\n";
    if (!ext_converged)
      std::cout << "*** out-of-core Bellman-Ford did not converge within " << n
                << " rounds ***\n";

    bench_drives::settle_drives();  // isolate the fast method's timer from this
                                    // run
  } else {
    std::cout << "out-of-core Bellman-Ford (per-vertex): skipped (edges "
              << to_gb(edge_bytes) << " GB exceed per-vertex budget "
              << to_gb(per_vertex_budget) << " GB)\n";
  }

  // external_bellman_ford_fast: same algorithm, one streaming pass over the
  // edges per round (delayed::segmented_reduce) instead of a per-vertex
  // reader setup -- see external_bellman_ford.h for why that's expected to
  // be dramatically faster.
  std::cout << "Running out-of-core Bellman-Ford (fast)..." << std::flush;
  trace_mark(("fast_op_start_" + label).c_str());
  t0 = Clock::now();
  size_t fast_rounds = 0;
  parlay::sequence<long double> d_fast =
      external_bellman_ford_fast(graph, start, &fast_rounds);
  const double fast_op_s = elapsed(t0);
  trace_mark(("fast_op_end_" + label).c_str());
  std::cout << " done\n";

  const bool fast_converged = d_fast.size() == n;
  size_t fast_reachable = 0;
  if (fast_converged)
    fast_reachable =
        parlay::reduce(parlay::map(d_fast, [](long double dv) -> size_t {
          return unreached(dv) ? 0 : 1;
        }));

  // Same correction as gb_s above: fast_op_s covers fast_rounds full
  // streaming passes over the edges, not just one.
  const double fast_gb_s = to_gb(edge_bytes) * (double)fast_rounds / fast_op_s;
  std::cout << fast_reachable << "/" << n << " vertices reachable   "
            << std::setprecision(4) << fast_op_s << "s   "
            << std::setprecision(2) << fast_gb_s << " GB/s (edges read)\n";
  if (!fast_converged)
    std::cout << "*** out-of-core Bellman-Ford (fast) did not converge within "
              << n << " rounds ***\n";

  // `inmem_ok` was decided at the top of run_case (before the build) and WG
  // was only populated if it held: a sweep's smaller points all run this
  // binary WITHOUT PLAID_TRACE (io_trace.py traces only the largest by
  // default), so a sweep whose smaller points still exceed a couple billion
  // vertices would otherwise pay full DRAM cost (RMAT generation +
  // bellman_ford<long double>, i.e. O(n) `long double` distance arrays with
  // an n^2/2-edge "dense" case) on every one of them, not just the largest.
  // Default matches the size at which that DRAM cost stops being a useful
  // baseline point; raise BELLMAN_FORD_INMEM_MAX_N on a box with enough
  // DRAM to still want the comparison at larger n (the baseline runs
  // regardless of PLAID_TRACE -- it appends a CPU-only, disk-idle tail to
  // a trace capture after fast_op_end, which is expected).
  std::optional<parlay::sequence<long double>> d_mem_opt;
  double inmem_op_s = 0;
  if (!inmem_ok) {
    std::cout << "Running in-memory bellman_ford: skipped (n " << n
              << ", inmem_max_n " << inmem_max_n << ")\n";
  } else {
    std::cout << "Running in-memory bellman_ford..." << std::flush;
    t0 = Clock::now();
    d_mem_opt = bellman_ford<long double>(start, WG);
    inmem_op_s = elapsed(t0);
    std::cout << " done (" << std::setprecision(4) << inmem_op_s << "s)\n";
  }

  // Cross-check one out-of-core result against the in-memory baseline
  // (unreached-aware: the two sides use different "unreached" sentinels).
  auto compare_to_mem = [&](const std::string& name,
                            const parlay::sequence<long double>& d_out,
                            bool out_converged) -> bool {
    bool ok = out_converged && d_mem_opt.has_value();
    if (ok) {
      const auto& d_mem = *d_mem_opt;
      if (d_mem.size() != d_out.size()) {
        std::cout << "*** MISMATCH (" << name << "): in-mem " << d_mem.size()
                  << " distances != out-of-core " << d_out.size() << " ***\n";
        ok = false;
      } else {
        for (size_t i = 0; i < n; i++) {
          bool out_u = unreached(d_out[i]), mem_u = unreached(d_mem[i]);
          if (out_u != mem_u || (!out_u && d_mem[i] != d_out[i])) {
            std::cout << "*** MISMATCH (" << name << ") at vertex " << i
                      << ": in-mem " << (double)d_mem[i] << " != out-of-core "
                      << (double)d_out[i] << " ***\n";
            ok = false;
            break;
          }
        }
      }
    } else if (!d_mem_opt.has_value() && !inmem_ok) {
      std::cout << name << " result skipped comparison (in-mem bellman_ford "
                << "not run)\n";
    } else if (!d_mem_opt.has_value()) {
      std::cout << "in-mem bellman_ford: did not converge within " << n
                << " rounds; skipped comparison\n";
    } else {
      std::cout << name << " result skipped comparison (no convergence)\n";
    }
    return ok;
  };

  // Skip the cross-check for a budget-skipped per-vertex run -- there's no
  // result to compare. Also skip entirely when d_mem_opt was never computed
  // (n past inmem_max_n -- see above).
  const bool no_inmem = !inmem_ok;
  const bool agree = no_inmem || !per_vertex_ok ||
                     compare_to_mem("out-of-core", d_ext, ext_converged);
  const bool fast_agree =
      no_inmem || compare_to_mem("out-of-core (fast)", d_fast, fast_converged);

  // Machine-readable line for benchmarks/run_benches.py.
  // Columns: case,n,m,build_s,op_s,inmem_op_s,reachable,throughput_gb_s,
  //          fast_op_s,fast_reachable,fast_throughput_gb_s
  // (op_s/reachable/throughput_gb_s blank when the per-vertex method is
  // skipped past the byte budget, so the plotted per-vertex line stops;
  // inmem_op_s blank past inmem_max_n vertices, where it was never run.)
  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  std::cout << "CSV," << label << ',' << n << ',' << m << ',' << f9(build_s)
            << ',' << (per_vertex_ok ? f9(op_s) : std::string()) << ','
            << (no_inmem ? std::string() : f9(inmem_op_s)) << ','
            << (per_vertex_ok ? std::to_string(reachable) : std::string())
            << ',' << (per_vertex_ok ? f9(gb_s) : std::string()) << ','
            << f9(fast_op_s) << ',' << fast_reachable << ',' << f9(fast_gb_s)
            << '\n';

  cleanup_prefix(edge_prefix);
  return agree && fast_agree;
}

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  // external_bellman_ford holds one SequentialReadContext per parlay worker
  // (external_bellman_ford.h), each with its own fd cache (bounded at
  // MAX_CACHED_FDS = 256, chunk_delayed.h), plus the CSR build's own
  // readers/writers -- workers * 256 alone blows past the common 1024 soft
  // RLIMIT_NOFILE.  Lift the soft limit to the hard limit before any I/O
  // starts (same fix every other external example applies).
  RaiseFdLimit();
  const size_t n_req = (argc > 1) ? std::stoull(argv[1]) : 200;
  const size_t balanced_avg_degree = (argc > 2) ? std::stoull(argv[2]) : 8;
  const std::string case_filter = (argc > 3) ? argv[3] : "all";

  // Sparse/balanced/dense span avg_degree from a bare-spanning-tree-ish 2,
  // through the parlaylib driver's own default of 8, up to n/2 (m ~ n^2/2)
  // -- "dense" is only a jump in edge count, not a different generator; see
  // the file header for why RMAT stays skewed even at that end.
  const size_t dense_avg_degree = std::max<size_t>(1, n_req / 2);

  struct GraphCase {
    std::string label;
    size_t avg_degree;
  };
  const GraphCase cases[] = {
      {"sparse", 2},
      {"balanced", balanced_avg_degree},
      {"dense", dense_avg_degree},
  };

  bool all_agree = true;
  bool ran_any = false;
  for (const auto& c : cases) {
    if (case_filter != "all" && case_filter != c.label) continue;
    all_agree &= run_case(c.label, n_req, c.avg_degree);
    ran_any = true;
  }
  CHECK(ran_any) << "unknown case " << case_filter
                 << " (expected all|sparse|balanced|dense)";

  return all_agree ? 0 : 1;
}
