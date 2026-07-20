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
// Graph construction is NOT out-of-core: there is no streaming graph-ingestion
// path yet (a general one is planned separately), so this generates + weights
// each RMAT graph entirely in DRAM (graph_utils' rmat_symmetric_graph +
// add_weights) and then does a one-off flatten into chunk_csr row order,
// written out with ChunkSequenceOps::tabulate.  So only the relaxation phase
// is "external"; the graph itself must fit DRAM to be built and to run the
// in-memory baseline.  The graph is symmetric (undirected), so its own
// adjacency doubles as its transpose GT, which both algorithms pull from.
//
// Defaults are deliberately tiny.  external_bellman_ford relaxes a vertex by
// calling delayed::materialize on a fresh delayed cut of the edge chunk_seq
// PER VERTEX PER ROUND (chunk_delayed.h's for_each_chunk spins up its own
// ChunkSequenceReader — 10 io_uring rings by default — on every call), so
// its cost scales as O(rounds * n) reader setups, not O(rounds * m) bytes
// read.  That is expected to make it dramatically slower than the in-memory
// baseline, and dramatically slower than external_bellman_ford_fast (which
// does one reader setup per round), even on small graphs; this benchmark
// exists to measure exactly that gap, not to hide it.  Scale n up
// cautiously via argv for the slow variant.
//
//   usage: bellman_fordExample [global --flags] [n] [balanced_avg_degree] [case]
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
//   the same fields for external_bellman_ford_fast).

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unistd.h>

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
#include "ChunkSequence/examples/in_memory/graph/graph_utils/graph_utils.h"
#include "ChunkSequence/examples/in_memory/graph/bellman_ford.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "utils/trace_marker.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/examples/external/external_bellman_ford.h"

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}
static double to_gb(size_t bytes) { return (double)bytes / (1024.0 * 1024.0 * 1024.0); }

static void cleanup_prefix(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++) unlink(GetFileName(prefix, d).c_str());
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
static bool run_case(const std::string& label, size_t n_req, size_t avg_degree) {
    const std::string edge_prefix = "bf_edges_" + label;

    std::cout << "\n=== case: " << label << " (avg_degree " << avg_degree
              << ") ===\n";
    std::cout << "Generating " << n_req << "-vertex RMAT graph (avg degree "
              << avg_degree << ")..." << std::flush;
    trace_mark(("build_start_" + label).c_str());
    auto t0 = Clock::now();

    // Symmetric (undirected) weighted graph in DRAM -- also its own transpose,
    // so it feeds the "pull" Bellman-Ford directly on both sides of the compare.
    auto G = vertex_utils::rmat_symmetric_graph((long)n_req, (long)(avg_degree * n_req));
    const size_t n = G.size();   // RMAT rounds n_req up to a power of two
    auto WG = vertex_utils::add_weights<long double>(G, 1.0L, 20.0L);

    // Exclusive degree prefix sum (chunk_csr's `degree_scan`, length n+1). n is
    // small enough (this algorithm's own per-vertex I/O pattern forces that)
    // that a sequential prefix sum is negligible next to everything else here.
    parlay::sequence<size_t> degree_scan(n + 1);
    degree_scan[0] = 0;
    for (size_t v = 0; v < n; v++)
        degree_scan[v + 1] = degree_scan[v] + WG[v].size();
    const size_t m = degree_scan[n];

    // Flatten to CSR row order (vertex-major) so the tabulate below can look up
    // edge i by a single flat index with no cross-vertex search.
    auto flat = parlay::flatten(parlay::tabulate(n, [&](size_t v) {
        return parlay::map(WG[v], [](const auto& p) {
            return weighted_edge{p.first, p.second};
        });
    }));

    chunk_csr graph;
    graph.degree_scan = degree_scan;
    graph.edges = ChunkSequenceOps::tabulate<weighted_edge>(
        m, edge_prefix, [&](size_t i) { return flat[i]; });

    const double build_s = elapsed(t0);
    trace_mark(("build_end_" + label).c_str());
    std::cout << " done (" << n << " vertices, " << m << " edges, "
              << std::fixed << std::setprecision(4) << build_s << "s)\n";

    const size_t start = 0;

    std::cout << "Running out-of-core Bellman-Ford..." << std::flush;
    trace_mark(("op_start_" + label).c_str());
    t0 = Clock::now();
    parlay::sequence<long double> d_ext = external_bellman_ford(graph, start);
    const double op_s = elapsed(t0);
    trace_mark(("op_end_" + label).c_str());
    std::cout << " done\n";

    // external_bellman_ford returns an empty sequence (rather than an
    // optional) if it doesn't converge within n rounds.
    const bool ext_converged = d_ext.size() == n;
    size_t reachable = 0;
    if (ext_converged)
        reachable = parlay::reduce(parlay::map(d_ext, [](long double dv) -> size_t {
            return unreached(dv) ? 0 : 1;
        }));

    const double gb_s = to_gb(m * sizeof(weighted_edge)) / op_s;
    std::cout << reachable << "/" << n << " vertices reachable   "
              << std::setprecision(4) << op_s << "s   "
              << std::setprecision(2) << gb_s << " GB/s (edges read)\n";
    if (!ext_converged)
        std::cout << "*** out-of-core Bellman-Ford did not converge within "
                  << n << " rounds ***\n";

    // external_bellman_ford_fast: same algorithm, one streaming pass over the
    // edges per round (ChunkSegmentedReduce) instead of a per-vertex reader
    // setup -- see external_bellman_ford.h for why that's expected to be
    // dramatically faster.
    std::cout << "Running out-of-core Bellman-Ford (fast)..." << std::flush;
    trace_mark(("fast_op_start_" + label).c_str());
    t0 = Clock::now();
    parlay::sequence<long double> d_fast = external_bellman_ford_fast(graph, start);
    const double fast_op_s = elapsed(t0);
    trace_mark(("fast_op_end_" + label).c_str());
    std::cout << " done\n";

    const bool fast_converged = d_fast.size() == n;
    size_t fast_reachable = 0;
    if (fast_converged)
        fast_reachable = parlay::reduce(parlay::map(d_fast, [](long double dv) -> size_t {
            return unreached(dv) ? 0 : 1;
        }));

    const double fast_gb_s = to_gb(m * sizeof(weighted_edge)) / fast_op_s;
    std::cout << fast_reachable << "/" << n << " vertices reachable   "
              << std::setprecision(4) << fast_op_s << "s   "
              << std::setprecision(2) << fast_gb_s << " GB/s (edges read)\n";
    if (!fast_converged)
        std::cout << "*** out-of-core Bellman-Ford (fast) did not converge within "
                  << n << " rounds ***\n";

    std::cout << "Running in-memory bellman_ford..." << std::flush;
    t0 = Clock::now();
    auto d_mem_opt = bellman_ford<long double>(start, WG);
    const double inmem_op_s = elapsed(t0);
    std::cout << " done (" << std::setprecision(4) << inmem_op_s << "s)\n";

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
        } else if (!d_mem_opt.has_value()) {
            std::cout << "in-mem bellman_ford: did not converge within " << n
                      << " rounds; skipped comparison\n";
        } else {
            std::cout << name << " result skipped comparison (no convergence)\n";
        }
        return ok;
    };

    const bool agree = compare_to_mem("out-of-core", d_ext, ext_converged);
    const bool fast_agree = compare_to_mem("out-of-core (fast)", d_fast, fast_converged);

    // Machine-readable line for benchmarks/run_benches.py.
    // Columns: case,n,m,build_s,op_s,inmem_op_s,reachable,throughput_gb_s,
    //          fast_op_s,fast_reachable,fast_throughput_gb_s
    auto f9 = [](double v) { std::ostringstream o; o << std::setprecision(9) << v; return o.str(); };
    std::cout << "CSV," << label << ',' << n << ',' << m << ',' << f9(build_s)
              << ',' << f9(op_s) << ',' << f9(inmem_op_s) << ',' << reachable
              << ',' << f9(gb_s) << ',' << f9(fast_op_s) << ',' << fast_reachable
              << ',' << f9(fast_gb_s) << '\n';

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

    struct GraphCase { std::string label; size_t avg_degree; };
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
