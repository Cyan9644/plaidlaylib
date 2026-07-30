// bfsExample — out-of-core BFS_simple vs the in-memory parlaylib reference,
// swept over sparse/balanced/dense RMAT graphs at the same n.
//
// BFS_simple (external_bfs.h) is the out-of-core analogue of parlaylib's BFS
// (examples/in_memory/graph/bfs.h, byte-identical to
// deps/parlaylib-examples/BFS.h): expand the frontier one level at a time,
// each round materializing every frontier vertex's adjacency and CAS-claiming
// unvisited neighbors into the next frontier. Unlike external_bellman_ford
// (which has a slow per-vertex method AND a fast streaming method), there is
// only one out-of-core BFS implementation here -- BFS_simple relaxes a vertex
// by calling chunk_csr::delay_get_adjacent + sequential_materialize (against a
// per-worker SequentialReadContext) PER FRONTIER VERTEX PER ROUND, the same
// "blocking pread per vertex, not buffered" cost profile external_bellman_ford's
// slow method has (see external_bfs.h). That is expected to make it dramatically slower than the
// in-memory baseline even at modest n; this benchmark exists to measure
// exactly that gap, not to hide it -- and, because there is no faster
// out-of-core alternative to fall back on, BFS_simple is never budget-gated
// off the way external_bellman_ford's slow method is: gating away the only
// out-of-core series would leave nothing to plot at larger n.
//
// Graph construction IS out-of-core (external_rmat.h), exactly as in
// bellman_ford.cpp: an RMAT edge is a pure function of its index, so the edge
// list is tabulated straight onto the drives, put into CSR order with
// direct_sample_sort, and deduped/projected/counted in one DensePackStream
// pass. The graph is symmetric (undirected), so its own adjacency doubles as
// its transpose, though BFS only ever walks it in one direction.
//
// BFS_simple ignores edge weights entirely (it only reads
// weighted_edge::connecting_vertex), so unlike bellman_ford.cpp the in-memory
// baseline needs no add_weights step: it builds directly with
// graph_utils<size_t>::rmat_symmetric_graph(n_req, m_req), the plain
// unweighted graph type BFS() (examples/in_memory/graph/bfs.h) expects. The
// out-of-core side still goes through external_rmat_symmetric_graph (which
// always produces a chunk_csr of weighted_edges), so weights exist on disk
// but are simply unused by BFS_simple -- both generators are still
// deterministic functions of (n_req, m_req) and use the same draws, so they
// produce the same graph (up to neighbor order within a row, which BFS
// membership cannot see), which is what makes the level-by-level cross-check
// below double as an end-to-end check that they agree.
//
// Defaults are deliberately tiny, for the same reason bellman_ford.cpp's are:
// BFS_simple's cost scales as O(sum of visited vertices' degrees) reader
// setups (a pread per vertex, once, in the round it's discovered), not O(m)
// bytes read, so it is expected to be dramatically slower than the in-memory
// baseline. Unlike bellman_ford.cpp, this benchmark does NOT skip the
// out-of-core method past a byte budget -- see above -- so keep any sweep's n
// small unless you're deliberately measuring the slow end of that gap. The
// in-memory baseline is still skipped past BFS_INMEM_MAX_N vertices (default
// 2^30, env-overridable), the same "stop paying DRAM cost once it stops being
// a useful baseline point" gate bellman_ford.cpp uses; it runs regardless of
// PLAID_TRACE (raise the env var on a box with enough DRAM to still want the
// comparison at larger n; see run_case()).
//
// A second, size-derived gate guards DRAM directly (BFS_BUILD_BUDGET_BYTES,
// default half of physical RAM, 0 disables), estimating the unavoidable
// out-of-core footprint (chunk_csr's degree_scan plus BFS_simple's
// atomic<bool> visited array) and the in-memory baseline's graph BEFORE
// allocating anything -- same reasoning as bellman_ford.cpp's build-budget
// gate. Both are far smaller here than Bellman-Ford's (no `long double`
// distance arrays, no weighted graph), so this budget is expected to bind
// only at vertex counts in the billions.
//
//   usage: bfsExample [global --flags] [n] [balanced_avg_degree] [case]
//     n                    requested vertex count, rounded up to a power of
//                          two by the RMAT generator (default 200); shared by
//                          all three cases
//     balanced_avg_degree  avg_degree for the "balanced" case only (default
//                          8, matching bellman_ford.cpp's driver); the
//                          "sparse" case always uses avg_degree 2, and
//                          "dense" always uses n/2
//     case                 "all" (default), or "sparse"/"balanced"/"dense" to
//                          run and print a CSV line for just that one case
//                          instead of all three -- used by
//                          benchmarks/run_benches.py's EXAMPLES registry /
//                          io_trace.py, which need exactly one CSV line per
//                          invocation (they keep only the last line seen)
//
// One CSV line per case:
//   CSV,case,n,m,build_s,op_s,inmem_op_s,levels,reachable,throughput_gb_s
//   throughput = edge bytes (m * sizeof(weighted_edge)) / op_s -- the same
//   "dataset size / op time" convention every other example here uses, not a
//   literal touched-bytes accounting (BFS_simple only ever reads edges
//   incident to a reached vertex, once). inmem_op_s is blank when the
//   in-memory baseline is skipped (n past BFS_INMEM_MAX_N).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unistd.h>

#include "absl/log/check.h"
#include "parlay/primitives.h"

// Upstream-shaped in-memory reference: RMAT graph generation and the plain
// parallel BFS, byte-identical to
// deps/parlaylib-examples/{helper/graph_utils.h,BFS.h}.
//
// Must be parsed BEFORE external_compressed_sparse_row.h below: that header
// #defines bare `vertex`/`weight` macros (size_t / long double) with no
// #undef, which would otherwise rewrite these headers' `vertex` template
// parameter (and every use of it) out from under them while they're being
// preprocessed. (external_bfs.h's own BFS_simple sidesteps this by naming its
// template parameter `V`, not `vertex` -- see its header comment.)
#include "ChunkSequence/examples/in_memory/graph/graph_utils/graph_utils.h"
#include "ChunkSequence/examples/in_memory/graph/bfs.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "utils/trace_marker.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/examples/external/bench_drives.h"
// Before external_bfs.h: this pulls in dense_pack/direct_samplesort, which
// must be parsed before external_compressed_sparse_row.h's bare macros.
#include "ChunkSequence/examples/external/graph_utils/external_rmat.h"
#include "ChunkSequence/examples/external/external_bfs.h"
#include "ChunkSequence/ExternalPrimitives/materialize.h"

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}
static double to_gb(size_t bytes) { return (double)bytes / (1024.0 * 1024.0 * 1024.0); }

static void cleanup_prefix(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++) unlink(GetFileName(prefix, d).c_str());
}

// chunk_seq only tracks byte counts per chunk (element type isn't known at
// that layer), so the element count of a size_t-element chunk_seq is the sum
// of each chunk's `used` bytes / sizeof(size_t) -- header-only, no I/O.
static size_t level_size(const chunk_seq& level) {
    size_t total = 0;
    for (const chunk& c : level.chunks) total += c.used / sizeof(size_t);
    return total;
}

using vertex_utils = graph_utils<size_t>;

// Builds one RMAT graph (n_req vertices, avg_degree * n_req edges before
// rounding/symmetrizing), runs BFS_simple and the in-memory BFS baseline from
// the same start vertex, cross-checks them level-by-level, prints a summary
// and a CSV line, and cleans up the files it wrote. Returns true iff the two
// sides agree (or the in-memory baseline was skipped -- nothing to compare).
static bool run_case(const std::string& label, size_t n_req, size_t avg_degree) {
    const std::string edge_prefix = "bfs_edges_" + label;

    std::cout << "\n=== case: " << label << " (avg_degree " << avg_degree
              << ") ===\n";

    // n is a pure function of n_req (RMAT rounds up to a power of two), so
    // every budget decision below can be made BEFORE anything is allocated.
    const size_t n = size_t{1} << (int)std::round(std::log2((double)n_req));
    const size_t m_req = avg_degree * n_req;   // directed edges, pre-dedup

    // Two DRAM gates, both evaluated up front -- same shape as
    // bellman_ford.cpp's, but far smaller: BFS_simple needs no `long double`
    // distance arrays and the in-memory baseline needs no weighted graph.
    //
    // ext_bytes is what the out-of-core path itself cannot avoid: chunk_csr's
    // degree_scan (8 bytes/vertex, DRAM-resident by construction --
    // external_compressed_sparse_row.h) plus BFS_simple's atomic<bool>
    // visited array (1 byte/vertex).
    //
    // inmem_bytes is the in-memory baseline's unweighted graph: ~16
    // bytes/vertex of parlay::sequence outer-array overhead (G plus
    // symmetrize's transpose) plus ~24 bytes per requested edge (8 in G, 8 in
    // GT, 8 in symmetrize's append/remove_duplicates copies).
    const size_t ext_bytes   = 8 * (n + 1) + n;
    const size_t inmem_bytes = 16 * n + 24 * m_req;

    size_t inmem_max_n = 1ull << 30;
    if (const char* e = getenv("BFS_INMEM_MAX_N"))
        inmem_max_n = std::stoull(e);
    bool inmem_ok = n <= inmem_max_n;

    const size_t phys = (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
    size_t budget = phys / 2;
    if (const char* e = getenv("BFS_BUILD_BUDGET_BYTES"))
        budget = std::stoull(e);

    if (budget != 0 && ext_bytes > budget) {
        // Nothing here is runnable. Print no CSV line at all and return
        // success: run_benches.py drops a CSV-less point with a warning and
        // keeps sweeping, which is what we want instead of an OOM kill.
        std::cout << "SKIPPED: n=" << n << " needs " << std::setprecision(3)
                  << to_gb(ext_bytes) << " GB of DRAM for degree_scan + visited "
                  << "alone, past the " << to_gb(budget) << " GB budget "
                  << "(BFS_BUILD_BUDGET_BYTES)\n";
        return true;
    }
    if (inmem_ok && budget != 0 && ext_bytes + inmem_bytes > budget) {
        // Only the baseline doesn't fit -- drop it and still run (and report)
        // BFS_simple rather than losing the whole point.
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
    // BFS_simple only ever reads connecting_vertex, so the weights exist on
    // disk but are unused. Nothing graph-sized is ever DRAM-resident here
    // except degree_scan; see external_rmat.h.
    chunk_csr graph = ExternalGraphUtils::external_rmat_symmetric_graph(
        n_req, m_req, edge_prefix, /*minw=*/1.0L, /*maxw=*/20.0L);
    const size_t m = graph.degree_scan[n];

    const double build_s = elapsed(t0);
    trace_mark(("build_end_" + label).c_str());
    std::cout << " done (" << n << " vertices, " << m << " edges, "
              << std::fixed << std::setprecision(4) << build_s << "s)\n";

    // The baseline's own copy of the same graph, in DRAM, unweighted (BFS
    // doesn't need add_weights). Built independently rather than shared with
    // the out-of-core path: both generators are deterministic functions of
    // (n_req, m_req), so building twice turns the level-by-level cross-check
    // below into an end-to-end check that they agree. Not counted in build_s,
    // which times the out-of-core build only.
    vertex_utils::graph G;
    if (inmem_ok) {
        std::cout << "Generating the in-memory baseline's copy..." << std::flush;
        G = vertex_utils::rmat_symmetric_graph((long)n_req, (long)m_req);
        std::cout << " done\n";
    }

    const size_t start = 0;

    bench_drives::settle_drives();   // isolate the timed read from the build's writeback

    // BFS_simple: same algorithm as the in-memory BFS below, but each round
    // materializes every frontier vertex's adjacency with a fresh delayed cut
    // (a blocking pread per vertex, not buffered -- see external_bfs.h), so
    // it's expected to be dramatically slower than the in-memory baseline
    // even at modest edge counts. Unlike external_bellman_ford's per-vertex
    // method, this is never budget-skipped: it's the only out-of-core series
    // here, so gating it off would leave nothing to plot.
    std::cout << "Running out-of-core BFS (BFS_simple)..." << std::flush;
    trace_mark(("op_start_" + label).c_str());
    t0 = Clock::now();
    parlay::sequence<chunk_seq> ext_frontiers = BFS_simple(start, graph);
    const double op_s = elapsed(t0);
    trace_mark(("op_end_" + label).c_str());
    std::cout << " done\n";

    const size_t ext_levels = ext_frontiers.size();
    size_t ext_reachable = 0;
    for (const auto& level : ext_frontiers) ext_reachable += level_size(level);

    const size_t edge_bytes = m * sizeof(weighted_edge);
    const double gb_s = to_gb(edge_bytes) / op_s;
    std::cout << ext_levels << " levels   " << ext_reachable << "/" << n
              << " vertices reachable   " << std::setprecision(4) << op_s
              << "s   " << std::setprecision(2) << gb_s << " GB/s (edges read)\n";

    // `inmem_ok` was decided at the top of run_case (before the build) and G
    // was only populated if it held -- same reasoning as bellman_ford.cpp:
    // a sweep's smaller points all run this binary WITHOUT PLAID_TRACE
    // (io_trace.py traces only the largest by default), so a sweep whose
    // smaller points still exceed a couple billion vertices would otherwise
    // pay full DRAM cost on every one of them, not just the largest.
    double inmem_op_s = 0;
    parlay::sequence<parlay::sequence<size_t>> mem_frontiers;
    if (!inmem_ok) {
        std::cout << "Running in-memory BFS: skipped (n " << n
                  << ", inmem_max_n " << inmem_max_n << ")\n";
    } else {
        std::cout << "Running in-memory BFS..." << std::flush;
        t0 = Clock::now();
        mem_frontiers = BFS(start, G);
        inmem_op_s = elapsed(t0);
        std::cout << " done (" << std::setprecision(4) << inmem_op_s << "s)\n";
    }

    // Cross-check: level counts must match, and for each level, the sorted
    // vertex sets must match. Level MEMBERSHIP is a deterministic function of
    // the graph (shortest hop-count from start), but the order vertices land
    // in within a level is racy (parallel CAS discovery among several
    // in-frontier neighbors), so an unsorted compare would spuriously fail.
    bool agree = true;
    if (inmem_ok) {
        if (mem_frontiers.size() != ext_levels) {
            std::cout << "*** MISMATCH: in-mem " << mem_frontiers.size()
                      << " levels != out-of-core " << ext_levels << " ***\n";
            agree = false;
        } else {
            for (size_t lvl = 0; lvl < ext_levels && agree; lvl++) {
                parlay::sequence<size_t> ext_level =
                    ChunkSequenceOps::materialize<size_t>(ext_frontiers[lvl]);
                parlay::sequence<size_t> mem_level = mem_frontiers[lvl];
                if (ext_level.size() != mem_level.size()) {
                    std::cout << "*** MISMATCH at level " << lvl << ": in-mem "
                              << mem_level.size() << " vertices != out-of-core "
                              << ext_level.size() << " ***\n";
                    agree = false;
                    break;
                }
                parlay::sort_inplace(ext_level);
                parlay::sort_inplace(mem_level);
                for (size_t i = 0; i < ext_level.size(); i++) {
                    if (ext_level[i] != mem_level[i]) {
                        std::cout << "*** MISMATCH at level " << lvl
                                  << ", position " << i << ": in-mem "
                                  << mem_level[i] << " != out-of-core "
                                  << ext_level[i] << " ***\n";
                        agree = false;
                        break;
                    }
                }
            }
        }
        if (agree) std::cout << "cross-check OK: " << ext_levels
                              << " levels agree\n";
    } else {
        std::cout << "cross-check skipped (in-memory BFS not run)\n";
    }

    // Machine-readable line for benchmarks/run_benches.py.
    // Columns: case,n,m,build_s,op_s,inmem_op_s,levels,reachable,throughput_gb_s
    // (inmem_op_s blank past inmem_max_n vertices, where it was never run.)
    auto f9 = [](double v) { std::ostringstream o; o << std::setprecision(9) << v; return o.str(); };
    std::cout << "CSV," << label << ',' << n << ',' << m << ',' << f9(build_s)
              << ',' << f9(op_s)
              << ',' << (inmem_ok ? f9(inmem_op_s) : std::string())
              << ',' << ext_levels << ',' << ext_reachable
              << ',' << f9(gb_s) << '\n';

    cleanup_prefix(edge_prefix);
    // BFS_simple leaves one chunk_seq per round ("bfs_frontier0",
    // "bfs_frontier1", ... -- an unknown count up front, each possibly
    // sharded across several drives), which cleanup_prefix's fixed
    // per-drive-index enumeration can't catch; a directory-scan cleanup does.
    bench_drives::clear_drives({"bfs_frontier"});
    return agree;
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    // BFS_simple materializes one delayed cut per frontier vertex per round
    // against a per-worker SequentialReadContext (bounded-LRU fd cache, see
    // external_bfs.h) -- num_workers() * SequentialReadContext::MAX_CACHED_FDS
    // fds at steady state, plus the CSR build's own readers/writers, can still
    // exceed the common 1024 soft RLIMIT_NOFILE. Lift the soft limit to the
    // hard limit before any I/O starts (same fix every other external example
    // applies).
    RaiseFdLimit();
    const size_t n_req = (argc > 1) ? std::stoull(argv[1]) : 200;
    const size_t balanced_avg_degree = (argc > 2) ? std::stoull(argv[2]) : 8;
    const std::string case_filter = (argc > 3) ? argv[3] : "all";

    // Sparse/balanced/dense span avg_degree from a bare-spanning-tree-ish 2,
    // through the parlaylib driver's own default of 8, up to n/2 (m ~ n^2/2)
    // -- "dense" is only a jump in edge count, not a different generator; see
    // bellman_ford.cpp's file header for why RMAT stays skewed even at that
    // end.
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
