// Correctness test for the out-of-core upper convex hull (ChunkSequenceOps::UpperHull).
//
// Each case builds an n-point cloud on the SSDs, runs UpperHull with a *tiny*
// DRAM budget so the recursion is forced through many out-of-core split levels
// (exercising ChunkReduce/ChunkFilter + the file plumbing rather than
// short-circuiting into the in-memory base case), and checks the result two ways:
//
//   1. Differential: it must be byte-for-byte the same hull (same vertex indices,
//      same left-to-right order) as parlaylib's in-DRAM upper_hull on the same
//      points.  This pits the whole out-of-core machinery against an independent
//      implementation.
//   2. Independent geometry: the hull must run strictly left to right from the
//      min-x to the max-x point, and no input point may lie strictly above any
//      hull edge (so it really is the upper boundary) — a check that would catch
//      a bug shared by both quickhull implementations.
//
// Exits 0 iff every case passes.

#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

#include "parlay/primitives.h"

// Upstream in-DRAM baseline (global point/area/quickhull/upper_hull, no guard).
#include "parlaylib-examples/quickhull.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/examples/chunk_convex_hull.h"

using ChunkSequenceOps::hpoint;
using ChunkSequenceOps::area;

static void cleanup_prefix(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++) unlink(GetFileName(prefix, d).c_str());
}

// Independent geometric validation of an upper hull over the point set `pts`.
static bool geometry_ok(const std::vector<hpoint>& pts,
                        const std::vector<uint64_t>& hull) {
    if (hull.size() < 2) {
        std::cout << "    FAIL geometry: degenerate hull of size " << hull.size() << "\n";
        return false;
    }
    // Recover hull vertex coordinates by original index.
    std::vector<hpoint> H;
    H.reserve(hull.size());
    for (uint64_t id : hull) H.push_back(pts[id]);

    // Endpoints must be the (x, then y)-min and -max points (upstream's
    // convention: the hull runs from the lowest leftmost to the highest
    // rightmost point, so a vertical run at an extreme x is allowed).
    auto pntless = [](const hpoint& a, const hpoint& b) {
        return (a.x < b.x) || (a.x == b.x && a.y < b.y);
    };
    hpoint mn = pts[0], mx = pts[0];
    for (const hpoint& p : pts) {
        if (pntless(p, mn)) mn = p;
        if (pntless(mx, p)) mx = p;
    }
    if (H.front().idx != mn.idx || H.back().idx != mx.idx) {
        std::cout << "    FAIL geometry: hull endpoints are not the (x,y) extremes\n";
        return false;
    }
    // x must be non-decreasing along the hull.
    for (size_t i = 0; i + 1 < H.size(); i++)
        if (H[i].x > H[i + 1].x) {
            std::cout << "    FAIL geometry: x decreases at hull vertex " << i << "\n";
            return false;
        }

    // No input point may sit strictly above the hull edge spanning its x.
    const double eps = 1e-9;
    size_t seg = 0;
    // Sort a copy of points by x to sweep edges in order.
    std::vector<hpoint> byx = pts;
    std::sort(byx.begin(), byx.end(),
              [](const hpoint& a, const hpoint& b) { return a.x < b.x; });
    for (const hpoint& p : byx) {
        while (seg + 2 < H.size() && p.x > H[seg + 1].x) seg++;
        // p.x within [H[seg].x, H[seg+1].x]; above means left of the directed edge.
        if (area(H[seg], H[seg + 1], p) > eps) {
            std::cout << "    FAIL geometry: point (" << p.x << "," << p.y
                      << ") lies above hull edge " << seg << "\n";
            return false;
        }
    }
    return true;
}

static bool run_case(const std::string& name, size_t n,
                     const std::function<hpoint(size_t)>& gen,
                     bool expect_ext_recursion) {
    std::cout << "  " << name << "  (n=" << n << ")\n" << std::flush;

    const std::string in_prefix = "cht_in";
    chunk_seq points = ChunkSequenceOps::tabulate<hpoint>(n, in_prefix, gen);

    // Tiny DRAM budget (128 points) so the recursion is forced out-of-core.
    const size_t budget = 128 * sizeof(hpoint);
    std::vector<uint64_t> hull = ChunkSequenceOps::UpperHull(points, budget, "cht_s");
    const size_t splits = ChunkSequenceOps::last_ext_splits();

    // Independent in-DRAM baseline on the same points.
    std::vector<hpoint> pts(n);
    for (size_t i = 0; i < n; i++) pts[i] = gen(i);
    pointseq Pts = parlay::tabulate(n, [&](size_t i) { return point{pts[i].x, pts[i].y}; });
    intseq hull_mem = upper_hull(Pts);

    bool pass = true;

    // 1. Differential: identical hull vs upstream.
    if (hull.size() != hull_mem.size()) {
        std::cout << "    FAIL count: out-of-core=" << hull.size()
                  << " in-mem=" << hull_mem.size() << "\n";
        pass = false;
    } else {
        for (size_t i = 0; i < hull.size(); i++)
            if (hull[i] != (uint64_t)hull_mem[i]) {
                std::cout << "    FAIL match at " << i << ": out-of-core=" << hull[i]
                          << " in-mem=" << hull_mem[i] << "\n";
                pass = false;
                break;
            }
        if (pass) std::cout << "    match  OK (" << hull.size() << " vertices)\n";
    }

    // 2. Independent geometry.
    if (pass && !geometry_ok(pts, hull)) pass = false;
    else if (pass) std::cout << "    geometry OK\n";

    // 3. The recursion actually left DRAM when we expected it to.
    if (expect_ext_recursion && splits == 0) {
        std::cout << "    FAIL: expected out-of-core recursion but none occurred\n";
        pass = false;
    } else {
        std::cout << "    out-of-core split levels: " << splits << "\n";
    }

    cleanup_prefix(in_prefix);
    return pass;
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 50'000;
    const double inv = 1.0 / 18446744073709551616.0;   // 1 / 2^64

    bool ok = true;

    // Uniform random cloud (interior points dominate; small hull).
    ok &= run_case("uniform random", n, [inv](size_t i) {
        double x = (double)parlay::hash64(2 * i)     * inv;
        double y = (double)parlay::hash64(2 * i + 1) * inv;
        return hpoint{x, y, (uint64_t)i, 0};
    }, /*expect_ext_recursion=*/true);

    // Points on a downward parabola: (almost) all are upper-hull vertices, which
    // stresses a deep, unbalanced quickhull recursion.
    ok &= run_case("parabola (mostly on hull)", n, [n](size_t i) {
        double x = (double)i / (double)n;          // distinct, increasing x
        double y = 1.0 - (x - 0.5) * (x - 0.5);    // concave: every point extreme
        return hpoint{x, y, (uint64_t)i, 0};
    }, /*expect_ext_recursion=*/true);

    // Points on a circle: all on the (full) hull; the upper arc is the result.
    ok &= run_case("circle", n, [n](size_t i) {
        double t = 6.283185307179586 * (double)i / (double)n;
        return hpoint{std::cos(t), std::sin(t), (uint64_t)i, 0};
    }, /*expect_ext_recursion=*/true);

    // Small hand case: a square with interior points (hull = the two top corners).
    ok &= run_case("small square + interior", 6, [](size_t i) {
        static const double xs[6] = {0, 1, 0, 1, 0.5, 0.3};
        static const double ys[6] = {0, 0, 1, 1, 0.5, 0.7};
        return hpoint{xs[i], ys[i], (uint64_t)i, 0};
    }, /*expect_ext_recursion=*/false);

    std::cout << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
}
