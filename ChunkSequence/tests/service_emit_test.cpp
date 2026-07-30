// serviceEmitTest — correctness + streaming-parity of the demand-driven IO service
// (indexed_io_service.h) and the emit primitive built on it (chunk_emit.h).
//
//  A. KMP three-way differential.  The SAME text is searched three ways:
//       - windowed  : ChunkKmp        (pre-read window + halo on DensePackStream)
//       - service   : ChunkKmpService (free in[g] indexing via the shared service)
//       - reference : a streaming char-by-char KMP in DRAM (regenerates the text)
//     All three must agree exactly (count + positions, text order).  This proves the
//     free-indexing body (which crosses chunk boundaries with plain in[g], no halo)
//     matches the bespoke windowed KMP.  Search times are printed so the service can
//     be checked against the windowed path (streaming parity is the goal; on tmpfs
//     both are memory-bound so the ratio is only indicative).
//
//  B. Irregular gather differential.  out[j] = data.get(idx[j]) for a data-dependent
//     idx (random, with repeats) — an access pattern a contiguous window cannot
//     express — driven straight through ServiceView, checked element-wise against a
//     DRAM gather.  Exercises the shared cache's coalescing on scattered 4 KiB reads.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

#include "parlay/primitives.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_emit.h"
#include "ChunkSequence/indexed_io_service.h"
#include "ChunkSequence/examples/chunk_kmp.h"

static constexpr size_t CHARS_PER_CHUNK = CHUNK_SIZE / sizeof(char);

using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}
static void cleanup(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++) unlink(GetFileName(prefix, d).c_str());
}

// Streaming reference KMP (regenerates the text from f, so no O(n) buffer needed).
static std::vector<uint64_t> reference_kmp(
    size_t n, const std::function<char(size_t)>& f, const std::string& pat) {
    const long m = (long)pat.size();
    std::vector<long> failure(m, -1);
    for (long r = 1, l = -1; r < m; r++) {
        while (l != -1 && pat[l + 1] != pat[r]) l = failure[l];
        if (pat[l + 1] == pat[r]) failure[r] = ++l;
    }
    std::vector<uint64_t> out;
    long tail = -1;
    for (size_t i = 0; i < n; i++) {
        const char c = f(i);
        while (tail != -1 && c != pat[tail + 1]) tail = failure[tail];
        if (c == pat[tail + 1]) tail++;
        if (tail == m - 1) { out.push_back((uint64_t)(i - (size_t)tail)); tail = failure[tail]; }
    }
    return out;
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    RaiseFdLimit();
    namespace ops = ChunkSequenceOps;

    int fails = 0;
    auto expect = [&](bool ok, const std::string& msg) {
        if (!ok) { std::cerr << "FAIL: " << msg << "\n"; fails++; }
    };
    auto same = [](const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); i++) if (a[i] != b[i]) return false;
        return true;
    };

    std::cout << "serviceEmitTest: workers=" << parlay::num_workers()
              << ", CHARS_PER_CHUNK=" << CHARS_PER_CHUNK << "\n";

    // ── A. KMP three-way differential (+ timing) ────────────────────────────────
    auto kmp_case = [&](const std::string& name, size_t n,
                        const std::function<char(size_t)>& f, const std::string& pat) {
        chunk_seq text = ops::tabulate<char>(n, "se_text", f);

        // Baseline: the bespoke windowed KMP (pre-read window + halo).
        auto t0 = Clock::now();
        chunk_seq mw = ops::ChunkKmp<char>(text, "se_kw", pat);
        const double tw = secs(t0);

        // Service KMP, written INLINE as a parallel-for over chunks — no bespoke
        // primitive, no halo, no templated helper.  KMP preprocessing (failure
        // function) happens once; then ChunkEmitService spawns one task per input
        // chunk, and the task scans its own range [in.lo(), in.hi()) with a plain
        // KMP automaton, reading in[g] (which transparently crosses into the next
        // chunk), and emits every match that STARTS in its chunk.
        const long m = (long)pat.size();
        std::vector<long> fail(m, -1);
        for (long r = 1, l = -1; r < m; r++) {
            while (l != -1 && pat[l + 1] != pat[r]) l = fail[l];
            if (pat[l + 1] == pat[r]) fail[r] = ++l;
        }
        t0 = Clock::now();
        chunk_seq ms = ops::ChunkEmitService<char, uint64_t>(text, "se_ks",
            [&](ops::ServiceView<char>& in) {
                parlay::sequence<uint64_t> out;
                long tail = -1;
                for (size_t g = in.lo(); g < in.n() && (long)g - tail <= (long)in.hi(); g++) {
                    const char c = in[g];
                    while (tail != -1 && c != pat[tail + 1]) tail = fail[tail];
                    if (c == pat[tail + 1]) tail++;
                    if (tail == m - 1) {
                        const size_t start = g - (size_t)(m - 1);
                        if (start < in.hi()) out.push_back(start);
                        tail = fail[tail];
                    }
                }
                return out;
            });
        const double ts = secs(t0);

        std::vector<uint64_t> vw = mw.to_vector<uint64_t>();
        std::vector<uint64_t> vs = ms.to_vector<uint64_t>();
        std::vector<uint64_t> ref = reference_kmp(n, f, pat);

        expect(same(vw, ref), name + ": windowed KMP != reference");
        expect(same(vs, ref), name + ": service KMP != reference");
        expect(same(vw, vs),  name + ": service KMP != windowed KMP");

        std::cout << "  A " << std::left << std::setw(22) << name
                  << " matches=" << ref.size()
                  << "  windowed " << std::fixed << std::setprecision(4) << tw << "s"
                  << "  service " << ts << "s"
                  << "  (service/windowed " << std::setprecision(2)
                  << (ts > 0 ? tw / ts : 0.0) << "x)\n";

        cleanup("se_text"); cleanup("se_kw"); cleanup("se_ks");
    };

    const size_t nA = (argc > 1) ? std::stoull(argv[1]) : (4 * CHARS_PER_CHUNK + 7777);
    // Random 4-letter text, self-overlapping pattern (boundary-spanning matches).
    kmp_case("random4_abab", nA,
             [](size_t i) { return (char)('a' + parlay::hash64(i) % 4); }, "abab");
    // All-'a' text, "aaaa": a match at nearly every position — heavy boundary
    // crossing and many full output chunks (stresses in[g] across chunk seams).
    kmp_case("all_a_dense", 2 * CHARS_PER_CHUNK + 5,
             [](size_t) { return 'a'; }, "aaaa");

    // ── B. Irregular gather differential ────────────────────────────────────────
    {
        const size_t ng = 4 * ELEMS_PER_CHUNK + 321;      // uint64 data
        chunk_seq data = ops::tabulate<uint64_t>(ng, "se_data",
                            [](size_t i) { return parlay::hash64(i); });
        std::vector<uint64_t> vals = data.to_vector<uint64_t>();   // DRAM reference

        const size_t k = std::min<size_t>(ng, 200000);
        std::vector<size_t>   idx(k);
        std::vector<uint64_t> ref(k), out(k);
        for (size_t j = 0; j < k; j++) {
            idx[j] = parlay::hash64(j * 2 + 1) % ng;               // random, with repeats
            ref[j] = vals[idx[j]];
        }

        auto& svc = ops::IndexedIoService::instance();
        ops::IndexedIoService::HandleState* h =
            svc.open<uint64_t>(data, /*block_bytes=*/O_DIRECT_MULTIPLE);  // 4 KiB scattered

        const size_t W = std::max<size_t>(1, std::min<size_t>(parlay::num_workers(), k));
        const size_t seg = (k + W - 1) / W;
        auto t0 = Clock::now();
        parlay::parallel_for(0, W, [&](size_t w) {
            const size_t lo = w * seg, hi = std::min(k, lo + seg);
            if (lo >= hi) return;
            ops::ServiceView<uint64_t> in(svc, h, 0, ng);
            for (size_t j = lo; j < hi; j++) out[j] = in.get(idx[j]);
        }, /*granularity=*/1);
        const double gs = secs(t0);
        svc.close(h);

        bool ok = true;
        for (size_t j = 0; j < k && ok; j++) ok = (out[j] == ref[j]);
        expect(ok, "irregular gather != DRAM gather");
        std::cout << "  B gather              k=" << k << "  " << std::fixed
                  << std::setprecision(4) << gs << "s  ("
                  << (ok ? "matches DRAM" : "MISMATCH") << ")\n";

        cleanup("se_data");
    }

    std::cout << (fails == 0 ? "PASS" : "FAIL") << "  fails=" << fails << "\n";
    return fails == 0 ? 0 : 1;
}
