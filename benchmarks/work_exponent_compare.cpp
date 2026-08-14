// Synthetic "work per element" benchmark.
//
// Every other benchmark in this repo times a REAL algorithm's fixed, trivial
// per-element cost (delayed_compare.cpp's add1/mul2 are O(1)/element).  That
// leaves no way to show, in general, the observed sweet spot that total work
// should scale like O(n log n) -- i.e. O(log n) work per element -- for an
// out-of-core run to stay both (a) tractable as n grows past the DRAM-resident
// regime and (b) compute-dense enough per chunk to hide io_uring/O_DIRECT read
// latency. This binary isolates "work per element" (and, separately, "number
// of full-sequence passes") as an explicit, dialable variable independent of
// any specific algorithm's semantics, so the two failure modes can be measured
// directly as n and the work exponent both vary.
//
// Two axes, modeling the two structurally different ways real O(n log n)
// algorithms in this repo realize their log-n factor:
//
//   elemwork  one ChunkMap call; f(x) = x ^ churn(x, k), k = k(n, mode).
//             Models arithmetic density growing within a single pass (as in
//             fft.cpp's four-step transform). Pushing k too high inflates CPU
//             time per pass at fixed I/O bytes -- visibly compute-bound.
//   rounds    R(n, mode) back-to-back ChunkReduce(sum) passes over the same
//             input. Models repeated full-sequence I/O passes (as in dc3's
//             prefix-doubling rounds, convex_hull's split levels,
//             external_bellman_ford_fast's rounds). Pushing R too high
//             inflates total bytes moved (R*n) -- stays I/O-bound throughout,
//             just does more I/O.
//
// modes: const (k or R independent of n), log (k0*log2(n)), sqrt (k0*sqrt(n)),
// linear (k0*n -- deliberately intractable at scale; only ever meant to be
// swept over a small n range to show the wall-clock cliff qualitatively, see
// the WORK_EXPONENT_MAX_TOTAL_OPS guardrail below).
//
// Correctness: an always-on cheap check reads back output chunk 0 (elemwork)
// or re-checks round-to-round agreement (rounds), independent of n; a
// RAM-budget-gated full check (WORK_EXPONENT_INMEM_BUDGET_BYTES, same pattern
// as delayed_compare.cpp's DELAYED_INMEM_BUDGET_BYTES) additionally verifies
// the full sum against an in-memory computation when n fits.
//
// CSV,<axis>,<mode>,<n>,<k0>,<k_effective>,<total_ops>,<build_s>,<op_s>,
//     <throughput_gb_s>,<agree>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "ChunkSequence/Primitives/map.h"
#include "ChunkSequence/Primitives/reduce.h"
#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/helper/bench_drives.h"
#include "absl/log/check.h"
#include "configs.h"
#include "parlay/primitives.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "utils/io_backend.h"
#include "utils/trace_marker.h"

// Data-dependent mixing loop: each iteration depends on the previous (no
// hoisting/CSE across iterations) and the result is written to output (no
// dead-code elimination of the loop itself). k=0 degenerates to the identity.
static inline uint64_t churn(uint64_t x, uint64_t k) {
  uint64_t s = x;
  for (uint64_t i = 0; i < k; i++) {
    s += 0x9E3779B97F4A7C15ULL;
    s = (s ^ (s >> 30)) * 0xBF58476D1CE4E5B9ULL;
  }
  return s;
}

struct SumMonoid {
  uint64_t identity = 0;
  uint64_t operator()(uint64_t a, uint64_t b) const { return a + b; }
};

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

static size_t chunk_seq_bytes(const chunk_seq& seq) {
  size_t total = 0;
  for (const auto& c : seq.chunks) total += c.used;
  return total;
}
static double to_gb(size_t bytes) {
  return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static void cleanup_prefix(const std::string& prefix) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    plaid::io::Unlink(GetFileName(prefix, d).c_str());
}

// k(n, mode) for the elemwork axis (also reused as R(n, mode) for rounds).
static size_t k_for(size_t n, const std::string& mode, size_t k0) {
  if (mode == "const") return k0;
  if (mode == "log")
    return k0 * (size_t)std::ceil(std::log2((double)std::max<size_t>(n, 2)));
  if (mode == "sqrt") return k0 * (size_t)std::ceil(std::sqrt((double)n));
  if (mode == "linear") return k0 * n;
  std::cerr << "unknown mode '" << mode << "' (want const|log|sqrt|linear)\n";
  std::exit(2);
}

// Read back output chunk 0 (the ChunkMap<uint64_t> output, index-ordered) and
// compare every element against churn() applied locally -- independent of n,
// catches an elided busy-loop or a wrong output regardless of scale.
static bool verify_chunk0(const chunk_seq& out, uint64_t k) {
  if (out.chunks.empty()) return true;
  const chunk& c = out.chunks[0];
  CHECK(c.index == 0);
  const size_t count = c.used / sizeof(uint64_t);
  if (count == 0) return true;

  int fd = plaid::io::Open(c.filename.c_str(), O_DIRECT | O_RDONLY);
  if (fd < 0) {
    std::cerr << "  FAIL chunk 0: open(" << c.filename
              << "): " << strerror(errno) << "\n";
    return false;
  }
  const size_t read_size = AlignUp(c.used);
  uint64_t* buf =
      (uint64_t*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, read_size);
  CHECK(buf != nullptr);
  const ssize_t got = plaid::io::Pread(fd, buf, read_size, (off_t)c.begin_addr);
  plaid::io::Close(fd);
  if (got < 0 || (size_t)got < c.used) {
    std::cerr << "  FAIL chunk 0: pread got " << got << " expected at least "
              << c.used << "\n";
    free(buf);
    return false;
  }

  bool ok = true;
  for (size_t i = 0; i < count && ok; i++) {
    const uint64_t expected =
        i ^ churn(i, k);  // input chunk 0 holds values [0, count)
    if (buf[i] != expected) {
      std::cerr << "  FAIL chunk 0 element " << i << ": got " << buf[i]
                << " expected " << expected << "\n";
      ok = false;
    }
  }
  free(buf);
  return ok;
}

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : (1ULL << 24);
  const std::string axis = (argc > 2) ? argv[2] : "elemwork";
  const std::string mode = (argc > 3) ? argv[3] : "log";
  const size_t k0 = (argc > 4) ? std::stoull(argv[4]) : 16;

  if (axis != "elemwork" && axis != "rounds") {
    std::cerr << "unknown axis '" << axis << "' (want elemwork|rounds)\n";
    return 2;
  }

  const size_t k_effective = k_for(n, mode, k0);
  const size_t total_ops = (axis == "elemwork") ? k_effective * n : k_effective;

  if (mode == "linear") {
    size_t max_total_ops = 1ULL << 34;
    if (const char* e = getenv("WORK_EXPONENT_MAX_TOTAL_OPS"))
      max_total_ops = std::stoull(e);
    if (total_ops > max_total_ops) {
      std::cerr << "refusing to run: mode=linear total_ops=" << total_ops
                << " exceeds WORK_EXPONENT_MAX_TOTAL_OPS=" << max_total_ops
                << " (this mode is only meant for small, separate n sweeps)\n";
      return 2;
    }
  }

  const std::string label = axis + "_" + mode;

  std::cout << "n=" << n << "  axis=" << axis << "  mode=" << mode
            << "  k0=" << k0 << "  k_effective=" << k_effective
            << "  total_ops=" << total_ops << "\n";

  // ── RAM budget for the in-memory cross-check ─────────────────────────────
  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys / 2;
  if (const char* e = getenv("WORK_EXPONENT_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  const size_t in_bytes_logical = n * sizeof(uint64_t);
  const bool inmem_ok = in_bytes_logical <= budget;

  // ── build (untimed) ───────────────────────────────────────────────────────
  trace_mark(("build_start_" + label).c_str());
  std::cout << "Generating chunk_seq iota(" << n << ")..." << std::flush;
  auto tb0 = Clock::now();
  const chunk_seq cseq = plaid::iota(n);
  const double build_s = elapsed(tb0);
  const size_t in_bytes = chunk_seq_bytes(cseq);
  std::cout << " " << cseq.chunks.size() << " chunks, " << std::fixed
            << std::setprecision(3) << to_gb(in_bytes) << " GB\n";
  trace_mark(("build_end_" + label).c_str());

  parlay::sequence<uint64_t> A;
  if (inmem_ok && axis == "elemwork") {
    A = parlay::tabulate(n, [](size_t i) { return (uint64_t)i; });
  }

  bench_drives::settle_drives();  // isolate the timed op from build writeback

  bool agree = true;
  auto check = [&](const char* what, bool ok) {
    if (!ok) {
      std::cout << "  *** MISMATCH: " << what << " ***\n";
      agree = false;
    }
  };

  double op_s = 0;
  uint64_t result_sum = 0;

  if (axis == "elemwork") {
    std::cout << "--- elemwork/" << mode << ": ChunkMap(x ^ churn(x,k)) ---\n";
    const uint64_t k = k_effective;
    auto f = [k](uint64_t x) { return x ^ churn(x, k); };

    trace_mark(("op_start_" + label).c_str());
    auto t0 = Clock::now();
    chunk_seq out = plaid::ChunkMap<uint64_t>(cseq, "bw_we_m", f);
    op_s = elapsed(t0);
    trace_mark(("op_end_" + label).c_str());

    agree = verify_chunk0(out, k);
    check("chunk0 readback matches churn()", agree);

    if (inmem_ok) {
      result_sum = plaid::ChunkReduce<uint64_t>(out, SumMonoid{});
      const parlay::plus<uint64_t> psum{};
      uint64_t expected = parlay::reduce(parlay::map(A, f), psum);
      check("full-sequence sum matches in-memory", result_sum == expected);
    }
    cleanup_prefix("bw_we_m");
  } else {
    std::cout << "--- rounds/" << mode << ": " << k_effective
              << " back-to-back ChunkReduce(sum) passes ---\n";
    const size_t R = k_effective;
    uint64_t acc = 0;
    uint64_t first_round = 0;
    bool rounds_agree = true;

    trace_mark(("op_start_" + label).c_str());
    auto t0 = Clock::now();
    for (size_t r = 0; r < std::max<size_t>(R, 1); r++) {
      uint64_t s = plaid::ChunkReduce<uint64_t>(cseq, SumMonoid{});
      if (r == 0)
        first_round = s;
      else if (s != first_round)
        rounds_agree = false;
      acc ^= s;  // prevents the loop from being optimized away
    }
    op_s = elapsed(t0);
    trace_mark(("op_end_" + label).c_str());
    std::cout << "  (xor-accumulated round results: " << acc << ")\n";

    check("every round's sum matches the first round's", rounds_agree);
    agree = rounds_agree;
    result_sum = first_round;

    if (inmem_ok) {
      const parlay::plus<uint64_t> psum{};
      uint64_t expected = parlay::reduce(
          parlay::tabulate(n, [](size_t i) { return (uint64_t)i; }), psum);
      check("round sum matches in-memory", result_sum == expected);
    }
  }

  cleanup_prefix("iota");

  const double throughput_gb_s = to_gb(in_bytes) / op_s;
  std::cout << (agree ? "agree=1 (all checks passed)"
                      : "agree=0 (MISMATCH -- see above)")
            << "   op_s=" << std::fixed << std::setprecision(4) << op_s << "   "
            << std::setprecision(2) << throughput_gb_s
            << " GB/s (eff. input)\n";

  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  std::cout << "CSV," << axis << ',' << mode << ',' << n << ',' << k0 << ','
            << k_effective << ',' << total_ops << ',' << f9(build_s) << ','
            << f9(op_s) << ',' << f9(throughput_gb_s) << ',' << (agree ? 1 : 0)
            << '\n';
  return agree ? 0 : 1;
}
