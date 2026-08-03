// nested_map_reduce_compare — time the FUSED inner map+reduce against the
// UNFUSED (materialize-the-intermediate) spelling of the same computation.
//
//   fused        : NestedMapReduce(ns, map, monoid)      -> 1 streaming read pass
//   materialized : NestedMap(ns, map) then NestedReduce  -> read + write + read
//
// Both compute, per inner sequence, sum_x (x % 100).  The driver builds a
// nested_seq, times each variant, verifies they agree (a differential check),
// and prints timings + a machine-readable CSV line.  Manual-only benchmark
// (not in TEST_BINARIES / bench recipes) — build and run by hand:
//     rm -f bin/nestedMapReduceCompare && make bin/nestedMapReduceCompare
//     bin/nestedMapReduceCompare [num_inner_seqs]

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>

#include "ChunkSequence/nested_ops.h"
#include "ChunkSequence/nested_seq.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"

struct SumMonoid {
  uint64_t identity = 0;
  uint64_t operator()(uint64_t a, uint64_t b) const { return a + b; }
};

// Each inner sequence is 512..1535 uint64_t elements = 4..12 KiB (avg ~1024
// elements ≈ 8 KiB), so the on-disk data volume — and thus the I/O the fusion
// saves — dominates the trivial per-element map.  `n` (argv[1]) is the number of
// such inner sequences; total data ≈ n * ~8 KiB.
static parlay::sequence<uint64_t> gen_seq(size_t i) {
  const size_t l = 512 + (size_t)((i * 2654435761ULL) % 1024);
  parlay::sequence<uint64_t> s(l);
  for (size_t k = 0; k < l; k++) s[k] = i * 1000003ULL + k;
  return s;
}

static double secs(std::chrono::steady_clock::time_point a,
                   std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);

  size_t n = 40000;  // number of inner sequences
  if (argc > 1) n = std::stoull(argv[1]);

  const std::string src = "nmrc_src";
  auto ns = ChunkSequenceOps::NestedTabulate<uint64_t>(n, src, gen_seq);

  const size_t total_elems = ns.total_seqs() == 0 ? 0 : ns.seq_len_scan.back();
  const double data_mb = total_elems * sizeof(uint64_t) / 1e6;
  std::cout << "nested map-reduce: fused vs materialized-intermediate\n"
            << "  n=" << n << " inner sequences, " << total_elems
            << " elements, " << data_mb << " MB, " << ns.chunks.size()
            << " chunks\n"
            << std::flush;

  auto mapf = [](uint64_t x) { return x % 100; };

  // Fused: single streaming read pass, no intermediate on disk.
  auto t0 = std::chrono::steady_clock::now();
  parlay::sequence<uint64_t> fused =
      ChunkSequenceOps::NestedMapReduce<uint64_t, uint64_t>(ns, mapf, SumMonoid{});
  auto t1 = std::chrono::steady_clock::now();

  // Materialized: write the mapped nested_seq, then read it back and reduce.
  parlay::sequence<uint64_t> mat =
      ChunkSequenceOps::NestedMapReduceMaterialized<uint64_t, uint64_t>(
          ns, mapf, SumMonoid{}, "nmrc_mapped");
  auto t2 = std::chrono::steady_clock::now();

  bool agree = fused.size() == mat.size();
  for (size_t i = 0; i < fused.size() && agree; i++)
    if (fused[i] != mat[i]) agree = false;

  const double fused_s = secs(t0, t1), mat_s = secs(t1, t2);
  const double bytes = total_elems * (double)sizeof(uint64_t);
  // fused moves the data once; materialized moves it ~3x (read+write+read).
  const double fused_gb = bytes / fused_s / 1e9;
  const double mat_gb = 3.0 * bytes / mat_s / 1e9;

  std::cout << "  fused        (NestedMapReduce)          : " << fused_s
            << " s   (~" << fused_gb << " GB/s, 1 read pass)\n"
            << "  materialized (NestedMap + NestedReduce)  : " << mat_s
            << " s   (~" << mat_gb << " GB/s, read+write+read)\n"
            << "  speedup (materialized / fused)           : " << (mat_s / fused_s)
            << "x\n"
            << "  agree: " << (agree ? "yes" : "NO") << "\n";

  std::cout << "CSV,n,total_mb,fused_s,materialized_s,speedup,agree\n"
            << "CSV," << n << "," << data_mb << "," << fused_s << "," << mat_s
            << "," << (mat_s / fused_s) << "," << (agree ? 1 : 0) << "\n";

  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++) unlink(GetFileName(src, d).c_str());
  return agree ? 0 : 1;
}
