// count_sortExample — out-of-core counting-sort benchmark/demo.
//
// Buckets an n-element uint64_t input via plaid::count_sort_by_key
// (Primitives/count_sort.h, a single streaming-reader pass that routes each
// element to its bucket's write stream) and fuses the per-bucket chunk_seqs
// back into one index-ordered sequence with plaid::fuse.  Compares against
// parlay::counting_sort (its identity-key overload, which requires elements
// already in [0, num_buckets)) on the identical input built in DRAM.
//
// The key IS the value here (hash64(i) % NUM_BUCKETS, so every element in
// bucket b is literally the integer b, repeated) -- this makes the
// bucket-grouped output fully determined regardless of within-bucket order,
// so the cross-check is a direct element-for-element equality (after reading
// the out-of-core result back with plaid::materialize) rather than a
// permutation-only check.
//
// Dual-purpose like the other examples: prints human-readable timings and
// ends with a machine-readable `CSV,` line that benchmarks/run_benches.py
// greps.  The in-memory baseline is gated by a RAM budget (input + baseline
// output + readback = ~24n), overridable via EXAMPLE_INMEM_BUDGET_BYTES.
//
//   usage: count_sortExample [global --flags] [n]

#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/Primitives/count_sort.h"
#include "ChunkSequence/Primitives/materialize.h"
#include "absl/log/check.h"
#include "parlay/primitives.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "utils/io_backend.h"
#include "utils/trace_marker.h"

static constexpr size_t NUM_BUCKETS = 4096;

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}
static double to_gb(size_t bytes) {
  return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static void quiesce_drives() {
  sync();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

static void cleanup_prefix(const std::string& prefix) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    plaid::io::Unlink(GetFileName(prefix, d).c_str());
}

static uint64_t key_at(size_t i) { return parlay::hash64(i) % NUM_BUCKETS; }

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
  CHECK(n > 0) << "need n > 0 (n=" << n << ")";

  // RAM budget: baseline input (8n) + baseline output (8n) + out-of-core
  // readback for the cross-check (8n) = ~24n.
  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys / 2;
  if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  const bool inmem_ok = n <= budget / 24;

  const std::string in_prefix = "csrt_in";
  const std::string bucket_prefix = "csrt_bucket";

  std::cout << "Building " << n << "-element input (mod " << NUM_BUCKETS
            << ")..." << std::flush;
  trace_mark("build_start");
  auto t0 = Clock::now();
  chunk_seq keys = plaid::tabulate<uint64_t>(n, in_prefix, key_at);
  const double build_s = elapsed(t0);
  trace_mark("build_end");
  std::cout << " done (" << std::fixed << std::setprecision(4) << build_s
            << "s)\n";
  quiesce_drives();

  std::cout << "Counting-sorting " << n << " elements into " << NUM_BUCKETS
            << " buckets..." << std::flush;
  trace_mark("op_start");
  t0 = Clock::now();
  std::vector<chunk_seq> buckets(NUM_BUCKETS);
  plaid::count_sort_by_key<uint64_t>(
      keys, NUM_BUCKETS, buckets, [](uint64_t v) { return v; }, bucket_prefix);
  chunk_seq sorted = plaid::fuse(buckets);
  const double sort_s = elapsed(t0);
  trace_mark("op_end");
  const double gb_s = to_gb(n * sizeof(uint64_t)) / sort_s;
  std::cout << " done   " << std::setprecision(4) << sort_s << "s   "
            << std::setprecision(2) << gb_s << " GB/s (input read)\n";

  bool agree = true;
  double inmem_sort_s = 0;
  if (inmem_ok) {
    auto keys_mem = parlay::tabulate(n, key_at);
    t0 = Clock::now();
    auto sorted_mem = parlay::counting_sort(keys_mem, NUM_BUCKETS).first;
    inmem_sort_s = elapsed(t0);
    std::cout << "in-mem parlay::counting_sort: " << std::setprecision(4)
              << inmem_sort_s << "s\n";

    auto ours = plaid::materialize<uint64_t>(sorted);
    if (ours.size() != sorted_mem.size()) {
      std::cout << "*** MISMATCH: out-of-core produced " << ours.size()
                << " elements, expected " << sorted_mem.size() << " ***\n";
      agree = false;
    } else {
      for (size_t i = 0; i < ours.size() && agree; i++) {
        if (ours[i] != sorted_mem[i]) {
          std::cout << "*** MISMATCH at index " << i << ": " << ours[i]
                    << " != " << sorted_mem[i] << " ***\n";
          agree = false;
        }
      }
      if (agree)
        std::cout << "cross-check: out-of-core counting sort matches in-mem "
                     "counting sort exactly\n";
    }
  } else {
    std::cout << "in-mem parlay::counting_sort: skipped (~24n footprint "
              << "exceeds RAM budget " << std::setprecision(2)
              << to_gb(budget) << " GB)\n";
  }

  // Machine-readable line for benchmarks/run_benches.py (examples sweep).
  // Columns: n,build_s,sort_s,inmem_sort_s,throughput_gb_s
  // (inmem_sort_s blank when the input exceeds the RAM budget).
  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  std::cout << "CSV," << n << ',' << f9(build_s) << ',' << f9(sort_s) << ','
            << (inmem_ok ? f9(inmem_sort_s) : std::string()) << ','
            << f9(gb_s) << '\n';

  cleanup_prefix(in_prefix);
  cleanup_prefix(bucket_prefix);
  return agree ? 0 : 1;
}
