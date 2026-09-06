// Benchmark: out-of-core deduplication (ChunkSequenceOps::unique,
// ExternalPrimitives/unique.h) against an in-memory sort+std::unique
// baseline on the identical key multiset.
//
//   1. disk       ChunkSequenceOps::unique -- hash-buckets every element via
//                 a delayed map + count_sort (identical values land in the
//                 same bucket), then per bucket: materialize into DRAM,
//                 parlay::sort_inplace, std::unique+resize, and rewrite.
//                 flatten concatenates the (now deduped) buckets. Output
//                 order is bucket-hash order, not value order.
//   2. in-memory  parlay::sort_inplace + std::unique on the same keys in
//                 DRAM -- the yardstick, and (once sorted) the correctness
//                 reference. Stops at the RAM cliff like every other
//                 example's in-mem series.
//
// unique.h has no second on-disk implementation yet (unlike reverse's
// legacy/fast pair or random_shuffle's primitives/direct pair), so this is a
// single out-of-core contestant against the DRAM baseline -- same shape as
// external_random_shuffle.cpp before Permutation existed, not a three-way
// comparison. If a real sweep shows this implementation meaningfully
// trailing the DRAM baseline by more than the expected out-of-core cost,
// that is the trigger for writing a second ("direct") implementation to
// compare against -- not done here.
//
// Unlike reverse/shuffle, unique's input needs actual duplicates to
// exercise dedup: key_at(i) = hash64(i) % modulus, modulus = max(1, n /
// DUP_FACTOR), so each distinct value appears DUP_FACTOR times on average.
//
// Correctness is neither an element-wise-in-place check (reverse) nor a
// permutation check (shuffle): the output is smaller than the input and its
// order is unspecified, so both outputs are SORTED and compared: the disk
// contestant's materialized, sorted output must equal the DRAM baseline's
// sorted-and-deduped key set exactly (same size, same elements).
//
//   usage: external_uniqueExample [global --flags] [n]
//     n   number of keys before dedup (default 1e6)
//
// CSV line:
//   CSV,<n>,<dup_factor>,<build_s>,<unique_s>,<inmem_s>,<result_count>,
//       <unique_gb_s>
//   throughput = input bytes (n*8) / unique_s (the disk contestant's own
//   time, input-side); inmem_s is blank past the RAM budget, so the plotted
//   in-mem line stops at the cliff.
//
// Dual-purpose like the other examples: prints human-readable output AND the
// machine-readable "CSV," line benchmarks/run_benches.py greps.

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "ChunkSequence/ExternalPrimitives/materialize.h"
#include "ChunkSequence/ExternalPrimitives/unique.h"
#include "ChunkSequence/chunk_seq.h"
#include "absl/log/check.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Average number of times each distinct value repeats in the input, so
// unique actually has duplicates to remove.
static constexpr uint64_t kDupFactor = 4;

// Flush dirty pages and let the drives settle between a write phase and a
// timed op, so still-draining writeback doesn't queue behind (and inflate)
// the op's own I/O.  Always outside a timed region -- see
// external_samplesort.cpp / external_random_shuffle.cpp.
static void quiesce_drives() {
  sync();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
}
static double to_gb(size_t bytes) {
  return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

// ── drive hygiene ───────────────────────────────────────────────────────────
// Same two mechanisms as external_random_shuffle.cpp: a working prefix sweep
// to isolate the timed op on drives holding only its input, and a
// snapshot/sweep backstop so any file left behind by a naming mismatch is
// caught rather than silently accumulating across sweep points.
static void remove_prefixes(const std::vector<std::string>& prefixes) {
  for (const std::string& dir : GetSSDList()) {
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
      const std::string name = e.path().filename().string();
      for (const std::string& p : prefixes) {
        if (name.rfind(p, 0) == 0) {  // name starts with p
          std::filesystem::remove(e.path(), ec);
          break;
        }
      }
    }
  }
}

static std::set<std::string> snapshot_drives() {
  std::set<std::string> files;
  for (const std::string& dir : GetSSDList()) {
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec))
      files.insert(e.path().string());
  }
  return files;
}

static std::vector<std::string> sweep_new_files(
    const std::set<std::string>& before) {
  std::vector<std::string> leaked;
  for (const std::string& path : snapshot_drives()) {
    if (before.count(path)) continue;
    std::error_code ec;
    std::filesystem::remove(path, ec);
    leaked.push_back(path);
  }
  return leaked;
}

// The input, and unique()'s own file family.  unique()'s result *is* its
// eu_res* files (count_sort's own bucket output, then each bucket rewritten
// in place under the same prefix + "_u<b>" -- both start with "eu_res"), so
// that prefix must be swept too, or the whole output leaks every sweep
// point.
static const std::string kInPrefix = "eu_in";
static const std::vector<std::string> kMethodPrefixes = {
    "eu_res"};  // ChunkSequenceOps::unique(seq, "eu_res")

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  RaiseFdLimit();

  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
  CHECK(n > 0) << "need n > 0 (n=" << n << ")";

  const uint64_t modulus = std::max<uint64_t>(1, n / kDupFactor);
  auto key_at = [modulus](size_t i) { return parlay::hash64(i) % modulus; };

  // RAM budget for the in-memory baseline + cross-check: the keys (8n), the
  // baseline's sorted-unique result (aliased in place, up to 8n), and the
  // disk output read back for the check (up to 8n) -- ~24n, the same order
  // of magnitude as external_random_shuffle's ~32n.
  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys;
  if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  const bool check_ok = n <= budget / 24;

  std::cout << std::fixed;
  std::cout << "Clearing stale unique files from the drives..." << std::flush;
  std::vector<std::string> all_prefixes = {kInPrefix};
  all_prefixes.insert(all_prefixes.end(), kMethodPrefixes.begin(),
                      kMethodPrefixes.end());
  remove_prefixes(all_prefixes);
  std::cout << " done\n";

  const std::set<std::string> pre_run = snapshot_drives();

  // ── the shared input ────────────────────────────────────────────────────
  std::cout << "Building " << n << "-key chunk_seq input (~" << kDupFactor
            << "x duplication)..." << std::flush;
  auto t0 = Clock::now();
  chunk_seq seq = ChunkSequenceOps::tabulate<uint64_t>(n, kInPrefix, key_at);
  const double build_s = elapsed(t0);
  std::cout << " done (" << std::setprecision(4) << build_s << "s)\n";
  quiesce_drives();  // isolate the op timer from the build's writeback

  // ── the disk contestant ──────────────────────────────────────────────────
  std::cout << "Deduplicating " << n << " keys (ChunkSequenceOps::unique)..."
            << std::flush;
  t0 = Clock::now();
  chunk_seq deduped = ChunkSequenceOps::unique<uint64_t>(seq, "eu_res");
  const double unique_s = elapsed(t0);
  const double unique_gb_s = to_gb(n * sizeof(uint64_t)) / unique_s;
  std::cout << " done   " << std::setprecision(4) << unique_s << "s   "
            << std::setprecision(2) << unique_gb_s << " GB/s (input read)\n";

  parlay::sequence<uint64_t> ours;
  if (check_ok) ours = ChunkSequenceOps::materialize<uint64_t>(deduped);
  remove_prefixes(kMethodPrefixes);
  std::cout << "unique's files cleared\n";
  quiesce_drives();

  // ── in-memory baseline + cross-check ────────────────────────────────────
  bool agree = true;
  double inmem_unique_s = 0;
  size_t result_count = 0;
  if (check_ok) {
    auto keys_mem = parlay::tabulate(n, key_at);
    t0 = Clock::now();
    parlay::sort_inplace(keys_mem);
    keys_mem.resize(std::unique(keys_mem.begin(), keys_mem.end()) -
                    keys_mem.begin());
    inmem_unique_s = elapsed(t0);
    result_count = keys_mem.size();
    std::cout << "in-mem sort+std::unique   " << std::setprecision(4)
              << inmem_unique_s << "s   (" << result_count
              << " distinct keys)\n";

    // The disk output, once sorted, must equal the sorted-unique reference
    // exactly -- catches a dropped, duplicated, or spuriously-kept element.
    parlay::sort_inplace(ours);
    if (ours.size() != keys_mem.size()) {
      std::cout << "*** MISMATCH: disk produced " << ours.size()
                << " distinct keys, expected " << keys_mem.size() << " ***\n";
      agree = false;
    } else {
      for (size_t i = 0; i < keys_mem.size(); i++) {
        if (ours[i] != keys_mem[i]) {
          std::cout << "*** MISMATCH at sorted index " << i << ": "
                    << ours[i] << " != " << keys_mem[i] << " ***\n";
          agree = false;
          break;
        }
      }
    }
    if (agree)
      std::cout << "cross-check: disk output matches the sorted-unique "
                   "in-mem reference\n";
  } else {
    std::cout << "in-mem baseline + cross-check: skipped (~24n footprint "
                 "exceeds RAM budget "
              << std::setprecision(2) << to_gb(budget) << " GB)\n";
  }

  // Machine-readable line for benchmarks/run_benches.py (examples sweep).
  // Columns: n,dup_factor,build_s,unique_s,inmem_s,result_count,unique_gb_s
  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  std::cout << "CSV," << n << ',' << kDupFactor << ',' << f9(build_s) << ','
            << f9(unique_s) << ','
            << (check_ok ? f9(inmem_unique_s) : std::string()) << ','
            << result_count << ',' << f9(unique_gb_s) << '\n';

  // ── leave the drives exactly as we found them ───────────────────────────
  remove_prefixes(all_prefixes);
  const std::vector<std::string> leaked = sweep_new_files(pre_run);
  if (!leaked.empty()) {
    std::cout << "*** LEAK: " << leaked.size()
              << " file(s) were left on the drives by this run (removed now); "
                 "the driver's prefix list is incomplete ***\n";
    for (size_t i = 0; i < leaked.size() && i < 20; i++)
      std::cout << "      " << leaked[i] << '\n';
    if (leaked.size() > 20)
      std::cout << "      ... and " << (leaked.size() - 20) << " more\n";
    agree = false;
  } else {
    std::cout << "drives clean: no files left behind by this run\n";
  }
  return agree ? 0 : 1;
}
