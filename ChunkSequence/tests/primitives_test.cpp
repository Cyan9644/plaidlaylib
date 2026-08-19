// primitives_test.cpp -- correctness tests for chunk_seq.h, primitives.h and
// sort.h, in one binary.
//
//   usage: primitivesTest [global --flags] [n]
//
// Each case is the test that used to be its own binary, moved here verbatim
// inside its own namespace with main() renamed to run().  Cheap/foundational
// cases run first so a substrate break surfaces before the expensive sorts.
// The optional element count is forwarded to every case exactly as it used to
// be forwarded to every binary (`make test TEST_ARGS=8000000`); a case with no
// argument uses its own default.
//
// Exit code 0 iff every case passed.
//
// The delayed layer has its own binary (delayed_test.cpp), as do the four
// examples with tests (kmp, rabin_karp, bigint_add, convex_hull).
//
// Cases: iota, map, reduce, scan, segmented_reduce, find_if, histogram, scalar,
// filter, flat_tabulate, flat_map, partition, group_by, chunk_operation,
// combined, samplesort

#include "parlay/primitives.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/Primitives/delayed.h"
#include "ChunkSequence/Primitives/secondary_primitives.h"
#include "ChunkSequence/Primitives/sort.h"
#include "absl/log/check.h"
#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "utils/bench_drives.h"
#include "utils/file_utils.h"

// ============================================================================
// iota -- tabulate / iota / consolidate round-trip
//
// (was ChunkSequence/tests/iota_test.cpp)
// ============================================================================

namespace test_iota {

/**
 * Verify that iota(n) writes the identity sequence correctly.
 *
 * For every chunk in the returned chunk_seq, this test opens chunk.filename
 * directly with open(O_DIRECT|O_RDONLY) and reads chunk.used bytes from
 * chunk.begin_addr using pread().  It then checks that element i of the
 * chunk equals  chunk.index * ELEMS_PER_CHUNK + i  for all i in [0, count).
 *
 * No ChunkSequenceReader, no UnorderedFileReader — raw syscalls only.
 */
int run(int argc, char* argv[]) {
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 5'000'000ULL;

  using T = uint64_t;

  // ── write ────────────────────────────────────────────────────────────────
  std::cout << "iota(" << n << ") writing...\n" << std::flush;
  const chunk_seq seq = plaid::iota(n);
  std::cout << "wrote " << seq.chunks.size() << " chunks across "
            << GetSSDList().size() << " drives\n"
            << std::flush;

  // ── assert all chunks live on configured SSD mounts ──────────────────────
  {
    const auto& ssds = GetSSDList();
    size_t ssd_count = 0;
    for (const auto& c : seq.chunks)
      for (const auto& ssd : ssds)
        if (c.filename.rfind(ssd, 0) == 0) {
          ssd_count++;
          break;
        }
    if (ssd_count != seq.chunks.size()) {
      std::cerr << "FAIL: only " << ssd_count << "/" << seq.chunks.size()
                << " chunks are on configured SSDs\n";
      return 1;
    }
    std::cout << "SSD check OK (sample: " << seq.chunks[0].filename << ")\n"
              << std::flush;
  }

  // ── verify each chunk via direct syscall reads ────────────────────────────
  std::atomic<size_t> pass_count{0};
  std::atomic<size_t> fail_count{0};

  parlay::parallel_for(
      0, seq.chunks.size(),
      [&](size_t ci) {
        const chunk& c = seq.chunks[ci];

        // Chunk metadata sanity checks.
        CHECK(c.used % sizeof(T) == 0)
            << "chunk " << c.index << ": used not T-aligned";
        CHECK(c.begin_addr % CHUNK_SIZE == 0)
            << "chunk " << c.index << ": begin_addr misaligned";

        const size_t count = c.used / sizeof(T);
        const size_t start = c.index * ELEMS_PER_CHUNK;
        const size_t read_size =
            AlignUp(c.used == 0 ? (size_t)O_DIRECT_MULTIPLE : c.used);

        if (count == 0) {
          pass_count++;
          return;
        }

        int fd = open(c.filename.c_str(), O_DIRECT | O_RDONLY);
        if (fd < 0) {
          std::cerr << "FAIL chunk " << c.index << ": open(" << c.filename
                    << ") failed: " << strerror(errno) << "\n";
          fail_count++;
          return;
        }

        T* buf = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, read_size);
        CHECK(buf != nullptr);

        const ssize_t got = pread(fd, buf, read_size, (off_t)c.begin_addr);
        close(fd);

        if (got < 0 || (size_t)got < c.used) {
          std::cerr << "FAIL chunk " << c.index << ": pread returned " << got
                    << " (expected " << c.used << ")\n";
          free(buf);
          fail_count++;
          return;
        }

        // Check each element.
        bool ok = true;
        for (size_t i = 0; i < count && ok; i++) {
          if (buf[i] != static_cast<T>(start + i)) {
            std::cerr << "FAIL chunk " << c.index << " element " << i
                      << ": got " << buf[i] << " expected " << (start + i)
                      << " (start=" << start << ")\n";
            fail_count++;
            ok = false;
          }
        }
        if (ok) pass_count++;

        free(buf);
      },
      /*granularity=*/1);

  // ── coverage check ────────────────────────────────────────────────────────
  // Every chunk index [0, num_chunks) must appear exactly once.
  const size_t num_chunks = seq.chunks.size();
  std::vector<bool> seen(num_chunks, false);
  bool coverage_ok = true;
  for (const auto& c : seq.chunks) {
    if (c.index >= num_chunks || seen[c.index]) {
      std::cerr << "FAIL: duplicate or out-of-range chunk index " << c.index
                << "\n";
      coverage_ok = false;
    }
    seen[c.index] = true;
  }
  for (size_t i = 0; i < num_chunks; i++) {
    if (!seen[i]) {
      std::cerr << "FAIL: chunk index " << i << " missing from chunk_seq\n";
      coverage_ok = false;
    }
  }

  // ── report ────────────────────────────────────────────────────────────────
  const bool all_pass = (fail_count == 0) && coverage_ok;
  std::cout << (all_pass ? "PASS" : "FAIL") << "  chunks=" << num_chunks
            << "  verified=" << pass_count.load()
            << "  failed=" << fail_count.load() << "\n";

  return all_pass ? 0 : 1;
}

}  // namespace test_iota

// ============================================================================
// map -- ChunkMap, incl. the sizeof(R) > sizeof(T) fanout
//
// (was ChunkSequence/tests/map_test.cpp)
// ============================================================================

namespace test_map {

// ── helpers ──────────────────────────────────────────────────────────────────

// Read one output chunk from disk via pread and verify every element.
//
// f_expected maps the global input index (= value written by iota) to the
// expected output value.  No assumption is made about begin_addr or which
// file the chunk lives in — we use the chunk descriptor as-is.
template <typename R>
bool verify_chunk(const chunk& c, size_t elems_per_input_chunk,
                  std::function<R(uint64_t)> f_expected) {
  const size_t count = c.used / sizeof(R);
  const size_t start_idx =
      c.index * elems_per_input_chunk;  // global start in [0,n)
  const size_t read_size =
      AlignUp(count == 0 ? (size_t)O_DIRECT_MULTIPLE : c.used);

  if (count == 0) return true;

  int fd = open(c.filename.c_str(), O_DIRECT | O_RDONLY);
  if (fd < 0) {
    std::cerr << "  FAIL chunk " << c.index << ": open(" << c.filename
              << "): " << strerror(errno) << "\n";
    return false;
  }

  R* buf = (R*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, read_size);
  CHECK(buf != nullptr);

  const ssize_t got = pread(fd, buf, read_size, (off_t)c.begin_addr);
  close(fd);

  if (got < 0 || (size_t)got < c.used) {
    std::cerr << "  FAIL chunk " << c.index << ": pread got " << got
              << " expected at least " << c.used << "\n";
    free(buf);
    return false;
  }

  bool ok = true;
  for (size_t i = 0; i < count && ok; i++) {
    const R expected = f_expected(start_idx + i);
    if (buf[i] != expected) {
      std::cerr << "  FAIL chunk " << c.index << " element " << i << ": got "
                << (uint64_t)buf[i] << " expected " << (uint64_t)expected
                << " (global idx " << start_idx + i << ")\n";
      ok = false;
    }
  }

  free(buf);
  return ok;
}

// Run one named map test case, return true iff all chunks verify correctly.
template <typename R>
bool run_test(const std::string& name, const chunk_seq& input,
              const std::string& result_prefix, size_t elems_per_input_chunk,
              std::function<R(uint64_t)> f) {
  std::cout << "  " << name << " ... " << std::flush;

  const chunk_seq output =
      plaid::ChunkMap<uint64_t, R>(input, result_prefix, f);

  // Assert output files are on configured SSD mounts.
  {
    const auto& ssds = GetSSDList();
    size_t ssd_count = 0;
    for (const auto& c : output.chunks)
      for (const auto& ssd : ssds)
        if (c.filename.rfind(ssd, 0) == 0) {
          ssd_count++;
          break;
        }
    if (ssd_count != output.chunks.size()) {
      std::cout << "FAIL (only " << ssd_count << "/" << output.chunks.size()
                << " output chunks on SSDs)\n";
      return false;
    }
  }

  // The output must describe the same number of chunks as the input.
  if (output.chunks.size() != input.chunks.size()) {
    std::cout << "FAIL (chunk count: got " << output.chunks.size() << " want "
              << input.chunks.size() << ")\n";
    return false;
  }

  // Every chunk index [0, n_chunks) must appear exactly once.
  const size_t n_chunks = output.chunks.size();
  std::vector<bool> seen(n_chunks, false);
  bool coverage_ok = true;
  for (const auto& c : output.chunks) {
    if (c.index >= n_chunks || seen[c.index]) {
      std::cerr << "  FAIL: duplicate/out-of-range index " << c.index << "\n";
      coverage_ok = false;
    } else {
      seen[c.index] = true;
    }
  }

  // Verify element values in parallel.
  std::atomic<size_t> pass_count{0}, fail_count{0};
  parlay::parallel_for(
      0, n_chunks,
      [&](size_t ci) {
        if (verify_chunk<R>(output.chunks[ci], elems_per_input_chunk, f))
          pass_count++;
        else
          fail_count++;
      },
      /*granularity=*/1);

  const bool ok = coverage_ok && (fail_count == 0);
  std::cout << (ok ? "PASS" : "FAIL") << "  chunks=" << n_chunks
            << "  verified=" << pass_count.load()
            << "  failed=" << fail_count.load() << "\n";

  // Free the per-drive output files now that they're verified.  At GiB-scale
  // inputs the configured "SSDs" may share one filesystem, so keeping every
  // test case's output around would exhaust space.
  std::set<std::string> out_files;
  for (const auto& c : output.chunks) out_files.insert(c.filename);
  for (const auto& fname : out_files) unlink(fname.c_str());

  return ok;
}

// ── main ─────────────────────────────────────────────────────────────────────

int run(int argc, char* argv[]) {
  // Default to 128 Mi uint64_t elements = exactly 1 GiB = 256 chunks, so the
  // chunk count comfortably exceeds the drive count (one file per SSD).
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 134'217'728ULL;

  using T = uint64_t;

  std::cout << "Building iota(" << n << ")...\n" << std::flush;
  const chunk_seq input = plaid::iota(n);
  std::cout << input.chunks.size() << " chunks across " << GetSSDList().size()
            << " drives\n\n";

  bool all_pass = true;

  // ── in-place cases (T == R == uint64_t) ──────────────────────────────────

  // identity: output[i] == input[i] == global_index
  all_pass &= run_test<T>("identity      x -> x", input, "map_id",
                          ELEMS_PER_CHUNK, [](T x) -> T { return x; });

  // increment: output[i] == global_index + 1
  all_pass &= run_test<T>("increment     x -> x+1", input, "map_incr",
                          ELEMS_PER_CHUNK, [](T x) -> T { return x + 1; });

  // double: output[i] == global_index * 2
  all_pass &= run_test<T>("double        x -> x*2", input, "map_double",
                          ELEMS_PER_CHUNK, [](T x) -> T { return x * 2; });

  // complement: output[i] == ~global_index
  all_pass &= run_test<T>("complement    x -> ~x", input, "map_compl",
                          ELEMS_PER_CHUNK, [](T x) -> T { return ~x; });

  // ── type-changing case (T=uint64_t, R=uint32_t) ───────────────────────────
  // Tests the non-in-place allocation path; for n < 2^32 the low 32 bits
  // are lossless.
  all_pass &= run_test<uint32_t>("narrow u64->u32 x -> uint32_t(x)", input,
                                 "map_narrow", ELEMS_PER_CHUNK,
                                 [](T x) -> uint32_t { return (uint32_t)x; });

  std::cout << "\n" << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
  return all_pass ? 0 : 1;
}

}  // namespace test_map

// ============================================================================
// reduce -- ChunkReduce folds
//
// (was ChunkSequence/tests/reduce_test.cpp)
// ============================================================================

namespace test_reduce {

// ── monoids ──────────────────────────────────────────────────────────────────

struct SumMonoid {
  uint64_t identity = 0;
  uint64_t operator()(uint64_t a, uint64_t b) const { return a + b; }
};

struct MaxMonoid {
  // iota(n) starts at 0, so 0 is the correct min-identity for max.
  uint64_t identity = 0;
  uint64_t operator()(uint64_t a, uint64_t b) const { return std::max(a, b); }
};

struct MinMonoid {
  uint64_t identity = UINT64_MAX;
  uint64_t operator()(uint64_t a, uint64_t b) const { return std::min(a, b); }
};

struct XorMonoid {
  uint64_t identity = 0;
  uint64_t operator()(uint64_t a, uint64_t b) const { return a ^ b; }
};

// ── helpers ──────────────────────────────────────────────────────────────────

// XOR(0 ^ 1 ^ … ^ k) using the 4-cycle closed form.
static uint64_t xor_prefix(uint64_t k) {
  switch (k % 4) {
    case 0:
      return k;
    case 1:
      return 1;
    case 2:
      return k + 1;
    default:
      return 0;
  }
}

static bool report(const std::string& name, uint64_t got, uint64_t expected) {
  const bool ok = (got == expected);
  std::cout << "  " << std::left << std::setw(32) << name
            << (ok ? "PASS" : "FAIL") << "  got=" << got
            << " expected=" << expected << "\n";
  return ok;
}

// ── main ─────────────────────────────────────────────────────────────────────

int run(int argc, char* argv[]) {
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 5'000'000ULL;

  std::cout << "Building iota(" << n << ")...\n" << std::flush;
  const chunk_seq input = plaid::iota(n);
  std::cout << input.chunks.size() << " chunks across " << GetSSDList().size()
            << " drives\n\n";

  bool all_pass = true;

  // sum of 0+1+…+(n-1) = n*(n-1)/2
  all_pass &= report("sum  0+1+…+(n-1)",
                     plaid::ChunkReduce<uint64_t>(input, SumMonoid{}),
                     (uint64_t)(n - 1) * n / 2);

  // max element of iota(n) = n-1
  all_pass &=
      report("max  element", plaid::ChunkReduce<uint64_t>(input, MaxMonoid{}),
             (uint64_t)(n - 1));

  // min element of iota(n) = 0
  all_pass &= report("min  element",
                     plaid::ChunkReduce<uint64_t>(input, MinMonoid{}), 0ULL);

  // XOR of 0^1^…^(n-1), computed via closed-form prefix formula
  all_pass &= report("xor  0^1^…^(n-1)",
                     plaid::ChunkReduce<uint64_t>(input, XorMonoid{}),
                     xor_prefix((uint64_t)(n - 1)));

  std::cout << "\n" << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
  return all_pass ? 0 : 1;
}

}  // namespace test_reduce

// ============================================================================
// scan -- ChunkScan, both passes + returned total
//
// (was ChunkSequence/tests/scan_test.cpp)
// ============================================================================

namespace test_scan {

// ── monoids ──────────────────────────────────────────────────────────────────

struct SumMonoid {
  uint64_t identity = 0;
  uint64_t operator()(uint64_t a, uint64_t b) const { return a + b; }
};

struct XorMonoid {
  uint64_t identity = 0;
  uint64_t operator()(uint64_t a, uint64_t b) const { return a ^ b; }
};

// XOR(0 ^ 1 ^ … ^ k) using the 4-cycle closed form.
static uint64_t xor_prefix(uint64_t k) {
  switch (k % 4) {
    case 0:
      return k;
    case 1:
      return 1;
    case 2:
      return k + 1;
    default:
      return 0;
  }
}

// ── helpers ──────────────────────────────────────────────────────────────────

// Read one output chunk from disk via pread and verify every element against
// f_expected, which maps the global element index (in [0,n)) to the expected
// scanned value.  No assumption about begin_addr or which file the chunk lives
// in — we use the chunk descriptor as-is.
template <typename R>
bool verify_chunk(const chunk& c, size_t elems_per_input_chunk,
                  std::function<R(uint64_t)> f_expected) {
  const size_t count = c.used / sizeof(R);
  const size_t start_idx =
      c.index * elems_per_input_chunk;  // global start in [0,n)
  const size_t read_size =
      AlignUp(count == 0 ? (size_t)O_DIRECT_MULTIPLE : c.used);

  if (count == 0) return true;

  int fd = open(c.filename.c_str(), O_DIRECT | O_RDONLY);
  if (fd < 0) {
    std::cerr << "  FAIL chunk " << c.index << ": open(" << c.filename
              << "): " << strerror(errno) << "\n";
    return false;
  }

  R* buf = (R*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, read_size);
  CHECK(buf != nullptr);

  const ssize_t got = pread(fd, buf, read_size, (off_t)c.begin_addr);
  close(fd);

  if (got < 0 || (size_t)got < c.used) {
    std::cerr << "  FAIL chunk " << c.index << ": pread got " << got
              << " expected at least " << c.used << "\n";
    free(buf);
    return false;
  }

  bool ok = true;
  for (size_t i = 0; i < count && ok; i++) {
    const R expected = f_expected(start_idx + i);
    if (buf[i] != expected) {
      std::cerr << "  FAIL chunk " << c.index << " element " << i << ": got "
                << (uint64_t)buf[i] << " expected " << (uint64_t)expected
                << " (global idx " << start_idx + i << ")\n";
      ok = false;
    }
  }

  free(buf);
  return ok;
}

// Run one named scan test case, return true iff all chunks verify correctly.
// f_expected(global_idx) gives the expected exclusive-scan value at global_idx;
// expected_total is the grand total ChunkScan must also return.
template <typename Monoid, typename R = uint64_t>
bool run_test(const std::string& name, const chunk_seq& input,
              const std::string& result_prefix, size_t elems_per_input_chunk,
              Monoid monoid, std::function<R(uint64_t)> f_expected,
              R expected_total) {
  std::cout << "  " << name << " ... " << std::flush;

  // Plain variables (not a structured binding) so `output` can be captured by
  // the parlay::parallel_for lambda below under -std=c++17.
  const auto scan_result =
      plaid::ChunkScan<uint64_t, R>(input, result_prefix, monoid);
  const chunk_seq& output = scan_result.first;
  const R total = scan_result.second;

  // Verify the returned grand total.
  bool total_ok = (total == expected_total);
  if (!total_ok)
    std::cerr << "\n  FAIL total: got " << (uint64_t)total << " expected "
              << (uint64_t)expected_total << "\n";

  // Assert output files are on configured SSD mounts.
  {
    const auto& ssds = GetSSDList();
    size_t ssd_count = 0;
    for (const auto& c : output.chunks)
      for (const auto& ssd : ssds)
        if (c.filename.rfind(ssd, 0) == 0) {
          ssd_count++;
          break;
        }
    if (ssd_count != output.chunks.size()) {
      std::cout << "FAIL (only " << ssd_count << "/" << output.chunks.size()
                << " output chunks on SSDs)\n";
      return false;
    }
  }

  // The output must describe the same number of chunks as the input.
  if (output.chunks.size() != input.chunks.size()) {
    std::cout << "FAIL (chunk count: got " << output.chunks.size() << " want "
              << input.chunks.size() << ")\n";
    return false;
  }

  // Every chunk index [0, n_chunks) must appear exactly once.
  const size_t n_chunks = output.chunks.size();
  std::vector<bool> seen(n_chunks, false);
  bool coverage_ok = true;
  for (const auto& c : output.chunks) {
    if (c.index >= n_chunks || seen[c.index]) {
      std::cerr << "  FAIL: duplicate/out-of-range index " << c.index << "\n";
      coverage_ok = false;
    } else {
      seen[c.index] = true;
    }
  }

  // Verify element values in parallel.
  std::atomic<size_t> pass_count{0}, fail_count{0};
  parlay::parallel_for(
      0, n_chunks,
      [&](size_t ci) {
        if (verify_chunk<R>(output.chunks[ci], elems_per_input_chunk,
                            f_expected))
          pass_count++;
        else
          fail_count++;
      },
      /*granularity=*/1);

  const bool ok = coverage_ok && total_ok && (fail_count == 0);
  std::cout << (ok ? "PASS" : "FAIL") << "  chunks=" << n_chunks
            << "  verified=" << pass_count.load()
            << "  failed=" << fail_count.load()
            << "  total=" << (total_ok ? "OK" : "BAD") << "\n";

  // Free the per-drive output files now that they're verified.  At GiB-scale
  // inputs the configured "SSDs" may share one filesystem, so keeping every
  // test case's output around would exhaust space.
  std::set<std::string> out_files;
  for (const auto& c : output.chunks) out_files.insert(c.filename);
  for (const auto& fname : out_files) unlink(fname.c_str());

  return ok;
}

// ── main ─────────────────────────────────────────────────────────────────────

int run(int argc, char* argv[]) {
  // Default to 128 Mi uint64_t elements = exactly 1 GiB = 256 chunks, so the
  // chunk count comfortably exceeds the drive count (one file per SSD).
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 134'217'728ULL;

  std::cout << "Building iota(" << n << ")...\n" << std::flush;
  const chunk_seq input = plaid::iota(n);
  std::cout << input.chunks.size() << " chunks across " << GetSSDList().size()
            << " drives\n\n";

  bool all_pass = true;

  // exclusive sum scan over iota(n): out[i] = 0+1+…+(i-1) = i*(i-1)/2
  // (out[0] = 0); total = 0+1+…+(n-1) = n*(n-1)/2.
  all_pass &= run_test<SumMonoid>(
      "sum  exclusive prefix", input, "scan_sum", ELEMS_PER_CHUNK, SumMonoid{},
      std::function<uint64_t(uint64_t)>(
          [](uint64_t i) -> uint64_t { return i * (i - 1) / 2; }),
      /*expected_total=*/(uint64_t)(n - 1) * n / 2);

  // exclusive xor scan over iota(n): out[i] = 0^1^…^(i-1) = xor_prefix(i-1)
  // (out[0] = 0); total = 0^1^…^(n-1) = xor_prefix(n-1).
  all_pass &= run_test<XorMonoid>(
      "xor  exclusive prefix", input, "scan_xor", ELEMS_PER_CHUNK, XorMonoid{},
      std::function<uint64_t(uint64_t)>([](uint64_t i) -> uint64_t {
        return i == 0 ? 0 : xor_prefix(i - 1);
      }),
      /*expected_total=*/xor_prefix(n - 1));

  std::cout << "\n" << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
  return all_pass ? 0 : 1;
}

}  // namespace test_scan

// ============================================================================
// segmented_reduce -- ChunkSegmentedReduce boundary merge
//
// (was ChunkSequence/tests/segmented_reduce_test.cpp)
// ============================================================================

namespace test_segmented_reduce {

// ── monoids ──────────────────────────────────────────────────────────────────

struct SumMonoid {
  uint64_t identity = 0;
  uint64_t operator()(uint64_t a, uint64_t b) const { return a + b; }
};

struct MinMonoid {
  uint64_t identity = UINT64_MAX;
  uint64_t operator()(uint64_t a, uint64_t b) const { return std::min(a, b); }
};

struct MaxMonoid {
  uint64_t identity = 0;
  uint64_t operator()(uint64_t a, uint64_t b) const { return std::max(a, b); }
};

// ── helpers ──────────────────────────────────────────────────────────────────

// Cycle a small pattern of segment sizes (including 0, sub-chunk, and
// multi-chunk-spanning sizes relative to `ept`) until it covers exactly `n`
// elements, so the resulting bounds exercise: segments fully inside one
// chunk, segments that straddle a chunk seam, an empty segment, and a "mega"
// segment spanning several whole chunks (the chained-boundary-merge path).
static std::vector<size_t> make_bounds(size_t n, size_t ept) {
  const std::vector<size_t> pattern = {
      0,       1, ept / 2, 2 * ept + 137, ept - 100, 3,
      ept + 1, 0, 12345,   ept * 3 + 7,   1,         ept / 3 + 9};
  std::vector<size_t> bounds{0};
  size_t total = 0, p = 0;
  while (total < n) {
    size_t s = pattern[p % pattern.size()];
    if (total + s > n) s = n - total;
    total += s;
    bounds.push_back(total);
    p++;
  }
  return bounds;
}

// Verify a ChunkSegmentedReduce result against a per-segment closed form
// expected_fn(lo, hi) -> R, reporting up to 5 mismatches.
template <typename R, typename ExpectedFn>
static bool verify(const std::string& name, const parlay::sequence<R>& got,
                   const std::vector<size_t>& bounds, ExpectedFn expected_fn) {
  const size_t num_segments = bounds.size() - 1;
  size_t fails = 0;
  for (size_t v = 0; v < num_segments; v++) {
    const R expected = expected_fn(bounds[v], bounds[v + 1]);
    if (got[v] != expected) {
      if (fails < 5)
        std::cerr << "  FAIL segment " << v << " [" << bounds[v] << ","
                  << bounds[v + 1] << "): got " << (uint64_t)got[v]
                  << " expected " << (uint64_t)expected << "\n";
      fails++;
    }
  }
  std::cout << "  " << std::left << std::setw(32) << name
            << (fails == 0 ? "PASS" : "FAIL") << "  segments=" << num_segments
            << "  failed=" << fails << "\n";
  return fails == 0;
}

// Closed forms for reducing elem_to_val(x) = x over iota(n)'s values [lo,hi).
static uint64_t expected_sum(size_t lo, size_t hi) {
  return lo >= hi ? 0 : (uint64_t)(lo + hi - 1) * (hi - lo) / 2;
}
static uint64_t expected_min(size_t lo, size_t hi) {
  return lo < hi ? (uint64_t)lo : UINT64_MAX;
}
static uint64_t expected_max(size_t lo, size_t hi) {
  return lo < hi ? (uint64_t)(hi - 1) : 0ULL;
}

// A 32-byte record type (sizeof(T) != sizeof(uint64_t)) so this test also
// covers ChunkSegmentedReduce addressing chunks by CHUNK_SIZE/sizeof(T)
// rather than the global (uint64_t-sized) ELEMS_PER_CHUNK -- the exact bug
// class previously seen in the delayed::cut path for 32-byte elements.
struct rec32 {
  uint64_t v;
  unsigned char pad[24];
};
static_assert(sizeof(rec32) == 32, "rec32 must be 32 bytes");

// ── main ─────────────────────────────────────────────────────────────────────

int run(int argc, char* argv[]) {
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 5'000'000ULL;
  bool all_pass = true;

  {
    std::cout << "uint64_t elements, n=" << n
              << " (ELEMS_PER_CHUNK=" << ELEMS_PER_CHUNK << ")\n"
              << std::flush;
    const chunk_seq input = plaid::iota(n);
    const auto bounds_v = make_bounds(n, ELEMS_PER_CHUNK);
    const parlay::sequence<size_t> bounds(bounds_v.begin(), bounds_v.end());
    auto id = [](uint64_t x) { return x; };

    all_pass &=
        verify<uint64_t>("sum",
                         plaid::ChunkSegmentedReduce<uint64_t, uint64_t>(
                             input, bounds, id, SumMonoid{}),
                         bounds_v, expected_sum);
    all_pass &=
        verify<uint64_t>("min",
                         plaid::ChunkSegmentedReduce<uint64_t, uint64_t>(
                             input, bounds, id, MinMonoid{}),
                         bounds_v, expected_min);
    all_pass &=
        verify<uint64_t>("max",
                         plaid::ChunkSegmentedReduce<uint64_t, uint64_t>(
                             input, bounds, id, MaxMonoid{}),
                         bounds_v, expected_max);
  }

  {
    const size_t n2 = std::min(n, (size_t)2'000'000ULL);
    const size_t ept2 = CHUNK_SIZE / sizeof(rec32);
    std::cout << "\n32-byte elements, n=" << n2 << " (elems_per_chunk=" << ept2
              << ", != ELEMS_PER_CHUNK=" << ELEMS_PER_CHUNK << ")\n"
              << std::flush;
    const chunk_seq input =
        plaid::tabulate<rec32>(n2, "segreduce_rec32", [](size_t i) {
          rec32 r;
          r.v = i;
          return r;
        });
    const auto bounds_v = make_bounds(n2, ept2);
    const parlay::sequence<size_t> bounds(bounds_v.begin(), bounds_v.end());
    auto id = [](const rec32& r) { return r.v; };

    all_pass &= verify<uint64_t>("sum (32B elems)",
                                 plaid::ChunkSegmentedReduce<rec32, uint64_t>(
                                     input, bounds, id, SumMonoid{}),
                                 bounds_v, expected_sum);
    all_pass &= verify<uint64_t>("min (32B elems)",
                                 plaid::ChunkSegmentedReduce<rec32, uint64_t>(
                                     input, bounds, id, MinMonoid{}),
                                 bounds_v, expected_min);
    all_pass &= verify<uint64_t>("max (32B elems)",
                                 plaid::ChunkSegmentedReduce<rec32, uint64_t>(
                                     input, bounds, id, MaxMonoid{}),
                                 bounds_v, expected_max);

    for (const auto& c : input.chunks) unlink(c.filename.c_str());
  }

  std::cout << "\n" << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
  return all_pass ? 0 : 1;
}

}  // namespace test_segmented_reduce

// ============================================================================
// find_if -- ChunkFindIf
//
// (was ChunkSequence/tests/find_if_test.cpp)
// ============================================================================

namespace test_find_if {

// iota(n) holds the sequence 0, 1, …, n-1 in order, so the element equal to
// `target` sits at logical position `target`.  find_if(== target) must return
// exactly `target`, and a predicate that never fires must return n.

static bool report(const std::string& name, size_t got, size_t expected) {
  const bool ok = (got == expected);
  std::cout << "  " << std::left << std::setw(36) << name
            << (ok ? "PASS" : "FAIL") << "  got=" << got
            << " expected=" << expected << "\n";
  return ok;
}

int run(int argc, char* argv[]) {
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 5'000'000ULL;

  std::cout << "Building iota(" << n << ")...\n" << std::flush;
  const chunk_seq input = plaid::iota(n);
  std::cout << input.chunks.size() << " chunks across " << GetSSDList().size()
            << " drives\n\n";

  bool all_pass = true;

  // Match at several positions: first, last, and a few interior/boundary ones.
  std::vector<size_t> targets = {0, 1, n / 2, n - 1};
  if (n > 524'288) targets.push_back(524'288);  // first element of chunk 1
  if (n > 524'289) targets.push_back(524'289);
  for (size_t t : targets) {
    all_pass &= report(
        "find_if(== " + std::to_string(t) + ")",
        plaid::ChunkFindIf<uint64_t>(input, [t](uint64_t x) { return x == t; }),
        t);
  }

  // Predicate that is never satisfied -> not found -> returns n.
  all_pass &= report(
      "find_if(no match)",
      plaid::ChunkFindIf<uint64_t>(input, [n](uint64_t x) { return x >= n; }),
      n);

  // Predicate satisfied by many elements -> first satisfying index.
  // x >= n/2 first holds at x == n/2, i.e. position n/2.
  all_pass &= report("find_if(>= n/2)",
                     plaid::ChunkFindIf<uint64_t>(
                         input, [n](uint64_t x) { return x >= n / 2; }),
                     n / 2);

  std::cout << "\n" << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
  return all_pass ? 0 : 1;
}

}  // namespace test_find_if

// ============================================================================
// histogram -- ChunkHistogramByIndex / ByKey
//
// (was ChunkSequence/tests/histogram_test.cpp)
// ============================================================================

namespace test_histogram {

static bool report(const std::string& name, bool ok) {
  std::cout << "  " << std::left << std::setw(40) << name
            << (ok ? "PASS" : "FAIL") << "\n";
  return ok;
}

int run(int argc, char* argv[]) {
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 2'000'000ULL;

  std::cout << "Building iota(" << n << ")...\n" << std::flush;
  const chunk_seq input = plaid::iota(n);
  std::cout << input.chunks.size() << " chunks across " << GetSSDList().size()
            << " drives\n\n";

  bool all_pass = true;

  // 1. Histogram of iota(n) with num_unique = n: every value 0..n-1 appears
  //    exactly once, so every bucket must be 1.
  {
    auto h = plaid::ChunkHistogramByIndex<uint64_t>(input, n);
    bool ok = (h.size() == n);
    for (size_t b = 0; ok && b < n; b++) ok = (h[b] == 1);
    all_pass &= report("iota(n): all buckets == 1", ok);
  }

  // 2. Histogram of (x % k): bucket b gets ceil((n-b)/k) elements.  For k that
  //    divides n, that's exactly n/k per bucket.
  {
    const size_t k = 10;
    chunk_seq mod = plaid::ChunkMap<uint64_t>(
        input, "hist_mod", [k](uint64_t x) { return x % k; });
    auto h = plaid::ChunkHistogramByIndex<uint64_t>(mod, k);
    bool ok = (h.size() == k);
    size_t sum = 0;
    for (size_t b = 0; b < k; b++) {
      const size_t expected = (n - b + k - 1) / k;  // count of x<n with x%k==b
      if (h[b] != expected) ok = false;
      sum += h[b];
    }
    ok = ok && (sum == n);  // counts must total n
    all_pass &= report("(x % 10): per-bucket counts + total", ok);
  }

  std::cout << "\n" << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
  return all_pass ? 0 : 1;
}

}  // namespace test_histogram

// ============================================================================
// scalar -- scalar probes (scan_find / linear_find)
//
// (was ChunkSequence/tests/scalar_test.cpp)
// ============================================================================

namespace test_scalar {

/**
 * Verify the scalar element ops on a materialized chunk_seq:
 *   plaid::size    — total element count (not chunk count)
 *   chunk_seq::operator[]     — read one element at a logical index
 *   chunk_seq::push_back      — append one element (in place)
 *
 * Uses iota(n) as ground truth (element i == i), then exercises both push_back
 * paths: appending into a partial last chunk (read-modify-write of one block)
 * and appending past a full last chunk (new-chunk allocation).  Correctness is
 * cross-checked with operator[] and, finally, consolidate() to a local file.
 */
int run(int argc, char* argv[]) {
  namespace ops = plaid;
  using T = uint64_t;

  int fails = 0;
  auto expect = [&](bool ok, const std::string& msg) {
    if (!ok) {
      std::cerr << "FAIL: " << msg << "\n";
      fails++;
    }
  };

  // ── size + operator[] on a partial-last-chunk iota ─────────────────────────
  const size_t n = (argc > 1)
                       ? std::stoull(argv[1])
                       : (3 * ELEMS_PER_CHUNK + 12345);  // partial last chunk
  std::cout << "iota(" << n << "), ELEMS_PER_CHUNK=" << ELEMS_PER_CHUNK << "\n";
  chunk_seq seq = ops::iota(n);

  expect(ops::size(seq) == n, "size(iota(n)) != n");

  for (size_t i :
       {(size_t)0, n / 2, n - 1, 3 * ELEMS_PER_CHUNK,  // last-chunk head
        3 * ELEMS_PER_CHUNK + 6000}) {                 // mid last chunk
    if (i >= n) continue;
    T got = seq[i];
    expect(got == (T)i,
           "seq[" + std::to_string(i) + "]=" + std::to_string(got));
  }

  // ── push_back into a partial last chunk (read-modify-write path) ───────────
  for (size_t k = 0; k < 5; k++) {
    const size_t before = ops::size(seq);
    const T v = 1'000'000'000ULL + k;
    seq.push_back(v);
    expect(ops::size(seq) == before + 1,
           "size did not grow by 1 after push_back");
    expect(seq[before] == v, "seq[new tail] != pushed value");
    // identity neighbors in the same RMW block/chunk stay untouched
    expect(seq[n - 1] == (T)(n - 1), "push_back corrupted neighbor n-1");
    expect(seq[0] == (T)0, "push_back corrupted element 0");
  }

  // ── push_back that spills into a brand-new chunk ───────────────────────────
  // Fill the current last chunk exactly, then push once more.
  {
    size_t last_used = seq.chunks.back().used;  // bytes
    size_t room = (CHUNK_SIZE - last_used) / sizeof(T);
    for (size_t j = 0; j < room; j++) seq.push_back((T)0xABCD0000 + j);
    expect(seq.chunks.back().used == CHUNK_SIZE,
           "last chunk not full after filling");

    const size_t nc_before = seq.chunks.size();
    const size_t idx = ops::size(seq);
    const T v = 0xFEED1234ULL;
    seq.push_back(v);
    expect(seq.chunks.size() == nc_before + 1,
           "push_back did not allocate a new chunk");
    expect(seq.chunks.back().index == nc_before, "new chunk index wrong");
    expect(seq.chunks.back().used == sizeof(T), "new chunk used != sizeof(T)");
    expect(ops::size(seq) == idx + 1, "size wrong after new-chunk push_back");
    expect(seq[idx] == v, "seq[new-chunk element] != pushed value");
  }

  // ── consolidate + verify the whole thing matches an in-memory model ────────
  // Per-pid path, unlinked below: a fixed shared name in a sticky world-
  // writable dir survives the run, and once one user's copy is left behind
  // every later run by a different user gets EACCES on the O_CREAT (Linux's
  // fs.protected_regular) -- which used to leave this case verifying the
  // STALE file and reporting PASS.
  const char* tmpdir = getenv("TMPDIR");
  const std::string out = std::string(tmpdir && *tmpdir ? tmpdir : "/tmp") +
                          "/scalar_test_consolidated." +
                          std::to_string((long)getpid()) + ".bin";
  seq.consolidate(out);
  {
    FILE* f = fopen(out.c_str(), "rb");
    CHECK(f != nullptr) << "could not reopen " << out << " after consolidate";
    const size_t total = ops::size(seq);
    std::vector<T> buf(total);
    size_t got = fread(buf.data(), sizeof(T), total, f);
    fclose(f);
    expect(got == total, "consolidated file wrong length");
    // Every index we can predict: [0, n) is the identity.
    bool ok = true;
    for (size_t i = 0; i < n && ok; i++)
      if (buf[i] != (T)i) {
        expect(false, "consolidated[" + std::to_string(i) + "] wrong");
        ok = false;
      }
  }
  unlink(out.c_str());

  // ── delayed::size (file / map / index / zip) ───────────────────────────────
  {
    namespace d = plaid::delayed;
    chunk_seq base = ops::iota(n);  // fresh, exactly n elems
    auto del = d::delay(base);
    expect(d::size(del) == n, "delayed::size(delay(seq)) != n");
    auto m = d::map(del, [](uint64_t x) { return x + 1; });
    expect(d::size(m) == n, "delayed::size(map) != n");  // map preserves count
    auto tab = d::tabulate(n + 7, [](size_t i) { return (uint64_t)i; });
    expect(d::size(tab) == n + 7, "delayed::size(tabulate) != n+7");
    auto z = d::zip(del, tab, (uint64_t)0);  // padded zip
    expect(d::size(z) == std::max(n, n + 7),
           "delayed::size(zip) != max(lenA,lenB)");
  }

  std::cout << (fails == 0 ? "PASS" : "FAIL") << "  size=" << ops::size(seq)
            << "  chunks=" << seq.chunks.size() << "  fails=" << fails << "\n";
  return fails == 0 ? 0 : 1;
}

}  // namespace test_scalar

// ============================================================================
// filter -- ChunkFilter dense packing + carry
//
// (was ChunkSequence/tests/filter_test.cpp)
// ============================================================================

namespace test_filter {

struct SumMonoid {
  uint64_t identity = 0;
  uint64_t operator()(uint64_t a, uint64_t b) const { return a + b; }
};

// Remove all per-drive files created under a given prefix (one file per drive).
// Used for both iota input files and filter output files.
static void cleanup_prefix(const std::string& prefix) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    unlink(GetFileName(prefix, d).c_str());
}

// Builds iota(n), applies ChunkFilter, consolidates the survivor stream in
// index order to a local file, and verifies every element equals expected_at(j)
// — i.e. that filter PRESERVES global element order across batch boundaries
// (which the order-insensitive sum check in run_filter_test cannot catch).
static bool run_order_test(const std::string& name, size_t n,
                           std::function<bool(uint64_t)> pred,
                           std::function<uint64_t(size_t)> expected_at,
                           size_t expected_count) {
  std::cout << "  " << name << "  (n=" << n << ", expected=" << expected_count
            << ")\n"
            << std::flush;

  chunk_seq seq = plaid::iota(n);

  const std::string out_prefix = "filter_test_out";
  const std::string consolidated = "filter_test_order_consolidated";
  chunk_seq filtered = plaid::ChunkFilter<uint64_t>(seq, out_prefix, pred);

  bool pass = true;

  // Count check up front (consolidate writes exactly the survivor stream).
  size_t actual_count = 0;
  for (const auto& c : filtered.chunks)
    actual_count += c.used / sizeof(uint64_t);
  if (actual_count != expected_count) {
    std::cout << "    FAIL count: got=" << actual_count
              << " expected=" << expected_count << "\n";
    pass = false;
  } else {
    std::cout << "    count  OK\n";
  }

  // Write survivors to a local file in index order, then read back sequentially
  // and compare each element to the expected in-order value.
  filtered.consolidate(consolidated);

  int fd = open(consolidated.c_str(), O_RDONLY);
  if (fd < 0) {
    std::cout << "    FAIL open(" << consolidated << "): " << strerror(errno)
              << "\n";
    pass = false;
  } else {
    constexpr size_t BUF_ELEMS = (1 << 20);  // 8 MiB worth of uint64_t per read
    std::vector<uint64_t> buf(BUF_ELEMS);
    size_t j = 0;
    bool order_ok = true;
    while (order_ok) {
      const ssize_t got = read(fd, buf.data(), BUF_ELEMS * sizeof(uint64_t));
      if (got < 0) {
        std::cout << "    FAIL read: " << strerror(errno) << "\n";
        pass = order_ok = false;
        break;
      }
      if (got == 0) break;  // EOF
      const size_t count = (size_t)got / sizeof(uint64_t);
      for (size_t i = 0; i < count; i++, j++) {
        const uint64_t expected = expected_at(j);
        if (buf[i] != expected) {
          std::cout << "    FAIL order: element " << j << " got " << buf[i]
                    << " expected " << expected << "\n";
          pass = order_ok = false;
          break;
        }
      }
    }
    close(fd);
    if (order_ok && j != expected_count) {
      std::cout << "    FAIL order: read " << j << " elements, expected "
                << expected_count << "\n";
      pass = false;
    } else if (order_ok) {
      std::cout << "    order  OK\n";
    }
  }

  std::cout << "    => " << (pass ? "PASS" : "FAIL") << "\n\n";

  cleanup_prefix("iota");
  cleanup_prefix(out_prefix);
  unlink(consolidated.c_str());

  return pass;
}

// Builds iota(n), applies ChunkFilter with pred, verifies count / packing /
// index-order / sum, cleans up everything, and returns true iff all checks
// pass.
//
// expected_chunks: if >= 0, also verifies the exact number of output chunks.
static bool run_filter_test(const std::string& name, size_t n,
                            std::function<bool(uint64_t)> pred,
                            size_t expected_count, uint64_t expected_sum,
                            int expected_chunks = -1) {
  std::cout << "  " << name << "  (n=" << n << ", expected=" << expected_count
            << ")\n"
            << std::flush;

  chunk_seq seq = plaid::iota(n);

  const std::string out_prefix = "filter_test_out";
  chunk_seq filtered = plaid::ChunkFilter<uint64_t>(seq, out_prefix, pred);

  bool pass = true;

  // 1. Element count.
  {
    size_t actual = 0;
    for (const auto& c : filtered.chunks) actual += c.used / sizeof(uint64_t);
    if (actual != expected_count) {
      std::cout << "    FAIL count: got=" << actual
                << " expected=" << expected_count << "\n";
      pass = false;
    } else {
      std::cout << "    count  OK\n";
    }
  }

  // 2. Tight packing: all chunks except the last must be full.
  {
    bool ok = true;
    for (size_t i = 0; i + 1 < filtered.chunks.size() && ok; i++) {
      if (filtered.chunks[i].used != CHUNK_SIZE) {
        std::cout << "    FAIL packing: chunk " << i
                  << " used=" << filtered.chunks[i].used
                  << " (expected CHUNK_SIZE=" << CHUNK_SIZE << ")\n";
        pass = ok = false;
      }
    }
    if (ok) std::cout << "    packing OK\n";
  }

  // 3. Index-ordered invariant: chunks[i].index == i.
  {
    bool ok = true;
    for (size_t i = 0; i < filtered.chunks.size() && ok; i++) {
      if (filtered.chunks[i].index != i) {
        std::cout << "    FAIL index order: chunks[" << i
                  << "].index=" << filtered.chunks[i].index << "\n";
        pass = ok = false;
      }
    }
    if (ok) std::cout << "    index  OK\n";
  }

  // 4. Optional exact chunk count.
  if (expected_chunks >= 0) {
    const size_t ec = static_cast<size_t>(expected_chunks);
    if (filtered.chunks.size() != ec) {
      std::cout << "    FAIL #chunks: got=" << filtered.chunks.size()
                << " expected=" << ec << "\n";
      pass = false;
    } else {
      std::cout << "    #chunks OK (" << ec << ")\n";
    }
  }

  // 5. Sum via ChunkReduce (skip for empty output to avoid reducing empty seq).
  if (expected_count > 0) {
    const uint64_t actual_sum =
        plaid::ChunkReduce<uint64_t>(filtered, SumMonoid{});
    if (actual_sum != expected_sum) {
      std::cout << "    FAIL sum: got=" << actual_sum
                << " expected=" << expected_sum << "\n";
      pass = false;
    } else {
      std::cout << "    sum    OK\n";
    }
  }

  std::cout << "    => " << (pass ? "PASS" : "FAIL") << "\n\n";

  // Always clean up so the next sub-test starts with a clean slate.
  cleanup_prefix("iota");
  cleanup_prefix(out_prefix);

  return pass;
}

int run(int argc, char* argv[]) {
  bool all_pass = true;

  // ── 1. No survivors ──────────────────────────────────────────────────────
  // pred always false → empty output chunk_seq; output files exist but are
  // empty.
  all_pass &= run_filter_test(
      "all_fail", ELEMS_PER_CHUNK, [](uint64_t) { return false; },
      /*expected_count=*/0,
      /*expected_sum=*/0,
      /*expected_chunks=*/0);

  // ── 2. All survivors, exactly one full output chunk ──────────────────────
  // n == ELEMS_PER_CHUNK → no carry.
  {
    const size_t n = ELEMS_PER_CHUNK;
    all_pass &= run_filter_test(
        "all_pass_exact_chunk", n, [](uint64_t) { return true; },
        /*expected_count=*/n,
        /*expected_sum=*/(uint64_t)(n - 1) * n / 2,
        /*expected_chunks=*/1);
  }

  // ── 3. Single survivor ───────────────────────────────────────────────────
  // Extreme sparsity: only element 0 passes.
  // Output: 1 partial chunk with used = sizeof(uint64_t) = 8 bytes.
  all_pass &= run_filter_test(
      "single_survivor", ELEMS_PER_CHUNK, [](uint64_t x) { return x == 0; },
      /*expected_count=*/1,
      /*expected_sum=*/0,
      /*expected_chunks=*/1);

  // ── 4. Partial last input chunk, all-pass ────────────────────────────────
  // n is not a multiple of ELEMS_PER_CHUNK: the second (last) input chunk
  // contains only 7 elements.  All-pass → output: 1 full + 1 partial (7 elems).
  {
    const size_t n = ELEMS_PER_CHUNK + 7;
    all_pass &= run_filter_test(
        "partial_input_all_pass", n, [](uint64_t) { return true; },
        /*expected_count=*/n,
        /*expected_sum=*/(uint64_t)(n - 1) * n / 2,
        /*expected_chunks=*/2);
  }

  // ── 5. Leftover carry at end ─────────────────────────────────────────────
  // 3 full input chunks (fits in one batch), x%2==0 keeps exactly half.
  // 3*ELEMS_PER_CHUNK/2 = 1.5*ELEMS_PER_CHUNK survivors
  // → 1 full output chunk + 1 partial (ELEMS_PER_CHUNK/2 elements).
  {
    const size_t n = 3 * ELEMS_PER_CHUNK;
    const size_t expected = n / 2;
    // sum of 0, 2, 4, …, n-2  =  expected*(expected-1)
    all_pass &= run_filter_test(
        "leftover_carry", n, [](uint64_t x) { return x % 2 == 0; }, expected,
        /*expected_sum=*/(uint64_t)(expected - 1) * expected,
        /*expected_chunks=*/2);
  }

  // ── 6. Carry propagates across a batch boundary ──────────────────────────
  // 129 input chunks → 2 batches (128 + 1).
  // x%3==0: batch-1 survivors (22,369,622) % ELEMS_PER_CHUNK = 349,526 ≠ 0,
  // so a non-zero carry flows from batch 1 into batch 2.
  // 129 = 3×43, so 129*ELEMS_PER_CHUNK is divisible by 3 → m = n/3 is exact.
  // sum of 0, 3, 6, … = 3*(m*(m-1)/2).
  {
    const size_t n = 129 * ELEMS_PER_CHUNK;
    const uint64_t m = n / 3;
    all_pass &= run_filter_test(
        "cross_batch_carry", n, [](uint64_t x) { return x % 3 == 0; },
        /*expected_count=*/m,
        /*expected_sum=*/3ULL * (m * (m - 1) / 2));
  }

  // ── 7. Two-batch dense filter (original test) ────────────────────────────
  // 160 chunks across 2 batches, x%2==0 → exactly 80*ELEMS_PER_CHUNK survivors,
  // no carry.  Accepts argv[1] to override n (forced even).
  {
    const size_t n_raw =
        (argc > 1) ? std::stoull(argv[1]) : 160ULL * ELEMS_PER_CHUNK;
    const size_t n = n_raw & ~size_t(1);
    const size_t expected = n / 2;
    all_pass &= run_filter_test(
        "even_dense_2batch", n, [](uint64_t x) { return x % 2 == 0; }, expected,
        /*expected_sum=*/(uint64_t)(expected - 1) * expected);
  }

  // ── 8. Order preservation across batch boundaries ────────────────────────
  // 256 input chunks = 2 full batches.  x%2==0 keeps exactly half; for iota the
  // in-order survivors are 0,2,4,… so element j must equal 2*j.  Any
  // cross-batch reordering (chunks arriving out of completion order) breaks
  // this.
  {
    const size_t n = 256 * ELEMS_PER_CHUNK;
    all_pass &= run_order_test(
        "order_cross_batch", n, [](uint64_t x) { return x % 2 == 0; },
        [](size_t j) -> uint64_t { return 2ULL * j; },
        /*expected_count=*/n / 2);
  }

  std::cout << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
  return all_pass ? 0 : 1;
}

}  // namespace test_filter

// ============================================================================
// flat_tabulate -- ChunkFlatTabulate
//
// (was ChunkSequence/tests/flat_tabulate_test.cpp)
// ============================================================================

namespace test_flat_tabulate {

static void cleanup_prefix(const std::string& prefix) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    unlink(GetFileName(prefix, d).c_str());
}

// ── shared helpers ──────────────────────────────────────────────────────────

// Count elements stored in a chunk_seq from its metadata (no read needed).
static size_t count_elems(const chunk_seq& seq) {
  size_t total = 0;
  for (const auto& c : seq.chunks) total += c.used / sizeof(uint64_t);
  return total;
}

// Check that all chunks except the last have used == CHUNK_SIZE.
static bool check_packing(const chunk_seq& seq, const std::string& label) {
  for (size_t i = 0; i + 1 < seq.chunks.size(); i++) {
    if (seq.chunks[i].used != CHUNK_SIZE) {
      std::cout << "    FAIL packing: " << label << " chunk " << i
                << " used=" << seq.chunks[i].used << " (expected CHUNK_SIZE)\n";
      return false;
    }
  }
  return true;
}

// Check the index-ordered invariant: chunks[i].index == i.
static bool check_index_order(const chunk_seq& seq, const std::string& label) {
  for (size_t i = 0; i < seq.chunks.size(); i++) {
    if (seq.chunks[i].index != i) {
      std::cout << "    FAIL index order: " << label << " chunks[" << i
                << "].index=" << seq.chunks[i].index << "\n";
      return false;
    }
  }
  return true;
}

// Consolidate seq to a local file, read it back sequentially, compare each
// element to expected[j].  Returns true if all elements match.
static bool check_order(const chunk_seq& seq,
                        const std::vector<uint64_t>& expected,
                        const std::string& tmp_path) {
  seq.consolidate(tmp_path);
  int fd = open(tmp_path.c_str(), O_RDONLY);
  if (fd < 0) {
    std::cout << "    FAIL open(" << tmp_path << "): " << strerror(errno)
              << "\n";
    return false;
  }
  constexpr size_t BUF_ELEMS = 1 << 20;
  std::vector<uint64_t> buf(BUF_ELEMS);
  size_t j = 0;
  bool ok = true;
  while (ok) {
    ssize_t got = read(fd, buf.data(), BUF_ELEMS * sizeof(uint64_t));
    if (got < 0) {
      std::cout << "    FAIL read: " << strerror(errno) << "\n";
      ok = false;
      break;
    }
    if (got == 0) break;
    size_t cnt = (size_t)got / sizeof(uint64_t);
    for (size_t i = 0; i < cnt && ok; i++, j++) {
      if (j >= expected.size() || buf[i] != expected[j]) {
        std::cout << "    FAIL order: element " << j << " got=" << buf[i]
                  << " expected=" << expected[j] << "\n";
        ok = false;
      }
    }
  }
  close(fd);
  unlink(tmp_path.c_str());
  if (ok && j != expected.size()) {
    std::cout << "    FAIL order: read " << j << " elements, expected "
              << expected.size() << "\n";
    ok = false;
  }
  return ok;
}

// ── in-memory sieve (for reference counts / golden values) ──────────────────

parlay::sequence<long> in_mem_primes(long n) {
  if (n < 2) return {};
  long sqrt_n = (long)std::sqrt((double)n);
  auto sp = in_mem_primes(sqrt_n);
  parlay::sequence<bool> flags(n + 1, true);
  parlay::parallel_for(
      0, n / sqrt_n + 1,
      [&](long i) {
        long start = sqrt_n * i;
        long end = (std::min)(start + sqrt_n, n + 1);
        for (long p : sp) {
          long first = (std::max)(2 * p, (((start - 1) / p) + 1) * p);
          for (long k = first; k < end; k += p) flags[k] = false;
        }
      },
      1);
  flags[0] = flags[1] = false;
  return parlay::filter(parlay::iota<long>(n + 1),
                        [&](long i) { return flags[i]; });
}

// ── tests ───────────────────────────────────────────────────────────────────

// Test 1: identity — f returns every index in [start, end).
// Output should be [0, 1, ..., n-1] in order with n elements.
static bool test_identity() {
  std::cout << "test_identity\n" << std::flush;
  const size_t n =
      3 * ELEMS_PER_CHUNK + 7;  // spans multiple chunks, non-aligned
  const std::string prefix = "ft_identity";

  chunk_seq result = plaid::ChunkFlatTabulate<uint64_t>(
      n, prefix, [](size_t start, size_t end) {
        parlay::sequence<uint64_t> out(end - start);
        for (size_t i = 0; i < end - start; i++) out[i] = (uint64_t)(start + i);
        return out;
      });

  bool pass = true;

  // Count
  size_t got = count_elems(result);
  if (got != n) {
    std::cout << "  FAIL count: got=" << got << " expected=" << n << "\n";
    pass = false;
  } else {
    std::cout << "  count  OK\n";
  }

  // Packing
  if (!check_packing(result, "identity"))
    pass = false;
  else
    std::cout << "  packing OK\n";

  // Index order invariant
  if (!check_index_order(result, "identity"))
    pass = false;
  else
    std::cout << "  index  OK\n";

  // Element order: should be [0,1,...,n-1]
  std::vector<uint64_t> expected(n);
  for (size_t i = 0; i < n; i++) expected[i] = (uint64_t)i;
  if (!check_order(result, expected, "ft_identity_consolidated"))
    pass = false;
  else
    std::cout << "  order  OK\n";

  std::cout << "  => " << (pass ? "PASS" : "FAIL") << "\n\n";
  cleanup_prefix(prefix);
  return pass;
}

// Test 2: odds — f returns odd numbers in [start, end).
// Output should be [1, 3, 5, ..., largest odd < n] in order.
static bool test_odds() {
  std::cout << "test_odds\n" << std::flush;
  // Use 2 full + 1 partial virtual chunk so carry propagates across a boundary
  const size_t n = 2 * ELEMS_PER_CHUNK + 5;
  const std::string prefix = "ft_odds";

  chunk_seq result = plaid::ChunkFlatTabulate<uint64_t>(
      n, prefix, [](size_t start, size_t end) {
        parlay::sequence<uint64_t> out;
        size_t s = (start % 2 == 0) ? start + 1 : start;
        for (size_t i = s; i < end; i += 2) out.push_back((uint64_t)i);
        return out;
      });

  bool pass = true;

  // Build expected list
  std::vector<uint64_t> expected;
  for (size_t i = 1; i < n; i += 2) expected.push_back((uint64_t)i);

  size_t got = count_elems(result);
  if (got != expected.size()) {
    std::cout << "  FAIL count: got=" << got << " expected=" << expected.size()
              << "\n";
    pass = false;
  } else {
    std::cout << "  count  OK\n";
  }

  if (!check_packing(result, "odds"))
    pass = false;
  else
    std::cout << "  packing OK\n";

  if (!check_index_order(result, "odds"))
    pass = false;
  else
    std::cout << "  index  OK\n";

  if (!check_order(result, expected, "ft_odds_consolidated"))
    pass = false;
  else
    std::cout << "  order  OK\n";

  std::cout << "  => " << (pass ? "PASS" : "FAIL") << "\n\n";
  cleanup_prefix(prefix);
  return pass;
}

// Test 3: empty — f returns nothing, output should be empty.
static bool test_empty_output() {
  std::cout << "test_empty_output\n" << std::flush;
  const std::string prefix = "ft_empty";

  chunk_seq result = plaid::ChunkFlatTabulate<uint64_t>(
      ELEMS_PER_CHUNK, prefix,
      [](size_t, size_t) { return parlay::sequence<uint64_t>{}; });

  bool pass = true;
  if (!result.chunks.empty()) {
    std::cout << "  FAIL: expected 0 output chunks, got "
              << result.chunks.size() << "\n";
    pass = false;
  } else {
    std::cout << "  OK: empty output\n";
  }
  std::cout << "  => " << (pass ? "PASS" : "FAIL") << "\n\n";
  cleanup_prefix(prefix);
  return pass;
}

// Test 4: primes count — compare chunk_primes output count against
// in_mem_primes.
static bool test_primes_count(size_t n) {
  std::cout << "test_primes_count  n=" << n << "\n" << std::flush;
  const std::string prefix = "ft_primes";

  // Compute small primes in memory
  long sqrt_n = (long)std::sqrt((double)n);
  while ((long long)(sqrt_n + 1) * (sqrt_n + 1) <= (long long)n) sqrt_n++;
  parlay::sequence<long> small = in_mem_primes(sqrt_n);

  chunk_seq result = plaid::ChunkFlatTabulate<uint64_t>(
      n + 1, prefix, [&](size_t start, size_t end) {
        std::vector<bool> flags(end - start, true);
        for (long p : small) {
          size_t first = std::max((size_t)(2 * p), (((start - 1) / p) + 1) * p);
          for (size_t k = first; k < end; k += (size_t)p)
            flags[k - start] = false;
        }
        parlay::sequence<uint64_t> out;
        size_t lo = (start < 2) ? 2 : start;
        for (size_t i = lo; i < end; i++)
          if (flags[i - start]) out.push_back((uint64_t)i);
        return out;
      });

  bool pass = true;

  // Reference: in-memory sieve (trusted for small n)
  parlay::sequence<long> ref = in_mem_primes((long)n);
  const size_t expected_count = ref.size();

  size_t got_count = count_elems(result);
  if (got_count != expected_count) {
    std::cout << "  FAIL count: got=" << got_count
              << " expected=" << expected_count << "\n";
    pass = false;
  } else {
    std::cout << "  count  OK  (pi(" << n << ")=" << expected_count << ")\n";
  }

  if (!check_packing(result, "primes"))
    pass = false;
  else
    std::cout << "  packing OK\n";

  if (!check_index_order(result, "primes"))
    pass = false;
  else
    std::cout << "  index  OK\n";

  // For small n also verify element order matches reference exactly
  if (n <= 10'000'000ULL) {
    std::vector<uint64_t> expected(ref.begin(), ref.end());
    if (!check_order(result, expected, "ft_primes_consolidated"))
      pass = false;
    else
      std::cout << "  order  OK\n";
  }

  std::cout << "  => " << (pass ? "PASS" : "FAIL") << "\n\n";
  cleanup_prefix(prefix);
  return pass;
}

// Test 5: chunk boundary — verify that 524287 (= 2^19 - 1, a Mersenne prime) is
// included and that 524288 (= 2^19, clearly composite) is absent.
// These values straddle the boundary of the first virtual chunk [0, 524288).
static bool test_chunk_boundary() {
  std::cout << "test_chunk_boundary\n" << std::flush;

  // Cover a range that includes both sides of the first chunk boundary.
  const size_t n = ELEMS_PER_CHUNK + 100;  // [0, 524388)
  const std::string prefix = "ft_boundary";

  long sqrt_n = (long)std::sqrt((double)n);
  while ((long long)(sqrt_n + 1) * (sqrt_n + 1) <= (long long)n) sqrt_n++;
  parlay::sequence<long> small = in_mem_primes(sqrt_n);

  chunk_seq result = plaid::ChunkFlatTabulate<uint64_t>(
      n + 1, prefix, [&](size_t start, size_t end) {
        std::vector<bool> flags(end - start, true);
        for (long p : small) {
          size_t first = std::max((size_t)(2 * p), (((start - 1) / p) + 1) * p);
          for (size_t k = first; k < end; k += (size_t)p)
            flags[k - start] = false;
        }
        parlay::sequence<uint64_t> out;
        size_t lo = (start < 2) ? 2 : start;
        for (size_t i = lo; i < end; i++)
          if (flags[i - start]) out.push_back((uint64_t)i);
        return out;
      });

  bool pass = true;

  // Consolidate and search for the boundary values.
  const std::string consolidated = "ft_boundary_consol";
  result.consolidate(consolidated);

  int fd = open(consolidated.c_str(), O_RDONLY);
  CHECK(fd >= 0) << "could not open consolidated file";

  constexpr size_t BUF_ELEMS = 1 << 20;
  std::vector<uint64_t> buf(BUF_ELEMS);
  std::set<uint64_t> found;
  while (true) {
    ssize_t got = read(fd, buf.data(), BUF_ELEMS * sizeof(uint64_t));
    if (got <= 0) break;
    size_t cnt = (size_t)got / sizeof(uint64_t);
    for (size_t i = 0; i < cnt; i++) found.insert(buf[i]);
  }
  close(fd);
  unlink(consolidated.c_str());

  // 524287 = 2^19 - 1 is a Mersenne prime — must be present.
  if (found.count(524287) == 0) {
    std::cout
        << "  FAIL: 524287 (2^19-1, Mersenne prime) missing from output\n";
    pass = false;
  } else {
    std::cout << "  OK: 524287 present\n";
  }

  // 524288 = 2^19 is composite — must be absent.
  if (found.count(524288) != 0) {
    std::cout << "  FAIL: 524288 (composite) incorrectly present in output\n";
    pass = false;
  } else {
    std::cout << "  OK: 524288 absent\n";
  }

  std::cout << "  => " << (pass ? "PASS" : "FAIL") << "\n\n";
  cleanup_prefix(prefix);
  return pass;
}

// Test 6: consolidate writes packed raw uint64_t with no padding or header.
// Runs chunk_primes(1000), consolidates to a temp file, reads it back, and
// checks element-by-element against in_mem_primes(1000).
static bool test_consolidate_output() {
  std::cout << "test_consolidate_output\n" << std::flush;
  const size_t n = 1000;
  const std::string prefix = "ft_consol_primes";
  const std::string out_path = "ft_consol_primes_output.bin";

  long sqrt_n = (long)std::sqrt((double)n);
  while ((long long)(sqrt_n + 1) * (sqrt_n + 1) <= (long long)n) sqrt_n++;
  parlay::sequence<long> small = in_mem_primes(sqrt_n);

  chunk_seq result = plaid::ChunkFlatTabulate<uint64_t>(
      n + 1, prefix, [&](size_t start, size_t end) {
        std::vector<bool> flags(end - start, true);
        for (long p : small) {
          size_t first = std::max((size_t)(2 * p), (((start - 1) / p) + 1) * p);
          for (size_t k = first; k < end; k += (size_t)p)
            flags[k - start] = false;
        }
        parlay::sequence<uint64_t> out;
        size_t lo = (start < 2) ? 2 : start;
        for (size_t i = lo; i < end; i++)
          if (flags[i - start]) out.push_back((uint64_t)i);
        return out;
      });

  result.consolidate(out_path);

  bool pass = true;
  parlay::sequence<long> ref = in_mem_primes((long)n);

  // Verify file size: must be exactly ref.size() * 8 bytes, no padding or
  // header.
  struct stat st;
  if (stat(out_path.c_str(), &st) != 0) {
    std::cout << "  FAIL: could not stat " << out_path << "\n";
    pass = false;
  } else {
    const size_t expected_bytes = ref.size() * sizeof(uint64_t);
    if ((size_t)st.st_size != expected_bytes) {
      std::cout << "  FAIL file size: got=" << st.st_size
                << " expected=" << expected_bytes << "\n";
      pass = false;
    } else {
      std::cout << "  file size OK (" << expected_bytes << " bytes)\n";
    }
  }

  // Read back and compare element-by-element.
  int fd = open(out_path.c_str(), O_RDONLY);
  if (fd < 0) {
    std::cout << "  FAIL open: " << strerror(errno) << "\n";
    pass = false;
  } else {
    std::vector<uint64_t> got_vals(ref.size());
    ssize_t bytes = read(fd, got_vals.data(), ref.size() * sizeof(uint64_t));
    close(fd);
    if (bytes != (ssize_t)(ref.size() * sizeof(uint64_t))) {
      std::cout << "  FAIL short read: got=" << bytes << "\n";
      pass = false;
    } else {
      bool ok = true;
      for (size_t i = 0; i < ref.size() && ok; i++) {
        if (got_vals[i] != (uint64_t)ref[i]) {
          std::cout << "  FAIL element " << i << ": got=" << got_vals[i]
                    << " expected=" << ref[i] << "\n";
          ok = pass = false;
        }
      }
      if (ok) std::cout << "  values OK (" << ref.size() << " primes)\n";
    }
  }

  std::cout << "  => " << (pass ? "PASS" : "FAIL") << "\n\n";
  unlink(out_path.c_str());
  cleanup_prefix(prefix);
  return pass;
}

// ── main ─────────────────────────────────────────────────────────────────────

int run(int argc, char* argv[]) {
  bool all_pass = true;

  all_pass &= test_identity();
  all_pass &= test_odds();
  all_pass &= test_empty_output();

  // Known pi(n) values; element-order check runs for n <= 10^7.
  all_pass &= test_primes_count(100);  // pi = 25
  all_pass &= test_primes_count(ELEMS_PER_CHUNK -
                                1);  // includes 524287 (Mersenne prime)
  all_pass &= test_primes_count(ELEMS_PER_CHUNK);  // exactly one virtual chunk
  all_pass &=
      test_primes_count(ELEMS_PER_CHUNK + 1);  // one elem in second chunk
  all_pass &= test_primes_count(1'000'000);    // pi = 78498
  all_pass &= test_primes_count(10'000'000);   // pi = 664579

  all_pass &= test_chunk_boundary();
  all_pass &= test_consolidate_output();

  // Optional large-n check from command line (count only, no order check).
  if (argc > 1) {
    size_t n = std::stoull(argv[1]);
    all_pass &= test_primes_count(n);
  }

  std::cout << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
  return all_pass ? 0 : 1;
}

}  // namespace test_flat_tabulate

// ============================================================================
// flat_map -- ChunkFlatMap, incl. the forward halo
//
// (was ChunkSequence/tests/flat_map_test.cpp)
// ============================================================================

namespace test_flat_map {

// One chunk of uint64_t holds EPCT elements.
static constexpr size_t EPCT = CHUNK_SIZE / sizeof(uint64_t);

// Remove all per-drive files created under a given prefix (one file per drive).
static void cleanup_prefix(const std::string& prefix) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    unlink(GetFileName(prefix, d).c_str());
}

// Verify `matches` against `expected`: element count, tight packing, the
// index-ordered invariant, and exact contents via consolidate + stream compare.
// Mirrors kmp_test's verifier.  Returns true iff PASS.
static bool verify(const std::string& name, const chunk_seq& matches,
                   const std::vector<uint64_t>& expected) {
  bool pass = true;

  size_t actual_count = 0;
  for (const auto& c : matches.chunks)
    actual_count += c.used / sizeof(uint64_t);
  if (actual_count != expected.size()) {
    std::cout << "    FAIL count: got=" << actual_count
              << " expected=" << expected.size() << "\n";
    pass = false;
  } else {
    std::cout << "    count  OK (" << actual_count << ")\n";
  }

  {
    bool ok = true;
    for (size_t i = 0; i + 1 < matches.chunks.size() && ok; i++)
      if (matches.chunks[i].used != CHUNK_SIZE) {
        std::cout << "    FAIL packing: chunk " << i
                  << " used=" << matches.chunks[i].used << "\n";
        pass = ok = false;
      }
    if (ok) std::cout << "    packing OK\n";
  }

  {
    bool ok = true;
    for (size_t i = 0; i < matches.chunks.size() && ok; i++)
      if (matches.chunks[i].index != i) {
        std::cout << "    FAIL index order: chunks[" << i
                  << "].index=" << matches.chunks[i].index << "\n";
        pass = ok = false;
      }
    if (ok) std::cout << "    index  OK\n";
  }

  if (pass && actual_count > 0) {
    const std::string consolidated = "flatmap_test_consolidated";
    matches.consolidate(consolidated);
    std::vector<uint64_t> got = [&] {
      std::vector<uint64_t> v(actual_count);
      int fd = open(consolidated.c_str(), O_RDONLY);
      CHECK(fd >= 0) << "open(" << consolidated << "): " << strerror(errno);
      size_t off = 0;
      while (off < actual_count * sizeof(uint64_t)) {
        ssize_t g = read(fd, (char*)v.data() + off,
                         actual_count * sizeof(uint64_t) - off);
        CHECK(g > 0) << "short read";
        off += (size_t)g;
      }
      close(fd);
      return v;
    }();
    unlink(consolidated.c_str());

    bool ok = true;
    for (size_t i = 0; i < actual_count && ok; i++)
      if (got[i] != expected[i]) {
        std::cout << "    FAIL position " << i << ": got " << got[i]
                  << " expected " << expected[i] << "\n";
        pass = ok = false;
      }
    if (ok) std::cout << "    positions OK\n";
  }

  std::cout << "    => " << (pass ? "PASS" : "FAIL") << "\n\n";
  return pass;
}

// ── Test 1: halo == 0, variable-length per-chunk output, no neighbor ─────────
// Body keeps even input values, doubled.  Asserts the engine passes a null,
// empty halo for every call when halo == 0.  Differential against an in-memory
// filter+map over the same generator.
static bool test_halo0(size_t n, const std::function<uint64_t(size_t)>& gen) {
  std::cout << "  halo0_even_doubled  (n=" << n << ")\n" << std::flush;
  const std::string in_prefix = "flatmap_in";
  const std::string out_prefix = "flatmap_out";

  chunk_seq input = plaid::tabulate<uint64_t>(n, in_prefix, gen);
  std::atomic<bool> saw_bad_halo{false};
  chunk_seq matches = plaid::ChunkFlatMap<uint64_t, uint64_t>(
      input, out_prefix, /*halo=*/0,
      [&](const uint64_t* data, size_t cnt, uint64_t /*gpos*/,
          const uint64_t* halo, size_t halo_n) {
        if (halo != nullptr || halo_n != 0) saw_bad_halo.store(true);
        parlay::sequence<uint64_t> out;
        for (size_t i = 0; i < cnt; i++)
          if (data[i] % 2 == 0) out.push_back(data[i] * 2);
        return out;
      });

  std::vector<uint64_t> expected;
  for (size_t i = 0; i < n; i++)
    if (gen(i) % 2 == 0) expected.push_back(gen(i) * 2);

  bool pass = verify("halo0_even_doubled", matches, expected);
  if (saw_bad_halo.load()) {
    std::cout << "    FAIL: body saw non-null/non-empty halo with halo==0\n";
    pass = false;
  } else {
    std::cout << "    halo-null OK\n";
  }

  cleanup_prefix(in_prefix);
  cleanup_prefix(out_prefix);
  return pass;
}

// ── Test 2/3: halo == 1, a 2-element "pattern" search over uint64_t ──────────
// The body reports every global position i (starting in its own chunk) where
// text[i]==P0 && text[i+1]==P1; the second element may live in the forward
// halo.  Also asserts the final chunk's halo is empty (halo_n == 0).  Planting
// a match across a chunk boundary exercises the in-batch neighbor; planting one
// across the 127/128 batch seam exercises the synchronous seam read.
static constexpr uint64_t P0 = 100, P1 = 101;

static bool test_halo1(const std::string& name, size_t n,
                       const std::function<uint64_t(size_t)>& gen) {
  std::cout << "  " << name << "  (n=" << n << ")\n" << std::flush;
  const std::string in_prefix = "flatmap_in";
  const std::string out_prefix = "flatmap_out";

  chunk_seq input = plaid::tabulate<uint64_t>(n, in_prefix, gen);
  std::atomic<bool> bad_final{false};
  chunk_seq matches = plaid::ChunkFlatMap<uint64_t, uint64_t>(
      input, out_prefix, /*halo=*/1,
      [&](const uint64_t* data, size_t cnt, uint64_t gpos, const uint64_t* halo,
          size_t halo_n) {
        // The very last chunk of the whole sequence must get an empty halo.
        if (gpos + cnt == n && halo_n != 0) bad_final.store(true);
        const size_t avail = cnt + halo_n;
        auto at = [&](size_t i) { return i < cnt ? data[i] : halo[i - cnt]; };
        parlay::sequence<uint64_t> out;
        for (size_t i = 0; i + 1 < avail && i < cnt; i++)
          if (at(i) == P0 && at(i + 1) == P1) out.push_back(gpos + i);
        return out;
      });

  std::vector<uint64_t> expected;
  for (size_t i = 0; i + 1 < n; i++)
    if (gen(i) == P0 && gen(i + 1) == P1) expected.push_back(i);

  bool pass = verify(name, matches, expected);
  if (bad_final.load()) {
    std::cout << "    FAIL: final chunk received a non-empty halo\n";
    pass = false;
  } else {
    std::cout << "    final-halo-empty OK\n";
  }

  cleanup_prefix(in_prefix);
  cleanup_prefix(out_prefix);
  return pass;
}

// ── Test 4: elementwise overload -- variable-length output per element ───────
// f(v) repeats v (v % 3) times, so output length varies 0..2 per input element
// (both the zero-output and multi-output carry cases are exercised).  n is
// chosen to straddle a chunk boundary.  Differential against an in-memory loop.
static bool test_elementwise(size_t n,
                             const std::function<uint64_t(size_t)>& gen) {
  std::cout << "  elementwise_repeat_mod3  (n=" << n << ")\n" << std::flush;
  const std::string in_prefix = "flatmap_in";
  const std::string out_prefix = "flatmap_out";

  chunk_seq input = plaid::tabulate<uint64_t>(n, in_prefix, gen);
  chunk_seq matches = plaid::ChunkFlatMap<uint64_t, uint64_t>(
      input, out_prefix, [](uint64_t v) {
        parlay::sequence<uint64_t> out;
        for (uint64_t k = 0; k < v % 3; k++) out.push_back(v);
        return out;
      });

  std::vector<uint64_t> expected;
  for (size_t i = 0; i < n; i++) {
    uint64_t v = gen(i);
    for (uint64_t k = 0; k < v % 3; k++) expected.push_back(v);
  }

  bool pass = verify("elementwise_repeat_mod3", matches, expected);

  cleanup_prefix(in_prefix);
  cleanup_prefix(out_prefix);
  return pass;
}

int run(int argc, char* argv[]) {
  bool all_pass = true;

  // Deterministic small-range noise (always < P0, so never a spurious match).
  auto noise = [](size_t i) -> uint64_t { return parlay::hash64(i) % 4; };

  // 1. halo == 0: variable-length flat-map with a partial last chunk.
  all_pass &= test_halo0(2 * EPCT + 5, noise);

  // 2. halo == 1, in-batch: matches planted inside a chunk and straddling the
  //    chunk-0/1 and chunk-1/2 boundaries (< DENSE_PACK_BATCH_SIZE chunks).
  {
    const size_t n = 3 * EPCT + 7;
    auto gen = [&](size_t i) -> uint64_t {
      if (i == EPCT / 2) return P0;  // fully inside chunk 0
      if (i == EPCT / 2 + 1) return P1;
      if (i == EPCT - 1) return P0;  // straddles chunk 0/1
      if (i == EPCT) return P1;
      if (i == 2 * EPCT - 1) return P0;  // straddles chunk 1/2
      if (i == 2 * EPCT) return P1;
      return noise(i);
    };
    all_pass &= test_halo1("halo1_inbatch", n, gen);
  }

  // 3. halo == 1, batch seam: > DENSE_PACK_BATCH_SIZE chunks with a match
  //    planted across the chunk-127/128 boundary, exercising the O_DIRECT
  //    seam read.  argv[1] overrides n (min 2 chunks).
  {
    const size_t seam = plaid::DENSE_PACK_BATCH_SIZE;  // 128
    size_t n = (argc > 1) ? std::stoull(argv[1]) : (seam + 1) * EPCT + 9;
    n = std::max(n, 2 * EPCT);
    const size_t last_of_batch = std::min<size_t>(seam, n / EPCT) - 1;
    const size_t plant = (last_of_batch + 1) * EPCT - 1;  // P0 at seam tail
    auto gen = [&](size_t i) -> uint64_t {
      if (i == plant) return P0;      // last elem of chunk `last_of_batch`
      if (i == plant + 1) return P1;  // first elem of next chunk (seam)
      return noise(i);
    };
    all_pass &= test_halo1("halo1_seam", n, gen);
  }

  // 4. Elementwise overload: variable-length (0..2) output per input element,
  //    straddling a chunk boundary.
  all_pass &= test_elementwise(2 * EPCT + 5, noise);

  std::cout << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
  return all_pass ? 0 : 1;
}

}  // namespace test_flat_map

// ============================================================================
// partition -- ChunkPartition k-way split + PARTITION_DROP
//
// (was ChunkSequence/tests/partition_test.cpp)
// ============================================================================

namespace test_partition {

// Correctness test for ChunkPartition (chunk_partition.h).
//
// Builds a uint64_t sequence 0..n-1, partitions it into k buckets by a key
// function that also drops some elements, and verifies:
//   1. every element in bucket b really has key == b (and was not a drop),
//   2. every kept input value appears exactly once across the buckets and every
//      dropped value appears nowhere (union == input minus drops, no dupes),
//   3. each bucket is a valid chunk_seq: index-ordered and dense-except-last.
//
// Exits 0 iff all checks pass.

static constexpr size_t U64_PER_CHUNK = CHUNK_SIZE / sizeof(uint64_t);

static void cleanup_prefix(const std::string& prefix) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    unlink(GetFileName(prefix, d).c_str());
}

// bucket for value v: drop multiples of 13, else v % k.
static size_t key_of(uint64_t v, size_t k) {
  if (v % 13 == 0) return plaid::PARTITION_DROP;
  return (size_t)(v % k);
}

int run(int argc, char* argv[]) {
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 500'000;
  const size_t k = 4;

  const std::string in_prefix = "pt_in";
  const std::string out_prefix = "pt_out";

  chunk_seq seq = plaid::tabulate<uint64_t>(
      n, in_prefix, [](size_t i) { return (uint64_t)i; });

  std::vector<chunk_seq> parts = plaid::ChunkPartition<uint64_t>(
      seq, k, out_prefix, [k](uint64_t v) { return key_of(v, k); });

  bool pass = true;

  if (parts.size() != k) {
    std::cout << "  FAIL: got " << parts.size() << " buckets, expected " << k
              << "\n";
    pass = false;
  }

  std::vector<char> seen(n, 0);  // which input values were returned
  size_t kept = 0;

  for (size_t b = 0; b < parts.size() && pass; b++) {
    const chunk_seq& bucket = parts[b];

    // Index-ordered + dense-except-last.
    for (size_t i = 0; i < bucket.chunks.size(); i++) {
      if (bucket.chunks[i].index != i) {
        std::cout << "  FAIL bucket " << b << ": chunk " << i << " has index "
                  << bucket.chunks[i].index << "\n";
        pass = false;
        break;
      }
      if (i + 1 < bucket.chunks.size() && bucket.chunks[i].used != CHUNK_SIZE) {
        std::cout << "  FAIL bucket " << b << ": non-last chunk " << i
                  << " used=" << bucket.chunks[i].used << " (not full)\n";
        pass = false;
        break;
      }
    }
    if (!pass) break;

    // Contents: every value routes to this bucket and is not a drop.
    std::vector<uint64_t> vals = bucket.to_vector<uint64_t>();
    for (uint64_t v : vals) {
      if (v >= n) {
        std::cout << "  FAIL bucket " << b << ": value " << v << " >= n\n";
        pass = false;
        break;
      }
      if (key_of(v, k) != b) {
        std::cout << "  FAIL bucket " << b << ": value " << v << " has key "
                  << key_of(v, k) << "\n";
        pass = false;
        break;
      }
      if (seen[v]) {
        std::cout << "  FAIL: value " << v << " appears more than once\n";
        pass = false;
        break;
      }
      seen[v] = 1;
      kept++;
    }
  }

  // Every kept input present exactly once; every dropped input absent.
  if (pass) {
    size_t expected_kept = 0;
    for (size_t v = 0; v < n; v++) {
      const bool dropped = (v % 13 == 0);
      if (!dropped) expected_kept++;
      if (seen[v] == (dropped ? 1 : 0)) {
        std::cout << "  FAIL: value " << v
                  << (dropped ? " dropped but present" : " kept but missing")
                  << "\n";
        pass = false;
        break;
      }
    }
    if (pass && kept != expected_kept) {
      std::cout << "  FAIL: kept " << kept << " != expected " << expected_kept
                << "\n";
      pass = false;
    }
    if (pass)
      std::cout << "  OK: " << kept << " kept across " << k
                << " buckets, drops absent, packing valid\n";
  }

  cleanup_prefix(in_prefix);
  cleanup_prefix(out_prefix);

  std::cout << (pass ? "PASS" : "FAIL") << "\n";
  return pass ? 0 : 1;
}

}  // namespace test_partition

// ============================================================================
// group_by -- group_by_index / group_by_key
//
// (was ChunkSequence/tests/group_by_test.cpp)
// ============================================================================

namespace test_group_by {

// Correctness test for group_by_index / group_by_key (group_by.h).
//
// group_by_index: builds a uint64_t sequence 0..n-1, groups it into k buckets
// by v % k (no drop -- group_by_index has no drop sentinel, unlike
// ChunkPartition), and verifies:
//   1. every element in bucket b really has v % k == b,
//   2. every input value appears exactly once across the buckets (full
//      coverage, since nothing can be dropped),
//   3. each bucket is a valid chunk_seq: index-ordered and dense-except-last.
//
// group_by_key: groups the same kind of sequence by an arbitrary key (first
// the identity key, then a coarser derived key v/1000), and verifies every
// returned value lands in the bucket its own hash(key)%num_buckets predicts,
// plus full coverage -- group_by_key is a thin hash-bucket wrapper over
// group_by_index, so it inherits the same no-drop/dense-except-last
// invariants.
//
// Exits 0 iff all checks pass.

static void cleanup_prefix(const std::string& prefix) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    unlink(GetFileName(prefix, d).c_str());
}

// Checks that every bucket in `parts` is index-ordered + dense-except-last,
// every value satisfies `bucket_of(value) == bucket index`, and every input
// value in [0, n) is returned exactly once across all buckets. Returns
// whether all checks passed; prints FAIL lines for anything that didn't.
template <typename BucketOf>
static bool check_grouping(const std::string& label,
                           const std::vector<chunk_seq>& parts, size_t n,
                           BucketOf bucket_of) {
  bool pass = true;
  std::vector<char> seen(n, 0);

  for (size_t b = 0; b < parts.size() && pass; b++) {
    const chunk_seq& bucket = parts[b];

    for (size_t i = 0; i < bucket.chunks.size(); i++) {
      if (bucket.chunks[i].index != i) {
        std::cout << "  FAIL " << label << " bucket " << b << ": chunk " << i
                  << " has index " << bucket.chunks[i].index << "\n";
        pass = false;
        break;
      }
      if (i + 1 < bucket.chunks.size() && bucket.chunks[i].used != CHUNK_SIZE) {
        std::cout << "  FAIL " << label << " bucket " << b
                  << ": non-last chunk " << i
                  << " used=" << bucket.chunks[i].used << " (not full)\n";
        pass = false;
        break;
      }
    }
    if (!pass) break;

    std::vector<uint64_t> vals = bucket.to_vector<uint64_t>();
    for (uint64_t v : vals) {
      if (v >= n) {
        std::cout << "  FAIL " << label << " bucket " << b << ": value " << v
                  << " >= n\n";
        pass = false;
        break;
      }
      if (bucket_of(v) != b) {
        std::cout << "  FAIL " << label << " bucket " << b << ": value " << v
                  << " expected bucket " << bucket_of(v) << "\n";
        pass = false;
        break;
      }
      if (seen[v]) {
        std::cout << "  FAIL " << label << ": value " << v
                  << " appears more than once\n";
        pass = false;
        break;
      }
      seen[v] = 1;
    }
  }

  if (pass) {
    for (size_t v = 0; v < n; v++) {
      if (!seen[v]) {
        std::cout << "  FAIL " << label << ": value " << v << " missing\n";
        pass = false;
        break;
      }
    }
  }
  if (pass)
    std::cout << "  OK " << label << ": " << n << " values covered across "
              << parts.size() << " buckets, packing valid\n";
  return pass;
}

int run(int argc, char* argv[]) {
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 500'000;
  const size_t k = 4;

  bool pass = true;

  // group_by_index: bucket = v % k.
  {
    const std::string in_prefix = "gbi_in";
    const std::string out_prefix = "gbi_out";
    chunk_seq seq = plaid::tabulate<uint64_t>(
        n, in_prefix, [](size_t i) { return (uint64_t)i; });

    std::vector<chunk_seq> parts = plaid::group_by_index<uint64_t>(
        seq, k, out_prefix, [k](uint64_t v) { return (size_t)(v % k); });

    if (parts.size() != k) {
      std::cout << "  FAIL group_by_index: got " << parts.size()
                << " buckets, expected " << k << "\n";
      pass = false;
    } else if (!check_grouping("group_by_index", parts, n,
                               [k](uint64_t v) { return (size_t)(v % k); })) {
      pass = false;
    }

    cleanup_prefix(in_prefix);
    cleanup_prefix(out_prefix);
  }

  // group_by_index with a bucket count in the thousands -- unlike the k=4
  // case above, this exercises BucketWriter's per-bucket setup cost
  // (Primitives/chunk_seq.h: a Request pool entry and scatter buffers sized
  // per bucket, not per element) at a bucket count large relative to n, the
  // regime that crashed benchmarks/summary_figure.py's group_by_index demo
  // (ChunkSequence/examples/primitive_demos.cpp) before it started scaling
  // its bucket count with n/RAM instead of using a fixed 4096. cleanup here
  // must unlink one file per bucket, not per drive -- reusing cleanup_prefix
  // (SSD_COUNT files) for a bucket prefix would strand almost all of them.
  {
    const size_t n_large = 50'000;
    const size_t k_large = 4096;
    const std::string in_prefix = "gbi_large_in";
    const std::string out_prefix = "gbi_large_out";
    chunk_seq seq = plaid::tabulate<uint64_t>(
        n_large, in_prefix, [](size_t i) { return (uint64_t)i; });

    std::vector<chunk_seq> parts = plaid::group_by_index<uint64_t>(
        seq, k_large, out_prefix,
        [k_large](uint64_t v) { return (size_t)(v % k_large); });

    if (parts.size() != k_large) {
      std::cout << "  FAIL group_by_index(large k): got " << parts.size()
                << " buckets, expected " << k_large << "\n";
      pass = false;
    } else if (!check_grouping(
                   "group_by_index(large k)", parts, n_large,
                   [k_large](uint64_t v) { return (size_t)(v % k_large); })) {
      pass = false;
    }

    cleanup_prefix(in_prefix);
    for (size_t i = 0; i < k_large; i++)
      unlink(GetFileName(out_prefix, i).c_str());
  }

  // group_by_key: identity key, default Hash = std::hash<uint64_t>.
  {
    const std::string in_prefix = "gbk_id_in";
    const std::string out_prefix = "gbk_id_out";
    chunk_seq seq = plaid::tabulate<uint64_t>(
        n, in_prefix, [](size_t i) { return (uint64_t)i; });

    auto key_of = [](uint64_t v) { return v; };
    auto expected_bucket = [k](uint64_t v) {
      return (size_t)(std::hash<uint64_t>{}(v) % k);
    };

    std::vector<chunk_seq> parts =
        plaid::group_by_key<uint64_t>(seq, k, out_prefix, key_of);

    if (parts.size() != k) {
      std::cout << "  FAIL group_by_key(identity): got " << parts.size()
                << " buckets, expected " << k << "\n";
      pass = false;
    } else if (!check_grouping("group_by_key(identity)", parts, n,
                               expected_bucket)) {
      pass = false;
    }

    cleanup_prefix(in_prefix);
    cleanup_prefix(out_prefix);
  }

  // group_by_key: derived key (v / 1000), same default Hash.
  {
    const std::string in_prefix = "gbk_derived_in";
    const std::string out_prefix = "gbk_derived_out";
    chunk_seq seq = plaid::tabulate<uint64_t>(
        n, in_prefix, [](size_t i) { return (uint64_t)i; });

    auto key_of = [](uint64_t v) { return v / 1000; };
    auto expected_bucket = [k](uint64_t v) {
      return (size_t)(std::hash<uint64_t>{}(v / 1000) % k);
    };

    std::vector<chunk_seq> parts =
        plaid::group_by_key<uint64_t>(seq, k, out_prefix, key_of);

    if (parts.size() != k) {
      std::cout << "  FAIL group_by_key(derived): got " << parts.size()
                << " buckets, expected " << k << "\n";
      pass = false;
    } else if (!check_grouping("group_by_key(derived)", parts, n,
                               expected_bucket)) {
      pass = false;
    }

    cleanup_prefix(in_prefix);
    cleanup_prefix(out_prefix);
  }

  std::cout << (pass ? "PASS" : "FAIL") << "\n";
  return pass ? 0 : 1;
}

}  // namespace test_group_by

// ============================================================================
// reverse -- in-place chunk_seq reversal
//
// (was ChunkSequence/Primitives/reverse.h)
// ============================================================================

namespace test_reverse {

// Correctness test for plaid::reverse (secondary_primitives.h): builds
// 0..n-1, reverses it out-of-core, and checks the readback matches
// std::reverse on a DRAM copy -- exercised at a size that spans multiple
// chunks with a partial last chunk, and again at a size that fits in exactly
// one chunk (n < ELEMS_PER_CHUNK), since reverse's chunk-swap approach
// deliberately relocates any partial chunk from last to first and the
// element-level result must still come out byte-correct either way.

static void cleanup_prefix(const std::string& prefix) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    unlink(GetFileName(prefix, d).c_str());
}

static bool check_reverse(const std::string& label, size_t n) {
  const std::string prefix = "rev_test_in";
  chunk_seq seq = plaid::tabulate<uint64_t>(
      n, prefix, [](size_t i) { return (uint64_t)i; });

  std::vector<uint64_t> expected = seq.to_vector<uint64_t>();
  std::reverse(expected.begin(), expected.end());

  plaid::reverse<uint64_t>(seq);
  std::vector<uint64_t> got = seq.to_vector<uint64_t>();

  cleanup_prefix(prefix);

  bool pass = (got == expected);
  if (!pass) {
    std::cout << "  FAIL " << label << ": n=" << n
              << " reversed contents mismatch\n";
    for (size_t i = 0; i < n && i < 5; i++) {
      std::cout << "    got[" << i << "]=" << got[i] << " expected[" << i
                << "]=" << expected[i] << "\n";
    }
  } else {
    std::cout << "  OK " << label << ": n=" << n
              << " reversed contents match std::reverse\n";
  }
  return pass;
}

int run(int argc, char* argv[]) {
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 500'000;

  bool pass = true;
  if (!check_reverse("reverse (multi-chunk, partial last)", n)) pass = false;
  if (!check_reverse("reverse (single chunk)",
                     std::min<size_t>(n, ELEMS_PER_CHUNK / 4)))
    pass = false;

  return pass ? 0 : 1;
}

}  // namespace test_reverse

// ============================================================================
// chunk_operation -- ChunkOperation apply<> + process_inplace_budgeted
//
// (was ChunkSequence/tests/chunk_operation_test.cpp)
// ============================================================================

namespace test_chunk_operation {

// Correctness test for ChunkOperation's dispatch front door
// (ExternalPrimitives/chunk_operation.h) and the DRAM-budget-checked,
// wave-batched process_inplace_budgeted engine it's built on
// (ExternalPrimitives/small_sequence_ops.h). Exercises Sort and Shuffle under
// both a default (single-wave) and an artificially small (forced multi-wave)
// DRAM budget.
//
// Exits 0 iff all checks pass.

using plaid::apply;
using plaid::ChunkOperation;

namespace {

void set_budget(const char* v) { setenv("PROCESS_INPLACE_BUDGET_BYTES", v, 1); }
void clear_budget() { unsetenv("PROCESS_INPLACE_BUDGET_BYTES"); }

// Per-bucket key generator: a pure function of (bucket, local index), so a
// bucket's original content can be regenerated for comparison without having
// to snapshot it before the in-place operation runs.
uint64_t key_at(size_t bucket, size_t i) {
  return parlay::hash64(bucket * 1000000007ULL + i);
}

}  // namespace

// Build `num_buckets` independent chunk_seqs of `elems_per_bucket` elements
// each, via key_at.
static std::vector<chunk_seq> build_buckets(size_t num_buckets,
                                            size_t elems_per_bucket,
                                            const std::string& prefix) {
  std::vector<chunk_seq> buckets(num_buckets);
  for (size_t b = 0; b < num_buckets; b++) {
    buckets[b] = plaid::tabulate<uint64_t>(
        elems_per_bucket, prefix + "_" + std::to_string(b),
        [b](size_t i) { return key_at(b, i); });
  }
  return buckets;
}

static bool check_sort(std::vector<chunk_seq>& buckets, size_t elems_per_bucket,
                       const char* label) {
  bool ok = true;
  for (size_t b = 0; b < buckets.size(); b++) {
    std::vector<uint64_t> got = buckets[b].to_vector<uint64_t>();
    auto ref = parlay::tabulate(elems_per_bucket,
                                [b](size_t i) { return key_at(b, i); });
    parlay::sort_inplace(ref);
    if (got.size() != ref.size() ||
        !std::equal(got.begin(), got.end(), ref.begin())) {
      std::cout << "  FAIL " << label << ": bucket " << b
                << " not sorted correctly\n";
      ok = false;
    }
  }
  if (ok) std::cout << "  OK " << label << ": all buckets sorted correctly\n";
  return ok;
}

static bool check_shuffle(std::vector<chunk_seq>& buckets,
                          size_t elems_per_bucket, const char* label) {
  bool ok = true;
  for (size_t b = 0; b < buckets.size(); b++) {
    std::vector<uint64_t> got = buckets[b].to_vector<uint64_t>();
    auto ref = parlay::tabulate(elems_per_bucket,
                                [b](size_t i) { return key_at(b, i); });
    if (got.size() != ref.size()) {
      std::cout << "  FAIL " << label << ": bucket " << b << " has "
                << got.size() << " elements, expected " << ref.size() << "\n";
      ok = false;
      continue;
    }
    // Permutation check (exact order is randomized): a valid shuffle's sorted
    // content must match the sorted original content exactly.
    std::sort(got.begin(), got.end());
    parlay::sort_inplace(ref);
    if (!std::equal(got.begin(), got.end(), ref.begin())) {
      std::cout << "  FAIL " << label << ": bucket " << b
                << " is not a permutation of its original content\n";
      ok = false;
    }
  }
  if (ok)
    std::cout << "  OK " << label
              << ": all buckets are permutations of their original content\n";
  return ok;
}

int run(int argc, char* argv[]) {
  const size_t num_buckets = 6;
  const size_t elems_per_bucket =
      (argc > 1) ? std::stoull(argv[1]) : 2'000'000;  // ~16MB/bucket (uint64_t)
  const size_t bucket_bytes = elems_per_bucket * sizeof(uint64_t);

  // A budget that fits exactly one bucket comfortably but not two -- forces
  // more than one wave across the 6 buckets built below, without ever
  // tripping process_inplace_budgeted's own-bucket-too-big CHECK (each
  // bucket alone is well under this budget).
  const size_t small_budget = bucket_bytes + bucket_bytes / 2;  // 1.5 buckets

  std::cout << "elems_per_bucket=" << elems_per_bucket
            << " bucket_bytes=" << bucket_bytes
            << " small_budget=" << small_budget << "\n";

  bool pass = true;

  // -- Sort, default (single-wave) budget --------------------------------
  clear_budget();
  {
    auto buckets =
        build_buckets(num_buckets, elems_per_bucket, "co_sort_default");
    apply<ChunkOperation::Sort, uint64_t>(buckets);
    pass &= check_sort(buckets, elems_per_bucket, "sort (default budget)");
    bench_drives::clear_drives({"co_sort_default"});
  }

  // -- Sort, forced multi-wave --------------------------------------------
  set_budget(std::to_string(small_budget).c_str());
  {
    auto buckets =
        build_buckets(num_buckets, elems_per_bucket, "co_sort_multiwave");
    apply<ChunkOperation::Sort, uint64_t>(buckets);
    pass &= check_sort(buckets, elems_per_bucket,
                       "sort (forced multi-wave budget)");
    bench_drives::clear_drives({"co_sort_multiwave"});
  }

  // -- Shuffle, default (single-wave) budget ------------------------------
  clear_budget();
  {
    auto buckets =
        build_buckets(num_buckets, elems_per_bucket, "co_shuf_default");
    apply<ChunkOperation::Shuffle, uint64_t>(buckets, {}, /*seed=*/42);
    pass &=
        check_shuffle(buckets, elems_per_bucket, "shuffle (default budget)");
    bench_drives::clear_drives({"co_shuf_default"});
  }

  // -- Shuffle, forced multi-wave ------------------------------------------
  set_budget(std::to_string(small_budget).c_str());
  {
    auto buckets =
        build_buckets(num_buckets, elems_per_bucket, "co_shuf_multiwave");
    apply<ChunkOperation::Shuffle, uint64_t>(buckets, {}, /*seed=*/1234);
    pass &= check_shuffle(buckets, elems_per_bucket,
                          "shuffle (forced multi-wave budget)");
    bench_drives::clear_drives({"co_shuf_multiwave"});
  }
  clear_budget();

  std::cout << (pass ? "PASS" : "FAIL") << "\n";
  return pass ? 0 : 1;
}

}  // namespace test_chunk_operation

// ============================================================================
// combined -- multi-primitive pipelines end to end
//
// (was ChunkSequence/tests/combined_test.cpp)
// ============================================================================

namespace test_combined {

// Combined / integration test for the ChunkSequence primitives.
//
// Where the per-primitive tests (map/reduce/filter/scan) check one operation in
// isolation against a closed form, this suite *chains* tabulate/iota → ChunkMap
// → ChunkFilter → ChunkScan → ChunkReduce and verifies the composed result
// against a plain serial reference computed over a std::vector.  Each pipeline
// is run across a battery of edge-case sizes: empty, single element, partial
// last chunk, exact chunk boundary, just over a chunk, several chunks, and (for
// filter) a size that spans multiple FILTER_BATCH_SIZE batches.
//
// The verification trick is chunk_seq::consolidate: it writes a chunk_seq's
// elements to a local file in index order, which we read back and compare to
// the reference vector element-by-element.  This makes deep equality checks
// trivial for arbitrary compositions.

// ── monoids (shared with the per-primitive tests) ────────────────────────────
struct SumMonoid {
  uint64_t identity = 0;
  uint64_t operator()(uint64_t a, uint64_t b) const { return a + b; }
};
struct MaxMonoid {
  uint64_t identity = 0;
  uint64_t operator()(uint64_t a, uint64_t b) const { return std::max(a, b); }
};
struct MinMonoid {
  uint64_t identity = UINT64_MAX;
  uint64_t operator()(uint64_t a, uint64_t b) const { return std::min(a, b); }
};
struct XorMonoid {
  uint64_t identity = 0;
  uint64_t operator()(uint64_t a, uint64_t b) const { return a ^ b; }
};

// ── global pass/fail bookkeeping ─────────────────────────────────────────────
static size_t g_pass = 0, g_fail = 0;

static bool report(const std::string& name, bool ok,
                   const std::string& detail = "") {
  std::cout << "    " << std::left << std::setw(44) << name
            << (ok ? "PASS" : "FAIL");
  if (!ok && !detail.empty()) std::cout << "  " << detail;
  std::cout << "\n";
  (ok ? g_pass : g_fail)++;
  return ok;
}

// ── helpers ──────────────────────────────────────────────────────────────────

// Remove the per-drive files created under a prefix (one per drive).
static void cleanup_prefix(const std::string& prefix) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    unlink(GetFileName(prefix, d).c_str());
}

// Read a chunk_seq's elements, in index order, into a vector<T> via
// consolidate.
template <typename T>
static std::vector<T> materialize(const chunk_seq& seq) {
  const std::string tmp = "combined_test_materialize.tmp";
  seq.consolidate(tmp);

  int fd = open(tmp.c_str(), O_RDONLY);
  CHECK(fd >= 0) << "materialize: open(" << tmp << "): " << strerror(errno);

  std::vector<T> out;
  std::vector<T> buf(1 << 20);  // 1 Mi elements per read
  while (true) {
    const ssize_t got = read(fd, buf.data(), buf.size() * sizeof(T));
    CHECK(got >= 0) << "materialize: read: " << strerror(errno);
    if (got == 0) break;
    out.insert(out.end(), buf.begin(), buf.begin() + (size_t)got / sizeof(T));
  }
  close(fd);
  unlink(tmp.c_str());
  return out;
}

// Deep-equality check: materialize got_seq and compare to expected.
template <typename T>
static bool expect_eq_vec(const std::string& name, const chunk_seq& got_seq,
                          const std::vector<T>& expected) {
  const std::vector<T> got = materialize<T>(got_seq);
  if (got.size() != expected.size())
    return report(name, false,
                  "size got=" + std::to_string(got.size()) +
                      " want=" + std::to_string(expected.size()));
  for (size_t i = 0; i < got.size(); i++)
    if (got[i] != expected[i])
      return report(name, false,
                    "elem " + std::to_string(i) +
                        " got=" + std::to_string((uint64_t)got[i]) +
                        " want=" + std::to_string((uint64_t)expected[i]));
  return report(name, true);
}

static bool expect_scalar(const std::string& name, uint64_t got,
                          uint64_t want) {
  return report(name, got == want,
                "got=" + std::to_string(got) + " want=" + std::to_string(want));
}

// ── serial reference implementations (operate on std::vector<uint64_t>)
// ───────

static std::vector<uint64_t> ref_iota(size_t n) {
  std::vector<uint64_t> v(n);
  for (size_t i = 0; i < n; i++) v[i] = (uint64_t)i;
  return v;
}
static std::vector<uint64_t> ref_map(
    std::vector<uint64_t> v, const std::function<uint64_t(uint64_t)>& f) {
  for (auto& x : v) x = f(x);
  return v;
}
static std::vector<uint64_t> ref_filter(
    const std::vector<uint64_t>& v, const std::function<bool(uint64_t)>& p) {
  std::vector<uint64_t> out;
  for (auto x : v)
    if (p(x)) out.push_back(x);
  return out;
}
// Exclusive scan; returns the prefix vector and writes the grand total to
// *total.
template <typename Monoid>
static std::vector<uint64_t> ref_scan_excl(const std::vector<uint64_t>& v,
                                           Monoid m, uint64_t* total) {
  std::vector<uint64_t> out(v.size());
  uint64_t run = m.identity;
  for (size_t i = 0; i < v.size(); i++) {
    out[i] = run;
    run = m(run, v[i]);
  }
  *total = run;
  return out;
}
template <typename Monoid>
static uint64_t ref_reduce(const std::vector<uint64_t>& v, Monoid m) {
  uint64_t acc = m.identity;
  for (auto x : v) acc = m(acc, x);
  return acc;
}

// ── per-size pipeline battery (deep equality via materialize)
// ─────────────────

static void run_size(size_t n) {
  std::cout << "  n=" << n << "  ("
            << ((n + ELEMS_PER_CHUNK - 1) / ELEMS_PER_CHUNK) << " chunks)\n";

  const std::vector<uint64_t> base = ref_iota(n);

  // ── ChunkMap: x -> x+1 (in-place, T==R) ──────────────────────────────────
  {
    chunk_seq seq = plaid::iota(n);
    chunk_seq out = plaid::ChunkMap<uint64_t>(
        seq, "comb_map",
        std::function<uint64_t(uint64_t)>([](uint64_t x) { return x + 1; }));
    expect_eq_vec<uint64_t>("map  x->x+1", out,
                            ref_map(base, [](uint64_t x) { return x + 1; }));
    cleanup_prefix("iota");
    cleanup_prefix("comb_map");
  }

  // ── ChunkMap: type-changing u64 -> u32 (non-in-place path) ───────────────
  {
    chunk_seq seq = plaid::iota(n);
    chunk_seq out = plaid::ChunkMap<uint64_t, uint32_t>(
        seq, "comb_map32", std::function<uint32_t(uint64_t)>([](uint64_t x) {
          return (uint32_t)(x & 0xFFFFFFFFu);
        }));
    std::vector<uint32_t> expected(n);
    for (size_t i = 0; i < n; i++)
      expected[i] = (uint32_t)(base[i] & 0xFFFFFFFFu);
    expect_eq_vec<uint32_t>("map  u64->u32", out, expected);
    cleanup_prefix("iota");
    cleanup_prefix("comb_map32");
  }

  // ── ChunkScan: exclusive sum, with returned total ────────────────────────
  {
    chunk_seq seq = plaid::iota(n);
    auto [out, total] =
        plaid::ChunkScan<uint64_t>(seq, "comb_scan", SumMonoid{});
    uint64_t ref_total = 0;
    auto ref = ref_scan_excl(base, SumMonoid{}, &ref_total);
    expect_eq_vec<uint64_t>("scan sum (exclusive)", out, ref);
    expect_scalar("scan sum total", total, ref_total);
    cleanup_prefix("iota");
    cleanup_prefix("comb_scan");
  }

  // ── ChunkFilter: keep evens, order-preserving ────────────────────────────
  {
    chunk_seq seq = plaid::iota(n);
    chunk_seq out = plaid::ChunkFilter<uint64_t>(
        seq, "comb_flt",
        std::function<bool(uint64_t)>([](uint64_t x) { return x % 2 == 0; }));
    expect_eq_vec<uint64_t>(
        "filter evens", out,
        ref_filter(base, [](uint64_t x) { return x % 2 == 0; }));
    cleanup_prefix("iota");
    cleanup_prefix("comb_flt");
  }

  // ── ChunkReduce: sum / max / min / xor (scalars, identity-correct on empty)
  // ─
  {
    chunk_seq seq = plaid::iota(n);
    expect_scalar("reduce sum", plaid::ChunkReduce<uint64_t>(seq, SumMonoid{}),
                  ref_reduce(base, SumMonoid{}));
    expect_scalar("reduce max", plaid::ChunkReduce<uint64_t>(seq, MaxMonoid{}),
                  ref_reduce(base, MaxMonoid{}));
    expect_scalar("reduce min", plaid::ChunkReduce<uint64_t>(seq, MinMonoid{}),
                  ref_reduce(base, MinMonoid{}));
    expect_scalar("reduce xor", plaid::ChunkReduce<uint64_t>(seq, XorMonoid{}),
                  ref_reduce(base, XorMonoid{}));
    cleanup_prefix("iota");
  }

  // ── Flagship: map -> filter -> scan -> reduce, all chained ────────────────
  // iota(n) -> (3x+1) -> keep even -> exclusive-sum scan -> max reduce.
  {
    chunk_seq seq = plaid::iota(n);
    chunk_seq mapped = plaid::ChunkMap<uint64_t>(
        seq, "comb_p_map", std::function<uint64_t(uint64_t)>([](uint64_t x) {
          return 3 * x + 1;
        }));
    chunk_seq filt = plaid::ChunkFilter<uint64_t>(
        mapped, "comb_p_flt",
        std::function<bool(uint64_t)>([](uint64_t x) { return x % 2 == 0; }));
    auto [scanned, total] =
        plaid::ChunkScan<uint64_t>(filt, "comb_p_scan", SumMonoid{});
    uint64_t mx = plaid::ChunkReduce<uint64_t>(scanned, MaxMonoid{});

    // Serial reference of the same chain.
    auto rv = ref_map(base, [](uint64_t x) { return 3 * x + 1; });
    rv = ref_filter(rv, [](uint64_t x) { return x % 2 == 0; });
    uint64_t ref_total = 0;
    auto rscan = ref_scan_excl(rv, SumMonoid{}, &ref_total);
    uint64_t ref_mx = ref_reduce(rscan, MaxMonoid{});

    expect_eq_vec<uint64_t>("pipeline map|filter|scan output", scanned, rscan);
    expect_scalar("pipeline scan total", total, ref_total);
    expect_scalar("pipeline reduce(max)", mx, ref_mx);

    cleanup_prefix("iota");
    cleanup_prefix("comb_p_map");
    cleanup_prefix("comb_p_flt");
    cleanup_prefix("comb_p_scan");
  }
}

// ── targeted edge cases not tied to the size sweep ───────────────────────────

static void run_edge_cases() {
  std::cout << "  edge cases\n";

  const size_t n = 3 * ELEMS_PER_CHUNK + 5;
  const std::vector<uint64_t> base = ref_iota(n);

  // filter that keeps everything → identity-shaped, order preserved.
  {
    chunk_seq seq = plaid::iota(n);
    chunk_seq out = plaid::ChunkFilter<uint64_t>(
        seq, "edge_flt_all",
        std::function<bool(uint64_t)>([](uint64_t) { return true; }));
    expect_eq_vec<uint64_t>("filter keep-all == input", out, base);
    cleanup_prefix("iota");
    cleanup_prefix("edge_flt_all");
  }

  // filter that drops everything → empty; chaining scan/reduce on the empty
  // seq.
  {
    chunk_seq seq = plaid::iota(n);
    chunk_seq empty = plaid::ChunkFilter<uint64_t>(
        seq, "edge_flt_none",
        std::function<bool(uint64_t)>([](uint64_t) { return false; }));
    expect_scalar("filter drop-all -> 0 chunks", empty.chunks.size(), 0);

    auto [sc, total] =
        plaid::ChunkScan<uint64_t>(empty, "edge_scan_empty", SumMonoid{});
    expect_scalar("scan(empty) -> 0 chunks", sc.chunks.size(), 0);
    expect_scalar("scan(empty) total == identity", total, 0);
    expect_scalar("reduce(empty) == identity",
                  plaid::ChunkReduce<uint64_t>(empty, SumMonoid{}), 0);
    cleanup_prefix("iota");
    cleanup_prefix("edge_flt_none");
    cleanup_prefix("edge_scan_empty");
  }

  // tabulate with a non-iota function: f(i) = i*i (mod 2^64).
  {
    chunk_seq seq = plaid::tabulate<uint64_t>(
        n, "edge_tab", std::function<uint64_t(size_t)>([](size_t i) {
          return (uint64_t)i * i;
        }));
    std::vector<uint64_t> expected(n);
    for (size_t i = 0; i < n; i++) expected[i] = (uint64_t)i * i;
    expect_eq_vec<uint64_t>("tabulate i*i", seq, expected);
    cleanup_prefix("edge_tab");
  }
}

// ── multi-batch filter+scan composition (scalar-verified, low memory) ────────
// Spans > FILTER_BATCH_SIZE input chunks so the cross-batch ordering path runs,
// but we verify with closed-form scalars to avoid a multi-GiB reference vector.
static void run_multibatch() {
  const size_t chunks = 130;  // 2 filter batches (128 + 2)
  const size_t n = chunks * ELEMS_PER_CHUNK;
  std::cout << "  multi-batch  n=" << n << "  (" << chunks << " chunks)\n";

  chunk_seq seq = plaid::iota(n);
  chunk_seq filt = plaid::ChunkFilter<uint64_t>(
      seq, "mb_flt",
      std::function<bool(uint64_t)>([](uint64_t x) { return x % 2 == 0; }));

  // Survivors of iota(n) with x%2==0 are 0,2,…,n-2: count = n/2.
  const uint64_t cnt = n / 2;
  expect_scalar("filter count == n/2",
                plaid::ChunkReduce<uint64_t, uint64_t>(
                    plaid::ChunkMap<uint64_t>(filt, "mb_ones",
                                              std::function<uint64_t(uint64_t)>(
                                                  [](uint64_t) { return 1; })),
                    SumMonoid{}),
                cnt);
  cleanup_prefix("mb_ones");

  // sum of survivors = 0+2+…+2(cnt-1) = cnt*(cnt-1).
  expect_scalar("filter survivor sum",
                plaid::ChunkReduce<uint64_t>(filt, SumMonoid{}),
                cnt * (cnt - 1));

  // Exclusive scan total over the survivors == survivor sum.
  auto [sc, total] = plaid::ChunkScan<uint64_t>(filt, "mb_scan", SumMonoid{});
  expect_scalar("scan total == survivor sum", total, cnt * (cnt - 1));
  // Largest exclusive prefix is the sum of all but the last survivor.
  expect_scalar("scan max prefix",
                plaid::ChunkReduce<uint64_t>(sc, MaxMonoid{}),
                cnt * (cnt - 1) - 2 * (cnt - 1));

  cleanup_prefix("iota");
  cleanup_prefix("mb_flt");
  cleanup_prefix("mb_scan");
}

// ── main ─────────────────────────────────────────────────────────────────────

int run(int argc, char* argv[]) {
  std::cout << "Combined ChunkSequence test  (" << GetSSDList().size()
            << " drives)\n\n";

  // Size sweep: empty, single, tiny, partial chunk, exact boundary, just over,
  // and a few chunks.  Deep-equality verified against a serial reference.
  const std::vector<size_t> sizes = {
      0,
      1,
      7,
      ELEMS_PER_CHUNK - 1,
      ELEMS_PER_CHUNK,
      ELEMS_PER_CHUNK + 1,
      2 * ELEMS_PER_CHUNK + 3,
  };
  for (size_t n : sizes) run_size(n);

  run_edge_cases();
  run_multibatch();

  std::cout << "\n"
            << g_pass << " passed, " << g_fail << " failed.  "
            << (g_fail == 0 ? "ALL PASS" : "SOME FAILED") << "\n";
  return g_fail == 0 ? 0 : 1;
}

}  // namespace test_combined

// ============================================================================
// samplesort -- sample_sort correctness + drive striping
//
// (was ChunkSequence/tests/samplesort_striped_test.cpp)
// ============================================================================

namespace test_samplesort {

// Correctness + placement test for external_samplesort.h's sample_sort.
//
// sample_sort's bucketing step routes through group_by_index
// (Primitives/group_by.h), which shares its bounded BucketWriter substrate
// with count_sort -- one file per bucket (same as sample_sort_random/
// sample_sort_singledrive).  A bucket's file is placed via
// GetFileName(prefix, bucket_index) (bucket_index % num_drives), so with
// more buckets than drives -- true well before this test's default n -- the
// overall output spreads across every drive even though each individual
// bucket lives in a single file.
//
// This test checks:
//   1. the output is fully sorted and element-for-element identical to
//      parlay::sort on the same (distinct) keys.
//   2. on the placement side, that the output touches more than one distinct
//      drive when more than one drive is available -- the regression check a
//      future change that quietly collapsed the output back onto a single
//      drive would fail.
//
// Exits 0 iff all checks pass.

using plaid::sample_sort;

// Which drive a chunk's file lives on, by matching its GetFileName-derived
// path prefix against GetSSDList(); -1 if no drive matches (shouldn't happen).
static int drive_of(const std::string& filename) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    if (filename.rfind(ssds[d] + "/", 0) == 0) return (int)d;
  return -1;
}

// Distinct drives touched anywhere in `seq`'s output.
static size_t distinct_drives(const chunk_seq& seq) {
  std::set<int> drives;
  for (const chunk& c : seq.chunks) drives.insert(drive_of(c.filename));
  return drives.size();
}

static bool check_sorted(const std::string& name, chunk_seq& result,
                         const parlay::sequence<uint64_t>& ref) {
  std::vector<uint64_t> got = result.to_vector<uint64_t>();
  if (got.size() != ref.size()) {
    std::cout << "  FAIL " << name << ": got " << got.size()
              << " elements, expected " << ref.size() << "\n";
    return false;
  }
  for (size_t i = 0; i < got.size(); i++) {
    if (got[i] != ref[i]) {
      std::cout << "  FAIL " << name << ": element " << i << " is " << got[i]
                << ", expected " << ref[i] << "\n";
      return false;
    }
  }
  std::cout << "  OK " << name << ": " << got.size()
            << " elements, sorted correctly\n";
  return true;
}

int run(int argc, char* argv[]) {
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 50'000'000;

  auto key_at = [](size_t i) { return parlay::hash64(i); };
  parlay::sequence<uint64_t> ref = parlay::tabulate(n, key_at);
  parlay::sort_inplace(ref);

  const size_t num_drives = GetSSDList().size();
  std::cout << "n=" << n << " drives=" << num_drives << "\n";

  bool pass = true;
  const std::string prefix = "sst_prim_in";

  chunk_seq in = plaid::tabulate<uint64_t>(n, prefix, key_at);
  chunk_seq out = sample_sort<uint64_t>(in);

  if (!check_sorted("sample_sort", out, ref)) pass = false;

  const size_t drives = distinct_drives(out);
  std::cout << "  .. sample_sort: output touches " << drives
            << " distinct drives\n";
  if (num_drives > 1) {
    if (drives <= 1) {
      std::cout << "  FAIL sample_sort: output touched only " << drives
                << " drive(s), expected more than 1\n";
      pass = false;
    } else {
      std::cout << "  OK sample_sort: output spread across " << drives
                << " drives\n";
    }
  }

  // Sweep every file this sort touched.  sample_sort has no prefix parameter
  // of its own (it always uses "ss_bucket_"/"ss_base_" + a process-global
  // counter), so those two fixed prefixes are swept alongside the input's.
  bench_drives::clear_drives({prefix, "ss_bucket_", "ss_base_"});

  std::cout << (pass ? "PASS" : "FAIL") << "\n";
  return pass ? 0 : 1;
}

}  // namespace test_samplesort

// ============================================================================
// driver
// ============================================================================

int main(int argc, char* argv[]) {
  // Parsed ONCE here, not per case: it consumes the global flags from argv
  // and populates the SSD list, and a second call would reset that list to
  // the defaults, discarding any --ssd= selection.
  ParseGlobalArguments(argc, argv);

  static const std::vector<std::pair<const char*, int (*)(int, char**)>>
      kCases = {
          {"iota", &test_iota::run},
          {"map", &test_map::run},
          {"reduce", &test_reduce::run},
          {"scan", &test_scan::run},
          {"segmented_reduce", &test_segmented_reduce::run},
          {"find_if", &test_find_if::run},
          {"histogram", &test_histogram::run},
          {"scalar", &test_scalar::run},
          {"filter", &test_filter::run},
          {"flat_tabulate", &test_flat_tabulate::run},
          {"flat_map", &test_flat_map::run},
          {"partition", &test_partition::run},
          {"group_by", &test_group_by::run},
          {"reverse", &test_reverse::run},
          {"chunk_operation", &test_chunk_operation::run},
          {"combined", &test_combined::run},
          {"samplesort", &test_samplesort::run},
      };

  size_t failed = 0;
  for (const auto& c : kCases) {
    std::cout << "\n==================== " << c.first
              << " ====================\n";
    if (c.second(argc, argv) != 0) failed++;
  }

  std::cout << "\n"
            << (kCases.size() - failed) << "/" << kCases.size()
            << " cases passed.  "
            << (failed == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << "\n";
  return failed == 0 ? 0 : 1;
}
