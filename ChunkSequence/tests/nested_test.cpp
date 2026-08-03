#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

#include "ChunkSequence/ExternalPrimitives/materialize.h"
#include "ChunkSequence/nested_gather.h"
#include "ChunkSequence/nested_ops.h"
#include "ChunkSequence/nested_seq.h"
#include "absl/log/check.h"
#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"

using ChunkSequenceOps::nested_seq;

static void cleanup_prefix(const std::string& prefix) {
  const auto& ssds = GetSSDList();
  for (size_t d = 0; d < ssds.size(); d++)
    unlink(GetFileName(prefix, d).c_str());
}

struct SumMonoid {
  uint64_t identity = 0;
  uint64_t operator()(uint64_t a, uint64_t b) const { return a + b; }
};

// ── the generator under test (deterministic; stresses the packer) ────────────
// Lengths: mostly small, sprinkled empties, and occasional full-chunk sequences
// (each its own chunk) that exercise the packing boundary.  Values are a
// function of (seq index, position) so the golden reference is exact.
static size_t gen_len(size_t i) {
  if (i % 50 == 0) return 0;                      // empty inner sequences
  if (i > 0 && i % 4000 == 0) return ELEMS_PER_CHUNK;  // a full-chunk sequence
  return (i * 2654435761ULL) % 300 + 1;           // 1..300, pseudo-random
}
static parlay::sequence<uint64_t> gen_seq(size_t i) {
  const size_t l = gen_len(i);
  parlay::sequence<uint64_t> s(l);
  for (size_t k = 0; k < l; k++) s[k] = i * 1000003ULL + k;
  return s;
}

// ── golden reference held in DRAM ────────────────────────────────────────────
static std::vector<parlay::sequence<uint64_t>> golden_of(size_t n) {
  std::vector<parlay::sequence<uint64_t>> g(n);
  parlay::parallel_for(0, n, [&](size_t i) { g[i] = gen_seq(i); });
  return g;
}

static bool eq_seq(const parlay::sequence<uint64_t>& a,
                   const parlay::sequence<uint64_t>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); i++)
    if (a[i] != b[i]) return false;
  return true;
}

// ── test 1: construction invariants ──────────────────────────────────────────
static bool test_invariants(const nested_seq<>& ns,
                            const std::vector<parlay::sequence<uint64_t>>& g) {
  std::cout << "test_invariants\n" << std::flush;
  bool pass = true;
  const size_t n = g.size();
  constexpr size_t epc = CHUNK_SIZE / sizeof(uint64_t);

  if (ns.total_seqs() != n) {
    std::cout << "  FAIL total_seqs: got=" << ns.total_seqs() << " expected=" << n
              << "\n";
    pass = false;
  }

  // seq_len_scan matches golden lengths.
  for (size_t i = 0; i < n && pass; i++) {
    if (ns.len(i) != g[i].size()) {
      std::cout << "  FAIL len[" << i << "]: got=" << ns.len(i)
                << " expected=" << g[i].size() << "\n";
      pass = false;
    }
  }

  // Chunks: index-ordered, contiguous partition, each within one read.
  size_t expect_first = 0;
  for (size_t c = 0; c < ns.chunks.size(); c++) {
    const auto& nc = ns.chunks[c];
    if (nc.raw.index != c) {
      std::cout << "  FAIL index order: chunks[" << c << "].raw.index="
                << nc.raw.index << "\n";
      pass = false;
    }
    if (nc.first_seq != expect_first) {
      std::cout << "  FAIL partition: chunk " << c << " first_seq="
                << nc.first_seq << " expected=" << expect_first << "\n";
      pass = false;
    }
    if (nc.num_seqs == 0) {
      std::cout << "  FAIL empty chunk " << c << "\n";
      pass = false;
    }
    const size_t elems = ns.seq_len_scan[nc.first_seq + nc.num_seqs] -
                         ns.seq_len_scan[nc.first_seq];
    if (elems > epc) {
      std::cout << "  FAIL chunk " << c << " holds " << elems
                << " elems > epc=" << epc << " (straddles a read)\n";
      pass = false;
    }
    if (nc.raw.used != elems * sizeof(uint64_t)) {
      std::cout << "  FAIL chunk " << c << " used=" << nc.raw.used
                << " expected=" << elems * sizeof(uint64_t) << "\n";
      pass = false;
    }
    expect_first += nc.num_seqs;
  }
  if (!ns.chunks.empty() && expect_first != n) {
    std::cout << "  FAIL partition total: covered " << expect_first
              << " expected " << n << "\n";
    pass = false;
  }

  std::cout << "  chunks=" << ns.chunks.size() << "  => "
            << (pass ? "PASS" : "FAIL") << "\n\n";
  return pass;
}

// ── test 2: get(i) round-trip on sampled indices ─────────────────────────────
static bool test_get(const nested_seq<>& ns,
                     const std::vector<parlay::sequence<uint64_t>>& g) {
  std::cout << "test_get\n" << std::flush;
  bool pass = true;
  const size_t n = g.size();

  std::vector<size_t> probes = {0, 1, 50, 2, n / 2, n - 1};
  // find the first full-chunk sequence and an empty one to probe explicitly
  for (size_t i = 0; i < n; i++)
    if (gen_len(i) == ELEMS_PER_CHUNK) {
      probes.push_back(i);
      break;
    }
  for (size_t i = 0; i < n; i++)
    if (gen_len(i) == 0) {
      probes.push_back(i);
      break;
    }

  for (size_t i : probes) {
    if (i >= n) continue;
    parlay::sequence<uint64_t> got = ns.get(i);
    if (!eq_seq(got, g[i])) {
      std::cout << "  FAIL get(" << i << "): size got=" << got.size()
                << " expected=" << g[i].size() << "\n";
      pass = false;
    }
  }
  std::cout << "  => " << (pass ? "PASS" : "FAIL") << "\n\n";
  return pass;
}

// ── test 3: consolidate_flat == flat concatenation of golden ─────────────────
static bool test_consolidate_flat(
    const nested_seq<>& ns,
    const std::vector<parlay::sequence<uint64_t>>& g) {
  std::cout << "test_consolidate_flat\n" << std::flush;
  bool pass = true;

  std::vector<uint64_t> expected;
  for (const auto& s : g)
    for (uint64_t x : s) expected.push_back(x);

  const std::string path = "nested_flat.bin";
  ns.consolidate_flat(path);

  struct stat st;
  if (stat(path.c_str(), &st) != 0) {
    std::cout << "  FAIL stat\n";
    pass = false;
  } else if ((size_t)st.st_size != expected.size() * sizeof(uint64_t)) {
    std::cout << "  FAIL size: got=" << st.st_size
              << " expected=" << expected.size() * sizeof(uint64_t) << "\n";
    pass = false;
  }

  if (pass) {
    int fd = open(path.c_str(), O_RDONLY);
    CHECK(fd >= 0);
    std::vector<uint64_t> got(expected.size());
    ssize_t rd = read(fd, got.data(), expected.size() * sizeof(uint64_t));
    close(fd);
    if (rd != (ssize_t)(expected.size() * sizeof(uint64_t))) {
      std::cout << "  FAIL short read\n";
      pass = false;
    } else {
      for (size_t i = 0; i < expected.size(); i++)
        if (got[i] != expected[i]) {
          std::cout << "  FAIL element " << i << ": got=" << got[i]
                    << " expected=" << expected[i] << "\n";
          pass = false;
          break;
        }
    }
  }
  unlink(path.c_str());
  std::cout << "  flat elems=" << expected.size() << "  => "
            << (pass ? "PASS" : "FAIL") << "\n\n";
  return pass;
}

// ── test 4: NestedMap (structure-preserving x -> 3x+1) ───────────────────────
static bool test_map(const nested_seq<>& ns,
                     const std::vector<parlay::sequence<uint64_t>>& g) {
  std::cout << "test_map\n" << std::flush;
  bool pass = true;
  const std::string prefix = "nested_map";

  nested_seq<> mapped = ChunkSequenceOps::NestedMap<uint64_t, uint64_t>(
      ns, prefix, [](const uint64_t* d, size_t l) {
        parlay::sequence<uint64_t> r(l);
        for (size_t i = 0; i < l; i++) r[i] = 3 * d[i] + 1;
        return r;
      });

  // Structure must be identical to the input.
  if (mapped.seq_len_scan.size() != ns.seq_len_scan.size() ||
      !std::equal(mapped.seq_len_scan.begin(), mapped.seq_len_scan.end(),
                  ns.seq_len_scan.begin())) {
    std::cout << "  FAIL: seq_len_scan differs from input\n";
    pass = false;
  }
  if (mapped.chunks.size() != ns.chunks.size()) {
    std::cout << "  FAIL: chunk count differs\n";
    pass = false;
  } else {
    for (size_t c = 0; c < mapped.chunks.size(); c++)
      if (mapped.chunks[c].first_seq != ns.chunks[c].first_seq ||
          mapped.chunks[c].num_seqs != ns.chunks[c].num_seqs) {
        std::cout << "  FAIL: chunk " << c << " bounds differ\n";
        pass = false;
        break;
      }
  }

  // Values must match g mapped.
  auto got = mapped.to_nested_vector();
  if (got.size() != g.size()) {
    std::cout << "  FAIL: seq count " << got.size() << " != " << g.size()
              << "\n";
    pass = false;
  } else {
    for (size_t i = 0; i < g.size() && pass; i++) {
      parlay::sequence<uint64_t> exp(g[i].size());
      for (size_t k = 0; k < g[i].size(); k++) exp[k] = 3 * g[i][k] + 1;
      if (!eq_seq(got[i], exp)) {
        std::cout << "  FAIL mapped seq " << i << "\n";
        pass = false;
      }
    }
  }

  cleanup_prefix(prefix);
  std::cout << "  => " << (pass ? "PASS" : "FAIL") << "\n\n";
  return pass;
}

// ── test 5: NestedReduce (sum per inner sequence) ────────────────────────────
static bool test_reduce(const nested_seq<>& ns,
                        const std::vector<parlay::sequence<uint64_t>>& g) {
  std::cout << "test_reduce\n" << std::flush;
  bool pass = true;

  parlay::sequence<uint64_t> sums =
      ChunkSequenceOps::NestedReduce<uint64_t, uint64_t>(
          ns, [](uint64_t x) { return x; }, SumMonoid{});

  if (sums.size() != g.size()) {
    std::cout << "  FAIL: result size " << sums.size() << " != " << g.size()
              << "\n";
    pass = false;
  } else {
    for (size_t i = 0; i < g.size() && pass; i++) {
      uint64_t exp = 0;
      for (uint64_t x : g[i]) exp += x;
      if (sums[i] != exp) {
        std::cout << "  FAIL reduce seq " << i << ": got=" << sums[i]
                  << " expected=" << exp << "\n";
        pass = false;
      }
    }
  }
  std::cout << "  => " << (pass ? "PASS" : "FAIL") << "\n\n";
  return pass;
}

// ── test 6: NestedFlatten == flat concatenation of golden ────────────────────
static bool test_flatten(const nested_seq<>& ns,
                         const std::vector<parlay::sequence<uint64_t>>& g) {
  std::cout << "test_flatten\n" << std::flush;
  bool pass = true;
  const std::string prefix = "nested_flat_op";

  chunk_seq flat = ChunkSequenceOps::NestedFlatten<uint64_t>(ns, prefix);

  std::vector<uint64_t> expected;
  for (const auto& s : g)
    for (uint64_t x : s) expected.push_back(x);

  std::vector<uint64_t> got;
  if (!flat.chunks.empty()) {
    auto m = ChunkSequenceOps::materialize<uint64_t>(flat);
    got.assign(m.begin(), m.end());
  }
  if (got.size() != expected.size()) {
    std::cout << "  FAIL size: got=" << got.size()
              << " expected=" << expected.size() << "\n";
    pass = false;
  } else {
    for (size_t i = 0; i < expected.size(); i++)
      if (got[i] != expected[i]) {
        std::cout << "  FAIL element " << i << ": got=" << got[i]
                  << " expected=" << expected[i] << "\n";
        pass = false;
        break;
      }
  }
  cleanup_prefix(prefix);
  std::cout << "  flat elems=" << expected.size() << "  => "
            << (pass ? "PASS" : "FAIL") << "\n\n";
  return pass;
}

// ── test 7: NestedGather (both directions) selects the right rows ────────────
static bool test_gather(const nested_seq<>& ns,
                        const std::vector<parlay::sequence<uint64_t>>& g) {
  std::cout << "test_gather\n" << std::flush;
  bool pass = true;
  const size_t n = g.size();

  // A scattered subset of ids (includes an empty row and the full-chunk one).
  parlay::sequence<uint64_t> ids;
  for (size_t i = 0; i < n; i += 7) ids.push_back(i);
  for (size_t i = 0; i < n; i++)
    if (gen_len(i) == 0 || gen_len(i) == ELEMS_PER_CHUNK) ids.push_back(i);

  // Expected: rows for the distinct, id-sorted selection (gather's output order).
  std::vector<uint64_t> sel(ids.begin(), ids.end());
  std::sort(sel.begin(), sel.end());
  sel.erase(std::unique(sel.begin(), sel.end()), sel.end());

  for (ChunkSequenceOps::GatherDir dir :
       {ChunkSequenceOps::GatherDir::kPull, ChunkSequenceOps::GatherDir::kPush}) {
    const bool is_push = dir == ChunkSequenceOps::GatherDir::kPush;
    const std::string prefix = is_push ? "nested_gather_push" : "nested_gather_pull";
    nested_seq<> got_ns = ChunkSequenceOps::NestedGather<uint64_t>(ns, ids, prefix, dir);
    auto got = got_ns.to_nested_vector();

    bool dpass = true;
    if (got.size() != sel.size()) {
      std::cout << "  FAIL " << (is_push ? "push" : "pull") << " rows "
                << got.size() << " != " << sel.size() << "\n";
      dpass = false;
    } else {
      for (size_t k = 0; k < sel.size() && dpass; k++)
        if (!eq_seq(got[k], g[sel[k]])) {
          std::cout << "  FAIL " << (is_push ? "push" : "pull") << " row " << k
                    << " (vertex " << sel[k] << ")\n";
          dpass = false;
        }
    }
    std::cout << "  " << (is_push ? "push" : "pull") << ": "
              << (dpass ? "OK" : "FAIL") << "\n";
    pass &= dpass;
    cleanup_prefix(prefix);
  }
  std::cout << "  => " << (pass ? "PASS" : "FAIL") << "\n\n";
  return pass;
}

// ── test 8: NestedMapReduce (fused inner map + reduce) ───────────────────────
// Per inner sequence, map x -> x%100 then sum.  Checked against the DRAM golden
// AND against the UNFUSED spelling NestedReduce(NestedMap(square-free map)),
// which materializes an intermediate nested_seq — same answer, extra I/O.
static bool test_map_reduce(const nested_seq<>& ns,
                            const std::vector<parlay::sequence<uint64_t>>& g) {
  std::cout << "test_map_reduce\n" << std::flush;
  bool pass = true;
  auto mapf = [](uint64_t x) { return x % 100; };

  parlay::sequence<uint64_t> fused =
      ChunkSequenceOps::NestedMapReduce<uint64_t, uint64_t>(ns, mapf, SumMonoid{});

  // golden
  if (fused.size() != g.size()) {
    std::cout << "  FAIL size " << fused.size() << " != " << g.size() << "\n";
    pass = false;
  } else {
    for (size_t i = 0; i < g.size() && pass; i++) {
      uint64_t exp = 0;
      for (uint64_t x : g[i]) exp += x % 100;
      if (fused[i] != exp) {
        std::cout << "  FAIL seq " << i << ": got=" << fused[i]
                  << " expected=" << exp << "\n";
        pass = false;
      }
    }
  }
  if (pass) std::cout << "  vs golden OK\n";

  // Unfused: materialize the mapped nested_seq, then reduce it.  Must agree.
  const std::string mp = "nmr_mapped";
  nested_seq<> mapped = ChunkSequenceOps::NestedMap<uint64_t, uint64_t>(
      ns, mp, [](const uint64_t* d, size_t l) {
        parlay::sequence<uint64_t> r(l);
        for (size_t k = 0; k < l; k++) r[k] = d[k] % 100;
        return r;
      });
  parlay::sequence<uint64_t> unfused =
      ChunkSequenceOps::NestedReduce<uint64_t, uint64_t>(
          mapped, [](uint64_t x) { return x; }, SumMonoid{});
  for (size_t i = 0; i < fused.size() && pass; i++)
    if (fused[i] != unfused[i]) {
      std::cout << "  FAIL fused vs unfused seq " << i << ": " << fused[i]
                << " != " << unfused[i] << "\n";
      pass = false;
    }
  if (pass) std::cout << "  fused == unfused OK\n";
  cleanup_prefix(mp);

  std::cout << "  => " << (pass ? "PASS" : "FAIL") << "\n\n";
  return pass;
}

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);

  // Enough inner sequences for multiple chunks AND multiple pack batches.
  size_t n = 20000;
  if (argc > 1) n = std::stoull(argv[1]);

  std::cout << "nested_seq test  n=" << n
            << "  ELEMS_PER_CHUNK=" << ELEMS_PER_CHUNK << "\n\n";

  const std::string prefix = "nested_src";
  auto g = golden_of(n);
  nested_seq<> ns =
      ChunkSequenceOps::NestedTabulate<uint64_t>(n, prefix, gen_seq);

  bool all_pass = true;
  all_pass &= test_invariants(ns, g);
  all_pass &= test_get(ns, g);
  all_pass &= test_consolidate_flat(ns, g);
  all_pass &= test_map(ns, g);
  all_pass &= test_reduce(ns, g);
  all_pass &= test_flatten(ns, g);
  all_pass &= test_gather(ns, g);
  all_pass &= test_map_reduce(ns, g);

  cleanup_prefix(prefix);

  std::cout << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
  return all_pass ? 0 : 1;
}
