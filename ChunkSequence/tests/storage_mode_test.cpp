#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <system_error>
#include <iostream>
#include <string>
#include <vector>

#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/Primitives/delayed.h"
#include "ChunkSequence/Primitives/primitives.h"
#include "ChunkSequence/Primitives/secondary_primitives.h"
#include "ChunkSequence/Primitives/sort.h"
#include "utils/file_utils.h"
#include "utils/vio.h"

// Differential test: run each primitive twice over identical input -- once with
// its chunks on the drives, once with them in DRAM -- and require the two to
// agree.
//
// The existing suites already prove each primitive computes the right answer;
// what this adds is that the *storage backend* is not observable.  That claim
// is the whole point of the memory mode: a benchmark comparing the two is only
// meaningful if the work either mode performs is the same work.  So besides
// comparing elements, it compares chunk structure -- chunk count and the `used`
// bytes per index.  If those ever diverge, the two modes are no longer running
// comparable amounts of I/O and the attribution the benchmark reports is void.
//
// Filenames and begin_addr are deliberately NOT compared: drive assignment is
// randomized per run (mt19937_64 seeded from random_device), so they differ
// between two runs of the *same* mode.  Nor are raw file bytes -- padding past
// a chunk's `used` is genuinely allowed to differ (process_inplace writes whole
// CHUNK_SIZE runs while reading only AlignUp(used)).

namespace {

namespace d = plaid::delayed;
using plaid::storage;

int failures = 0;
size_t N = 400'000;

void report(const std::string& what, bool ok) {
  std::cout << (ok ? "  OK   " : "  FAIL ") << what << "\n";
  if (!ok) failures++;
}

// Element-wise plus structural agreement between the two modes.
template <typename T>
bool agree(const std::string& what, const chunk_seq& a, const chunk_seq& b,
           bool order_matters = true) {
  bool ok = true;
  if (a.chunks.size() != b.chunks.size()) {
    std::cout << "    chunk count " << a.chunks.size() << " vs "
              << b.chunks.size() << "\n";
    ok = false;
  } else {
    for (size_t i = 0; i < a.chunks.size(); i++) {
      if (a.chunks[i].used != b.chunks[i].used) {
        std::cout << "    chunk " << i << " used " << a.chunks[i].used << " vs "
                  << b.chunks[i].used << "\n";
        ok = false;
        break;
      }
      if (a.chunks[i].index != i || b.chunks[i].index != i) {
        std::cout << "    chunk " << i << " index invariant broken\n";
        ok = false;
        break;
      }
    }
  }
  auto va = a.to_vector<T>();
  auto vb = b.to_vector<T>();
  if (va.size() != vb.size()) {
    std::cout << "    length " << va.size() << " vs " << vb.size() << "\n";
    ok = false;
  } else {
    // A few primitives (ChunkPartition, the bucket sorts) emit in completion
    // order, which is not reproducible; for those only the multiset matters.
    if (!order_matters) {
      std::sort(va.begin(), va.end());
      std::sort(vb.begin(), vb.end());
    }
    for (size_t i = 0; i < va.size(); i++)
      if (!(va[i] == vb[i])) {
        std::cout << "    element " << i << " differs\n";
        ok = false;
        break;
      }
  }
  report(what, ok);
  return ok;
}

void cleanup(const std::string& prefix) {
  for (size_t i = 0; i < GetSSDList().size(); i++) {
    vio::unlink(GetFileName(prefix, i, storage::disk).c_str());
    vio::unlink(GetFileName(prefix, i, storage::memory).c_str());
  }
}

// Sweep everything whose name carries `prefix`, however many files it turned
// out to be.  Needed for the sort intermediates, whose names embed a run tag
// and whose count depends on the bucket fanout rather than the drive count --
// the library never unlinks them itself, which on disk merely leaves files on
// the drives but in DRAM would accumulate until the cap trips.
void sweep(const std::string& prefix) {
  plaid::vio::release_prefix(prefix);
  for (const std::string& dir : GetSSDList()) {
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
      const std::string name = e.path().filename().string();
      if (name.rfind(prefix, 0) == 0) std::filesystem::remove(e.path(), ec);
    }
  }
}

struct SumMonoid {
  uint64_t identity = 0;
  uint64_t operator()(uint64_t a, uint64_t b) const { return a + b; }
};

// Build the same logical input in a given mode.
chunk_seq make_input(storage st, const std::string& prefix) {
  return plaid::tabulate<uint64_t>(
      N, prefix, [](size_t i) { return (uint64_t)((i * 2654435761u) % 1000003); },
      /*io_threads=*/0, st);
}

// Run `body` in both modes over a freshly built input and compare the results.
template <typename Body>
void both(const std::string& what, Body body, bool order_matters = true) {
  chunk_seq in_d = make_input(storage::disk, "sm_in_d");
  chunk_seq in_m = make_input(storage::memory, "sm_in_m");
  chunk_seq out_d = body(in_d, std::string("sm_out_d"));
  chunk_seq out_m = body(in_m, std::string("sm_out_m"));
  if (out_m.mode() != storage::memory && !out_m.chunks.empty()) {
    std::cout << "    output did not inherit memory mode\n";
    failures++;
  }
  agree<uint64_t>(what, out_d, out_m, order_matters);
  cleanup("sm_in_d");
  cleanup("sm_in_m");
  cleanup("sm_out_d");
  cleanup("sm_out_m");
}

}  // namespace

int main(int argc, char** argv) {
  InitLogger();
  ParseGlobalArguments(argc, argv);
  if (argc > 1) N = std::stoull(argv[1]);
  std::cout << "==================== storage mode differential  n=" << N
            << " ====================\n";

  // The input itself: does tabulate lay out identically in both backends?
  {
    chunk_seq a = make_input(storage::disk, "sm_in_d");
    chunk_seq b = make_input(storage::memory, "sm_in_m");
    report("input is memory-backed", b.mode() == storage::memory);
    report("input is disk-backed", a.mode() == storage::disk);
    agree<uint64_t>("tabulate", a, b);
    cleanup("sm_in_d");
    cleanup("sm_in_m");
  }

  both("ChunkMap", [](const chunk_seq& s, const std::string& p) {
    return plaid::ChunkMap<uint64_t, uint64_t>(s, p,
                                               [](uint64_t x) { return x * 3 + 1; });
  });

  both("ChunkFilter", [](const chunk_seq& s, const std::string& p) {
    return plaid::ChunkFilter<uint64_t>(s, p,
                                        [](uint64_t x) { return (x & 1) == 0; });
  });

  both("ChunkScan", [](const chunk_seq& s, const std::string& p) {
    return plaid::ChunkScan<uint64_t, uint64_t>(s, p, SumMonoid{}).first;
  });

  both("pack_value", [](const chunk_seq& s, const std::string& p) {
    return plaid::pack_value<uint64_t>(s, p,
                                       [](uint64_t x) { return x % 3 == 0; });
  });

  // ChunkReduce returns a scalar, so compare it directly.
  {
    chunk_seq a = make_input(storage::disk, "sm_in_d");
    chunk_seq b = make_input(storage::memory, "sm_in_m");
    report("ChunkReduce",
           plaid::ChunkReduce<uint64_t>(a, SumMonoid{}) ==
               plaid::ChunkReduce<uint64_t>(b, SumMonoid{}));
    cleanup("sm_in_d");
    cleanup("sm_in_m");
  }

  // ChunkFlatTabulate generates rather than reads, so its mode is explicit.
  {
    auto gen = [](size_t start, size_t end) {
      parlay::sequence<uint64_t> out;
      for (size_t i = start; i < end; i++)
        if (i % 3 == 0) out.push_back((uint64_t)i);
      return out;
    };
    chunk_seq a = plaid::ChunkFlatTabulate<uint64_t>(N, "sm_ft_d", gen,
                                                     storage::disk);
    chunk_seq b = plaid::ChunkFlatTabulate<uint64_t>(N, "sm_ft_m", gen,
                                                     storage::memory);
    report("ChunkFlatTabulate inherits explicit mode",
           b.mode() == storage::memory);
    agree<uint64_t>("ChunkFlatTabulate", a, b);
    cleanup("sm_ft_d");
    cleanup("sm_ft_m");
  }

  // The delayed layer: force's output should land where its leaves live.
  both("delayed force(map)", [](const chunk_seq& s, const std::string& p) {
    auto chain = d::map(d::delay<uint64_t>(s), [](uint64_t x) { return x ^ 5; });
    return d::force(chain, p);
  });

  // ChunkPartition returns k sequences and emits in completion order.
  {
    chunk_seq in_d = make_input(storage::disk, "sm_in_d");
    chunk_seq in_m = make_input(storage::memory, "sm_in_m");
    auto key = [](uint64_t x) { return (size_t)(x % 4); };
    auto pd = plaid::ChunkPartition<uint64_t>(in_d, 4, "sm_part_d", key);
    auto pm = plaid::ChunkPartition<uint64_t>(in_m, 4, "sm_part_m", key);
    bool ok = pd.size() == pm.size();
    for (size_t b = 0; ok && b < pd.size(); b++)
      ok &= agree<uint64_t>("ChunkPartition bucket " + std::to_string(b), pd[b],
                            pm[b], /*order_matters=*/false);
    cleanup("sm_in_d");
    cleanup("sm_in_m");
    cleanup("sm_part_d");
    cleanup("sm_part_m");
  }

  // sample_sort drives BucketWriter, count_sort and process_inplace -- the
  // deepest path, and the one with run coalescing and two rings per worker.
  {
    chunk_seq in_d = make_input(storage::disk, "sm_in_d");
    chunk_seq in_m = make_input(storage::memory, "sm_in_m");
    chunk_seq sd = plaid::sample_sort<uint64_t>(in_d);
    chunk_seq sm = plaid::sample_sort<uint64_t>(in_m);
    report("sample_sort output is memory-backed", sm.mode() == storage::memory);
    agree<uint64_t>("sample_sort", sd, sm);
    auto v = sm.to_vector<uint64_t>();
    report("sample_sort actually sorted", std::is_sorted(v.begin(), v.end()));
    cleanup("sm_in_d");
    cleanup("sm_in_m");
    sweep("ss_");
  }

  std::cout << "resident after all cleanup: " << plaid::vio::resident_bytes()
            << " bytes\n";
  report("no memory-backed storage leaked", plaid::vio::resident_bytes() == 0);

  std::cout << (failures == 0 ? "ALL PASS" : "SOME FAILED") << "\n";
  return failures == 0 ? 0 : 1;
}
