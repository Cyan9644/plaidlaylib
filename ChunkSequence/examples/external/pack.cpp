// packExample — out-of-core pack benchmark/demo.
//
// Packs the elements of an n-element uint64_t input flagged true by a
// deterministic boolean sequence (i%3 != 0, ~2/3 survive) via plaid::pack
// (Primitives/pack.h, the DRAM-boolean-sequence overload, a DensePack
// producer) and compares against parlay::pack on the identical input and
// flags built in DRAM.
//
// Dual-purpose like the other examples: prints human-readable timings and
// ends with a machine-readable `CSV,` line that benchmarks/run_benches.py
// greps.  Both implementations preserve input order, so the cross-check is a
// direct element-for-element equality (after reading the out-of-core result
// back with plaid::materialize).  The in-memory baseline is gated by a RAM
// budget (input + flags + baseline output + readback = ~25n, rounded to /32
// for headroom), overridable via EXAMPLE_INMEM_BUDGET_BYTES.
//
//   usage: packExample [global --flags] [n]

#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/Primitives/materialize.h"
#include "ChunkSequence/Primitives/pack.h"
#include "absl/log/check.h"
#include "parlay/primitives.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "utils/io_backend.h"
#include "utils/trace_marker.h"

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

static uint64_t elem_at(size_t i) { return parlay::hash64(i); }
static bool flag_at(size_t i) { return (i % 3) != 0; }

int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
  CHECK(n > 0) << "need n > 0 (n=" << n << ")";

  // RAM budget: baseline input (8n) + flags (~1n, bool) + baseline output
  // (up to 8n) + out-of-core readback for the cross-check (up to 8n) =~ 25n;
  // rounded down to /32 for headroom.
  const size_t phys =
      (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
  size_t budget = phys / 2;
  if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES"))
    budget = std::stoull(e);
  const bool inmem_ok = n <= budget / 32;

  const std::string in_prefix = "pck_in";
  const std::string out_prefix = "pck_out";

  std::cout << "Building " << n << "-element input..." << std::flush;
  trace_mark("build_start");
  auto t0 = Clock::now();
  chunk_seq seq = plaid::tabulate<uint64_t>(n, in_prefix, elem_at);
  parlay::sequence<bool> flags = parlay::tabulate(n, flag_at);
  const double build_s = elapsed(t0);
  trace_mark("build_end");
  std::cout << " done (" << std::fixed << std::setprecision(4) << build_s
            << "s)\n";
  quiesce_drives();

  std::cout << "Packing " << n << " elements (keep i%3!=0)..." << std::flush;
  trace_mark("op_start");
  t0 = Clock::now();
  chunk_seq packed = plaid::pack<uint64_t>(seq, out_prefix, flags);
  const double pack_s = elapsed(t0);
  trace_mark("op_end");
  size_t out_elems = 0;
  for (const auto& c : packed.chunks) out_elems += c.used / sizeof(uint64_t);
  const double gb_s = to_gb(n * sizeof(uint64_t)) / pack_s;
  std::cout << " done   " << out_elems << " kept   " << std::setprecision(4)
            << pack_s << "s   " << std::setprecision(2) << gb_s
            << " GB/s (input read)\n";

  bool agree = true;
  double inmem_pack_s = 0;
  if (inmem_ok) {
    auto seq_mem = parlay::tabulate(n, elem_at);
    t0 = Clock::now();
    auto packed_mem = parlay::pack(seq_mem, flags);
    inmem_pack_s = elapsed(t0);
    std::cout << "in-mem parlay::pack: " << packed_mem.size() << " kept   "
              << std::setprecision(4) << inmem_pack_s << "s\n";

    if (out_elems != packed_mem.size()) {
      std::cout << "*** MISMATCH: out-of-core count " << out_elems
                << " != in-mem count " << packed_mem.size() << " ***\n";
      agree = false;
    } else {
      auto ours = plaid::materialize<uint64_t>(packed);
      for (size_t i = 0; i < ours.size() && agree; i++) {
        if (ours[i] != packed_mem[i]) {
          std::cout << "*** MISMATCH at index " << i << ": " << ours[i]
                    << " != " << packed_mem[i] << " ***\n";
          agree = false;
        }
      }
      if (agree)
        std::cout << "cross-check: out-of-core pack matches in-mem pack "
                     "exactly (order-preserving)\n";
    }
  } else {
    std::cout << "in-mem parlay::pack: skipped (~25n footprint exceeds "
              << "RAM budget " << std::setprecision(2) << to_gb(budget)
              << " GB)\n";
  }

  // Machine-readable line for benchmarks/run_benches.py (examples sweep).
  // Columns: n,build_s,pack_s,inmem_pack_s,out_elems,throughput_gb_s
  // (inmem_pack_s blank when the input exceeds the RAM budget).
  auto f9 = [](double v) {
    std::ostringstream o;
    o << std::setprecision(9) << v;
    return o.str();
  };
  std::cout << "CSV," << n << ',' << f9(build_s) << ',' << f9(pack_s) << ','
            << (inmem_ok ? f9(inmem_pack_s) : std::string()) << ','
            << out_elems << ',' << f9(gb_s) << '\n';

  cleanup_prefix(in_prefix);
  cleanup_prefix(out_prefix);
  return agree ? 0 : 1;
}
