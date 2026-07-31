// even_squaresExample — the five `external_even_squares.h` implementations of
// "sum of squares of the even elements" head-to-head on the same input:
//
//   sum_of_even_squares_eager   out-of-core, ChunkFilter (materializes
//                                survivors to disk under "even_squares_tmp")
//                                then a fused delayed map+reduce.
//   sum_of_even_squares_delay   out-of-core, fully fused delayed chain
//                                (lazy_filter -> map -> reduce): one read
//                                pass, zero writes.
//   sum_of_even_squares_parlay_eager    in-memory parlay::filter+map+reduce.
//   sum_of_even_squares_parlay_delayed  in-memory parlay::delayed
//                                        filter+map+reduce (parlay::delayed::filter
//                                        + map, two delayed stages fused at reduce).
//   sum_of_even_squares_parlay_actually_delayed  in-memory parlay::delayed
//                                        filter_op: filter and square fused
//                                        into a single predicate returning
//                                        std::optional<T>, one delayed stage.
//
// All five run over the same values (value_at(i), masked to 16 bits so the
// summed result can't overflow size_t at any example-scale n) and must agree
// exactly -- this is an integer sum, so the cross-check is exact equality,
// not a tolerance compare. Exits non-zero on any mismatch.
//
// The two out-of-core methods share one input build (build_s); the eager
// method is the only one of the five that writes to disk, so its
// "even_squares_tmp" output is cleared (and the drives settled) before the
// delayed method's timed region, matching the fairness discipline
// bench_drives.h documents for samplesort_three_way. The three in-memory
// baselines only run below a DRAM budget (EXAMPLE_INMEM_BUDGET_BYTES,
// default half of physical RAM; ~24n bytes for a parlay::sequence<uint64_t>
// plus the eager path's filter/map intermediates -- same cliff convention as
// samplesort's in-mem series), past which their CSV fields are left blank.
//
//   usage: even_squaresExample [global --flags] [n]
//     n   element count (default 1e6)
//
// CSV line:
//   CSV,n,build_s,eager_op_s,delay_op_s,inmem_eager_op_s,inmem_delay_op_s,inmem_actually_delay_op_s,result,eager_gb_s,delay_gb_s
//   throughput = input bytes (n * 8) / op_s; inmem_*_op_s blank past the
//   DRAM budget.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

#include "absl/log/check.h"

#include "parlay/primitives.h"
#include "parlay/sequence.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "utils/trace_marker.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/examples/external/bench_drives.h"
#include "ChunkSequence/examples/external/external_even_squares.h"

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}
static double to_gb(size_t bytes) { return (double)bytes / (1024.0 * 1024.0 * 1024.0); }

// 16-bit range: per-element squares stay <= ~2^32, so the summed result
// can't silently overflow size_t at any example-scale n.
static uint64_t value_at(size_t i) { return parlay::hash64(i) & 0xFFFFull; }

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    RaiseFdLimit();
    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;

    // DRAM budget for the in-memory baselines: a parlay::sequence<uint64_t>
    // (8n bytes) plus the eager in-mem path's filter()+map() intermediates --
    // ~24n total, the same cliff convention documented for samplesort's
    // in-mem series.
    const size_t phys = (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
    size_t budget = phys / 2;
    if (const char* e = getenv("EXAMPLE_INMEM_BUDGET_BYTES")) budget = std::stoull(e);
    const bool inmem_ok = n * 24 <= budget;

    std::cout << "Building " << n << "-element out-of-core input..." << std::flush;
    const std::string in_prefix = "es_in";
    trace_mark("build_start");
    auto t0 = Clock::now();
    chunk_seq input = ChunkSequenceOps::tabulate<uint64_t>(n, in_prefix, value_at);
    const double build_s = elapsed(t0);
    trace_mark("build_end");
    std::cout << " done (" << std::fixed << std::setprecision(4) << build_s << "s)\n";

    bench_drives::settle_drives();   // isolate the first timed op from the build's writeback

    std::cout << "Running out-of-core eager (ChunkFilter)..." << std::flush;
    trace_mark("eager_op_start");
    t0 = Clock::now();
    const size_t eager_result = ChunkSequenceOps::sum_of_even_squares_eager<uint64_t>(input);
    const double eager_op_s = elapsed(t0);
    trace_mark("eager_op_end");
    const double eager_gb_s = to_gb(n * sizeof(uint64_t)) / eager_op_s;
    std::cout << " done (" << std::setprecision(4) << eager_op_s << "s, "
              << std::setprecision(2) << eager_gb_s << " GB/s)\n";

    // The eager method is the only one of the four that writes to disk
    // (ChunkFilter's survivors); clear its output and settle before timing
    // the delayed method, so the delayed run isn't paying for the eager
    // run's still-in-flight file removal (see bench_drives.h).
    bench_drives::clear_drives({"even_squares_tmp"});

    std::cout << "Running out-of-core delayed (fused)..." << std::flush;
    trace_mark("delay_op_start");
    t0 = Clock::now();
    const size_t delay_result = ChunkSequenceOps::sum_of_even_squares_delay<uint64_t>(input);
    const double delay_op_s = elapsed(t0);
    trace_mark("delay_op_end");
    const double delay_gb_s = to_gb(n * sizeof(uint64_t)) / delay_op_s;
    std::cout << " done (" << std::setprecision(4) << delay_op_s << "s, "
              << std::setprecision(2) << delay_gb_s << " GB/s)\n";

    size_t inmem_eager_result = 0, inmem_delay_result = 0, inmem_actually_delay_result = 0;
    double inmem_eager_op_s = 0, inmem_delay_op_s = 0, inmem_actually_delay_op_s = 0;
    if (inmem_ok) {
        std::cout << "Building " << n << "-element in-memory input..." << std::flush;
        parlay::sequence<uint64_t> mem_input = parlay::tabulate(n, value_at);
        std::cout << " done\n";

        std::cout << "Running in-memory eager..." << std::flush;
        t0 = Clock::now();
        inmem_eager_result = ChunkSequenceOps::sum_of_even_squares_parlay_eager<uint64_t>(mem_input);
        inmem_eager_op_s = elapsed(t0);
        std::cout << " done (" << std::setprecision(4) << inmem_eager_op_s << "s)\n";

        std::cout << "Running in-memory delayed..." << std::flush;
        t0 = Clock::now();
        inmem_delay_result = ChunkSequenceOps::sum_of_even_squares_parlay_delayed<uint64_t>(mem_input);
        inmem_delay_op_s = elapsed(t0);
        std::cout << " done (" << std::setprecision(4) << inmem_delay_op_s << "s)\n";

        std::cout << "Running in-memory actually-delayed (filter_op)..." << std::flush;
        t0 = Clock::now();
        inmem_actually_delay_result = ChunkSequenceOps::sum_of_even_squares_parlay_actually_delayed<uint64_t>(mem_input);
        inmem_actually_delay_op_s = elapsed(t0);
        std::cout << " done (" << std::setprecision(4) << inmem_actually_delay_op_s << "s)\n";
    } else {
        std::cout << "in-memory baselines: skipped (n*24=" << to_gb(n * 24)
                  << " GB exceeds budget " << to_gb(budget) << " GB)\n";
    }

    // Cross-check: an exact integer sum, so exact equality is the right
    // comparison (no tolerance). eager vs delay always; either against the
    // in-memory results when they ran.
    bool agree = true;
    auto check = [&](const std::string& name, size_t got, size_t expected) {
        if (got != expected) {
            std::cout << "*** MISMATCH (" << name << "): got " << got
                      << ", expected " << expected << " ***\n";
            agree = false;
        }
    };
    check("delay vs eager", delay_result, eager_result);
    if (inmem_ok) {
        check("in-mem eager vs eager", inmem_eager_result, eager_result);
        check("in-mem delayed vs eager", inmem_delay_result, eager_result);
        check("in-mem actually-delayed vs eager", inmem_actually_delay_result, eager_result);
    }

    std::cout << "sum of even squares = " << eager_result
              << (agree ? "   all methods agree\n" : "   METHODS DISAGREE\n");

    auto f9 = [](double v) { std::ostringstream o; o << std::setprecision(9) << v; return o.str(); };
    std::cout << "CSV," << n << ',' << f9(build_s) << ',' << f9(eager_op_s)
              << ',' << f9(delay_op_s)
              << ',' << (inmem_ok ? f9(inmem_eager_op_s) : std::string())
              << ',' << (inmem_ok ? f9(inmem_delay_op_s) : std::string())
              << ',' << (inmem_ok ? f9(inmem_actually_delay_op_s) : std::string())
              << ',' << eager_result
              << ',' << f9(eager_gb_s) << ',' << f9(delay_gb_s) << '\n';

    bench_drives::clear_drives({in_prefix});

    return agree ? 0 : 1;
}
