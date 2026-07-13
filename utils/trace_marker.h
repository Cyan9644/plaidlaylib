#ifndef TRACE_MARKER_H
#define TRACE_MARKER_H

#include <cstdio>
#include <cstdlib>
#include <ctime>

/**
 * Phase-boundary markers for the io_trace.py profiler.
 *
 * trace_mark("op_start") prints one line
 *
 *     TRACE,<label>,<seconds>
 *
 * to stdout, where <seconds> is CLOCK_MONOTONIC at the call — the *same* clock
 * Python's time.monotonic() samples in benchmarks/io_trace.py, so the marker
 * lands directly on the trace's sample timeline (both run on one machine).
 *
 * Markers are emitted only when the environment variable PLAID_TRACE is set
 * (to any value); otherwise trace_mark is a no-op.  Normal runs and the
 * `make bench-examples` sweep are therefore byte-for-byte unchanged, and the
 * `CSV,` line the sweep greps is never affected.
 *
 * Header-only so examples pick it up with no Makefile/object changes.
 */
inline bool trace_enabled() {
    // Cached once: getenv is cheap but this keeps repeated marks free.
    static const bool on = (std::getenv("PLAID_TRACE") != nullptr);
    return on;
}

inline void trace_mark(const char* label) {
    if (!trace_enabled()) return;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    double secs = (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
    // Leading '\n': the examples print progress with std::flush and no trailing
    // newline (e.g. "Sieving..."), so without this the marker would glue onto
    // that line and no longer start at column 0 for the profiler's `^TRACE,`
    // parse.  Trailing '\n' + flush keeps it a clean standalone line on stdout.
    std::printf("\nTRACE,%s,%.9f\n", label, secs);
    std::fflush(stdout);
}

#endif  // TRACE_MARKER_H
