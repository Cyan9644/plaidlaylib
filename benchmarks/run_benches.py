#!/usr/bin/env python3
"""Run the chunk-sequence benchmarks and plot them — single-file driver.

Replaces the predecessor project's two shell drivers (bench_delayed_scale.sh,
bench_chunk_size.sh) with one script that both *runs* the sweeps and *plots*
them.  It shells out to `make` to build the C++ binaries (all compilation stays
in the Makefile), runs them across a parameter sweep, parses the `CSV,` line
each binary prints, and writes PNGs + raw CSVs into a timestamped run directory.

Two benchmarks:

  * delayed scale  (benchmarks/delayed_compare.cpp -> bin/delayedCompare)
      Fixed chunk size, sweep n.  Compares in-mem delayed / chunk-eager /
      chunk-delayed for map|reduce, map|map|reduce, and force(map|map).

  * chunk size     (benchmarks/chunk_size_compare.cpp -> bin/chunkSizeCompare_<bytes>)
      Fixed n, sweep CHUNK_SIZE (one binary compiled per size via
      -DCHUNK_SIZE_BYTES).  Compares chunk-eager vs chunk-delayed.

The written CSVs use the same descriptive column names / units as the
predecessor's standalone plotters, and the plots follow the same two-panel
(read-bound map|map|reduce vs write-bound force(map|map)) base-2 log-log style.

To keep data off the drives, the script deletes the benchmarks' data files
(`iota<drive>` inputs + `bw_*` intermediates) between every sweep point and after
the run, so nothing accumulates (the C++ binaries clean their own intermediates,
but delayed_compare leaves its `iota` input behind).  It also best-effort
`fstrim`s the mounts once at startup (fstrim can be slow on real SSDs, so it is
not repeated between points; a no-op on a tmpfs dev box, where FITRIM is
unsupported — skipped quietly).  `--no-clean` / `--no-fstrim` disable each.

Every run carries a cross-substrate correctness check (agree=1).  If any point
reports agree=0 or a binary exits non-zero, this script prints the offending
output and exits non-zero — so `make bench` doubles as a differential test.
The examples sweep is softer: each example binary verifies its result against
the in-memory parlaylib baseline itself (when it fits in RAM) and a problem
(mismatch, crash) is warned about — immediately and again at the end of the
run — but does not stop the sweep; a point that produced no CSV is dropped.

Defaults are sized for a ~5 GiB tmpfs dev box; override via flags or env.
"""

import argparse
import glob
import math
import os
import re
import signal
import subprocess
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINDIR = os.path.join(REPO_ROOT, "bin")

# Defaults tuned for a ~5 GiB tmpfs (peak disk use in a step is ~3*n*8 bytes).
DEFAULT_N_VALUES = "1M 2M 4M 8M 16M 32M 64M"          # element counts
DEFAULT_CHUNK_SIZES = "512KiB 1MiB 2MiB 4MiB 8MiB 16MiB"
DEFAULT_CHUNK_N = "32M"                                # fixed n for chunk sweep
DEFAULT_EXAMPLE_SIZES = "128MiB 256MiB 512MiB 1GiB"    # examples sweep (dev/tmpfs); input size, not element count
DEFAULT_FSTRIM_GLOB = "/mnt/ssd*"

def _configs_chunk_bytes():
    """The examples' compiled CHUNK_SIZE, read from configs.h so it can't go stale.

    The examples don't pass -DCHUNK_SIZE_BYTES, so they use the `#define
    CHUNK_SIZE_BYTES` default in configs.h; parse and evaluate that arithmetic
    expression (e.g. `(1024 * 1024 * 4)` or `(4 << 20)`).  Falls back to 4 MiB if
    the define can't be found/parsed.
    """
    path = os.path.join(REPO_ROOT, "configs.h")
    try:
        with open(path) as f:
            text = f.read()
        # last matching #define wins (matches the C preprocessor)
        expr = None
        for m in re.finditer(r"#define\s+CHUNK_SIZE_BYTES\s+(.+)", text):
            expr = m.group(1).split("//")[0].split("/*")[0].strip()
        if expr and re.fullmatch(r"[0-9()*+\-<>x a-fA-F\t]+", expr):
            return int(eval(expr, {"__builtins__": {}}, {}))   # digits/operators only
    except (OSError, ValueError, SyntaxError):
        pass
    print("  !!! could not read CHUNK_SIZE_BYTES from configs.h; assuming 4 MiB",
          flush=True)
    return 4 << 20


# Compiled CHUNK_SIZE the examples are built with (from configs.h); the
# size->count conversion aligns n to this grid.
EXAMPLE_CHUNK_BYTES = _configs_chunk_bytes()


def _bellman_ford_dense_n(size_bytes):
    """n_from_size for bellman_ford_dense: invert size_bytes = 16*n^2.

    bellman_ford.cpp's dense case uses avg_degree = n/2, so edge count
    m ~= n^2/2 and edge bytes = m * sizeof(weighted_edge) = m*32 = 16*n^2 --
    quadratic in n, unlike every other example (see size_to_n). Rounds m
    itself down to a whole number of the real 32-byte-element chunk grid
    (at least one chunk) before inverting back to n, so sweep points still
    land on a chunk boundary the way every other example's do -- just
    computed in m-space instead of n-space, since n isn't the on-disk
    element count here.
    """
    edge_epc = EXAMPLE_CHUNK_BYTES // 32   # elements per chunk for a 32-byte weighted_edge
    m = max(edge_epc, (size_bytes // 32) // edge_epc * edge_epc)
    return int((2 * m) ** 0.5)


# bfs.cpp's dense case is the same avg_degree = n/2 shape as bellman_ford's
# (same chunk_csr / weighted_edge build via external_rmat_symmetric_graph),
# so the same size <-> n inversion applies verbatim.
_bfs_dense_n = _bellman_ford_dense_n

# ── examples registry ───────────────────────────────────────────────────────
# Each example is a dual-purpose binary (bin/<name>Example) that prints a
# `CSV,<cols...>` line the sweep greps.  `cols` names those fields in order;
# `time_col`/`inmem_col` pick the plotted out-of-core / in-memory series;
# `data_globs` lists the per-mount globs of files the example leaves on the
# drives (cleared between sweep points).  Add a new example by appending one
# entry (and, if it lives in examples/external/, an explicit Makefile rule).
#
# The sweep is parameterized by *input size in bytes*, not the binary's element
# count: `elem_bytes` is the primary on-disk sequence's element size and
# `input_seqs` how many such input sequences the example reads, so its total
# input footprint is `elem_bytes*input_seqs` bytes per element of `n`.  A target
# size S is converted to the binary's argv[1] via size_to_n() (chunk-aligned),
# so e.g. bigint_add (two 8-byte-limb operands) gets n = S/16 — half a single
# 8-byte sequence's n, since its input is split across two operands.
EXAMPLES = [
    {"name": "primes", "target": "bin/primesExample",
     "cols": ["n", "time_s", "inmem_time_s", "count", "throughput_gb_s"],
     "inmem_col": "inmem_time_s",
     "elem_bytes": 1, "input_seqs": 1,
     "xlabel": "input size",
     "title": "Prime sieve: out-of-core (ChunkFlatTabulate) vs in-mem parlaylib",
     "data_globs": ["primes[0-9]*"]},
    # kmpExample sweeps n with the pattern length m at its constant built-in
    # default; the plotted time is the search pass only (text build excluded).
    {"name": "kmp", "target": "bin/kmpExample",
     "cols": ["n", "m", "build_s", "search_s", "inmem_search_s", "count",
              "throughput_gb_s"],
     "time_col": "search_s", "inmem_col": "inmem_search_s",
     "elem_bytes": 1, "input_seqs": 1,
     "xlabel": "input size",
     "title": "KMP search: out-of-core (ChunkKmp) vs in-mem parlaylib",
     "data_globs": ["kmp_*"]},
    # rabin_karpExample: same driver shape as kmp (constant m, sweep n),
    # rolling-hash search instead of the KMP automaton.
    {"name": "rabin_karp", "target": "bin/rabin_karpExample",
     "cols": ["n", "m", "build_s", "search_s", "inmem_search_s", "count",
              "throughput_gb_s"],
     "time_col": "search_s", "inmem_col": "inmem_search_s",
     "elem_bytes": 1, "input_seqs": 1,
     "xlabel": "input size",
     "title": "Rabin-Karp search: out-of-core (ChunkRabinKarp) vs in-mem parlaylib",
     "data_globs": ["rk_*"]},
    # convex_hullExample sweeps n (32-byte points); the plotted time is the hull
    # pass only (point-cloud build excluded).  Its recursion leaves ch_scratch*
    # split intermediates in addition to the ch_in input.
    {"name": "convex_hull", "target": "bin/convex_hullExample",
     "cols": ["n", "build_s", "hull_s", "inmem_hull_s", "count", "throughput_gb_s"],
     "time_col": "hull_s", "inmem_col": "inmem_hull_s",
     "elem_bytes": 32, "input_seqs": 1,
     "xlabel": "input size",
     "title": "Upper convex hull: out-of-core (quickhull) vs in-mem parlaylib",
     "data_globs": ["ch_in*", "ch_scratch*"]},
    # convex_hull_lazy_filterExample: a SEPARATE, opt-in benchmark (deliberately
    # NOT in the Makefile's bench-examples rules, same convention as
    # bigint_add_eager) that adds a third line -- the same hull computed with
    # UpperHullLazyFilter (ChunkSequence/examples/chunk_convex_hull_lazy_filter.h,
    # every ChunkPartition call replaced by delayed::lazy_filter +
    # materialize_wide) -- alongside the existing ChunkPartition-based hull and
    # the in-mem baseline. Run it explicitly:
    # `run_benches.py --example convex_hull_lazy_filter`.
    {"name": "convex_hull_lazy_filter", "target": "bin/convex_hull_lazy_filterExample",
     "cols": ["n", "build_s", "hull_s", "lazyfilter_hull_s", "inmem_hull_s",
              "count", "throughput_gb_s"],
     "time_col": "hull_s", "inmem_col": "inmem_hull_s",
     "series_labels": ("in-mem parlaylib (DRAM)", "out-of-core (ChunkPartition)"),
     "extra_series": [("lazyfilter_hull_s", "out-of-core (delayed lazy_filter)", "^-")],
     "elem_bytes": 32, "input_seqs": 1,
     "xlabel": "input size",
     "title": "Upper convex hull: ChunkPartition vs delayed lazy_filter vs in-mem parlaylib",
     "data_globs": ["ch_in*", "ch_scratch*", "ch_lazy*"]},
    # suffix_arrayExample sweeps n; the plotted time is the construction (text
    # build excluded).  Prefix-doubling does ~2 external sorts per round over
    # ~log2(n) rounds, so its I/O (and peak disk residency) is many times the
    # input -- run it standalone at SMALL --example-sizes (e.g. "8MiB 16MiB
    # 32MiB"); it is deliberately NOT in the aggregate bench-examples list, whose
    # 128MiB+ sizes would exceed a dev tmpfs.  All intermediates live under the
    # sa_out prefix and are swept by the algorithm; sa_text/sa_out are the driver's.
    {"name": "suffix_array", "target": "bin/suffix_arrayExample",
     "cols": ["n", "build_s", "sa_s", "inmem_sa_s", "count", "throughput_gb_s"],
     "time_col": "sa_s", "inmem_col": "inmem_sa_s",
     "elem_bytes": 1, "input_seqs": 1,
     "xlabel": "input size",
     "title": "Suffix array: out-of-core (prefix doubling) vs in-mem parlaylib",
     "data_globs": ["sa_text*", "sa_out*"]},
    # dc3Example sweeps n; the plotted time is the DC3 construction (text build
    # excluded).  DC3 recurses on a 2/3-shrinking problem, so its total I/O is a
    # constant multiple of the input (not the O(log n) multiple prefix doubling
    # pays) -- the direct head-to-head against suffix_array on identical text.
    # Still several sorts per level, so like suffix_array it is kept OUT of the
    # aggregate bench-examples list; run standalone at SMALL --example-sizes.  All
    # intermediates live under the dc3_out prefix and are swept by the algorithm;
    # dc3_text/dc3_out are the driver's.
    {"name": "dc3", "target": "bin/dc3Example",
     "cols": ["n", "build_s", "sa_s", "inmem_sa_s", "count", "throughput_gb_s"],
     "time_col": "sa_s", "inmem_col": "inmem_sa_s",
     "elem_bytes": 1, "input_seqs": 1,
     "xlabel": "input size",
     "title": "Suffix array: out-of-core (DC3 / skew) vs in-mem parlaylib",
     "data_globs": ["dc3_text*", "dc3_out*"]},
    # fftExample sweeps N (16-byte complex<double>); the plotted time is total_s
    # (streaming length-A pass + random length-B pass; input build excluded).  Its
    # value is the streaming-vs-random I/O contrast (trace it with io_trace.py, and
    # the two stages are separately timed in the CSV).  The in-mem baseline is the
    # SAME transpose-free four-step FFT run in DRAM (same kernel), so the comparison
    # isolates I/O cost; it is kept OUT of the aggregate bench-examples list -- run
    # standalone via `--example fft`.  Input lives under fft_in, stage-1 out fft_s1.
    {"name": "fft", "target": "bin/fftExample",
     "cols": ["n", "build_s", "stage1_s", "stage2_s", "total_s", "inmem_s",
              "count", "throughput_gb_s"],
     "time_col": "total_s", "inmem_col": "inmem_s",
     "elem_bytes": 16, "input_seqs": 1,
     "xlabel": "input size",
     "title": "FFT: out-of-core vs in-mem (same transpose-free four-step)",
     "data_globs": ["fft_in*", "fft_s1*"]},
    # fft_transposeExample: the counterpart that DOES the on-disk transpose (all
    # streaming: stage1 + transpose + stage2T = 6N) instead of the random length-B
    # pass -- the classic external-memory tradeoff to compare against `fft` (4N with
    # random I/O).  Same in-mem four-step baseline.  Kept OUT of the aggregate list;
    # run via `--example fft_transpose`.  Leaves fft_in/fft_s1/fft_t/fft_t2 files.
    {"name": "fft_transpose", "target": "bin/fft_transposeExample",
     "cols": ["n", "build_s", "stage1_s", "transpose_s", "stage2t_s", "total_s",
              "inmem_s", "count", "throughput_gb_s"],
     "time_col": "total_s", "inmem_col": "inmem_s",
     "elem_bytes": 16, "input_seqs": 1,
     "xlabel": "input size",
     "title": "FFT: out-of-core transpose (streaming 6N) vs in-mem four-step",
     "data_globs": ["fft_in*", "fft_s1*", "fft_t*", "fft_t2*"]},
    # kth_smallestExample sweeps n with k at the median (n/2); the plotted time
    # is the selection pass only (input build excluded).  Its recursion leaves
    # id_/flags_/next_ intermediates in addition to the kth_in input.
    {"name": "kth_smallest", "target": "bin/kth_smallestExample",
     "cols": ["n", "k", "build_s", "select_s", "inmem_select_s", "result",
              "throughput_gb_s"],
     "time_col": "select_s", "inmem_col": "inmem_select_s",
     "elem_bytes": 8, "input_seqs": 1,
     "xlabel": "input size",
     "title": "kth-smallest: out-of-core (ChunkSequenceOps) vs in-mem parlaylib",
     "data_globs": ["kth_in*", "id_*", "flags_*", "next_*"]},
    # kth_smallest_delayedExample: head-to-head comparison, same precedent as
    # samplesort_three_way -- ChunkPartition (kth_smallest_fast, writes all 32
    # buckets every level) vs. delayed::lazy_filter (kth_smallest_delayed,
    # writes only the ONE surviving bucket, or nothing at all once it already
    # fits the DRAM budget) vs. in-mem parlaylib, all on the same keys.  This
    # is the positive counterpart to convex_hull_lazy_filter's quickhull
    # rewrite: quickhull's 2-way split keeps both branches (lazy_filter adds a
    # pass with no offsetting savings there), while kth-smallest's 32-way
    # split discards 31 of 32 branches every level, so skipping their writes
    # is a real saving. eager_write_bytes/delayed_write_bytes (raw byte
    # counts, not a rate) quantify that saving directly; each contestant
    # builds and times its own input, same fairness discipline as
    # samplesort_three_way (bench_drives.h settle/clear between them).
    {"name": "kth_smallest_delayed", "target": "bin/kth_smallest_delayedExample",
     "cols": ["n", "k", "fast_build_s", "fast_select_s", "delayed_build_s",
              "delayed_select_s", "inmem_select_s", "result",
              "eager_write_bytes", "delayed_write_bytes", "throughput_gb_s"],
     "time_col": "delayed_select_s", "inmem_col": "inmem_select_s",
     "series_labels": ("in-mem parlaylib kth_smallest (DRAM)",
                       "delayed::lazy_filter (kth_smallest_delayed)"),
     "extra_series": [("fast_select_s", "ChunkPartition (kth_smallest_fast)", "^-")],
     "elem_bytes": 8, "input_seqs": 1,
     "xlabel": "input size",
     "title": "kth-smallest: ChunkPartition vs delayed lazy_filter vs in-mem parlaylib",
     "data_globs": ["kthd_fast_in*", "kth_next_*", "kthd_delayed_in*", "kdl_next_*"]},
    # external_samplesortExample sweeps n; the plotted time is the sort pass only
    # (input build excluded).  Its recursion leaves ss_id_/ss_bucket_/ss_base_/
    # ss_deg_ intermediates plus the per-bucket base sorter's qs_base_ output
    # (the sorted result the driver returns *is* the qs_base_ files, via flatten)
    # in addition to the ss_in input.
    {"name": "external_samplesort", "target": "bin/external_samplesortExample",
     "cols": ["n", "build_s", "sort_s", "inmem_sort_s", "throughput_gb_s"],
     "time_col": "sort_s", "inmem_col": "inmem_sort_s",
     "elem_bytes": 8, "input_seqs": 1,
     "xlabel": "input size",
     "title": "sample sort: out-of-core (ChunkSequenceOps) vs in-mem parlaylib",
     "data_globs": ["ss_in*", "ss_id_*", "ss_bucket_*", "ss_base_*", "ss_deg_*",
                    "qs_base_*"]},

    # external_samplesort_vs_peterExample sweeps n and times BOTH out-of-core
    # sorts on the identical key multiset: ours (ChunkSequenceOps::sample_sort)
    # and Peter's (peter_samplesort/, via peter_shim).  Unlike the other
    # examples the "baseline" series is not in-memory — both series are disk
    # sorts and plot across the whole sweep (no RAM cliff).  Intermediates: our
    # ss_* recursion files plus Peter's pss_in/pss_out inputs+outputs and his
    # hard-coded spfx_ intermediate buckets.
    {"name": "external_samplesort_vs_peter",
     "target": "bin/external_samplesort_vs_peterExample",
     "cols": ["n", "ext_build_s", "ext_sort_s", "peter_build_s", "peter_sort_s",
              "ext_gb_s", "peter_gb_s"],
     "time_col": "ext_sort_s", "inmem_col": "peter_sort_s",
     "series_labels": ("Peter's sort (out-of-core)", "our sort (out-of-core)"),
     "no_ram_cliff": True,
     "elem_bytes": 8, "input_seqs": 1,
     "xlabel": "input size",
     "title": "sample sort: ours (ChunkSequenceOps) vs Peter's — both out-of-core",
     "data_globs": ["ss_in*", "ss_id_*", "ss_bucket_*", "ss_base_*", "ss_deg_*",
                    "qs_base_*", "pss_in*", "pss_out*", "spfx_*"]},

    # direct_samplesort_vs_peterExample: the same head-to-head as above, but our
    # contestant is direct_samplesort.h (ChunkSequenceOps::direct_sample_sort —
    # the same algorithm written straight against io_uring/O_DIRECT, in Peter's
    # scatter-gather shape, chunk_seq in/out) rather than the sort built on the
    # primitives.  Run both sweeps to separate the algorithm from the substrate:
    # this one is what the chunk_seq model costs when it is NOT paying for the
    # primitives' generality.  Same CSV columns, so it plots identically.
    # Intermediates: our dss_in input + the dss<tag>_<b> bucket files (which ARE
    # the sorted output), plus Peter's pss_in/pss_out and his spfx_ buckets.
    {"name": "direct_samplesort_vs_peter",
     "target": "bin/direct_samplesort_vs_peterExample",
     "cols": ["n", "ext_build_s", "ext_sort_s", "peter_build_s", "peter_sort_s",
              "ext_gb_s", "peter_gb_s"],
     "time_col": "ext_sort_s", "inmem_col": "peter_sort_s",
     "series_labels": ("Peter's sort (out-of-core)", "our direct-I/O sort (out-of-core)"),
     "no_ram_cliff": True,
     "elem_bytes": 8, "input_seqs": 1,
     "xlabel": "input size",
     "title": "sample sort: our direct-I/O sort vs Peter's — both out-of-core",
     "data_globs": ["dss_in*", "dss*", "pss_in*", "pss_out*", "spfx_*"]},

    # samplesort_three_wayExample: all THREE out-of-core sorts on the same keys in
    # one run — Peter's (peter_samplesort/, via peter_shim), ours written straight
    # against io_uring (direct_samplesort.h) and ours built on the primitives
    # (external_samplesort.h) — plus in-memory parlay::sort as a fourth series.
    # The two pairwise sweeps above measure one gap each against a separately timed
    # Peter run; this one puts everything side by side, so "Peter's vs our direct"
    # reads as the cost of the chunk_seq *substrate*, "our direct vs our
    # primitives" as the cost of the *primitives*, and the DRAM line as what all of
    # them are chasing.  The in-mem series stops at the RAM cliff (~24n, gated by
    # EXAMPLE_INMEM_BUDGET_BYTES); the three disk series continue past it, which is
    # the point of the project.
    #
    # Each sort runs ONCE per point (one input build + one sort each — no repeats:
    # these are consumer SSDs and every extra round is real write endurance).  What
    # makes the three comparable despite sharing the drives is the teardown between
    # them: the previous sort's files are removed and then every mount is synced and
    # left to settle (SS3_SETTLE_MS, default 2000 ms), because unlink() returns long
    # before ext4 has freed the blocks and a sort started on top of that background
    # work runs 15-25% slow.  SS3_FIRST=0|1|2 rotates which sort goes first; it is a
    # check knob (the times must not move), not a measurement one.
    #
    # Below ~512 MiB of input the pivot count drops to <= 3 and Peter's GetPivots
    # underflows (it takes a garbage pivot, unbalancing his buckets), so his series
    # is only meaningful from 512 MiB up — read the dev-box sizes with that in
    # mind, or sweep this example from 512MiB.
    {"name": "samplesort_three_way",
     "target": "bin/samplesort_three_wayExample",
     "cols": ["n", "peter_sort_s", "direct_sort_s", "prim_sort_s", "inmem_sort_s",
              "peter_build_s", "direct_build_s", "prim_build_s", "peter_gb_s",
              "direct_gb_s", "prim_gb_s"],
     "time_col": "direct_sort_s", "inmem_col": "inmem_sort_s",
     "series_labels": ("parlay::sort (DRAM)",
                       "Ours, Direct Port"),
     "extra_series": [("peter_sort_s", "Li et al. 2025", "d-"),
                      ("prim_sort_s", "Ours, Composed External", "^-")],
     "elem_bytes": 8, "input_seqs": 1,
     "xlabel": "Input Size",
     "title": "Sample Sort: External vs Native ParlayLib",
     "data_globs": ["dss_in*", "dss*", "ss_in*", "ss_id_*", "ss_bucket_*",
                    "ss_base_*", "ss_deg_*", "qs_base_*",
                    "pss_in*", "pss_out*", "spfx_*"]},

    # apply_sort_vs_samplesortExample: ChunkSequenceOps::apply<ChunkOperation::Sort>
    # (whole-sequence, DRAM-budgeted, no bucketing -- process_inplace_budgeted's
    # own CHECK caps how large an input it can take) vs sample_sort (recursive
    # out-of-core, external_samplesort.h), which itself uses apply<Sort> as its
    # per-bucket base case.  apply_sort_s/apply_build_s/apply_gb_s are left BLANK
    # by the driver once n exceeds apply<Sort>'s own DRAM budget (mirrors how
    # inmem_sort_s goes blank past the RAM cliff), so the extra_series line stops
    # at that cliff instead of dropping to zero.
    {"name": "apply_sort_vs_samplesort",
     "target": "bin/apply_sort_vs_samplesortExample",
     "cols": ["n", "apply_sort_s", "samplesort_sort_s", "inmem_sort_s",
              "apply_build_s", "samplesort_build_s", "apply_gb_s",
              "samplesort_gb_s"],
     "time_col": "samplesort_sort_s", "inmem_col": "inmem_sort_s",
     "extra_series": [("apply_sort_s",
                       "apply<Sort> (whole-sequence, DRAM-budgeted)", "^-")],
     "elem_bytes": 8, "input_seqs": 1,
     "xlabel": "input size",
     "title": "sort: recursive external_samplesort vs whole-sequence apply<Sort>",
     "data_globs": ["as_in*", "ss_in*", "ss_id_*", "ss_bucket_*", "ss_base_*",
                    "ss_deg_*", "qs_base_*"]},

    # samplesort_vs_samplesort_randomExample: ChunkSequenceOps::sample_sort
    # (pivot sampling via the shared sample<T> helper, ExternalPrimitives/
    # chunk_sample.h) vs sample_sort_random (the pre-refactor sibling kept in
    # external_samplesort.h for comparison -- same count_sort/apply<Sort>/
    # flatten pipeline, only the pivot-sampling bookkeeping differs). Both
    # always run (neither has a DRAM-budget CHECK like apply<Sort>'s), so no
    # column ever goes blank except inmem_sort_s past the RAM cliff.
    {"name": "samplesort_vs_samplesort_random",
     "target": "bin/samplesort_vs_samplesort_randomExample",
     "cols": ["n", "sample_sort_s", "sample_sort_random_s", "inmem_sort_s",
              "sample_sort_build_s", "sample_sort_random_build_s",
              "sample_sort_gb_s", "sample_sort_random_gb_s"],
     "time_col": "sample_sort_s", "inmem_col": "inmem_sort_s",
     "extra_series": [("sample_sort_random_s",
                       "sample_sort_random (pre-refactor pair sampling)", "^-")],
     "elem_bytes": 8, "input_seqs": 1,
     "xlabel": "input size",
     "title": "sample_sort: shared sample<T> helper vs pre-refactor pair sampling",
     "data_globs": ["sspv_in*", "ssrd_in*", "ss_bucket_*", "ss_base_*",
                    "ss_deg_*", "qs_base_*"]},

    # external_random_shuffleExample sweeps n and times THREE shuffles of the same
    # keys: random_shuffle_method (the bucketing shuffle on the high-level
    # abstractions), ChunkSequenceOps::Permutation (the same algorithm on the
    # low-level reader/writer, rewriting each bucket in place), and the in-mem
    # parlay::random_shuffle baseline (stops at the RAM cliff).  The plotted times
    # are the shuffle passes only (the shared input build is excluded).  Each
    # method's result *is* its bucket files (rs_out_/perm), so those prefixes are
    # swept too; the driver additionally snapshot-diffs the drives and fails if
    # anything at all is left behind.
    {"name": "external_random_shuffle",
     "target": "bin/external_random_shuffleExample",
     "cols": ["n", "build_s", "shuffle_s", "perm_s", "inmem_shuffle_s",
              "shuffle_gb_s", "perm_gb_s"],
     "time_col": "shuffle_s", "inmem_col": "inmem_shuffle_s",
     "series_labels": ("in-mem parlay::random_shuffle (DRAM)",
                       "random_shuffle_method (out-of-core)"),
     "extra_series": [("perm_s", "Permutation (out-of-core)", "^-")],
     "xlabel": "input size",
     "title": "random shuffle: two out-of-core methods vs in-mem parlaylib",
     "data_globs": ["rs_in*", "rs_bucket_*", "rs_out_*", "rs_base_*", "perm*"]},

    # random_shuffle_three_wayExample: the shuffle counterpart to
    # samplesort_three_way -- our primitives-based shuffle
    # (external_random_shuffle.h) vs our direct-I/O shuffle
    # (direct_random_shuffle.h) vs in-mem parlay::random_shuffle, all on the
    # same keys in one run. No vendored reference shuffle exists (unlike
    # sample sort's Peter's leg), so this is two out-of-core contestants
    # instead of three. Correctness is a permutation check (the keys are
    # distinct), not element-wise equality. inmem_shuffle_s is left BLANK
    # past the RAM budget, so the plotted DRAM line stops at the cliff.
    {"name": "random_shuffle_three_way",
     "target": "bin/random_shuffle_three_wayExample",
     "cols": ["n", "prim_shuffle_s", "direct_shuffle_s", "inmem_shuffle_s",
              "prim_build_s", "direct_build_s", "prim_gb_s", "direct_gb_s"],
     "time_col": "direct_shuffle_s", "inmem_col": "inmem_shuffle_s",
     "series_labels": ("parlay::random_shuffle (DRAM)", "Ours, Direct Port"),
     "extra_series": [("prim_shuffle_s", "Ours, Composed External", "^-")],
     "elem_bytes": 8, "input_seqs": 1,
     "xlabel": "input size",
     "title": "Random Shuffle: Direct I/O vs Composed Primitives vs In-Memory",
     "data_globs": ["rs3_in*", "ss_bucket_*", "drs3_in*", "drs3*"]},

    # fitmem_kth_smallestExample: same driver shape as kth_smallest, but the
    # single-level "fitmem" variant (one bucketing round, then select the winning
    # bucket in DRAM).  Its intermediates are fk_id_/fk_next_ alongside fk_in.
    {"name": "fitmem_kth_smallest", "target": "bin/fitmem_kth_smallestExample",
     "cols": ["n", "k", "build_s", "select_s", "inmem_select_s", "result",
              "throughput_gb_s"],
     "time_col": "select_s", "inmem_col": "inmem_select_s",
     "elem_bytes": 8, "input_seqs": 1,
     "xlabel": "input size",
     "title": "fitmem kth-smallest: out-of-core (ChunkSequenceOps) vs in-mem parlaylib",
     "data_globs": ["fk_in*", "fk_id_*", "fk_next_*"]},

    # fitmem_sortExample: same driver shape as external_samplesort, but the
    # single-level "fitmem" variant (one bucketing round, then each bucket is
    # sorted directly in DRAM).  Its intermediates are fs_id_/fs_bucket_/fs_base_/
    # fs_sorted_ (the sorted output references the fs_sorted_ files) alongside fs_in.
    {"name": "fitmem_sort", "target": "bin/fitmem_sortExample",
     "cols": ["n", "build_s", "sort_s", "inmem_sort_s", "throughput_gb_s"],
     "time_col": "sort_s", "inmem_col": "inmem_sort_s",
     "elem_bytes": 8, "input_seqs": 1,
     "xlabel": "input size",
     "title": "fitmem sample sort: out-of-core (ChunkSequenceOps) vs in-mem parlaylib",
     "data_globs": ["fs_in*", "fs_id_*", "fs_bucket_*", "fs_base_*", "fs_sorted_*"]},

    # external_linefitExample sweeps n; the plotted time is the fit itself
    # (input build excluded).  Both passes are fully delayed, so the fit leaves
    # no intermediates beyond the lf_x/lf_y inputs.
    {"name": "external_linefit", "target": "bin/external_linefitExample",
     "cols": ["n", "build_s", "fit_s", "inmem_fit_s", "offset", "slope",
              "throughput_gb_s"],
     "time_col": "fit_s", "inmem_col": "inmem_fit_s",
     "elem_bytes": 8, "input_seqs": 2,
     "xlabel": "input size",
     "title": "line fit: out-of-core (ChunkSequenceOps) vs in-mem parlaylib",
     "data_globs": ["lf_x*", "lf_y*"]},

    # bigint_addExample sweeps n (limb count); the plotted time is the add pass
    # only (operand build excluded).  Baseline is our own parlaylib reference.
    {"name": "bigint_add", "target": "bin/bigint_addExample",
     "cols": ["n", "build_s", "add_s", "inmem_add_s", "result_limbs",
              "throughput_gb_s"],
     "time_col": "add_s", "inmem_col": "inmem_add_s",
     "elem_bytes": 8, "input_seqs": 2,
     "xlabel": "input size",
     "title": "big-integer add: out-of-core (ChunkSequenceOps) vs in-mem parlaylib",
     "data_globs": ["bi_a*", "bi_b*", "bi_sum*"]},

    # bigint_mulExample sweeps n (limb count); the plotted time is the multiply
    # pass only.  Baseline is our own verified in-memory Karatsuba (upstream
    # parlaylib karatsuba is broken).  Set BIGINT_MUL_DRAM_BUDGET_BYTES small to
    # exercise the out-of-core recursion at modest n on a dev box.
    {"name": "bigint_mul", "target": "bin/bigint_mulExample",
     "cols": ["n", "build_s", "mul_s", "inmem_mul_s", "result_limbs",
              "throughput_gb_s"],
     "time_col": "mul_s", "inmem_col": "inmem_mul_s",
     "elem_bytes": 8, "input_seqs": 2,
     "xlabel": "input size",
     "title": "big-integer mul: out-of-core Karatsuba (ChunkSequenceOps) vs in-mem",
     "data_globs": ["bm_a*", "bm_b*", "bm_prod*", "bimul_zero*"]},

    # bigint_add_eagerExample: a SEPARATE, opt-in benchmark (deliberately NOT in
    # the Makefile's bench-examples rules) that adds a third line — the same add
    # done WITHOUT delayed fusion (intermediate classify/scan materialized to
    # disk) — alongside the fused out-of-core add and the in-mem baseline.  Run
    # it explicitly: `run_benches.py --example bigint_add_eager`.
    {"name": "bigint_add_eager", "target": "bin/bigint_add_eagerExample",
     "cols": ["n", "build_s", "add_s", "eager_add_s", "inmem_add_s",
              "result_limbs", "throughput_gb_s"],
     "time_col": "add_s", "inmem_col": "inmem_add_s",
     "series_labels": ("in-mem parlaylib (DRAM)", "out-of-core delayed (fused)"),
     "extra_series": [("eager_add_s", "out-of-core eager (no fusion)", "^-")],
     "elem_bytes": 8, "input_seqs": 2,
     "xlabel": "input size",
     "title": "big-integer add: fused vs eager out-of-core vs in-mem parlaylib",
     "data_globs": ["bie_a*", "bie_b*", "bie_sum*", "bie_eager*"]},

    # chunk_cutExample sweeps n; the plotted time is the cut itself (input build
    # excluded).  It cuts the middle ~half (k = n/2) with both endpoints in the
    # MIDDLE of a chunk (never on a chunk boundary), so the cut length scales with
    # n and every sweep point does the same real seam-rewrite work regardless of
    # how n aligns to the chunk grid (see chunk_cut.cpp for why naive n/4, 3n/4
    # endpoints alias the grid for power-of-two n).  Baseline is parlaylib's
    # slice::cut materialized into an independent DRAM sequence; the out-of-core
    # cut symmetrically materializes the range into fresh on-disk files.  Three
    # file sets: the "cut_in<d>" input,
    # "cut_in<d>_cut" seam scratch (matched by "cut_in*"), and the materialized
    # "cut_out<d>" output.
    {"name": "chunk_cut", "target": "bin/chunk_cutExample",
     "cols": ["n", "start", "end", "build_s", "cut_s", "inmem_cut_s",
              "out_elems", "throughput_gb_s"],
     "time_col": "cut_s", "inmem_col": "inmem_cut_s",
     "elem_bytes": 8, "input_seqs": 1,
     "xlabel": "input size",
     "title": "cut / slice: out-of-core (ChunkSequenceOps) vs in-mem parlaylib",
     "data_globs": ["cut_in*", "cut_out*"]},

    # bellman_fordExample: three registry entries, one per RMAT graph-density
    # case (sparse/balanced/dense avg_degree) that bellman_ford.cpp already
    # builds and cross-checks. The binary normally runs and prints a CSV line
    # for ALL THREE cases per invocation; `extra_argv`'s trailing case name
    # selects bellman_ford.cpp's argv[3] case filter so each entry's run does
    # and prints exactly ONE case -- required because run_binary/io_trace.py
    # keep only the last "CSV," line, so without the filter every entry would
    # silently report the dense case's numbers.  The leading "8" in each
    # extra_argv is a placeholder occupying argv[2] (balanced_avg_degree)'s
    # position so argv[3] lands correctly; sparse/dense ignore it (their
    # avg_degree is fixed independent of that arg).
    #
    # Each compares THREE implementations, like samplesort_three_way /
    # external_random_shuffle: external_bellman_ford (per-vertex, O(rounds*n)
    # reader setups -- op_s), external_bellman_ford_fast (one streaming
    # ChunkSegmentedReduce pass per round -- fast_op_s), and in-memory
    # parlaylib bellman_ford (inmem_op_s).
    #
    # NOT in the make bench-examples/-mid/-full target lists (opt-in via
    # --example only, same precedent as samplesort_three_way /
    # external_random_shuffle / kth_smallest): external_bellman_ford is
    # documented as dramatically slower than the in-memory baseline even at
    # small n, so it doesn't belong in the default dev-box sweep.
    {"name": "bellman_ford_sparse", "target": "bin/bellman_fordExample",
     "cols": ["case", "n", "m", "build_s", "op_s", "inmem_op_s", "reachable",
              "throughput_gb_s", "fast_op_s", "fast_reachable", "fast_throughput_gb_s"],
     "time_col": "op_s", "inmem_col": "inmem_op_s",
     "series_labels": ("in-mem parlaylib (DRAM)", "out-of-core, per-vertex"),
     "extra_series": [("fast_op_s", "out-of-core, streaming (fast)", "^-")],
     "extra_argv": ["8", "sparse"],
     "elem_bytes": 2 * 32, "input_seqs": 1,   # avg_degree(2) * sizeof(weighted_edge)
     "xlabel": "input size (edge bytes)",
     "title": "Bellman-Ford (sparse, avg_degree=2): out-of-core vs in-mem",
     "data_globs": ["bf_edges_sparse*"]},

    {"name": "bellman_ford_balanced", "target": "bin/bellman_fordExample",
     "cols": ["case", "n", "m", "build_s", "op_s", "inmem_op_s", "reachable",
              "throughput_gb_s", "fast_op_s", "fast_reachable", "fast_throughput_gb_s"],
     "time_col": "op_s", "inmem_col": "inmem_op_s",
     "series_labels": ("in-mem parlaylib (DRAM)", "out-of-core, per-vertex"),
     "extra_series": [("fast_op_s", "out-of-core, streaming (fast)", "^-")],
     "extra_argv": ["8", "balanced"],
     "elem_bytes": 8 * 32, "input_seqs": 1,   # avg_degree(8) * sizeof(weighted_edge)
     "xlabel": "input size (edge bytes)",
     "title": "Bellman-Ford (balanced, avg_degree=8): out-of-core vs in-mem",
     "data_globs": ["bf_edges_balanced*"]},

    # dense: avg_degree = n/2 (bellman_ford.cpp), so edge bytes ~= (n^2/2) *
    # sizeof(weighted_edge) = 16*n^2 -- quadratic in n, not linear, so this
    # entry overrides size_to_n's default formula via n_from_size
    # (_bellman_ford_dense_n, defined above the registry). elem_bytes/
    # input_seqs below are unused whenever n_from_size is present (see
    # size_to_n) -- kept only as documentation of the on-disk element size.
    {"name": "bellman_ford_dense", "target": "bin/bellman_fordExample",
     "cols": ["case", "n", "m", "build_s", "op_s", "inmem_op_s", "reachable",
              "throughput_gb_s", "fast_op_s", "fast_reachable", "fast_throughput_gb_s"],
     "time_col": "op_s", "inmem_col": "inmem_op_s",
     "series_labels": ("in-mem parlaylib (DRAM)", "out-of-core, per-vertex"),
     "extra_series": [("fast_op_s", "out-of-core, streaming (fast)", "^-")],
     "extra_argv": ["8", "dense"],
     "n_from_size": _bellman_ford_dense_n,
     "elem_bytes": 32, "input_seqs": 1,
     "xlabel": "input size (edge bytes)",
     "title": "Bellman-Ford (dense, avg_degree=n/2): out-of-core vs in-mem",
     "data_globs": ["bf_edges_dense*"]},

    # bfsExample: same three-case (sparse/balanced/dense) RMAT sweep as
    # bellman_ford, same extra_argv placeholder/case-filter reasoning and same
    # THREE series (in-mem, per-vertex, fast streaming): external_bfs.h's
    # external_bfs is the streaming counterpart to BFS_simple's per-vertex
    # pread implementation, same slow-vs-fast split as
    # external_bellman_ford/external_bellman_ford_fast.
    #
    # NOT in the make bench-examples/-mid/-full target lists (opt-in via
    # --example only, same precedent as bellman_ford_*): BFS_simple's
    # per-vertex reader-setup cost is documented as dramatically slower than
    # the in-memory baseline even at small n, so it doesn't belong in the
    # default dev-box sweep.
    {"name": "bfs_sparse", "target": "bin/bfsExample",
     "cols": ["case", "n", "m", "build_s", "op_s", "inmem_op_s", "levels",
              "reachable", "throughput_gb_s", "fast_op_s", "fast_levels",
              "fast_reachable", "fast_throughput_gb_s"],
     "time_col": "op_s", "inmem_col": "inmem_op_s",
     "series_labels": ("in-mem parlaylib (DRAM)", "out-of-core, per-vertex (BFS_simple)"),
     "extra_series": [("fast_op_s", "out-of-core, streaming (fast)", "^-")],
     "extra_argv": ["8", "sparse"],
     "elem_bytes": 2 * 32, "input_seqs": 1,   # avg_degree(2) * sizeof(weighted_edge)
     "xlabel": "input size (edge bytes)",
     "title": "BFS (sparse, avg_degree=2): out-of-core vs in-mem",
     "data_globs": ["bfs_edges_sparse*", "bfs_frontier*"]},

    {"name": "bfs_balanced", "target": "bin/bfsExample",
     "cols": ["case", "n", "m", "build_s", "op_s", "inmem_op_s", "levels",
              "reachable", "throughput_gb_s", "fast_op_s", "fast_levels",
              "fast_reachable", "fast_throughput_gb_s"],
     "time_col": "op_s", "inmem_col": "inmem_op_s",
     "series_labels": ("in-mem parlaylib (DRAM)", "out-of-core, per-vertex (BFS_simple)"),
     "extra_series": [("fast_op_s", "out-of-core, streaming (fast)", "^-")],
     "extra_argv": ["8", "balanced"],
     "elem_bytes": 8 * 32, "input_seqs": 1,   # avg_degree(8) * sizeof(weighted_edge)
     "xlabel": "input size (edge bytes)",
     "title": "BFS (balanced, avg_degree=8): out-of-core vs in-mem",
     "data_globs": ["bfs_edges_balanced*", "bfs_frontier*"]},

    # dense: avg_degree = n/2 (bfs.cpp), so edge bytes ~= 16*n^2, quadratic in
    # n -- see _bfs_dense_n above (an alias of bellman_ford's own inverse,
    # since both build the same chunk_csr shape). elem_bytes/input_seqs below
    # are unused whenever n_from_size is present (see size_to_n) -- kept only
    # as documentation of the on-disk element size.
    {"name": "bfs_dense", "target": "bin/bfsExample",
     "cols": ["case", "n", "m", "build_s", "op_s", "inmem_op_s", "levels",
              "reachable", "throughput_gb_s", "fast_op_s", "fast_levels",
              "fast_reachable", "fast_throughput_gb_s"],
     "time_col": "op_s", "inmem_col": "inmem_op_s",
     "series_labels": ("in-mem parlaylib (DRAM)", "out-of-core, per-vertex (BFS_simple)"),
     "extra_series": [("fast_op_s", "out-of-core, streaming (fast)", "^-")],
     "extra_argv": ["8", "dense"],
     "n_from_size": _bfs_dense_n,
     "elem_bytes": 32, "input_seqs": 1,
     "xlabel": "input size (edge bytes)",
     "title": "BFS (dense, avg_degree=n/2): out-of-core vs in-mem",
     "data_globs": ["bfs_edges_dense*", "bfs_frontier*"]},

    # even_squaresExample: the four external_even_squares.h implementations of
    # "sum of squares of the even elements" head-to-head on one shared
    # input -- out-of-core eager (ChunkFilter, then a fused delayed map+reduce)
    # vs out-of-core fully-fused delayed (lazy_filter -> map -> reduce, one
    # read pass, zero writes) vs the two in-memory parlay baselines (eager,
    # delayed). All four are cross-checked for exact equality (an integer
    # sum, so no tolerance compare); the in-mem pair stops at the ~24n DRAM
    # cliff (EXAMPLE_INMEM_BUDGET_BYTES), same convention as samplesort's
    # in-mem series.
    #
    # NOT in the make bench-examples/-mid/-full target lists (opt-in via
    # --example only, same precedent as samplesort_three_way/bellman_ford_*/
    # bfs_*): a comparison-focused example, not part of the default dev-box
    # sweep.
    {"name": "even_squares", "target": "bin/even_squaresExample",
     "cols": ["n", "build_s", "eager_op_s", "delay_op_s", "inmem_eager_op_s",
              "inmem_delay_op_s", "result", "eager_gb_s", "delay_gb_s"],
     "time_col": "delay_op_s", "inmem_col": "inmem_delay_op_s",
     "series_labels": ("in-mem parlaylib delayed (DRAM)", "out-of-core, fused delayed"),
     "extra_series": [("eager_op_s", "out-of-core, eager (ChunkFilter)", "^-"),
                      ("inmem_eager_op_s", "in-mem parlaylib eager (DRAM)", "d-")],
     "elem_bytes": 8, "input_seqs": 1,
     "xlabel": "input size",
     "title": "Sum of even squares: fused-delayed vs eager, out-of-core vs in-mem",
     "data_globs": ["es_in*", "even_squares_tmp*"]},

]


# ── size parsing ────────────────────────────────────────────────────────────
def parse_count(s):
    """Element count: `2^k`/`2**k`, decimal suffixes K=1e3/M=1e6/G=1e9, or raw."""
    s = s.strip()
    m = re.fullmatch(r"2(?:\^|\*\*)(\d+)", s)     # 2^30, 2**30
    if m:
        return 2 ** int(m.group(1))
    m = re.fullmatch(r"(\d+)([kKmMgG]?)", s)
    if not m:
        raise ValueError(f"bad count {s!r}")
    mult = {"": 1, "k": 10**3, "m": 10**6, "g": 10**9}[m.group(2).lower()]
    return int(m.group(1)) * mult


def parse_bytes(s):
    """Byte size: `2^k`/`2**k`, binary suffixes KiB/MiB/GiB/TiB (or K/M/G/T =
    *1024^x), or raw.

    The `2^k` form is bytes like every other form here, *not* an element count —
    an examples sweep is parameterized by input size (see size_to_n), so for an
    8-byte-element example `2^23` is the point that sorts 2^20 keys.  The sweep
    prints both at each point ("size=8 MiB (n=1048576)") so the two can't be
    confused for long.
    """
    s = s.strip()
    m = re.fullmatch(r"2(?:\^|\*\*)(\d+)", s)     # 2^30, 2**30
    if m:
        return 2 ** int(m.group(1))
    m = re.fullmatch(r"(\d+)\s*([kKmMgGtT]?)(i?[bB]?)", s)
    if not m:
        raise ValueError(f"bad byte size {s!r}")
    mult = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}[m.group(2).lower()]
    return int(m.group(1)) * mult


def size_to_n(entry, size_bytes):
    """Convert a target input size (bytes) to an example's argv[1] element count.

    An example's total on-disk input footprint is `elem_bytes*input_seqs` bytes
    per element of n (e.g. bigint_add reads two 8-byte-limb operands, so 16 B/n),
    so n = size / (elem_bytes*input_seqs).  n is rounded down to a whole number of
    chunks (at least one) to preserve the O_DIRECT chunk-aligned invariant: the
    engine tolerates a partial final chunk, but the original design keeps sweep
    points on the ELEMS_PER_CHUNK grid.

    An entry with an `n_from_size` callable overrides the linear formula above
    with `n_from_size(size_bytes) -> n` instead -- for examples where
    `elem_bytes` isn't a true per-n constant (e.g. bellman_ford_dense, whose
    edge count scales as n^2 since avg_degree itself grows with n).  The
    chunk-grid rounding below does NOT apply to this path: it assumes n counts
    elem_bytes-sized on-disk elements directly, which isn't true when n and
    the on-disk footprint are related non-linearly, so `n_from_size` is
    responsible for its own rounding if it wants any.
    """
    if "n_from_size" in entry:
        return max(1, entry["n_from_size"](size_bytes))
    per_n = entry["elem_bytes"] * entry["input_seqs"]
    epc = EXAMPLE_CHUNK_BYTES // entry["elem_bytes"]   # elements per chunk for this seq
    n = (size_bytes // per_n) // epc * epc
    return max(epc, n)


def _f(s):
    """Parse a CSV time field; blank (skipped in-memory line) -> None."""
    s = s.strip()
    return float(s) if s else None


# ── device maintenance ──────────────────────────────────────────────────────
# File-name globs (per mount) of every prefix the benchmarks write: the shared
# `iota<drive>` input plus each pipeline's intermediates.  Used to clear the
# drives between points so nothing accumulates — the C++ binaries clean their
# own intermediates, but delayed_compare leaves its `iota` input behind.
BENCH_FILE_GLOBS = ("iota[0-9]*", "bw_dl_*", "bw_cs_*") + \
    tuple(g for e in EXAMPLES for g in e["data_globs"])


def clear_bench_data(glob_pat, enabled):
    """Force-unlink leftover bench files under every mount in glob_pat.

    Called after every sweep point so nothing accumulates between runs.  A first
    unlink can fail if a prior (possibly crashed or sudo) run left a file without
    the owner write bit -- rather than silently skipping it (which lets files
    pile up invisibly), chmod it writable and retry, and if it *still* will not
    go, surface the path loudly instead of swallowing the error.
    """
    if not enabled:
        return
    removed = 0
    failed = []
    for m in sorted(glob.glob(glob_pat)):
        for pat in BENCH_FILE_GLOBS:
            for f in glob.glob(os.path.join(m, pat)):
                try:
                    os.unlink(f)
                    removed += 1
                except OSError:
                    try:
                        os.chmod(f, 0o644)
                        os.unlink(f)
                        removed += 1
                    except OSError as e:
                        failed.append(f"{f}: {e}")
    if removed:
        print(f"  cleared {removed} leftover bench files", flush=True)
    if failed:
        print(f"  !!! could NOT remove {len(failed)} bench file(s) "
              f"(they will accumulate):", flush=True)
        for line in failed[:10]:
            print(f"      {line}", flush=True)
        if len(failed) > 10:
            print(f"      ... and {len(failed) - 10} more", flush=True)


def fstrim_mounts(glob_pat, enabled):
    """Best-effort `fstrim` of every mount matching glob_pat.

    Announces itself (fstrim can be slow, so an unexplained pause is worse than
    a line of output) but returns the outcome as a note string for the
    end-of-run summary instead of printing it here, where it would scroll away
    (fstrim may need privileges on real SSDs, and on a tmpfs dev box FITRIM is
    unsupported — both worth noticing).  Returns None when disabled or no
    mounts match.
    """
    if not enabled:
        return None
    mounts = sorted(glob.glob(glob_pat))
    if not mounts:
        return None
    print(f"  fstrim {len(mounts)} mount(s) matching {glob_pat} ...", flush=True)
    ok, failed = 0, None
    for m in mounts:
        r = subprocess.run(["fstrim", m], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        if r.returncode == 0:
            ok += 1
        else:
            failed = r.stdout.strip() or f"fstrim {m} exit {r.returncode}"
    note = f"fstrim: ok on {ok} of {len(mounts)} mount(s)"
    if failed:
        note += f"; skipped/unsupported on the rest: {failed}"
    return note


# ── running binaries ────────────────────────────────────────────────────────
def make(target):
    print(f"  $ make {target}", flush=True)
    r = subprocess.run(["make", target], cwd=REPO_ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout)
        sys.exit(f"make {target} failed (exit {r.returncode})")


def run_binary(path, args, fatal=True):
    """Run a benchmark binary, echo its output, return (csv_fields, problem).

    `problem` is None on a clean run, else a short description (crash,
    verification mismatch, missing CSV line).  With fatal=True (the substrate
    benchmarks) any problem aborts the whole run; with fatal=False (the
    examples sweep) it is returned so the caller can warn and keep sweeping.
    `csv_fields` is None if the binary printed no CSV line (e.g. it crashed).
    """
    cmd = [path] + [str(a) for a in args]
    print(f"  $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=REPO_ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout, end="", flush=True)
    csv = None
    for line in r.stdout.splitlines():
        if line.startswith("CSV,"):
            csv = line[len("CSV,"):].split(",")
    problem = None
    if r.returncode != 0:
        if r.returncode == -signal.SIGKILL:
            # Nothing here ever sends SIGKILL and there is no timeout, so this
            # is the kernel OOM killer essentially every time.  Say so: an OOM
            # looks nothing like a crash to debug (no core, no stack, and any
            # trace file stops at whatever phase was allocating).
            why = "killed by SIGKILL — almost certainly the OOM killer"
        elif r.returncode < 0:
            why = f"killed by signal {-r.returncode} (crash)"
        else:
            why = "correctness mismatch or error"
        problem = f"exited {r.returncode} ({why})"
    elif csv is None:
        problem = "no CSV line in output"
    if problem and fatal:
        sys.exit(f"\n*** {os.path.basename(path)} {problem} — aborting ***")
    return csv, problem


# ── delayed scale sweep ─────────────────────────────────────────────────────
# CSV cols emitted by delayed_compare.cpp, in order.
DELAYED_COLS = ["n", "raw_read_s", "eager_mr_s", "delayed_mr_s", "inmem_mr_s",
                "eager_mmr_s", "delayed_mmr_s", "inmem_mmr_s",
                "eager_fmm_s", "delayed_fmm_s", "inmem_fmm_s", "agree"]


def run_delayed(n_values, extra_args, clear_glob, clear_enabled):
    make("bin/delayedCompare")
    binary = os.path.join(BINDIR, "delayedCompare")
    rows = []
    for n in n_values:
        print(f"\n=== delayed scale: n={n} ===", flush=True)
        fields, _ = run_binary(binary, [n] + extra_args)
        row = dict(zip(DELAYED_COLS, fields))
        if row["agree"].strip() != "1":
            sys.exit(f"\n*** agree={row['agree']} at n={n} — aborting ***")
        rows.append(row)
        clear_bench_data(clear_glob, clear_enabled)   # don't leave input on the drives
    return rows


# ── chunk size sweep ────────────────────────────────────────────────────────
# CSV cols emitted by chunk_size_compare.cpp, in order.
CHUNK_COLS = ["chunk_size_bytes", "n", "raw_s", "eager_mr_s", "delayed_mr_s",
              "eager_mmr_s", "delayed_mmr_s", "eager_fmm_s", "delayed_fmm_s", "agree"]


def run_chunk_size(chunk_sizes, n, extra_args, clear_glob, clear_enabled):
    rows = []
    for cs in chunk_sizes:
        make(f"bin/chunkSizeCompare_{cs}")
        binary = os.path.join(BINDIR, f"chunkSizeCompare_{cs}")
        print(f"\n=== chunk size: {cs} bytes, n={n} ===", flush=True)
        fields, _ = run_binary(binary, [n] + extra_args)
        row = dict(zip(CHUNK_COLS, fields))
        if row["agree"].strip() != "1":
            sys.exit(f"\n*** agree={row['agree']} at chunk_bytes={cs} — aborting ***")
        rows.append(row)
        clear_bench_data(clear_glob, clear_enabled)   # don't leave input on the drives
    return rows


# ── examples sweep ──────────────────────────────────────────────────────────
def run_example(entry, sizes, extra_args, clear_glob, clear_enabled, warnings,
                n_values=None):
    """Sweep one example over input `sizes` (bytes); return parsed rows.

    If `n_values` is given (a list of element counts), it takes precedence over
    `sizes`: each n is passed to the binary exactly as given, with no
    size_to_n/chunk-grid rounding — `input_bytes` (used only for the CSV column
    and the plot x-axis) is then computed as the inverse, n*elem_bytes*input_seqs,
    rather than driving n.

    Correctness is checked inside the binary: when the in-memory parlaylib
    baseline runs (sizes within its RAM budget) the binary cross-checks the
    count and the full contents against the out-of-core result and exits
    non-zero on a mismatch.  Unlike the substrate benchmarks' hard agree=1
    enforcement, a problem here (mismatch, crash) does NOT stop the sweep: it
    is appended to `warnings` (echoed again at the end of the run) and the
    sweep continues — a crashed point that printed no CSV line is dropped.

    Each point is run once: the binaries do real disk writes, so repeating a
    point to keep the fastest sample would multiply the write endurance cost for
    a modest noise win.  Instead the timed operation is isolated from its main
    noise source in-binary (a sync()+settle between the input build and the op,
    so the build's writeback doesn't inflate the op timer — see quiesce_drives()
    in each example).
    """
    make(entry["target"])
    binary = os.path.join(BINDIR, os.path.basename(entry["target"]))
    rows = []
    points = ([(n, n * entry["elem_bytes"] * entry["input_seqs"]) for n in n_values]
             if n_values is not None else
             [(size_to_n(entry, size), size) for size in sizes])
    for n, size in points:
        print(f"\n=== example {entry['name']}: size={_bytes_fmt(size, None)} "
              f"(n={n}) ===", flush=True)
        fields, problem = run_binary(binary, [n] + entry.get("extra_argv", []) + extra_args,
                                     fatal=False)
        if problem:
            w = (f"example {entry['name']} at size={_bytes_fmt(size, None)} (n={n}): "
                 f"{problem}" + ("" if fields else " — point dropped"))
            print(f"  !!! {w}", flush=True)
            warnings.append(w)
        if fields:
            row = dict(zip(entry["cols"], fields))
            row["input_bytes"] = str(size)
            rows.append(row)
        clear_bench_data(clear_glob, clear_enabled)   # don't leave output on the drives
    return rows


# ── output: raw CSV + plots ─────────────────────────────────────────────────
def write_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r[c] for c in header) + "\n")
    print(f"  wrote {path}", flush=True)


def _pow2_fmt(val, _):
    if val <= 0:
        return ""
    return f"$2^{{{round(math.log2(val))}}}$"


def _bytes_fmt(val, _):
    """Human-readable power-of-two byte size, e.g. 256 KiB / 1 MiB / 16 MiB."""
    if val <= 0:
        return ""
    for factor, unit in ((1024**4, "TiB"), (1024**3, "GiB"), (1024**2, "MiB"), (1024, "KiB")):
        if val >= factor:
            q = val / factor
            s = str(int(round(q))) if abs(q - round(q)) < 1e-6 else f"{q:g}"
            return f"{s} {unit}"
    return f"{int(round(val))} B"


def _series(rows, name):
    return [_f(r[name]) for r in rows]


def _runtime(xs, times):
    """(xs, ys) of the non-blank, positive points only."""
    out_x, out_y = [], []
    for x, t in zip(xs, times):
        if t is not None and t > 0.0:
            out_x.append(x)
            out_y.append(t)
    return out_x, out_y


def _draw_panel(ax, xs, lines, xlabel, title, xfmt=None):
    import matplotlib.ticker as ticker
    for label, times, style in lines:
        px, py = _runtime(xs, times)
        if px:
            ax.plot(px, py, style, label=label, markersize=5)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_pow2_fmt))
    if xfmt is not None:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(xfmt))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Operation Time (s)")
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.legend()


def plot_delayed(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style
    plot_style.apply()

    xs = [int(r["n"]) for r in rows]
    raw = _series(rows, "raw_read_s")

    fig, (ax_r, ax_w) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    _draw_panel(ax_r, xs, [
        ("in-mem delayed (DRAM)", _series(rows, "inmem_mmr_s"), "o-"),
        ("chunk-delayed",         _series(rows, "delayed_mmr_s"), "s-"),
        ("chunk-eager",           _series(rows, "eager_mmr_s"), "^-"),
        ("raw read (1 pass)",     raw, "k:"),
    ], "n (elements)", "map(x+1) | map(2x) | reduce(sum)  — read-bound")
    _draw_panel(ax_w, xs, [
        ("in-mem materialize (DRAM)", _series(rows, "inmem_fmm_s"), "o-"),
        ("chunk-delayed",            _series(rows, "delayed_fmm_s"), "s-"),
        ("chunk-eager",              _series(rows, "eager_fmm_s"), "^-"),
        ("raw read (1 pass)",        raw, "k:"),
    ], "n (elements)", "force(map(x+1) | map(2x))  — write-bound")

    fig.suptitle("Delayed sequences: in-memory vs chunk-eager vs chunk-delayed — scaling\n"
                 "(in-memory line stops where the input exceeds the RAM budget)")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


def plot_chunk_size(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style
    plot_style.apply()

    xs = [int(r["chunk_size_bytes"]) for r in rows]
    raw = _series(rows, "raw_s")
    n = rows[0]["n"] if rows else "?"

    fig, (ax_r, ax_w) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    _draw_panel(ax_r, xs, [
        ("chunk-delayed",     _series(rows, "delayed_mmr_s"), "s-"),
        ("chunk-eager",       _series(rows, "eager_mmr_s"), "^-"),
        ("raw read (1 pass)", raw, "k:"),
    ], "chunk size", "map(x+1) | map(2x) | reduce(sum)  — read-bound", xfmt=_bytes_fmt)
    _draw_panel(ax_w, xs, [
        ("chunk-delayed",     _series(rows, "delayed_fmm_s"), "s-"),
        ("chunk-eager",       _series(rows, "eager_fmm_s"), "^-"),
        ("raw read (1 pass)", raw, "k:"),
    ], "chunk size", "force(map(x+1) | map(2x))  — write-bound", xfmt=_bytes_fmt)

    fig.suptitle(f"Chunk-size sensitivity: chunk-eager vs chunk-delayed  (n={n})\n"
                 "(read-bound map|map|reduce vs write-bound force(map|map))")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


def plot_example(rows, entry, path):
    """Single-panel log-log plot of an example's runtime vs input size.

    Two series, styled like plot_delayed: the in-memory parlaylib baseline
    (which stops at the RAM cliff — blank CSV fields are dropped) and the
    out-of-core chunk implementation.  The x-axis is the uniform input size in
    bytes (see size_to_n), so examples with different element sizes are directly
    comparable.
    Two series by default, styled like plot_delayed: the in-memory parlaylib
    baseline (which stops at the RAM cliff — blank CSV fields are dropped) and
    the out-of-core chunk implementation.  An entry with `extra_series` plots
    additional implementations of the same operation alongside them (e.g.
    external_random_shuffle's second out-of-core method).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style
    plot_style.apply()

    xs = [int(r["input_bytes"]) for r in rows]
    # Default: out-of-core chunk impl vs in-memory parlaylib baseline (which
    # stops at the RAM cliff).  An entry may override the two series labels and
    # suppress the RAM-cliff note when the "baseline" series is itself
    # out-of-core (e.g. external_samplesort_vs_peter compares two disk sorts).
    base_label, cmp_label = entry.get(
        "series_labels", ("in-mem parlaylib (DRAM)", "out-of-core (chunk)"))
    subtitle = "" if entry.get("no_ram_cliff") else \
        "\n(In-Memory Goes to DRAM Limit)"
    lines = [
        (base_label, _series(rows, entry["inmem_col"]), "o-"),
        (cmp_label, _series(rows, entry.get("time_col", "time_s")), "s-"),
    ]
    lines += [(label, _series(rows, col), style)
              for col, label, style in entry.get("extra_series", [])]
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    # x is the input size in bytes (see size_to_n), ticked in binary units
    # (KiB/MiB/GiB/TiB) since the sweep is parameterized by input size.
    _draw_panel(ax, xs, lines, entry["xlabel"], entry["title"] + subtitle,
                xfmt=_bytes_fmt)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="run both benchmarks")
    ap.add_argument("--delayed", action="store_true", help="run the delayed-scale sweep")
    ap.add_argument("--chunk-size", action="store_true", help="run the chunk-size sweep")
    ap.add_argument("--examples", action="store_true",
                    help="run the examples sweep (all registered examples)")
    ap.add_argument("--example", default="",
                    help="run only these example(s) by name (comma/space-separated, "
                         f"e.g. 'external_samplesort'); implies --examples. "
                         f"choices: {', '.join(e['name'] for e in EXAMPLES)}")
    ap.add_argument("--outdir", default=os.environ.get("BENCH_OUTDIR", "results"),
                    help="parent dir for the timestamped run (default: results)")
    ap.add_argument("--n-values", default=os.environ.get("BENCH_N_VALUES", DEFAULT_N_VALUES),
                    help="delayed-scale n sweep (space-separated, e.g. '1M 8M 64M')")
    ap.add_argument("--chunk-sizes",
                    default=os.environ.get("BENCH_CHUNK_SIZES", DEFAULT_CHUNK_SIZES),
                    help="chunk-size sweep (space-separated bytes, e.g. '512KiB 4MiB' "
                         "or '2^19 2^22')")
    ap.add_argument("--n", default=os.environ.get("BENCH_CHUNK_N", DEFAULT_CHUNK_N),
                    help="fixed n for the chunk-size sweep (default: 32M)")
    ap.add_argument("--example-sizes",
                    default=os.environ.get("BENCH_EXAMPLE_SIZES", DEFAULT_EXAMPLE_SIZES),
                    help="examples input-size sweep, in BYTES (space-separated, e.g. "
                         "'256MiB 1GiB' or '2^28 2^30'); converted per example to its "
                         "element count (see size_to_n) — for an 8-byte-element example "
                         "'2^30' is the point that sorts 2^27 keys")
    ap.add_argument("--example-n-values",
                    default=os.environ.get("BENCH_EXAMPLE_N_VALUES", ""),
                    help="examples sweep by element count (n) directly, bypassing "
                         "--example-sizes/size_to_n entirely (space-separated, e.g. "
                         "'2^32 2^33 2^34'); each n is passed to the binary exactly as "
                         "given (no chunk-grid rounding). Takes precedence over "
                         "--example-sizes when non-empty")
    ap.add_argument("--ssd-args", default=os.environ.get("BENCH_SSD_ARGS", ""),
                    help="extra global flags passed to each binary (e.g. '--num_ssd=4')")
    ap.add_argument("--fstrim-glob",
                    default=os.environ.get("BENCH_FSTRIM_GLOB", DEFAULT_FSTRIM_GLOB),
                    help="glob of mounts to fstrim once at startup (default: /mnt/ssd*)")
    ap.add_argument("--no-fstrim", action="store_true",
                    help="disable the startup fstrim")
    ap.add_argument("--no-clean", action="store_true",
                    help="leave bench data files on the mounts (default: clear between points)")
    args = ap.parse_args()

    # --example NAME[,NAME...] selects a subset of the registry (and implies the
    # examples sweep); with no --example the sweep runs every registered example.
    selected = [x for x in re.split(r"[,\s]+", args.example) if x]
    known = {e["name"] for e in EXAMPLES}
    for name in selected:
        if name not in known:
            ap.error(f"unknown --example {name!r}; choices: {', '.join(sorted(known))}")
    examples_to_run = [e for e in EXAMPLES if not selected or e["name"] in selected]

    do_delayed = args.all or args.delayed
    do_chunk = args.all or args.chunk_size
    do_examples = args.examples or bool(selected)   # opt-in only; not part of --all
    if not (do_delayed or do_chunk or do_examples):
        ap.error("nothing to run: pass --all, --delayed, --chunk-size, "
                 "--examples, and/or --example NAME")

    extra = args.ssd_args.split() if args.ssd_args else []
    n_values = [parse_count(x) for x in args.n_values.split()]
    chunk_sizes = [parse_bytes(x) for x in args.chunk_sizes.split()]
    chunk_n = parse_count(args.n)
    example_sizes = [parse_bytes(x) for x in args.example_sizes.split()]
    example_n_values = ([parse_count(x) for x in args.example_n_values.split()]
                        if args.example_n_values.strip() else None)
    fstrim_enabled = not args.no_fstrim
    clear_enabled = not args.no_clean

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(REPO_ROOT, args.outdir, stamp)
    os.makedirs(outdir, exist_ok=True)
    print(f"Run directory: {outdir}\n")

    # Start from clean drives, then trim once up front (fstrim can be slow on
    # real SSDs, so we don't repeat it between points); both no-ops on tmpfs.
    # The fstrim outcome is reported in the end-of-run summary.
    clear_bench_data(args.fstrim_glob, clear_enabled)
    fstrim_note = fstrim_mounts(args.fstrim_glob, fstrim_enabled)

    if do_delayed:
        print("######## delayed scale ########")
        rows = run_delayed(n_values, extra, args.fstrim_glob, clear_enabled)
        write_csv(os.path.join(outdir, "delayed_scale.csv"), DELAYED_COLS, rows)
        plot_delayed(rows, os.path.join(outdir, "delayed_scale.png"))

    if do_chunk:
        print("\n######## chunk size ########")
        rows = run_chunk_size(chunk_sizes, chunk_n, extra, args.fstrim_glob, clear_enabled)
        write_csv(os.path.join(outdir, "chunk_size.csv"), CHUNK_COLS, rows)
        plot_chunk_size(rows, os.path.join(outdir, "chunk_size.png"))

    warnings = []
    if do_examples:
        for entry in examples_to_run:
            print(f"\n######## example: {entry['name']} ########")
            rows = run_example(entry, example_sizes, extra,
                               args.fstrim_glob, clear_enabled, warnings,
                               n_values=example_n_values)
            write_csv(os.path.join(outdir, f"{entry['name']}_scale.csv"),
                      ["input_bytes"] + entry["cols"], rows)
            # The CSV is the result; the plot is a convenience.  A plotting
            # failure (e.g. no matplotlib on a dev box) warns and lets the sweep
            # continue to the next example, like every other example problem —
            # it must not discard the timings we just spent the I/O to collect.
            try:
                plot_example(rows, entry,
                             os.path.join(outdir, f"{entry['name']}_scale.png"))
            except Exception as exc:
                warnings.append(f"{entry['name']}: plotting failed ({exc}); "
                                "CSV was written")
                print(f"  !!! plotting failed ({exc}); CSV was written", flush=True)

    # ── end-of-run summary — repeated here (and warnings persisted next to the
    # results) so problems can't get lost in the sweep output above.
    print("\n======== run summary ========")
    if fstrim_note:
        print(f"  {fstrim_note}")
    if warnings:
        print(f"  !!! {len(warnings)} example warning(s) "
              "(sweep continued past them):")
        for w in warnings:
            print(f"  !!!   {w}")
        wpath = os.path.join(outdir, "warnings.txt")
        with open(wpath, "w") as f:
            f.write("\n".join(warnings) + "\n")
        print(f"  !!! (also written to {wpath})")
    else:
        print("  no example warnings")
    print(f"\nDone. Results in {outdir}")


if __name__ == "__main__":
    main()
