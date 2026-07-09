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
import subprocess
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINDIR = os.path.join(REPO_ROOT, "bin")

# Defaults tuned for a ~5 GiB tmpfs (peak disk use in a step is ~3*n*8 bytes).
DEFAULT_N_VALUES = "1M 2M 4M 8M 16M 32M 64M"          # element counts
DEFAULT_CHUNK_SIZES = "512KiB 1MiB 2MiB 4MiB 8MiB 16MiB"
DEFAULT_CHUNK_N = "32M"                                # fixed n for chunk sweep
DEFAULT_EXAMPLE_N_VALUES = "2^24 2^25 2^26 2^27 2^28 2^29 2^30"  # examples sweep (dev/tmpfs)
DEFAULT_FSTRIM_GLOB = "/mnt/ssd*"

# ── examples registry ───────────────────────────────────────────────────────
# Each example is a dual-purpose binary (bin/<name>Example) that prints a
# `CSV,<cols...>` line the sweep greps.  `cols` names those fields in order;
# `time_col`/`inmem_col` pick the plotted out-of-core / in-memory series;
# `data_globs` lists the per-mount globs of files the example leaves on the
# drives (cleared between sweep points).  Add a new example by appending one
# entry (and, if it lives in examples/external/, an explicit Makefile rule).
EXAMPLES = [
    {"name": "primes", "target": "bin/primesExample",
     "cols": ["n", "time_s", "inmem_time_s", "count", "throughput_gb_s"],
     "inmem_col": "inmem_time_s",
     "xlabel": "n (sieve range)",
     "title": "Prime sieve: out-of-core (ChunkFlatTabulate) vs in-mem parlaylib",
     "data_globs": ["primes[0-9]*"]},
    # kmpExample sweeps n with the pattern length m at its constant built-in
    # default; the plotted time is the search pass only (text build excluded).
    {"name": "kmp", "target": "bin/kmpExample",
     "cols": ["n", "m", "build_s", "search_s", "inmem_search_s", "count",
              "throughput_gb_s"],
     "time_col": "search_s", "inmem_col": "inmem_search_s",
     "xlabel": "n (text length, chars)",
     "title": "KMP search: out-of-core (ChunkKmp) vs in-mem parlaylib",
     "data_globs": ["kmp_*"]},
    # rabin_karpExample: same driver shape as kmp (constant m, sweep n),
    # rolling-hash search instead of the KMP automaton.
    {"name": "rabin_karp", "target": "bin/rabin_karpExample",
     "cols": ["n", "m", "build_s", "search_s", "inmem_search_s", "count",
              "throughput_gb_s"],
     "time_col": "search_s", "inmem_col": "inmem_search_s",
     "xlabel": "n (text length, chars)",
     "title": "Rabin-Karp search: out-of-core (ChunkRabinKarp) vs in-mem parlaylib",
     "data_globs": ["rk_*"]},
    # kth_smallestExample sweeps n with k at the median (n/2); the plotted time
    # is the selection pass only (input build excluded).  Its recursion leaves
    # id_/flags_/next_ intermediates in addition to the kth_in input.
    {"name": "kth_smallest", "target": "bin/kth_smallestExample",
     "cols": ["n", "k", "build_s", "select_s", "inmem_select_s", "result",
              "throughput_gb_s"],
     "time_col": "select_s", "inmem_col": "inmem_select_s",
     "xlabel": "n (number of keys)",
     "title": "kth-smallest: out-of-core (ChunkSequenceOps) vs in-mem parlaylib",
     "data_globs": ["kth_in*", "id_*", "flags_*", "next_*"]},
    # external_samplesortExample sweeps n; the plotted time is the sort pass only
    # (input build excluded).  Its recursion leaves ss_id_/ss_bucket_/ss_base_/
    # ss_deg_ intermediates (the sorted output references the bucket files) in
    # addition to the ss_in input.
    {"name": "external_samplesort", "target": "bin/external_samplesortExample",
     "cols": ["n", "build_s", "sort_s", "inmem_sort_s", "throughput_gb_s"],
     "time_col": "sort_s", "inmem_col": "inmem_sort_s",
     "xlabel": "n (number of keys)",
     "title": "sample sort: out-of-core (ChunkSequenceOps) vs in-mem parlaylib",
     "data_globs": ["ss_in*", "ss_id_*", "ss_bucket_*", "ss_base_*", "ss_deg_*"]},

    # fitmem_kth_smallestExample: same driver shape as kth_smallest, but the
    # single-level "fitmem" variant (one bucketing round, then select the winning
    # bucket in DRAM).  Its intermediates are fk_id_/fk_next_ alongside fk_in.
    {"name": "fitmem_kth_smallest", "target": "bin/fitmem_kth_smallestExample",
     "cols": ["n", "k", "build_s", "select_s", "inmem_select_s", "result",
              "throughput_gb_s"],
     "time_col": "select_s", "inmem_col": "inmem_select_s",
     "xlabel": "n (number of keys)",
     "title": "fitmem kth-smallest: out-of-core (ChunkSequenceOps) vs in-mem parlaylib",
     "data_globs": ["fk_in*", "fk_id_*", "fk_next_*"]},

    # fitmem_sortExample: same driver shape as external_samplesort, but the
    # single-level "fitmem" variant (one bucketing round, then each bucket is
    # sorted directly in DRAM).  Its intermediates are fs_id_/fs_bucket_/fs_base_/
    # fs_sorted_ (the sorted output references the fs_sorted_ files) alongside fs_in.
    {"name": "fitmem_sort", "target": "bin/fitmem_sortExample",
     "cols": ["n", "build_s", "sort_s", "inmem_sort_s", "throughput_gb_s"],
     "time_col": "sort_s", "inmem_col": "inmem_sort_s",
     "xlabel": "n (number of keys)",
     "title": "fitmem sample sort: out-of-core (ChunkSequenceOps) vs in-mem parlaylib",
     "data_globs": ["fs_in*", "fs_id_*", "fs_bucket_*", "fs_base_*", "fs_sorted_*"]},

    # external_linefitExample sweeps n; the plotted time is the fit itself
    # (input build excluded).  Both passes are fully delayed, so the fit leaves
    # no intermediates beyond the lf_x/lf_y inputs.
    {"name": "external_linefit", "target": "bin/external_linefitExample",
     "cols": ["n", "build_s", "fit_s", "inmem_fit_s", "offset", "slope",
              "throughput_gb_s"],
     "time_col": "fit_s", "inmem_col": "inmem_fit_s",
     "xlabel": "n (number of points)",
     "title": "line fit: out-of-core (ChunkSequenceOps) vs in-mem parlaylib",
     "data_globs": ["lf_x*", "lf_y*"]},

    # bigint_addExample sweeps n (limb count); the plotted time is the add pass
    # only (operand build excluded).  Baseline is our own parlaylib reference.
    {"name": "bigint_add", "target": "bin/bigint_addExample",
     "cols": ["n", "build_s", "add_s", "inmem_add_s", "result_limbs",
              "throughput_gb_s"],
     "time_col": "add_s", "inmem_col": "inmem_add_s",
     "xlabel": "n (64-bit limbs)",
     "title": "big-integer add: out-of-core (ChunkSequenceOps) vs in-mem parlaylib",
     "data_globs": ["bi_a*", "bi_b*", "bi_sum*"]},

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
     "xlabel": "n (number of keys)",
     "title": "cut / slice: out-of-core (ChunkSequenceOps) vs in-mem parlaylib",
     "data_globs": ["cut_in*", "cut_out*"]},

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
    """Byte size: binary suffixes KiB/MiB/GiB (or K/M/G = *1024^x), or raw."""
    s = s.strip()
    m = re.fullmatch(r"(\d+)\s*([kKmMgG]?)(i?[bB]?)", s)
    if not m:
        raise ValueError(f"bad byte size {s!r}")
    mult = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3}[m.group(2).lower()]
    return int(m.group(1)) * mult


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
    """Best-effort unlink of leftover bench files under every mount in glob_pat."""
    if not enabled:
        return
    removed = 0
    for m in sorted(glob.glob(glob_pat)):
        for pat in BENCH_FILE_GLOBS:
            for f in glob.glob(os.path.join(m, pat)):
                try:
                    os.unlink(f)
                    removed += 1
                except OSError:
                    pass
    if removed:
        print(f"  cleared {removed} leftover bench files", flush=True)


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
        why = ("killed by a signal (crash)" if r.returncode < 0
               else "correctness mismatch or error")
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
def run_example(entry, n_values, extra_args, clear_glob, clear_enabled, warnings):
    """Sweep one example over n_values; return parsed rows.

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
    for n in n_values:
        print(f"\n=== example {entry['name']}: n={n} ===", flush=True)
        fields, problem = run_binary(binary, [n] + extra_args, fatal=False)
        if problem:
            w = (f"example {entry['name']} at n={n}: {problem}"
                 + ("" if fields else " — point dropped"))
            print(f"  !!! {w}", flush=True)
            warnings.append(w)
        if fields:
            rows.append(dict(zip(entry["cols"], fields)))
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
    for factor, unit in ((1024**3, "GiB"), (1024**2, "MiB"), (1024, "KiB")):
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
    ax.set_ylabel("operation time (s)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend()


def plot_delayed(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    """Single-panel log-log plot of an example's runtime vs n.

    Two series, styled like plot_delayed: the in-memory parlaylib baseline
    (which stops at the RAM cliff — blank CSV fields are dropped) and the
    out-of-core chunk implementation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [int(r["n"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    _draw_panel(ax, xs, [
        ("in-mem parlaylib (DRAM)", _series(rows, entry["inmem_col"]), "o-"),
        ("out-of-core (chunk)", _series(rows, entry.get("time_col", "time_s")), "s-"),
    ], entry["xlabel"],
       entry["title"] + "\n(in-mem line stops where the input exceeds the RAM budget)",
       xfmt=_pow2_fmt)
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
                    help="chunk-size sweep (space-separated, e.g. '512KiB 4MiB')")
    ap.add_argument("--n", default=os.environ.get("BENCH_CHUNK_N", DEFAULT_CHUNK_N),
                    help="fixed n for the chunk-size sweep (default: 32M)")
    ap.add_argument("--example-n-values",
                    default=os.environ.get("BENCH_EXAMPLE_N_VALUES", DEFAULT_EXAMPLE_N_VALUES),
                    help="examples n sweep (space-separated, e.g. '2^24 2^28 2^30')")
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
    example_n_values = [parse_count(x) for x in args.example_n_values.split()]
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
            rows = run_example(entry, example_n_values, extra,
                               args.fstrim_glob, clear_enabled, warnings)
            write_csv(os.path.join(outdir, f"{entry['name']}_scale.csv"),
                      entry["cols"], rows)
            plot_example(rows, entry, os.path.join(outdir, f"{entry['name']}_scale.png"))

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
