#!/usr/bin/env python3
"""Standalone sweep/plot driver for the two exploratory benchmarks:

  * filter compare   (benchmarks/filter_compare.cpp -> bin/filterCompare)
      Sweeps n. Compares eager (DensePack) ChunkFilter vs the pre-node-tree
      delayed filter (benchmarks/old_filter/old_chunk_delayed.h, frozen at
      commit c5c3406) vs the current node/Planner-based delayed filter.

  * zip depth        (benchmarks/zip_depth_compare.cpp -> bin/zipDepthCompare)
      A single run self-sweeps the zip chain depth k (1,2,4,8,...) internally
      and prints one CSV line per depth; this script just builds, runs once,
      and plots all the depths from that one run.

Deliberately separate from run_benches.py: no EXAMPLES registry entry, never
invoked by `make bench` / `make bench-full`. Mirrors that script's `make` /
CSV-parsing / matplotlib log-log-panel conventions on a smaller scale.
"""

import argparse
import glob
import os
import re
import subprocess
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINDIR = os.path.join(REPO_ROOT, "bin")

DEFAULT_N_VALUES = "1M 2M 4M 8M 16M 32M 64M"
DEFAULT_ZIP_N = "1M"
DEFAULT_MAX_K = 64

FILTER_COLS = ["n", "in_bytes", "eager_s", "old_delayed_s", "new_delayed_s", "agree"]
ZIP_COLS = ["k", "n", "reduce_s", "throughput_gb_s", "agree"]

BENCH_FILE_GLOBS = ("perm[0-9]*", "bw_fc_*", "zsrc*")


def parse_count(s):
    """Element count: `2^k`/`2**k`, decimal suffixes K=1e3/M=1e6/G=1e9, or raw."""
    s = s.strip()
    m = re.fullmatch(r"2(?:\^|\*\*)(\d+)", s)
    if m:
        return 2 ** int(m.group(1))
    m = re.fullmatch(r"(\d+)([kKmMgG]?)", s)
    if not m:
        raise ValueError(f"bad count {s!r}")
    mult = {"": 1, "k": 10**3, "m": 10**6, "g": 10**9}[m.group(2).lower()]
    return int(m.group(1)) * mult


def clear_bench_data(glob_pat, enabled):
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


def make(target):
    print(f"  $ make {target}", flush=True)
    r = subprocess.run(["make", target], cwd=REPO_ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout)
        sys.exit(f"make {target} failed (exit {r.returncode})")


def run_binary(path, args):
    """Run a benchmark binary; return the list of CSV field-lists, one per
    'CSV,' line printed (filter emits exactly one; zip-depth emits one per
    depth). Aborts the whole run on a non-zero exit or missing CSV output —
    these are cross-substrate differential checks, same as run_benches.py's
    substrate benchmarks."""
    cmd = [path] + [str(a) for a in args]
    print(f"  $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=REPO_ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout, end="", flush=True)
    rows = [line[len("CSV,"):].split(",") for line in r.stdout.splitlines()
            if line.startswith("CSV,")]
    if r.returncode != 0:
        sys.exit(f"\n*** {os.path.basename(path)} exited {r.returncode} — aborting ***")
    if not rows:
        sys.exit(f"\n*** {os.path.basename(path)} printed no CSV line — aborting ***")
    return rows


def write_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r[c] for c in header) + "\n")
    print(f"  wrote {path}", flush=True)


def _draw_panel(ax, xs, lines, xlabel, title):
    import matplotlib.ticker as ticker
    import math
    for label, ys, style in lines:
        ax.plot(xs, ys, style, label=label, markersize=5)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"$2^{{{round(math.log2(v))}}}$" if v > 0 else ""))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("time (s)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend()


def run_filter(n_values, extra_args, clear_glob, clear_enabled, outdir):
    make("bin/filterCompare")
    binary = os.path.join(BINDIR, "filterCompare")
    rows = []
    for n in n_values:
        print(f"\n=== filter compare: n={n} ===", flush=True)
        fields = run_binary(binary, [n] + extra_args)[0]
        row = dict(zip(FILTER_COLS, fields))
        if row["agree"].strip() != "1":
            sys.exit(f"\n*** agree={row['agree']} at n={n} — aborting ***")
        rows.append(row)
        clear_bench_data(clear_glob, clear_enabled)
    write_csv(os.path.join(outdir, "filter_compare.csv"), FILTER_COLS, rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [int(r["n"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    _draw_panel(ax, xs, [
        ("eager (DensePack)", [float(r["eager_s"]) for r in rows], "^-"),
        ("old-delayed (pre-node-tree)", [float(r["old_delayed_s"]) for r in rows], "o-"),
        ("new-delayed (node/Planner)", [float(r["new_delayed_s"]) for r in rows], "s-"),
    ], "n (elements)", "filter(keep even): eager vs old-delayed vs new-delayed windowing")
    path = os.path.join(outdir, "filter_compare.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


def run_zip_depth(n, max_k, extra_args, clear_glob, clear_enabled, outdir):
    make("bin/zipDepthCompare")
    binary = os.path.join(BINDIR, "zipDepthCompare")
    print(f"\n=== zip depth sweep: n={n} max_k={max_k} ===", flush=True)
    fields_list = run_binary(binary, [n, max_k] + extra_args)
    rows = [dict(zip(ZIP_COLS, fields)) for fields in fields_list]
    for row in rows:
        if row["agree"].strip() != "1":
            sys.exit(f"\n*** agree={row['agree']} at k={row['k']} — aborting ***")
    clear_bench_data(clear_glob, clear_enabled)
    write_csv(os.path.join(outdir, "zip_depth.csv"), ZIP_COLS, rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [int(r["k"]) for r in rows]
    ys = [float(r["reduce_s"]) for r in rows]
    ref = [ys[0] * (k / xs[0]) for k in xs]   # linear-in-k reference from the k=1 point
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    _draw_panel(ax, xs, [
        ("chunk-delayed reduce(sum)", ys, "s-"),
        ("linear reference (k * t(k=1))", ref, "k:"),
    ], "k (number of zipped sequences / chain depth)",
       f"Zip chain depth scaling  (n={xs and rows[0]['n']} per source)")
    path = os.path.join(outdir, "zip_depth.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="run both sweeps")
    ap.add_argument("--filter", action="store_true", help="run the filter-compare sweep")
    ap.add_argument("--zip-depth", action="store_true", help="run the zip-depth sweep")
    ap.add_argument("--n-values", default=DEFAULT_N_VALUES,
                    help="filter-compare n sweep (space-separated, e.g. '1M 8M 64M')")
    ap.add_argument("--n", default=DEFAULT_ZIP_N,
                    help="elements per source for the zip-depth sweep (default: 1M)")
    ap.add_argument("--max-k", type=int, default=DEFAULT_MAX_K,
                    help="max number of zipped sequences (default: 64)")
    ap.add_argument("--outdir", default="results",
                    help="parent dir for the timestamped run (default: results)")
    ap.add_argument("--ssd-args", default="",
                    help="extra global flags passed to each binary (e.g. '--num_ssd=4')")
    ap.add_argument("--fstrim-glob", default="/mnt/ssd*",
                    help="glob of mounts to clear bench data from (default: /mnt/ssd*)")
    ap.add_argument("--no-clean", action="store_true",
                    help="leave bench data files on the mounts")
    args = ap.parse_args()

    do_filter = args.all or args.filter
    do_zip = args.all or args.zip_depth
    if not (do_filter or do_zip):
        ap.error("nothing to run: pass --all, --filter, and/or --zip-depth")

    extra = args.ssd_args.split() if args.ssd_args else []
    n_values = [parse_count(x) for x in args.n_values.split()]
    zip_n = parse_count(args.n)
    clear_enabled = not args.no_clean

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(REPO_ROOT, args.outdir, stamp)
    os.makedirs(outdir, exist_ok=True)
    print(f"Run directory: {outdir}\n")

    clear_bench_data(args.fstrim_glob, clear_enabled)

    if do_filter:
        print("######## filter compare ########")
        run_filter(n_values, extra, args.fstrim_glob, clear_enabled, outdir)

    if do_zip:
        print("\n######## zip depth ########")
        run_zip_depth(zip_n, args.max_k, extra, args.fstrim_glob, clear_enabled, outdir)

    print(f"\nDone. Results in {outdir}")


if __name__ == "__main__":
    main()
