#!/usr/bin/env python3
"""Sweep/plot driver for the synthetic work-per-element benchmark
(benchmarks/work_exponent_compare.cpp -> bin/workExponentCompare).

Every other benchmark in this repo times a real algorithm's fixed, trivial
per-element cost. This one instead isolates "work per element" (and,
separately, "number of full-sequence passes") as a free variable, independent
of any specific algorithm's semantics, so the crossover between I/O-bound,
balanced, and compute-bound regimes can be measured directly as a function of
n and a work exponent (const/log/sqrt/linear).

Two axes (see work_exponent_compare.cpp's header for the full rationale):
  elemwork  one ChunkMap call with a k-iteration busy loop per element --
            models arithmetic density growing within a single pass (as in
            fft.cpp's four-step transform).
  rounds    R back-to-back ChunkReduce passes over the same input -- models
            repeated full-sequence I/O passes (as in dc3's prefix-doubling
            rounds, convex_hull's split levels, external_bellman_ford_fast's
            rounds).

For each (axis, mode) combination the largest n in the sweep is profiled with
benchmarks/io_trace.py's disk/CPU sampler (same PLAID_TRACE=1 mechanism, same
trace_{throughput,cpu,drives}.png output) so the I/O-bound vs compute-bound
crossover is visible at the device level, not just inferred from wall-clock;
every point (traced or not) still contributes a work_exponent.csv row.

Deliberately separate from run_benches.py: no EXAMPLES registry entry (every
entry there assumes an in-memory baseline column this synthetic knob doesn't
have), never invoked by `make bench` / `make bench-full`. Reuses
benchmarks/io_trace.py's registry-independent internals by import (device
resolution, the diskstats/cpu sampler, trace CSV/plot writers) rather than
duplicating them.
"""

import argparse
import glob
import os
import re
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_trace as it  # sibling module; also imports run_benches as it.rb

REPO_ROOT = it.rb.REPO_ROOT
BINDIR = it.rb.BINDIR

DEFAULT_MODES = "const log sqrt"
DEFAULT_N_VALUES = "1M 4M 16M 64M 256M 1G 4G"
DEFAULT_LINEAR_N_VALUES = "256 1K 4K 16K 64K"
DEFAULT_K0 = 16

WORK_EXPONENT_COLS = ["axis", "mode", "n", "k0", "k_effective", "total_ops",
                      "build_s", "op_s", "throughput_gb_s", "agree"]
TRACE_EXTRA_COLS = ["avg_util_pct", "avg_cpu_pct"]

BENCH_FILE_GLOBS = ("iota[0-9]*", "bw_we_*")


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


def _parse_csv_row(stdout, desc):
    fields = None
    for line in stdout.splitlines():
        if line.startswith("CSV,"):
            fields = line[len("CSV,"):].split(",")
    if fields is None or len(fields) != len(WORK_EXPONENT_COLS):
        sys.exit(f"\n*** no (or malformed) CSV line for {desc} — aborting ***")
    row = dict(zip(WORK_EXPONENT_COLS, fields))
    if row["agree"].strip() != "1":
        sys.exit(f"\n*** agree={row['agree']} for {desc} — aborting ***")
    return row


def run_binary(path, args, desc):
    """Run once, untraced; return the parsed CSV row. Aborts the whole sweep
    on a non-zero exit, missing CSV line, or agree != 1 -- same differential-
    test-style hard stop as every other benchmark driver in this repo."""
    cmd = [path] + [str(a) for a in args]
    print(f"  $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    print(r.stdout, end="", flush=True)
    if r.returncode != 0:
        sys.exit(f"\n*** {os.path.basename(path)} exited {r.returncode} ({desc}) — aborting ***")
    return _parse_csv_row(r.stdout, desc)


def run_traced_point(binary, args, devices, interval, outdir, axis, mode, n, point_idx, desc):
    """Run under io_trace.py's sampler; write a per-point trace_work_.../
    trace_{throughput,cpu,drives}.png (identical shape to io_trace.py's own
    output) and fold the traced op-window's avg util/CPU into the CSV row."""
    samples, markers, stdout = it.run_traced(binary, args, devices, interval)
    row = _parse_csv_row(stdout, desc)

    if len(samples) < 2:
        print(f"  !!! too few samples for {desc} (run too short for --interval); "
              "trace skipped, perf row still recorded", flush=True)
        return row

    ser_full = it.compute_series(samples, devices)
    t0 = samples[0][0]
    last_end = it.last_end_marker_mono(markers)
    ser = it.slice_ser(ser_full, devices, t0, last_end) if last_end is not None else ser_full

    label = f"{axis}_{mode}"
    trace_dir = os.path.join(outdir, f"trace_work_{axis}_{mode}_{n}_{point_idx}")
    os.makedirs(trace_dir, exist_ok=True)
    print(f"Trace directory: {trace_dir}", flush=True)
    it.write_trace_csv(os.path.join(trace_dir, "trace.csv"), ser, devices, t0)
    it.plot_trace(ser, markers, devices, t0, os.path.join(trace_dir, "trace.png"))

    start_mono = next((mono for lbl, mono in markers if lbl == f"op_start_{label}"), None)
    end_mono = next((mono for lbl, mono in markers if lbl == f"op_end_{label}"), None)
    if start_mono is not None and end_mono is not None:
        stats = it.window_series_stats(ser_full, t0, start_mono, end_mono)
        row["avg_util_pct"] = f"{stats['avg_util_pct']:.3f}"
        row["avg_cpu_pct"] = f"{stats['avg_cpu_pct']:.3f}"
    return row


def run_sweep(axes, modes, n_values, linear_n_values, k0, trace_mode, interval,
             mount_glob, outdir, ssd_args, clean):
    make("bin/workExponentCompare")
    binary = os.path.join(BINDIR, "workExponentCompare")

    devices = []
    if trace_mode != "none":
        devices, mapping = it.resolve_devices(mount_glob)
        if devices:
            print(f"Resolved {len(mapping)} mount(s) -> {len(devices)} device(s): "
                  f"{', '.join(devices)}", flush=True)
        else:
            print(f"  !!! no block devices resolved from {mount_glob!r} — disk "
                  "panels will be EMPTY (tmpfs dev box? wrong --mount-glob?). "
                  "CPU still recorded.", flush=True)

    rows = []
    for axis in axes:
        for mode in modes:
            values = linear_n_values if mode == "linear" else n_values
            if not values:
                continue
            largest_idx = values.index(max(values))
            for idx, n in enumerate(values):
                trace_this = trace_mode == "all" or (trace_mode == "largest" and idx == largest_idx)
                desc = f"axis={axis} mode={mode} n={n}"
                print(f"\n--- {desc}  [{idx + 1}/{len(values)}]"
                      f" {'[traced]' if trace_this else '[perf only]'} ---", flush=True)
                args = ssd_args + [n, axis, mode, k0]
                if trace_this:
                    row = run_traced_point(binary, args, devices, interval, outdir,
                                           axis, mode, n, idx, desc)
                else:
                    row = run_binary(binary, args, desc)
                rows.append(row)
                clear_bench_data(mount_glob, clean)
    return rows


def write_csv(path, rows):
    header = WORK_EXPONENT_COLS + TRACE_EXTRA_COLS
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r.get(c, "") for c in header) + "\n")
    print(f"  wrote {path}", flush=True)


def _series_by_mode(rows, axis):
    """{mode: [(n, op_s, throughput_gb_s, total_ops, avg_util_pct, avg_cpu_pct), ...]}
    sorted by n, for one axis."""
    out = {}
    for r in rows:
        if r["axis"] != axis:
            continue
        out.setdefault(r["mode"], []).append((
            int(r["n"]), float(r["op_s"]), float(r["throughput_gb_s"]),
            int(r["total_ops"]),
            float(r["avg_util_pct"]) if r.get("avg_util_pct") else None,
            float(r["avg_cpu_pct"]) if r.get("avg_cpu_pct") else None,
        ))
    for mode in out:
        out[mode].sort(key=lambda t: t[0])
    return out


def plot_vs_n(rows, axes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style
    plot_style.apply()

    fig, axs = plt.subplots(2, len(axes), figsize=(6.5 * len(axes), 9),
                            constrained_layout=True, squeeze=False)
    for col, axis in enumerate(axes):
        by_mode = _series_by_mode(rows, axis)
        ax_time, ax_thr = axs[0][col], axs[1][col]
        for mode, pts in by_mode.items():
            ns = [p[0] for p in pts]
            ax_time.plot(ns, [p[1] for p in pts], "o-", label=mode, markersize=5)
            ax_thr.plot(ns, [p[2] for p in pts], "o-", label=mode, markersize=5)
        for ax, ylabel, title in (
            (ax_time, "op time (s)", f"{axis}: wall-clock vs n"),
            (ax_thr, "throughput (GB/s, eff. input)", f"{axis}: throughput vs n"),
        ):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("n (elements)")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


def plot_vs_total_ops(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style
    plot_style.apply()

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    markers = {"elemwork": "o", "rounds": "^"}
    seen = set()
    for r in rows:
        axis, mode = r["axis"], r["mode"]
        key = (axis, mode)
        label = f"{axis}/{mode}" if key not in seen else None
        seen.add(key)
        ax.scatter(int(r["total_ops"]), float(r["op_s"]),
                  marker=markers.get(axis, "o"), label=label, s=30)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total ops (k_effective * n for elemwork, k_effective for rounds)")
    ax.set_ylabel("op time (s)")
    ax.set_title("wall-clock vs total ops — both axes on one shared x-axis")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


def plot_util_cpu(rows, axes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style
    plot_style.apply()

    fig, axs = plt.subplots(1, len(axes), figsize=(6.5 * len(axes), 4.5),
                            constrained_layout=True, squeeze=False)
    any_data = False
    for col, axis in enumerate(axes):
        by_mode = _series_by_mode(rows, axis)
        ax = axs[0][col]
        for mode, pts in by_mode.items():
            traced = [p for p in pts if p[4] is not None]
            if not traced:
                continue
            any_data = True
            ns = [p[0] for p in traced]
            ax.plot(ns, [p[4] for p in traced], "o-", label=f"{mode} drive %util")
            ax.plot(ns, [p[5] for p in traced], "s--", label=f"{mode} CPU %")
        ax.set_xscale("log")
        ax.set_ylim(0, 105)
        ax.set_xlabel("n (elements)")
        ax.set_ylabel("percent")
        ax.set_title(f"{axis}: traced-point drive %util vs CPU %")
        ax.legend(fontsize=8)
    if not any_data:
        fig.text(0.5, 0.5, "no traced points (--no-trace, or tmpfs dev box)",
                 ha="center", va="center")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", default="both", choices=["elemwork", "rounds", "both"],
                    help="which axis/axes to sweep (default: both)")
    ap.add_argument("--modes", default=DEFAULT_MODES,
                    help="space-separated modes to sweep, e.g. 'const log sqrt' "
                         "(default) or 'const log sqrt linear'. 'linear' uses "
                         "--linear-n-values instead of --n-values.")
    ap.add_argument("--n-values", default=DEFAULT_N_VALUES,
                    help="main multi-decade n sweep (space-separated, e.g. "
                         "'1M 4M 16M 64M 256M 1G 4G')")
    ap.add_argument("--linear-n-values", default=DEFAULT_LINEAR_N_VALUES,
                    help="separate, small n sweep used only for mode=linear "
                         "(deliberately intractable at scale)")
    ap.add_argument("--k0", type=int, default=DEFAULT_K0,
                    help=f"base op-count multiplier (default {DEFAULT_K0})")
    ap.add_argument("--trace", choices=["largest", "all", "none"], default="largest",
                    help="disk/CPU-sample the largest n per (axis,mode) [default], "
                         "every point (--trace all), or none")
    ap.add_argument("--interval", type=float, default=0.1,
                    help="sampling interval in seconds (default 0.1)")
    ap.add_argument("--mount-glob", default="/mnt/ssd*",
                    help="glob of SSD mounts to resolve to block devices for tracing")
    ap.add_argument("--outdir", default="results",
                    help="parent dir for the timestamped run (default: results)")
    ap.add_argument("--ssd-args", default="",
                    help="extra global flags passed to the binary (e.g. '--num_ssd=4')")
    ap.add_argument("--no-clean", action="store_true",
                    help="leave bench data files on the mounts")
    args = ap.parse_args()

    axes = ["elemwork", "rounds"] if args.axis == "both" else [args.axis]
    modes = args.modes.split()
    n_values = [parse_count(x) for x in args.n_values.split()]
    linear_n_values = [parse_count(x) for x in args.linear_n_values.split()]
    ssd_args = args.ssd_args.split() if args.ssd_args else []
    clean = not args.no_clean

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(REPO_ROOT, args.outdir, stamp)
    os.makedirs(outdir, exist_ok=True)
    print(f"Run directory: {outdir}\n")

    clear_bench_data(args.mount_glob, clean)

    rows = run_sweep(axes, modes, n_values, linear_n_values, args.k0, args.trace,
                     args.interval, args.mount_glob, outdir, ssd_args, clean)

    write_csv(os.path.join(outdir, "work_exponent.csv"), rows)
    plot_vs_n(rows, axes, os.path.join(outdir, "work_exponent_vs_n.png"))
    plot_vs_total_ops(rows, os.path.join(outdir, "work_exponent_vs_total_ops.png"))
    plot_util_cpu(rows, axes, os.path.join(outdir, "work_exponent_util_cpu.png"))

    print(f"\nDone. Results in {outdir}")


if __name__ == "__main__":
    main()
