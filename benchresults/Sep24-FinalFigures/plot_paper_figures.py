#!/usr/bin/env python3
"""Paper figures for the two-column Overleaf layout, from the CSVs frozen in
this directory (no binaries, no `make`, no results/ lookups).

Each figure is drawn at its real printed width -- the summary at \\textwidth
(figure*), the trace and ablation at \\columnwidth -- so the rcParams point
sizes below are the sizes that actually print; include them with
\\includegraphics[width=\\textwidth] / [width=\\columnwidth] and nothing is
rescaled.

  paper_summary.{pdf,png}   out-of-core / DRAM time ratio per entry
  paper_summary_core.{...}  the same without fft / bellman-ford / convex-hull
  paper_trace.{pdf,png}     bigint_add 4 TiB: CPU/util over throughput,
                            cropped to the algorithm window
  paper_ablation.{pdf,png}  bigint_add drive-placement policy scaling

  usage:
    python3 plot_paper_figures.py
    python3 plot_paper_figures.py --only trace --smooth 0
"""

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "benchmarks"))
import plot_style  # noqa: E402  (shared palette + serif look)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

SUMMARY_CSV = os.path.join(HERE, "summary_figure.csv")
TRACE_CSV = os.path.join(HERE, "trace_bigint_add_4TiB_0", "trace.csv")
# The ablation is stitched from four sweeps under ablation_data/, each carrying
# whichever substrate(s) it measured (add_s = fused delayed, eager_add_s =
# materialized eager):
#   141415  all three policies, both substrates, 128 / 256 / 512 GiB
#   102303  random + round_robin, both substrates, 1 / 2 / 4 TiB
#   212756  blocked, delayed only, 1 TiB     (its eager arm is not measured
#   201015  blocked, delayed only, 2 TiB      past 512 GiB)
#   230836  random,  delayed only, 8 TiB     (the deepest point measured)
# Merged per (policy, size, column) keeping the last non-blank value, so a
# sweep that left a column empty never erases one that filled it.  Each curve
# then spans exactly what exists for it.
ABLATION_CSVS = [
    os.path.join(HERE, "ablation_data", d, "drive_policy_bigint_add.csv")
    for d in ("20260925-141415-drive-policy", "20260925-102303-drive-policy",
              "20260925-212756-drive-policy", "20260925-201015-drive-policy",
              "20260925-230836-drive-policy")
]

# Printed widths (inches) for a typical two-column paper.
TEXT_WIDTH = 7.0
COLUMN_WIDTH = 3.33

# The trace's phase markers only ever reached io_trace.py's stdout, not
# trace.csv, so the window is pinned here, read off the throughput columns:
# staging's writes stop at ~212.5 s (the scan's read-only pass 1 ramps up
# after a ~2 s drive settle), the fused add+force (read+write) starts at
# ~282 s, and End lands at ~397.7 s.
TRACE_T0 = 212.5
TRACE_T1 = 400.0
PHASE_SPLIT = 282.0

# Summary bar order (summary_figure.csv `name` column): primitives, then
# examples.  Every CSV row must appear here exactly once (asserted).
SUMMARY_PRIMITIVES = [
    "map", "reduce", "filter", "scan", "tabulate", "group_by_index",
    "histogram_by_index", "pack", "random_shuffle", "reverse",
]
SUMMARY_EXAMPLES = [
    "bigint_add", "linefit", "kmp", "rabin_karp", "primes", "samplesort",
    "kth_smallest", "fft", "bellman_ford_sparse", "convex_hull",
]
SUMMARY_LABELS = {"bellman_ford_sparse": "bellman-ford",
                  "group_by_index": "group-by",
                  "histogram_by_index": "histogram-by"}
# The reduced variant (paper_summary_core) drops these.
SUMMARY_REDUCED_EXCLUDE = ("fft", "bellman_ford_sparse", "convex_hull")

# Same order/colors as drive_policy_ablation.py's plot, so the figure matches.
POLICIES = [("random", "blue"), ("round_robin", "orange"), ("blocked", "aqua")]

PAPER_RC = {
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 0.8,
    "lines.markersize": 3,
    "lines.markeredgewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "axes.labelpad": 2,
    "legend.borderpad": 0.3,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.0,
    "legend.frameon": False,
}


def label(s):
    """cmr10 draws '_' as a dot accent, so never let one reach the canvas."""
    return s.replace("_", "-")


def hb(n):
    for unit, size in (("TiB", 1 << 40), ("GiB", 1 << 30), ("MiB", 1 << 20)):
        if n >= size:
            v = n / size
            return f"{v:g} {unit}"
    return f"{n} B"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save(fig, outdir, stem):
    for ext in ("pdf", "png"):
        path = os.path.join(outdir, f"{stem}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"  wrote {path}")
    plt.close(fig)


def plot_summary(outdir, exclude=(), stem="paper_summary"):
    by_name = {r["name"]: r for r in read_csv(SUMMARY_CSV)}
    missing = [n for n in SUMMARY_PRIMITIVES + SUMMARY_EXAMPLES
               if n not in by_name]
    unplaced = set(by_name) - set(SUMMARY_PRIMITIVES + SUMMARY_EXAMPLES)
    assert not missing and not unplaced, (missing, unplaced)
    prims = [n for n in SUMMARY_PRIMITIVES if n not in exclude]
    exs = [n for n in SUMMARY_EXAMPLES if n not in exclude]
    rows = [by_name[n] for n in prims + exs]
    names = [SUMMARY_LABELS.get(r["name"], label(r["name"])) for r in rows]
    ratios = [float(r["ratio"]) for r in rows]
    x = np.arange(len(rows))

    # Kept deliberately short: this figure is a wide strip across the full text
    # width, and the rotated tick labels below the axes already add ~0.35in, so
    # the axes themselves carry the page cost.
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 1.0))
    ax.bar(x, ratios, 0.7, color=plot_style.PALETTE["blue"], zorder=2)
    ax.axhline(1.0, color=plot_style.PALETTE["red"], linestyle="--",
               linewidth=0.7, zorder=3)
    ax.text(x[-1] + 0.5, 1.0, " DRAM", color=plot_style.PALETTE["red"],
            va="center", ha="left", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0, labelsize=6.5, pad=1)
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_ylim(0, max(ratios) * 1.08)
    ax.set_ylabel("Time / DRAM time")
    ax.grid(True, axis="y", zorder=0)
    ax.grid(False, axis="x")
    save(fig, outdir, stem)


def _smooth(ys, window_s, dt):
    k = int(round(window_s / dt)) if window_s > 0 else 1
    if k <= 1:
        return ys
    kernel = np.ones(k) / k
    # Edge-pad so the ends aren't dragged toward zero.
    padded = np.pad(ys, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_trace(outdir, t0, t1, split, smooth_s):
    rows = read_csv(TRACE_CSV)
    t = np.array([float(r["time_s"]) for r in rows])
    keep = (t >= t0) & (t <= t1)
    t = t[keep]
    dt = float(np.median(np.diff(t)))

    def col(name, scale=1.0):
        ys = np.array([float(r[name]) for r in rows])[keep] * scale
        return _smooth(ys, smooth_s, dt)

    xs = t - t0
    P = plot_style.PALETTE
    fig, (ax_cpu, ax_bw) = plt.subplots(
        2, 1, figsize=(COLUMN_WIDTH, 2.5), sharex=True,
        gridspec_kw={"hspace": 0.12})

    ax_cpu.plot(xs, col("mean_util_pct"), color=P["violet"], label="drive util")
    ax_cpu.plot(xs, col("cpu_pct"), color=P["green"], label="CPU")
    ax_cpu.plot(xs, col("iowait_pct"), color=P["orange"], label="iowait")
    ax_cpu.set_ylim(0, 100)
    ax_cpu.set_yticks([0, 25, 50, 75, 100])
    ax_cpu.set_ylabel("Percent")

    ax_bw.plot(xs, col("agg_read_mbps", 1e-3), color=P["blue"], label="read")
    ax_bw.plot(xs, col("agg_write_mbps", 1e-3), color=P["red"], label="write")
    ax_bw.set_ylim(0, 75)
    ax_bw.set_ylabel("GB/s")
    ax_bw.set_xlabel("Seconds since algorithm start")
    ax_bw.set_xlim(0, xs[-1])

    for ax in (ax_cpu, ax_bw):
        ax.axvline(split - t0, color="#555555", linewidth=0.6, linestyle=":")
    # One key for both panels (all five colors are distinct), above the top.
    h1, l1 = ax_cpu.get_legend_handles_labels()
    h2, l2 = ax_bw.get_legend_handles_labels()
    ax_cpu.legend(h1 + h2, l1 + l2, loc="lower center",
                  bbox_to_anchor=(0.5, 1.0), ncol=5, borderaxespad=0.1,
                  handlelength=1.2, columnspacing=0.8)
    # Phase names in the throughput panel's empty bands: above the read-only
    # plateau's floor on the left, above the ~38 GB/s read line on the right.
    ax_bw.text((split - t0) / 2, 50, "scan pass 1\n(read)", ha="center",
               va="center", fontsize=6)
    ax_bw.text((split - t0 + xs[-1]) / 2, 55, "add + force (read + write)",
               ha="center", va="center", fontsize=6)
    save(fig, outdir, "paper_trace")


def plot_ablation(outdir):
    # (policy, size, column) -> time, last non-blank wins across the sweeps.
    merged = {}
    for path in ABLATION_CSVS:
        for r in read_csv(path):
            for col in ("add_s", "eager_add_s"):
                if r.get(col, "").strip():
                    merged[(r["policy"], int(r["input_bytes"]), col)] = \
                        float(r[col])

    P = plot_style.PALETTE
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.1))
    for policy, color in POLICIES:
        for col, style in (("add_s", "-o"), ("eager_add_s", "--s")):
            pts = sorted((size, t) for (pol, size, c), t in merged.items()
                         if pol == policy and c == col)
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, style, color=P[color])

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: hb(int(v))))
    ax.set_xlabel("Input size")
    ax.set_ylabel("Time (s)")
    ax.grid(True, which="major", alpha=0.5)

    handles = [Line2D([], [], color=P[c], linewidth=2.5, label=label(p))
               for p, c in POLICIES]
    handles += [Line2D([], [], color="#333333", linestyle="-", marker="o",
                       label="delayed"),
                Line2D([], [], color="#333333", linestyle="--", marker="s",
                       label="eager")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=5, borderaxespad=0.1, handlelength=1.5, columnspacing=0.8)
    save(fig, outdir, "paper_ablation")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["summary", "trace", "ablation"],
                    help="draw just one figure")
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--t-start", type=float, default=TRACE_T0)
    ap.add_argument("--t-end", type=float, default=TRACE_T1)
    ap.add_argument("--phase-split", type=float, default=PHASE_SPLIT)
    ap.add_argument("--smooth", type=float, default=0.5,
                    help="trace rolling-mean window in seconds (0 = raw)")
    args = ap.parse_args()

    plot_style.apply()
    with matplotlib.rc_context(PAPER_RC):
        if args.only in (None, "summary"):
            plot_summary(args.outdir)
            plot_summary(args.outdir, SUMMARY_REDUCED_EXCLUDE,
                         "paper_summary_core")
        if args.only in (None, "trace"):
            plot_trace(args.outdir, args.t_start, args.t_end,
                       args.phase_split, args.smooth)
        if args.only in (None, "ablation"):
            plot_ablation(args.outdir)


if __name__ == "__main__":
    main()
