#!/usr/bin/env python3
"""paper_summary_core redrawn in summary_figure.png's two-bar style: per entry, a
light-green ParlayLib bar fixed at 1 beside a dark-green PLAID bar at the
out-of-core / DRAM time ratio.  No gridlines, no red DRAM line.  Same data,
order, labels and fonts as plot_paper_figures.py's paper_summary_core (fft /
bellman-ford / convex-hull dropped), and the output is pinned to
paper_summary_core.png's exact pixel size.

  usage:
    python3 plot_paper_summary_bars.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_paper_figures as ppf  # noqa: E402  (also puts benchmarks/ on sys.path)
import plot_style  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

DPI = 300
REFERENCE_PNG = os.path.join(HERE, "paper_summary_core.png")
OUT_PNG = os.path.join(HERE, "paper_summary_bars.png")
# Sampled from image.png (seaborn-style Greens pair): light = ParlayLib,
# dark = PLAID.
PARLAY_COLOR = "#b3d495"
PLAID_COLOR = "#40923a"


def reference_size():
    if os.path.exists(REFERENCE_PNG):
        return Image.open(REFERENCE_PNG).size
    return (1804, 394)


def main():
    by_name = {r["name"]: r for r in ppf.read_csv(ppf.SUMMARY_CSV)}
    order = ppf.SUMMARY_PRIMITIVES + ppf.SUMMARY_EXAMPLES
    missing = [n for n in order if n not in by_name]
    unplaced = set(by_name) - set(order)
    assert not missing and not unplaced, (missing, unplaced)
    rows = [by_name[n] for n in order
            if n not in ppf.SUMMARY_REDUCED_EXCLUDE]
    names = [ppf.SUMMARY_LABELS.get(r["name"], ppf.label(r["name"]))
             for r in rows]
    ratios = [float(r["ratio"]) for r in rows]
    x = np.arange(len(rows))
    width = 0.46

    w_px, h_px = reference_size()
    plot_style.apply()
    with matplotlib.rc_context(ppf.PAPER_RC):
        fig, ax = plt.subplots(figsize=(w_px / DPI, h_px / DPI), dpi=DPI)
        fig.subplots_adjust(left=0.045, right=0.995, top=0.96, bottom=0.36)
        ax.bar(x - width / 2, [1.0] * len(x), width,
               color=PARLAY_COLOR, label="ParlayLib (in-memory)",
               zorder=2)
        ax.bar(x + width / 2, ratios, width,
               color=PLAID_COLOR, label="PLAID (external)",
               zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha="right",
                           rotation_mode="anchor")
        ax.tick_params(axis="x", length=0, labelsize=6.5, pad=1)
        ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
        ax.set_ylim(0, max(ratios) * 1.08)
        ax.set_ylabel("Time / DRAM time")
        ax.grid(False)
        ax.legend(loc="upper right", ncol=2, frameon=True)
        fig.savefig(OUT_PNG, dpi=DPI)
        plt.close(fig)

    got = Image.open(OUT_PNG).size
    assert got == (w_px, h_px), (got, (w_px, h_px))
    print(f"  wrote {OUT_PNG} ({got[0]}x{got[1]})")


if __name__ == "__main__":
    main()
