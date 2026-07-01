#!/usr/bin/env python3
"""Plot the in-memory vs chunk-eager vs chunk-delayed scaling harness.

Reads the CSV produced by bench_delayed_scale.sh:

    n,raw_read_s,eager_mr_s,delayed_mr_s,inmem_mr_s,
      eager_mmr_s,delayed_mmr_s,inmem_mmr_s,
      eager_fmm_s,delayed_fmm_s,inmem_fmm_s,agree

and renders one figure with two panels, y = operation time (s):

  * left  (read-bound):  map|map|reduce — in-mem delayed, chunk-delayed,
                         chunk-eager, and the raw-read ceiling;
  * right (write-bound): force(map|map) — in-mem materialize, chunk-delayed,
                         chunk-eager, with the raw-read line as a reference.

Blank in-memory fields (n past the RAM budget) are skipped, so the in-memory
line simply stops at the cliff.  Both axes are base-2 log with tick labels
formatted as 2^k.

Usage:
    python3 plot_delayed_scale.py [csv_path] [png_path]

Requires matplotlib:
    pip install matplotlib
"""
import csv
import math
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")  # headless: write a file, no display needed
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    sys.stderr.write(
        "error: matplotlib is required to plot.\n"
        "  install it with:  pip install matplotlib\n"
        "The CSV from bench_delayed_scale.sh is still usable without plotting.\n"
    )
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))


def _pow2_fmt(val, _):
    n = round(math.log2(val))
    return f'$2^{{{n}}}$'


def _f(s):
    """Parse a CSV time field; blank (skipped in-memory line) -> None."""
    s = s.strip()
    return float(s) if s else None


def runtime(ns, times):
    """(xs, ys) of wall-clock seconds for the non-blank points only."""
    xs, ys = [], []
    for n, t in zip(ns, times):
        if t is not None and t > 0.0:
            xs.append(n)
            ys.append(t)
    return xs, ys


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "delayed_scale.csv")
    png_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results", "delayed_scale.png")

    cols = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for key in reader.fieldnames:
            cols[key] = []
        for row in reader:
            for key in reader.fieldnames:
                cols[key].append(row[key])

    if not cols.get("n"):
        sys.stderr.write(f"error: no data rows in {csv_path}\n")
        sys.exit(1)

    n = [int(v) for v in cols["n"]]
    raw = [_f(v) for v in cols["raw_read_s"]]

    def series(name):
        return [_f(v) for v in cols[name]]

    fig, (ax_r, ax_w) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    # left: read-bound map|map|reduce
    read_lines = [
        ("in-mem delayed (DRAM)", series("inmem_mmr_s"), "o-"),
        ("chunk-delayed",         series("delayed_mmr_s"), "s-"),
        ("chunk-eager",           series("eager_mmr_s"), "^-"),
        ("raw read (1 pass)",     raw, "k:"),
    ]
    # right: write-bound force(map|map)
    write_lines = [
        ("in-mem materialize (DRAM)", series("inmem_fmm_s"), "o-"),
        ("chunk-delayed",            series("delayed_fmm_s"), "s-"),
        ("chunk-eager",              series("eager_fmm_s"), "^-"),
        ("raw read (1 pass)",        raw, "k:"),
    ]

    for ax, title, lines in [
        (ax_r, "map(x+1) | map(2x) | reduce(sum)  — read-bound", read_lines),
        (ax_w, "force(map(x+1) | map(2x))  — write-bound", write_lines),
    ]:
        for label, times, style in lines:
            xs, ys = runtime(n, times)
            if xs:
                ax.plot(xs, ys, style, label=label, markersize=5)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_pow2_fmt))
        ax.set_xlabel("n (elements)")
        ax.set_ylabel("operation time (s)")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        ax.legend()

    fig.suptitle("Delayed sequences: in-memory vs chunk-eager vs chunk-delayed — scaling\n"
                 "(in-memory line stops where the input exceeds the RAM budget)")
    fig.savefig(png_path, dpi=150)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
