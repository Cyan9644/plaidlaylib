#!/usr/bin/env python3
"""Plot the chunk-size sensitivity benchmark as a two-panel log-log figure.

Reads the CSV produced by bench_chunk_size.sh:

    chunk_size_bytes,n,raw_s,eager_mr_s,delayed_mr_s,
    eager_mmr_s,delayed_mmr_s,eager_fmm_s,delayed_fmm_s,agree

and renders one figure with two panels — left: map|map|reduce (read-bound),
right: force(map|map) (write-bound) — both with base-2 log scales on x
(chunk size in bytes, ticks labelled in KB or MB) and y (operation time in
seconds, ticks labelled as 2^k).

Usage:
    python3 plot_chunk_size.py [csv_path] [png_path]

Requires matplotlib:
    pip install matplotlib
"""
import csv
import math
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    sys.stderr.write(
        "error: matplotlib is required.\n"
        "  pip install matplotlib\n"
        "The CSV from bench_chunk_size.sh is still usable without plotting.\n"
    )
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))


def _pow2_time_fmt(val, _):
    n = round(math.log2(val))
    return f'$2^{{{n}}}$'


def _chunk_size_fmt(val, _):
    if val >= 1024 * 1024:
        return f'{int(val // (1024*1024))} MB'
    return f'{int(val // 1024)} KB'


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 \
        else os.path.join(HERE, "results", "chunk_size.csv")
    png_path = sys.argv[2] if len(sys.argv) > 2 \
        else os.path.join(HERE, "results", "chunk_size.png")

    cs, raw_s = [], []
    eager_mr, delayed_mr = [], []
    eager_mmr, delayed_mmr = [], []
    eager_fmm, delayed_fmm = [], []
    n_val = None

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            cs.append(int(row["chunk_size_bytes"]))
            n_val = int(row["n"])
            raw_s.append(float(row["raw_s"]))
            eager_mr.append(float(row["eager_mr_s"]))
            delayed_mr.append(float(row["delayed_mr_s"]))
            eager_mmr.append(float(row["eager_mmr_s"]))
            delayed_mmr.append(float(row["delayed_mmr_s"]))
            eager_fmm.append(float(row["eager_fmm_s"]))
            delayed_fmm.append(float(row["delayed_fmm_s"]))

    if not cs:
        sys.stderr.write(f"error: no data rows in {csv_path}\n")
        sys.exit(1)

    fig, (ax_r, ax_w) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    read_lines = [
        ("chunk-eager",        eager_mmr,  "^-"),
        ("chunk-delayed",      delayed_mmr, "s-"),
        ("raw read (ceiling)", raw_s,       "k:"),
    ]
    write_lines = [
        ("chunk-eager",   eager_fmm,   "^-"),
        ("chunk-delayed", delayed_fmm, "s-"),
    ]

    for ax, title, lines in [
        (ax_r, "map(x+1) | map(2x) | reduce(sum)  — read-bound", read_lines),
        (ax_w, "force(map(x+1) | map(2x))  — write-bound",       write_lines),
    ]:
        for label, ys, style in lines:
            ax.plot(cs, ys, style, label=label, markersize=6)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_chunk_size_fmt))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_pow2_time_fmt))
        ax.set_xlabel("chunk size")
        ax.set_ylabel("operation time (s)")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        ax.legend()

    n_label = f"n = {n_val:,}" if n_val else ""
    fig.suptitle(f"Chunk-size sensitivity: eager vs delayed  ({n_label})\n"
                 "eager = full SSD round-trips per intermediate; "
                 "delayed = fused, fewer I/O passes")
    fig.savefig(png_path, dpi=150)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
