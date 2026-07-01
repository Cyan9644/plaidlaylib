#!/usr/bin/env python3
"""Plot the chunk-vs-seq scaling benchmark as a two-panel log-log figure.

Reads the CSV produced by bench_chunk_vs_seq.sh:

    n,map_s,chunkmap_s,reduce_s,chunkreduce_s

and renders one figure with two panels — left: Map vs ChunkMap, right:
Reduce vs ChunkReduce — both with base-2 log scales on x (problem size) and
y (operation time), so tick marks land on powers of two.

Usage:
    python3 plot_chunk_bw.py [csv_path] [png_path]

Requires matplotlib (not bundled on this box):
    pip install matplotlib      # or add python3Packages.matplotlib to your Nix env
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
        "  (or add python3Packages.matplotlib to your Nix shell)\n"
        "The CSV from bench_chunk_vs_seq.sh is still usable without plotting.\n"
    )
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))


def _pow2_fmt(val, _):
    n = round(math.log2(val))
    return f'$2^{{{n}}}$'


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "chunk_bw.csv")
    png_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results", "chunk_bw.png")

    n, map_s, chunkmap_s, reduce_s, chunkreduce_s = [], [], [], [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            n.append(int(row["n"]))
            map_s.append(float(row["map_s"]))
            chunkmap_s.append(float(row["chunkmap_s"]))
            reduce_s.append(float(row["reduce_s"]))
            chunkreduce_s.append(float(row["chunkreduce_s"]))

    if not n:
        sys.stderr.write(f"error: no data rows in {csv_path}\n")
        sys.exit(1)

    fig, (ax_map, ax_red) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    panels = [
        (ax_map, "Map (x -> x+1, read + write)",
         [("Map (file-based)", map_s, "o-"), ("ChunkMap", chunkmap_s, "s--")]),
        (ax_red, "Reduce (sum, read-only)",
         [("Reduce (file-based)", reduce_s, "o-"), ("ChunkReduce", chunkreduce_s, "s--")]),
    ]
    for ax, title, series in panels:
        for label, ys, style in series:
            ax.plot(n, ys, style, label=label)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_pow2_fmt))
        ax.set_xlabel("n (elements)")
        ax.set_ylabel("operation time (s)")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        ax.legend()

    fig.suptitle("ChunkSequence vs file-based primitives — scaling")
    fig.savefig(png_path, dpi=150)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
