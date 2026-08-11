#!/usr/bin/env python3
"""Combined direct-vs-primitives samplesort I/O trace, sourced from
benchmarks/io_trace.py's io_sweep.csv (one row per input size, per-phase
averages for each algorithm).  Reproduces the look of the first two panels of
trace_direct.png/trace_primitives.png (benchresults/July17-SampleSortIO/) --
aggregate read/write throughput, and drive-util-vs-CPU -- but with direct and
primitives overlaid on the same two axes instead of separate files, and only
the op (sort) phase, not build.
"""
import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALGOS = [("primitives", "-"), ("direct", "--")]


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def series(rows, key):
    return [float(row[key]) for row in rows]


def build_figure(rows):
    ns = series(rows, "n")

    fig, (ax_bw, ax_bn) = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)

    for algo, ls in ALGOS:
        rd = series(rows, f"{algo}_op_avg_read_mbps")
        wr = series(rows, f"{algo}_op_avg_write_mbps")
        ax_bw.plot(ns, rd, "o", linestyle=ls, color="tab:blue",
                   label=f"{algo} read")
        ax_bw.plot(ns, wr, "o", linestyle=ls, color="tab:red",
                   label=f"{algo} write")
    ax_bw.set_xscale("log")
    ax_bw.set_ylabel("aggregate MB/s")
    ax_bw.set_xlabel("n (elements)")
    ax_bw.set_title("Read/Write throughput")
    ax_bw.grid(True, linestyle=":", linewidth=0.5)
    ax_bw.legend(fontsize=8)

    for algo, ls in ALGOS:
        util = series(rows, f"{algo}_op_avg_util_pct")
        cpu = series(rows, f"{algo}_op_avg_cpu_pct")
        ax_bn.plot(ns, util, "o", linestyle=ls, color="tab:purple",
                   label=f"{algo} drive %util")
        ax_bn.plot(ns, cpu, "o", linestyle=ls, color="tab:green",
                   label=f"{algo} CPU %")
    ax_bn.set_xscale("log")
    ax_bn.set_ylabel("percent")
    ax_bn.set_ylim(0, 105)
    ax_bn.set_xlabel("n (elements)")
    ax_bn.set_title("CPU/Drive Utilization")
    ax_bn.grid(True, linestyle=":", linewidth=0.5)
    ax_bn.legend(fontsize=8)

    fig.suptitle("samplesort_three_way: direct (dashed) vs primitives (solid), op phase")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="/home/acmyr/plaidlaylib/io_sweep.csv",
                         help="input io_sweep.csv path (default: %(default)s)")
    parser.add_argument("--out", default="figures/samplesort_throughput_utilization.png",
                         help="output PNG path (default: %(default)s)")
    args = parser.parse_args()

    rows = load_rows(args.csv)
    fig = build_figure(rows)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
