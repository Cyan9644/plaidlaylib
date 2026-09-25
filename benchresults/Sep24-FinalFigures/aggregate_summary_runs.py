#!/usr/bin/env python3
"""Mean / median of several summary_figure.csv runs, per entry.

Each input is a summary_figure.csv as benchmarks/summary_figure.py writes it
(label,name,n,input_bytes,time_s,inmem_time_s,ratio).  Rows are keyed by
`name`.  An entry measured at a different input_bytes in different runs is
flagged (size_mismatch=1) and warned about -- its stats mix sizes.

  usage:
    python3 aggregate_summary_runs.py RUN1.csv RUN2.csv [...] [--out DIR]

Writes to DIR (default: cwd):
  summary_stats.csv            per-entry per-run ratios + mean/median/min/max/stdev
  summary_figure_mean.csv      summary_figure.csv schema, mean times + mean ratio
  summary_figure_median.csv    summary_figure.csv schema, median times + median ratio
and prints a Markdown table to stdout.  The two summary_figure_*.csv files
redraw with plot_paper_summary_bars.py by pointing ppf.SUMMARY_CSV at them.
"""

import argparse
import csv
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_HEADER = ["label", "name", "n", "input_bytes", "time_s", "inmem_time_s",
              "ratio"]


def read_csv(path):
    with open(path, newline="") as f:
        return {r["name"]: r for r in csv.DictReader(f)}


def entry_order(runs):
    """plot_paper_figures' order if importable, else first-seen CSV order."""
    try:
        sys.path.insert(0, HERE)
        import plot_paper_figures as ppf
        order = list(ppf.SUMMARY_PRIMITIVES + ppf.SUMMARY_EXAMPLES)
    except Exception:
        order = []
    for run in runs:
        for name in run:
            if name not in order:
                order.append(name)
    return order


def stats(xs):
    return {"mean": statistics.mean(xs), "median": statistics.median(xs),
            "min": min(xs), "max": max(xs),
            "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+", help="summary_figure.csv files, one per run")
    ap.add_argument("--out", default=".", help="output directory (default: cwd)")
    args = ap.parse_args()

    runs = [read_csv(p) for p in args.csvs]
    k = len(runs)
    warnings = []
    stat_rows, mean_rows, median_rows = [], [], []

    for name in entry_order(runs):
        present = [(i, run[name]) for i, run in enumerate(runs) if name in run]
        if not present:
            continue
        if len(present) < k:
            missing = [args.csvs[i] for i in range(k) if name not in runs[i]]
            warnings.append(f"{name}: missing from {', '.join(missing)}")
        sizes = {r["input_bytes"] for _, r in present}
        mismatch = len(sizes) > 1
        if mismatch:
            warnings.append(f"{name}: input_bytes differs across runs "
                            f"({', '.join(sorted(sizes, key=int))}) -- stats mix sizes")

        ratios = [float(r["ratio"]) for _, r in present]
        times = [float(r["time_s"]) for _, r in present]
        inmems = [float(r["inmem_time_s"]) for _, r in present]
        rs, ts, ms = stats(ratios), stats(times), stats(inmems)
        first = present[0][1]

        row = {"label": first["label"], "name": name,
               "input_bytes": first["input_bytes"], "n_runs": len(present),
               "size_mismatch": int(mismatch)}
        per_run = {i: float(r["ratio"]) for i, r in present}
        for i in range(k):
            row[f"ratio_run{i + 1}"] = f"{per_run[i]:.6g}" if i in per_run else ""
        for prefix, s in (("ratio", rs), ("time_s", ts), ("inmem_time_s", ms)):
            for stat, v in s.items():
                row[f"{prefix}_{stat}"] = f"{v:.6g}"
        row["ratio_of_means"] = f"{ts['mean'] / ms['mean']:.6g}"
        stat_rows.append(row)

        for out, stat in ((mean_rows, "mean"), (median_rows, "median")):
            out.append({"label": first["label"], "name": name, "n": first["n"],
                         "input_bytes": first["input_bytes"],
                         "time_s": f"{ts[stat]:.9g}",
                         "inmem_time_s": f"{ms[stat]:.9g}",
                         "ratio": f"{rs[stat]:.9g}"})

    os.makedirs(args.out, exist_ok=True)
    stats_path = os.path.join(args.out, "summary_stats.csv")
    with open(stats_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stat_rows[0].keys()))
        w.writeheader()
        w.writerows(stat_rows)
    for fname, rows in (("summary_figure_mean.csv", mean_rows),
                        ("summary_figure_median.csv", median_rows)):
        with open(os.path.join(args.out, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIG_HEADER)
            w.writeheader()
            w.writerows(rows)

    run_cols = " | ".join(f"run{i + 1}" for i in range(k))
    print(f"| entry | size (GiB) | {run_cols} | mean | median | stdev | ratio of means |")
    print("|" + "---|" * (k + 6))
    for r in stat_rows:
        gib = int(r["input_bytes"]) / (1 << 30)
        per = " | ".join(r[f"ratio_run{i + 1}"] or "-" for i in range(k))
        flag = " (!)" if r["size_mismatch"] else ""
        print(f"| {r['label']}{flag} | {gib:g} | {per} | {float(r['ratio_mean']):.3f} | "
              f"{float(r['ratio_median']):.3f} | {float(r['ratio_stdev']):.3f} | "
              f"{float(r['ratio_of_means']):.3f} |")
    all_mean = [float(r["ratio_mean"]) for r in stat_rows]
    all_med = [float(r["ratio_median"]) for r in stat_rows]
    print(f"\nacross entries: mean-of-means={statistics.mean(all_mean):.3f}  "
          f"median-of-medians={statistics.median(all_med):.3f}  "
          f"geomean-of-means={statistics.geometric_mean(all_mean):.3f}")
    print(f"\nwrote {stats_path}, summary_figure_mean.csv, summary_figure_median.csv "
          f"in {os.path.abspath(args.out)}")
    if warnings:
        print(f"\n!!! {len(warnings)} warning(s):", file=sys.stderr)
        for w_ in warnings:
            print(f"  - {w_}", file=sys.stderr)


if __name__ == "__main__":
    main()
