#!/usr/bin/env python3
"""Combined relative-performance bar chart: for each label in
SUMMARY_ENTRIES (11 primitives + 9 examples), plots the largest input size
at which that entry's own recorded run still has an in-memory parlaylib
baseline, against a pinned in-mem reference at 1.0 -- in the spirit of
parlaylib's own "ParlayLib vs ParallelSTL" figure (see
benchresults/example_figure/), but comparing this library's out-of-core
primitives/examples against their in-memory parlaylib counterparts instead.

This script runs NO binaries and calls `make` on nothing: it only reads the
`<name>_scale.csv` files benchmarks/run_benches.py's example sweep already
writes to `<results-root>/<timestamp>/<name>_scale.csv` (one row per swept
size). Populate those with `make bench-examples-full` on the benchmark
machine: it sweeps exactly this script's entries (via `--list`) with the
in-memory baseline uncapped (run_benches.py --inmem-uncapped), so each
entry's last in-mem row is the largest size its baseline actually survived,
then `make bench-summary RUN=results/<timestamp>` to plot that run.

Reuses run_benches.py by import (the EXAMPLES registry and REPO_ROOT) -- the
same `import run_benches as rb` precedent io_trace.py / csv_from_log.py /
work_exponent_bench.py already use.

For each entry, the plotted point is the largest-n row in its
`<name>_scale.csv` whose in-mem column is non-blank (i.e. the biggest input
that sweep recorded before run_benches.py's own RAM-cliff budget skipped the
in-memory baseline for that point). An entry with no matching CSV, or whose
CSV has no row with a usable in-mem column, is skipped with a warning --
never fabricated.

Without --dir, every entry is looked up independently: each
`<name>_scale.csv` is found by globbing every `<results-root>/*/` directory
and taking the most recent match, so bars for entries swept at different
times (or via separate `--example` runs) still combine into one chart.
--dir pins the search to one specific run directory instead.

  usage:
    python3 benchmarks/summary_figure.py
    python3 benchmarks/summary_figure.py --only "reduce,tabulate,zip"
    python3 benchmarks/summary_figure.py --dir results/20260101-000000
"""

import argparse
import csv
import glob
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benches as rb  # noqa: E402  (sibling module; import-safe)

# (display_label, EXAMPLES registry name) pairs for the 20 bars: 11 primitives
# + 9 examples, "sort"/"samplesort" collapsed to the single samplesort entry
# and "bellman_ford" mapping to the bellman_ford_sparse registry entry (the
# sparse RMAT case). Already alphabetical by display_label; asserted below so
# a future edit can't silently desync the claim from the actual order.
#
# The whitelist cleanup dropped four former bars -- bellman_ford, fft,
# kth_smallest and count_sort's demo -- and `cut` is omitted because its demo
# segfaults (a pre-existing break, unchanged by the cleanup -- the subcommand
# is still there in primitive_demos.cpp for whoever fixes it). All three
# dropped examples have since been recovered from 9c96e4a: kth_smallest as an
# example-side primitive (chunk_kth_smallest.h); fft as chunk_fft.h + fft.cpp,
# the transpose-free variant (the counterpart fft_transpose.cpp is swept by
# run_benches.py but not plotted here); bellman_ford as the CSR-graph
# subsystem under ChunkSequence/helper/ + chunk_bellman_ford.h + bellman_ford.cpp,
# with the RMAT graph generator's own numeric_limits gap around sample_sort's
# pivot padding worked around by reusing direct_sample_sort (examples/
# direct_samplesort.h) instead. count_sort's demo was repointed to showcase
# the already-existing group_by_index primitive instead of being restored
# as-is.
SUMMARY_ENTRIES = [
    ("bellman_ford", "bellman_ford_sparse"),
    ("bigint_add", "bigint_add"),
    ("convex_hull", "convex_hull"),
    ("fft", "fft"),
    ("filter", "filter"),
    ("group_by_index", "group_by_index"),
    ("histogram_by_index", "histogram_by_index"),
    ("kmp", "kmp"),
    ("kth_smallest", "kth_smallest"),
    ("linefit", "linefit"),
    ("map", "map"),
    ("pack", "pack"),
    ("primes", "primes"),
    ("rabin_karp", "rabin_karp"),
    ("random_shuffle", "random_shuffle"),
    ("reduce", "reduce"),
    ("reverse", "reverse"),
    ("scan", "scan"),
    ("sort / samplesort", "samplesort"),
    ("tabulate", "tabulate"),
]
# `zip` is deliberately not a bar: its demo is a fused zip+map+reduce (a dot
# product) over two sequences -- the same shape the linefit bar already shows
# -- so it adds no new information.
assert sorted(SUMMARY_ENTRIES) == SUMMARY_ENTRIES, \
    "SUMMARY_ENTRIES must stay alphabetical by display_label"

REGISTRY = {e["name"]: e for e in rb.EXAMPLES}
for _label, _name in SUMMARY_ENTRIES:
    assert _name in REGISTRY, f"SUMMARY_ENTRIES: {_name!r} not in rb.EXAMPLES"


def find_csv(name, dir_arg, results_root):
    """Locate the on-disk <name>_scale.csv run_benches.py's example sweep
    writes. With an explicit --dir, only that directory is checked.
    Otherwise every <results_root>/<timestamp>/ directory is searched and the
    most recent match wins (timestamp directory names sort chronologically as
    strings), so entries swept at different times still combine into one
    chart. Returns None if nothing matches.
    """
    fname = f"{name}_scale.csv"
    if dir_arg is not None:
        path = os.path.join(dir_arg, fname)
        return path if os.path.isfile(path) else None
    candidates = sorted(glob.glob(os.path.join(results_root, "*", fname)))
    return candidates[-1] if candidates else None


def _positive(row, col):
    try:
        return float(row.get(col, "")) > 0
    except ValueError:  # blank field
        return False


def load_entry_row(csv_path, entry):
    """Return the row to plot from one entry's CSV: the largest-n row whose
    out-of-core and in-mem times are both present and positive (the biggest
    size at which the in-memory baseline still ran), or None if the file has
    no such row.
    """
    time_col = entry.get("time_col", "time_s")
    inmem_col = entry["inmem_col"]
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if _positive(r, time_col) and _positive(r, inmem_col)]
    if not rows:
        return None
    return max(rows, key=lambda r: int(r["n"]))


def collect_rows(dir_arg, results_root, warnings):
    """Read every SUMMARY_ENTRIES entry's <name>_scale.csv and return the
    list of {label, name, n, time_s, inmem_time_s, ratio} rows to plot. An
    entry with no matching CSV, or no row with a usable in-mem column, is
    skipped with a warning rather than fabricated.
    """
    rows = []
    for label, name in SUMMARY_ENTRIES:
        entry = REGISTRY[name]
        csv_path = find_csv(name, dir_arg, results_root)
        if csv_path is None:
            where = dir_arg if dir_arg is not None else os.path.join(results_root, "*")
            warnings.append(f"{label} ({name}): no {name}_scale.csv found under {where}")
            continue

        row = load_entry_row(csv_path, entry)
        if row is None:
            warnings.append(f"{label} ({name}): {csv_path} has no row with a "
                            f"non-blank {entry['inmem_col']}")
            continue

        time_col = entry.get("time_col", "time_s")
        inmem_col = entry["inmem_col"]
        t = float(row[time_col])
        t_inmem = float(row[inmem_col])
        ratio = t / t_inmem if t_inmem > 0 else float("nan")
        rows.append({"label": label, "name": name, "n": row["n"],
                     "time_s": f"{t:.9g}", "inmem_time_s": f"{t_inmem:.9g}",
                     "ratio": f"{ratio:.9g}"})
        print(f"  {label}: n={row['n']}  out-of-core={t:.4g}s  "
              f"in-mem={t_inmem:.4g}s  ratio={ratio:.4g}x  (from {csv_path})")
    return rows


def write_csv(path, rows):
    header = ["label", "name", "n", "time_s", "inmem_time_s", "ratio"]
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r[c] for c in header) + "\n")
    print(f"  wrote {path}", flush=True)


def plot_summary(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import plot_style
    plot_style.apply()

    labels = [r["label"] for r in rows]
    ratios = [float(r["ratio"]) for r in rows]
    x = np.arange(len(labels))
    width = 0.46  # wide relative to the 1.0 group spacing -> tight gap between bar groups

    # Sized for a LaTeX double-column figure: fixed width near a typical
    # two-column \textwidth, short enough not to dominate the page (half the
    # already-short height of the pre-aggregation design, since this is now a
    # cheap, no-execution summary rather than a heavy standalone run). This
    # figure is much shorter than plot_style's other consumers are tuned for
    # (they run ~5.5-6.5in tall), so left at plot_style's absolute point
    # sizes the text would dwarf a 1.25in-tall plot area. \includegraphics
    # rescales the whole PNG to a target width in the final document, so what
    # actually matters for legibility there is font-size-relative-to-width,
    # not the raw pt numbers -- scale every text/line rcParam down by how much
    # narrower this figure is than the old (pre-compaction) design so that
    # ratio, and hence the printed appearance, is preserved.
    fig_width = max(7.0, 0.32 * len(labels))
    old_fig_width = max(10, 0.55 * len(labels))
    text_scale = fig_width / old_fig_width
    base_rc = matplotlib.rcParams
    scaled_rc = {
        "font.size": base_rc["font.size"] * text_scale,
        "axes.labelsize": base_rc["axes.labelsize"] * text_scale,
        "xtick.labelsize": base_rc["xtick.labelsize"] * text_scale,
        "ytick.labelsize": base_rc["ytick.labelsize"] * text_scale,
        "legend.fontsize": base_rc["legend.fontsize"] * text_scale,
        "axes.linewidth": base_rc["axes.linewidth"] * text_scale,
        "grid.linewidth": base_rc["grid.linewidth"] * text_scale,
        "lines.linewidth": base_rc["lines.linewidth"] * text_scale,
    }
    with matplotlib.rc_context(scaled_rc):
        fig, ax = plt.subplots(figsize=(fig_width, 1.25), constrained_layout=True)
        ax.bar(x - width / 2, [1.0] * len(labels), width,
              label="ParlayLib DRAM", color=plot_style.PALETTE["green"])
        ax.bar(x + width / 2, ratios, width,
              label="PLAID external", color=plot_style.PALETTE["blue"])
        ax.axhline(1.0, color=plot_style.PALETTE["red"], linestyle="--",
                  linewidth=scaled_rc["lines.linewidth"] * 0.5, zorder=0)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)
        ax.set_ylabel("Relative Performance")
        ax.grid(True, axis="y")
        ax.legend()

        fig.savefig(path, dpi=300)
        plt.close(fig)
    print(f"  wrote {path}", flush=True)


def main():
    global SUMMARY_ENTRIES
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=os.environ.get("BENCH_OUTDIR", "results"),
                    help="parent dir for the timestamped output (default: results)")
    ap.add_argument("--only", default="",
                    help="plot only these display label(s) (comma/space-separated, "
                         f"e.g. 'reduce,tabulate,zip'); choices: "
                         f"{', '.join(l for l, _ in SUMMARY_ENTRIES)}")
    ap.add_argument("--dir", default=None,
                    help="read every entry's <name>_scale.csv from exactly this "
                         "directory instead of searching --results-root for the "
                         "most recent match per entry")
    ap.add_argument("--results-root", default=os.environ.get("BENCH_OUTDIR", "results"),
                    help="parent dir to search for <timestamp>/<name>_scale.csv "
                         "files when --dir isn't given (default: results)")
    ap.add_argument("--list", action="store_true",
                    help="print the entries' run_benches.py registry names "
                         "(comma-separated, for run_benches.py --example) and exit")
    args = ap.parse_args()

    if args.list:
        print(",".join(name for _, name in SUMMARY_ENTRIES))
        return

    selected = [x for x in re.split(r"[,\s]+", args.only) if x]
    known = {l for l, _ in SUMMARY_ENTRIES}
    for name in selected:
        if name not in known:
            ap.error(f"unknown --only label {name!r}; choices: {', '.join(sorted(known))}")
    if selected:
        SUMMARY_ENTRIES = [(l, n) for l, n in SUMMARY_ENTRIES if l in selected]

    dir_arg = os.path.join(rb.REPO_ROOT, args.dir) if args.dir else None
    results_root = os.path.join(rb.REPO_ROOT, args.results_root)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(rb.REPO_ROOT, args.outdir, stamp)
    os.makedirs(outdir, exist_ok=True)
    print(f"Output directory: {outdir}\n")

    warnings = []
    rows = collect_rows(dir_arg, results_root, warnings)

    write_csv(os.path.join(outdir, "summary_figure.csv"), rows)
    if rows:
        try:
            plot_summary(rows, os.path.join(outdir, "summary_figure.png"))
        except Exception as exc:
            warnings.append(f"plotting failed ({exc}); CSV was written")
            print(f"  !!! plotting failed ({exc}); CSV was written", flush=True)
    else:
        print("  !!! no bars collected; skipping the plot", flush=True)

    print("\n======== run summary ========")
    print(f"  {len(rows)}/{len(SUMMARY_ENTRIES)} entries produced a bar")
    if warnings:
        print(f"  !!! {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  !!!   {w}")
        wpath = os.path.join(outdir, "warnings.txt")
        with open(wpath, "w") as f:
            f.write("\n".join(warnings) + "\n")
        print(f"  !!! (also written to {wpath})")
    else:
        print("  no warnings")
    print(f"\nDone. Results in {outdir}")


if __name__ == "__main__":
    main()
