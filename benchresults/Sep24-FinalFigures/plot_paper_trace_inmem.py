#!/usr/bin/env python3
"""paper_trace with small-size CPU lines overlaid: the same bigint_add 4 TiB
out-of-core trace (drive util / CPU / iowait over read/write GB/s, cropped to
the algorithm window), plus two see-through green CPU lines from one smaller
(64 GiB) traced run -- that size's out-of-core add (dashed) and the in-memory
parlaylib add (solid).

The small lines come from a SEPARATE run because at 4 TiB the in-memory add
needs ~16 TiB of DRAM.  That run's out-of-core add is the bare
op_start..op_end window and the in-memory add is op_start_inmem..op_end_inmem
(bigint_add.cpp); both are cropped from its trace.csv using the markers.csv
io_trace.py writes beside it.  Only CPU is drawn for the small run.  Every
line starts at x = 0 = its own op start; the legend states each run's size.

  usage:
    python3 benchmarks/io_trace.py bigint_add --size 64GiB   # on the SSD box
    # copy that run's trace_bigint_add_* dir to trace_bigint_add_64GiB_inmem/,
    # then:
    python3 plot_paper_trace_inmem.py
    python3 plot_paper_trace_inmem.py --normalize   # x = fraction of op window
"""

import argparse
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

SMALL_DIR = os.path.join(HERE, "trace_bigint_add_64GiB_inmem")
SMALL_ALPHA = 0.4
# (start marker, end marker) of each small-run window, from bigint_add.cpp.
SMALL_EXTERNAL = ("op_start", "op_end")
SMALL_INMEM = ("op_start_inmem", "op_end_inmem")


def load_series(path, t0=None, t1=None):
    rows = ppf.read_csv(path)
    t = np.array([float(r["time_s"]) for r in rows])
    keep = np.ones_like(t, dtype=bool)
    if t0 is not None:
        keep &= t >= t0
    if t1 is not None:
        keep &= t <= t1
    t = t[keep]
    if len(t) < 2:
        sys.exit(f"{path}: fewer than two samples in [{t0}, {t1}]")
    dt = float(np.median(np.diff(t)))
    cols = {name: np.array([float(r[name]) for r in rows])[keep]
            for name in rows[0] if name != "time_s"}
    return t - t[0], dt, cols


def marker_window(markers, start, end):
    missing = [m for m in (start, end) if m not in markers]
    if missing:
        sys.exit(f"markers.csv lacks {', '.join(missing)} (have: "
                 f"{', '.join(markers)})")
    return markers[start], markers[end]


def plot(outdir, t0, t1, split, smooth_s, small_dir, normalize):
    small_trace = os.path.join(small_dir, "trace.csv")
    markers = {r["label"]: float(r["time_s"])
               for r in ppf.read_csv(os.path.join(small_dir, "markers.csv"))}

    xs, dt, pc = load_series(ppf.TRACE_CSV, t0, t1)
    exs, edt, ec = load_series(small_trace,
                               *marker_window(markers, *SMALL_EXTERNAL))
    ixs, idt, ic = load_series(small_trace,
                               *marker_window(markers, *SMALL_INMEM))
    split_x = split - t0
    span = xs[-1]
    print(f"  PLAID 4 TiB {span:.1f} s, PLAID 64 GiB {exs[-1]:.1f} s, "
          f"in-memory 64 GiB {ixs[-1]:.1f} s")
    if normalize:
        split_x /= span
        xs, exs, ixs = xs / span, exs / exs[-1], ixs / ixs[-1]

    def sm(ys, step):
        return ppf._smooth(ys, smooth_s, step)

    P = plot_style.PALETTE
    fig, (ax_cpu, ax_bw) = plt.subplots(
        2, 1, figsize=(ppf.COLUMN_WIDTH, 2.5), sharex=True,
        gridspec_kw={"hspace": 0.12})

    ax_cpu.plot(xs, sm(pc["mean_util_pct"], dt), color=P["violet"],
                label="drive util")
    ax_cpu.plot(xs, sm(pc["cpu_pct"], dt), color=P["green"],
                label="CPU (PLAID, 4 TiB)")
    ax_cpu.plot(exs, sm(ec["cpu_pct"], edt), color=P["green"],
                alpha=SMALL_ALPHA, linestyle="--", label="CPU (PLAID, 64 GiB)")
    ax_cpu.plot(ixs, sm(ic["cpu_pct"], idt), color=P["green"],
                alpha=SMALL_ALPHA, label="CPU (in-memory, 64 GiB)")
    ax_cpu.plot(xs, sm(pc["iowait_pct"], dt), color=P["orange"],
                label="iowait")
    ax_cpu.set_ylim(0, 100)
    ax_cpu.set_yticks([0, 25, 50, 75, 100])
    ax_cpu.set_ylabel("Percent")

    ax_bw.plot(xs, sm(pc["agg_read_mbps"], dt) * 1e-3, color=P["blue"],
               label="read")
    ax_bw.plot(xs, sm(pc["agg_write_mbps"], dt) * 1e-3, color=P["red"],
               label="write")
    ax_bw.set_ylim(0, 75)
    ax_bw.set_ylabel("GB/s")
    ax_bw.set_xlabel("Fraction of algorithm time" if normalize
                     else "Seconds since algorithm start")
    ax_bw.set_xlim(0, xs[-1])

    for ax in (ax_cpu, ax_bw):
        ax.axvline(split_x, color="#555555", linewidth=0.6, linestyle=":")
    h1, l1 = ax_cpu.get_legend_handles_labels()
    h2, l2 = ax_bw.get_legend_handles_labels()
    ax_cpu.legend(h1 + h2, l1 + l2, loc="lower center",
                  bbox_to_anchor=(0.5, 1.0), ncol=3, borderaxespad=0.1,
                  handlelength=1.6, columnspacing=0.8)
    ax_bw.text(split_x / 2, 50, "scan pass 1\n(read)", ha="center",
               va="center", fontsize=6)
    ax_bw.text((split_x + xs[-1]) / 2, 55, "add + force (read + write)",
               ha="center", va="center", fontsize=6)
    ppf.save(fig, outdir, "paper_trace_inmem")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--small-dir", default=SMALL_DIR,
                    help="io_trace.py point dir of the 64 GiB run "
                         "(needs trace.csv + markers.csv)")
    ap.add_argument("--t-start", type=float, default=ppf.TRACE_T0)
    ap.add_argument("--t-end", type=float, default=ppf.TRACE_T1)
    ap.add_argument("--phase-split", type=float, default=ppf.PHASE_SPLIT)
    ap.add_argument("--smooth", type=float, default=0.5,
                    help="rolling-mean window in seconds (0 = raw)")
    ap.add_argument("--normalize", action="store_true",
                    help="x = fraction of each run's own op window (0..1)")
    args = ap.parse_args()

    for f in ("trace.csv", "markers.csv"):
        if not os.path.exists(os.path.join(args.small_dir, f)):
            sys.exit(f"missing {os.path.join(args.small_dir, f)}: run "
                     "`io_trace.py bigint_add --size 64GiB` and copy its trace "
                     "dir there (see --help)")
    plot_style.apply()
    with matplotlib.rc_context(ppf.PAPER_RC):
        plot(args.outdir, args.t_start, args.t_end, args.phase_split,
             args.smooth, args.small_dir, args.normalize)


if __name__ == "__main__":
    main()
