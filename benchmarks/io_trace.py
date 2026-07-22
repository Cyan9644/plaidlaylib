#!/usr/bin/env python3
"""IO/CPU profiler for the examples — one detailed time-series trace per size,
plus (when swept) a combined summary CSV/PNG across sizes.

Where run_benches.py sweeps `n` and records one aggregate number per point,
this runs an example at one or more `n` and samples what the machine is
actually doing over the course of each run: per-SSD read/write throughput and
device utilization (from /proc/diskstats) and CPU (from /proc/stat), every
`--interval` seconds. The example is launched with PLAID_TRACE=1 so it emits
`TRACE,<phase>,<mono>` markers (see utils/trace_marker.h); those land on the
same CLOCK_MONOTONIC timeline the sampler uses, so build/op phases (and,
for multi-algorithm drivers like samplesort_three_way, each algorithm's own
phases — see MARKER_KINDS below) are drawn as labelled bands.

It answers, for a given algorithm:
  * IO-bound vs CPU-bound — device %util pinned near 100 with CPU headroom is
    IO-bound; the inverse is CPU-bound.
  * read/write asymmetry — aggregate read MB/s vs write MB/s over time.
  * per-drive access pattern — a device x time heatmap of read+write MB/s.

`--size` accepts space-separated tokens (like samplesort_phase_bench.py's
`--n-values`) to sweep several sizes in one command. By default, disk/CPU
sampling (and its trace.csv/trace.png) is only actually done for the LARGEST
size in the sweep — that's the point where the drives are most saturated and
the I/O profile is most informative, and repeating the sampler at every
smaller point mostly just adds wall-clock time (running the example itself,
which sampling doesn't speed up) for traces nobody looks at. Every point,
traced or not, still contributes one CSV performance row (that comparison
DOES benefit from every point — see below), so the smaller points aren't
wasted, just not disk/CPU-sampled. Pass `--trace-all` to sample every point
anyway, which also restores the combined summary across sizes: io_sweep.csv
(one row per TRACED size, with per-phase duration/util/CPU/throughput columns
derived by pairing each `..._start_<label>`/`..._end_<label>` marker) plus
io_sweep.png, so trends across n can be read the same way as the other
benchmark CSVs, without re-parsing every point's raw trace. With only the
largest point traced (the default), that summary would be a single-point
"trend", so it's skipped — the per-size trend to look at instead is
`{name}_scale.png`, built from every point's CSV row regardless of tracing.

For the single largest size in the run (always, sweep or not), if its markers
carry 2+ distinct labels (e.g. samplesort_three_way's peter/direct/primitives),
each label's own build_start..op_end window is additionally sliced out and
plotted/saved standalone (trace_<label>_throughput/_cpu/_drives.png, plus
trace_<label>.csv, same per-panel breakdown as the combined trace) — the
per-algorithm breakdown, generated once rather than at every sweep point since
it's mainly useful at the scale you actually care about.

MEANINGFUL ONLY ON REAL BLOCK DEVICES.  On the tmpfs dev box the "SSDs" are
RAM-backed and generate no /proc/diskstats traffic, so the disk panels come out
empty (the script warns and still records CPU).  Run it on the 30-SSD machine.

  usage:
    python3 benchmarks/io_trace.py <example> [--size 1GiB | --n 8192]
        [--interval 0.1] [--mount-glob /mnt/ssd*] [--outdir results]
        [--ssd-args '...'] [-- <extra args passed through to the example binary>]

    python3 benchmarks/io_trace.py samplesort_three_way \\
        --size "2^30 2^32 2^34 2^36 2^38"

    # --n bypasses --size/size_to_n and passes raw counts straight through as
    # argv[1] -- for examples (like bellman_ford's slow per-vertex variant)
    # where the byte-size model picks an impractically large n:
    python3 benchmarks/io_trace.py bellman_ford_sparse --n "512 1024 2048 4096 8192"

Reuses run_benches.py by import: the EXAMPLES registry, parse_bytes, parse_count,
size_to_n, make(), clear_bench_data(), REPO_ROOT/BINDIR.  Compilation stays in
the Makefile.
"""

import argparse
import glob
import os
import sys
import threading
import time
import subprocess
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benches as rb  # noqa: E402  (sibling module; import-safe)

SECTOR_BYTES = 512  # /proc/diskstats sectors are always 512 B


# ── device resolution ───────────────────────────────────────────────────────
def parse_mountinfo():
    """Return [(mount_point, device_basename), ...] for block-backed mounts.

    /proc/self/mountinfo lines are ` ... - <fstype> <source> <superopts>`; the
    mount point is the pre-`-` field 4 and the backing source the post-`-`
    field 1.  Only /dev/-backed sources have a /proc/diskstats row, so tmpfs
    and friends are dropped here.
    """
    out = []
    with open("/proc/self/mountinfo") as f:
        for line in f:
            left, _, right = line.partition(" - ")
            lf, rf = left.split(), right.split()
            if len(lf) < 5 or len(rf) < 2:
                continue
            mount_point, source = lf[4], rf[1]
            if source.startswith("/dev/"):
                out.append((mount_point, os.path.basename(source)))
    return out


def resolve_devices(mount_glob):
    """Map each mount matching mount_glob to its /proc/diskstats device name.

    Uses the longest mount-point prefix, so a path that is itself a mount and a
    path that is just a directory on a bigger filesystem both resolve.  Returns
    (sorted unique device names, {mount: device}); the device list is empty on
    a tmpfs box (nothing block-backed), which the caller warns about.
    """
    table = parse_mountinfo()
    mapping = {}
    for path in sorted(glob.glob(mount_glob)):
        rp = os.path.realpath(path)
        best_mp, best_dev = "", None
        for mp, dev in table:
            if (rp == mp or rp.startswith(mp.rstrip("/") + "/")) and len(mp) >= len(best_mp):
                best_mp, best_dev = mp, dev
        if best_dev:
            mapping[path] = best_dev
    devices = sorted(set(mapping.values()))
    return devices, mapping


# ── /proc sampling ──────────────────────────────────────────────────────────
def read_diskstats(devices):
    """{dev: (read_sectors, write_sectors, io_ticks_ms)} for the given devices."""
    want = set(devices)
    snap = {}
    with open("/proc/diskstats") as f:
        for line in f:
            p = line.split()
            if len(p) < 14:
                continue
            name = p[2]
            if name not in want:
                continue
            # After name: reads rmerged rsectors rms writes wmerged wsectors ...
            #             io_ticks is field index 9 (p[12]).
            snap[name] = (int(p[5]), int(p[9]), int(p[12]))
    return snap


def read_cpu():
    """(idle+iowait, non_idle, iowait) jiffies from the aggregate cpu line."""
    with open("/proc/stat") as f:
        parts = f.readline().split()
    v = [int(x) for x in parts[1:]]  # user nice system idle iowait irq softirq steal ...
    user, nice, system, idle, iowait = v[0], v[1], v[2], v[3], v[4]
    irq, softirq, steal = v[5], v[6], v[7]
    idle_all = idle + iowait
    non_idle = user + nice + system + irq + softirq + steal
    return idle_all, non_idle, iowait


class Sampler(threading.Thread):
    """Background thread: snapshot diskstats + cpu every `interval` seconds."""

    def __init__(self, devices, interval):
        super().__init__(daemon=True)
        self.devices = devices
        self.interval = interval
        self.samples = []  # (mono_time, disk_snap, cpu_snap)
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            self.samples.append((time.monotonic(),
                                 read_diskstats(self.devices), read_cpu()))
            self._stop.wait(self.interval)
        # one final sample so the last interval is closed off
        self.samples.append((time.monotonic(),
                             read_diskstats(self.devices), read_cpu()))

    def stop(self):
        self._stop.set()


# ── run the example under the sampler ───────────────────────────────────────
def run_traced(binary, args, devices, interval):
    """Launch the binary with PLAID_TRACE=1, sampling concurrently.

    Returns (samples, markers, stdout).  markers = [(label, mono_seconds)].
    """
    env = dict(os.environ, PLAID_TRACE="1")
    sampler = Sampler(devices, interval)
    sampler.start()
    cmd = [binary] + [str(a) for a in args]
    print(f"  $ PLAID_TRACE=1 {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=rb.REPO_ROOT, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sampler.stop()
    sampler.join()
    print(proc.stdout, end="", flush=True)
    if proc.returncode != 0:
        print(f"  !!! binary exited {proc.returncode} (trace still written)", flush=True)

    markers = []
    for line in proc.stdout.splitlines():
        if line.startswith("TRACE,"):
            _, label, mono = line.split(",", 2)
            markers.append((label, float(mono)))
    return sampler.samples, markers, proc.stdout


# ── reduce raw counters to per-interval rates ───────────────────────────────
def compute_series(samples, devices):
    """Turn raw counter snapshots into per-interval rate series.

    Each output point is anchored at the right edge of an interval; rates are
    delta/dt.  Returns a dict of parallel lists (times[], agg_read_mbps[], ...)
    plus per-device read/write/util keyed by device name.
    """
    ser = {"t": [], "agg_read": [], "agg_write": [], "mean_util": [],
           "cpu": [], "iowait": [],
           "dev_read": {d: [] for d in devices},
           "dev_write": {d: [] for d in devices},
           "dev_util": {d: [] for d in devices}}
    for (t0, d0, c0), (t1, d1, c1) in zip(samples, samples[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        ser["t"].append(t1)
        tot_r = tot_w = 0.0
        utils = []
        for dev in devices:
            r0, tk0 = d0.get(dev, (0, 0, 0))[0], d0.get(dev, (0, 0, 0))[2]
            w0 = d0.get(dev, (0, 0, 0))[1]
            r1, w1, tk1 = d1.get(dev, (r0, w0, tk0))
            rd = max(0, r1 - r0) * SECTOR_BYTES / 1e6 / dt   # MB/s
            wr = max(0, w1 - w0) * SECTOR_BYTES / 1e6 / dt
            ut = min(100.0, max(0, tk1 - tk0) / (dt * 1000.0) * 100.0)  # %busy
            ser["dev_read"][dev].append(rd)
            ser["dev_write"][dev].append(wr)
            ser["dev_util"][dev].append(ut)
            tot_r += rd
            tot_w += wr
            utils.append(ut)
        ser["agg_read"].append(tot_r)
        ser["agg_write"].append(tot_w)
        ser["mean_util"].append(sum(utils) / len(utils) if utils else 0.0)

        idle0, non0, iow0 = c0
        idle1, non1, iow1 = c1
        d_non = non1 - non0
        d_iow = iow1 - iow0
        d_tot = (idle1 + non1) - (idle0 + non0)
        ser["cpu"].append(100.0 * d_non / d_tot if d_tot > 0 else 0.0)
        ser["iowait"].append(100.0 * d_iow / d_tot if d_tot > 0 else 0.0)
    return ser


# ── marker pairing (per-algorithm phase windows, for the sweep summary) ─────
# Recognized start/end prefixes, matching utils/trace_marker.h's callers
# (bellman_ford.cpp, samplesort_three_way.cpp): "<kind>_start_<label>" /
# "<kind>_end_<label>" for kind in {build, op, fast_op}. Longest-first so
# "fast_op_start_" isn't mistaken for a plain "op_..." prefix.
MARKER_KINDS = ["fast_op", "build", "op"]


def split_marker(label):
    """'op_start_primitives' -> ('op', 'start', 'primitives'); else None."""
    for kind in sorted(MARKER_KINDS, key=len, reverse=True):
        for edge in ("start", "end"):
            prefix = f"{kind}_{edge}_"
            if label.startswith(prefix):
                return kind, edge, label[len(prefix):]
    return None


def pair_windows(markers):
    """{label: {kind: (start_mono, end_mono)}} from a flat marker list.

    Unmatched starts (no corresponding end — e.g. the binary crashed mid-phase)
    are dropped rather than guessed at.
    """
    pending = {}
    windows = {}
    for marker_label, mono in markers:
        parsed = split_marker(marker_label)
        if parsed is None:
            continue
        kind, edge, label = parsed
        key = (label, kind)
        if edge == "start":
            pending[key] = mono
        elif key in pending:
            windows.setdefault(label, {})[kind] = (pending.pop(key), mono)
    return windows


def window_series_stats(ser, t0, start_mono, end_mono):
    """Average agg_read/agg_write/mean_util/cpu over samples inside a window."""
    lo, hi = start_mono - t0, end_mono - t0
    idxs = [i for i, t in enumerate(ser["t"]) if lo <= (t - t0) <= hi]
    dur = max(0.0, end_mono - start_mono)
    if not idxs:
        return {"dur_s": dur, "avg_util_pct": 0.0, "avg_cpu_pct": 0.0,
                "avg_read_mbps": 0.0, "avg_write_mbps": 0.0}
    n = len(idxs)
    return {
        "dur_s": dur,
        "avg_util_pct": sum(ser["mean_util"][i] for i in idxs) / n,
        "avg_cpu_pct": sum(ser["cpu"][i] for i in idxs) / n,
        "avg_read_mbps": sum(ser["agg_read"][i] for i in idxs) / n,
        "avg_write_mbps": sum(ser["agg_write"][i] for i in idxs) / n,
    }


def slice_ser(ser, devices, lo_mono, hi_mono):
    """Restrict a compute_series() result to samples within [lo_mono, hi_mono].

    Same shape as compute_series' output, so it can be fed straight back into
    write_trace_csv/plot_trace — used to carve one algorithm's own window out
    of a multi-algorithm run (e.g. samplesort_three_way's three sorts) and
    plot/save it exactly like a standalone single-algorithm trace.
    """
    idxs = [i for i, t in enumerate(ser["t"]) if lo_mono <= t <= hi_mono]
    return {
        "t": [ser["t"][i] for i in idxs],
        "agg_read": [ser["agg_read"][i] for i in idxs],
        "agg_write": [ser["agg_write"][i] for i in idxs],
        "mean_util": [ser["mean_util"][i] for i in idxs],
        "cpu": [ser["cpu"][i] for i in idxs],
        "iowait": [ser["iowait"][i] for i in idxs],
        "dev_read": {d: [ser["dev_read"][d][i] for i in idxs] for d in devices},
        "dev_write": {d: [ser["dev_write"][d][i] for i in idxs] for d in devices},
        "dev_util": {d: [ser["dev_util"][d][i] for i in idxs] for d in devices},
    }


# ── output ──────────────────────────────────────────────────────────────────
def write_trace_csv(path, ser, devices, t0):
    header = ["time_s", "agg_read_mbps", "agg_write_mbps", "mean_util_pct",
              "cpu_pct", "iowait_pct"]
    for d in devices:
        header += [f"{d}_read_mbps", f"{d}_write_mbps", f"{d}_util_pct"]
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for i, t in enumerate(ser["t"]):
            row = [f"{t - t0:.4f}", f"{ser['agg_read'][i]:.3f}",
                   f"{ser['agg_write'][i]:.3f}", f"{ser['mean_util'][i]:.2f}",
                   f"{ser['cpu'][i]:.2f}", f"{ser['iowait'][i]:.2f}"]
            for d in devices:
                row += [f"{ser['dev_read'][d][i]:.3f}",
                        f"{ser['dev_write'][d][i]:.3f}",
                        f"{ser['dev_util'][d][i]:.2f}"]
            f.write(",".join(row) + "\n")
    print(f"  wrote {path}", flush=True)


PHASE_STYLE = {"build_start": ("tab:orange", "Setup"),
               "build_end": ("tab:gray", "Settling"),
               "op_start": ("tab:green", "Begin"),
               "op_end": ("tab:red", "End")}


def _overlay_phases(ax, markers, t0):
    for label, mono in markers:
        color, _ = PHASE_STYLE.get(label, ("k", label))
        ax.axvline(mono - t0, color=color, linestyle="--", linewidth=1.0, alpha=0.8)


def _phase_legend(fig, markers):
    """Bottom phase-boundary legend, if any markers fall on this panel."""
    seen = {}
    for label, _ in markers:
        color, name = PHASE_STYLE.get(label, ("k", label))
        seen[name] = color
    if not seen:
        return
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, linestyle="--", label=n)
               for n, c in seen.items()]
    # Below the x-axis label (not just the axes) so it doesn't collide with
    # it now that each panel is its own standalone figure; bbox_inches="tight"
    # keeps it in frame.
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.15), title="Phase Boundaries")


def plot_trace(ser, markers, devices, t0, path):
    """Write three standalone panels — throughput / CPU / per-drive access
    pattern — as separate PNGs derived from `path` (<base>_throughput.png,
    <base>_cpu.png, <base>_drives.png) instead of one combined figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style
    plot_style.apply()

    base, ext = os.path.splitext(path)
    xs = [t - t0 for t in ser["t"]]

    # A — aggregate throughput (read/write asymmetry).
    fig, ax_bw = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax_bw.plot(xs, ser["agg_read"], "-", color=plot_style.PALETTE["blue"], label="read")
    ax_bw.plot(xs, ser["agg_write"], "-", color=plot_style.PALETTE["red"], label="write")
    ax_bw.set_ylabel("Aggregate MB/s")
    ax_bw.set_xlabel("Seconds Since Initial Sample")
    ax_bw.set_title("Aggregate Throughput Over All Drives")
    ax_bw.grid(True)
    _overlay_phases(ax_bw, markers, t0)
    ax_bw.legend(loc="upper right", fontsize=9)
    _phase_legend(fig, markers)
    throughput_path = f"{base}_throughput{ext}"
    fig.savefig(throughput_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {throughput_path}", flush=True)

    # B — CPU/disk utilization (io-bound vs cpu-bound).
    fig, ax_bn = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax_bn.plot(xs, ser["mean_util"], "-", color=plot_style.PALETTE["violet"], label="mean drive %util")
    ax_bn.plot(xs, ser["cpu"], "-", color=plot_style.PALETTE["green"], label="CPU %")
    ax_bn.plot(xs, ser["iowait"], "-", color=plot_style.PALETTE["orange"], label="iowait %")
    ax_bn.set_ylabel("Percent")
    ax_bn.set_xlabel("Seconds Since Initial Sample")
    ax_bn.set_ylim(0, 105)
    ax_bn.set_title("CPU/Disk Utilization")
    ax_bn.grid(True)
    _overlay_phases(ax_bn, markers, t0)
    ax_bn.legend(loc="upper left", fontsize=9)
    _phase_legend(fig, markers)
    cpu_path = f"{base}_cpu{ext}"
    fig.savefig(cpu_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {cpu_path}", flush=True)

    # C — per-drive access pattern heatmap (read+write MB/s).
    fig, ax_hm = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    if devices and xs:
        import numpy as np
        mat = np.array([[ser["dev_read"][d][i] + ser["dev_write"][d][i]
                         for i in range(len(xs))] for d in devices])
        im = ax_hm.imshow(mat, aspect="auto", origin="lower", cmap="viridis",
                          extent=[xs[0], xs[-1], -0.5, len(devices) - 0.5],
                          interpolation="nearest")
        ax_hm.set_yticks(range(len(devices)))
        ax_hm.set_yticklabels([f"drive {i}" for i in range(len(devices))], fontsize=6)
        fig.colorbar(im, ax=ax_hm, label="Read, Write MB/s")
    else:
        ax_hm.text(0.5, 0.5, "no block-device traffic\n(tmpfs? wrong --mount-glob?)",
                   ha="center", va="center", transform=ax_hm.transAxes)
    ax_hm.set_title("Drive Access (MB/s)")
    ax_hm.set_xlabel("Seconds Since Initial Sample")
    _overlay_phases(ax_hm, markers, t0)
    _phase_legend(fig, markers)
    drives_path = f"{base}_drives{ext}"
    fig.savefig(drives_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {drives_path}", flush=True)


def write_sweep_csv(path, rows):
    """One row per size; columns are the union of every row's <label>_<kind>_*
    window-stat keys (missing ones blank), so a size that didn't reach a later
    phase still produces a well-formed row instead of breaking the header.
    """
    metric_cols = []
    seen = set()
    for row in rows:
        for col in row:
            if col in ("size", "size_bytes", "n") or col in seen:
                continue
            seen.add(col)
            metric_cols.append(col)
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else ("" if v is None else str(v))

    header = ["size", "size_bytes", "n"] + metric_cols
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(fmt(row.get(c)) for c in header) + "\n")
    print(f"  wrote {path}", flush=True)


def plot_sweep(rows, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style
    plot_style.apply()

    ns = [row["n"] for row in rows]
    # Discover which (label, kind) windows actually appear anywhere in the
    # sweep, e.g. [("peter","op"), ("direct","op"), ("primitives","op"), ...].
    combos = []
    for row in rows:
        for col in row:
            if col.endswith("_dur_s"):
                combo = col[: -len("_dur_s")]
                if combo not in combos:
                    combos.append(combo)
    if not combos:
        print("  (no paired start/end markers found; skipping sweep PNG)", flush=True)
        return

    def series(row_key):
        # A size that never reached this phase (e.g. the binary crashed
        # earlier) has no such key; plot a gap there rather than crashing.
        return [row.get(row_key) if row.get(row_key) is not None else float("nan")
                for row in rows]

    fig, (ax_dur, ax_bn, ax_bw) = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, combo in enumerate(combos):
        c = colors[i % len(colors)]
        durs = series(f"{combo}_dur_s")
        ax_dur.plot(ns, durs, "o-", color=c, label=combo)
        util = series(f"{combo}_avg_util_pct")
        cpu = series(f"{combo}_avg_cpu_pct")
        ax_bn.plot(ns, util, "o-", color=c, label=f"{combo} util%")
        ax_bn.plot(ns, cpu, "o--", color=c, label=f"{combo} cpu%")
        rd = series(f"{combo}_avg_read_mbps")
        wr = series(f"{combo}_avg_write_mbps")
        thr = [r + w for r, w in zip(rd, wr)]
        ax_bw.plot(ns, thr, "o-", color=c, label=combo)

    ax_dur.set_xscale("log"); ax_dur.set_ylabel("duration (s)")
    ax_dur.set_title("Phase duration vs n"); ax_dur.grid(True)
    ax_dur.legend(fontsize=8)

    ax_bn.set_xscale("log"); ax_bn.set_ylabel("Percent"); ax_bn.set_ylim(0, 105)
    ax_bn.set_title("Mean drive %util (solid) vs CPU% (dashed) during each phase")
    ax_bn.grid(True)
    ax_bn.legend(fontsize=7, ncol=2)

    ax_bw.set_xscale("log"); ax_bw.set_yscale("log")
    ax_bw.set_ylabel("Aggregate MB/s"); ax_bw.set_xlabel("n (elements)")
    ax_bw.set_title("Mean throughput during each phase")
    ax_bw.grid(True)
    ax_bw.legend(fontsize=8)

    fig.suptitle(title)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


def _desc(label, n, n_mode):
    """Human-readable point description for print()/titles: 'n=8192' in --n
    mode (label already IS the n token, so showing n again would be
    redundant), 'size=1MiB (n=1048576)' in --size mode."""
    return f"n={label}" if n_mode else f"size={label} (n={n})"


# ── main ────────────────────────────────────────────────────────────────────
def main():
    argv = sys.argv[1:]
    passthrough = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("example", help="example name; choices: "
                    + ", ".join(e["name"] for e in rb.EXAMPLES))
    ap.add_argument("--size", default=None,
                    help="one size, or a space-separated list to sweep (e.g. "
                         "'2^30 2^32 2^34'); each converted per example to an "
                         "element count (see run_benches.size_to_n). Default "
                         "'512MiB' if neither --size nor --n is given. Mutually "
                         "exclusive with --n.")
    ap.add_argument("--n", default=None,
                    help="one count, or a space-separated list to sweep, as the "
                         "example's raw argv[1] (e.g. bellman_ford's vertex "
                         "count) -- bypasses --size/size_to_n entirely. Use this "
                         "when the byte-size model picks an impractically large "
                         "n (e.g. '512 1024 2048 4096 8192' for bellman_ford's "
                         "slow per-vertex variant, instead of a size like "
                         "'1MiB' that size_to_n would turn into a six-figure "
                         "vertex count). Accepts the same forms as "
                         "run_benches.py's --n-values (raw digits, 2^k, K/M/G "
                         "suffixes). Mutually exclusive with --size.")
    ap.add_argument("--interval", type=float, default=0.1,
                    help="sampling interval in seconds (default 0.1)")
    ap.add_argument("--mount-glob", default="/mnt/ssd*",
                    help="glob of SSD mounts to resolve to block devices")
    ap.add_argument("--outdir", default=os.environ.get("BENCH_OUTDIR", "results"),
                    help="parent dir for the timestamped run (default: results)")
    ap.add_argument("--ssd-args", default=os.environ.get("BENCH_SSD_ARGS", ""),
                    help="extra global flags for the binary (e.g. '--num_ssd=4')")
    ap.add_argument("--no-clean", action="store_true",
                    help="leave the example's data files on the mounts")
    ap.add_argument("--trace-all", action="store_true",
                    help="disk/CPU-sample every --size point, not just the largest "
                         "(also restores the multi-point io_sweep.csv/.png summary)")
    args = ap.parse_args(argv)

    if args.size and args.n:
        ap.error("--size and --n are mutually exclusive")

    entry = next((e for e in rb.EXAMPLES if e["name"] == args.example), None)
    if entry is None:
        ap.error(f"unknown example {args.example!r}; choices: "
                 + ", ".join(e["name"] for e in rb.EXAMPLES))

    n_mode = args.n is not None
    if n_mode:
        point_labels = args.n.split()
        if not point_labels:
            ap.error("--n must not be empty")
        ns = [rb.parse_count(t) for t in point_labels]
        # Best-effort bytes estimate for perf_row["input_bytes"] (the *_scale.png
        # x-axis) and the trace title -- exact for entries without n_from_size
        # (the same linear formula size_to_n itself uses, just inverted), an
        # approximation for entries like bellman_ford_dense where it's cosmetic
        # only (see size_to_n / the registry entry's own comment).
        sizes_bytes = [n * entry["elem_bytes"] * entry["input_seqs"] for n in ns]
    else:
        point_labels = (args.size or "512MiB").split()
        if not point_labels:
            ap.error("--size must not be empty")
        sizes_bytes = [rb.parse_bytes(s) for s in point_labels]
        ns = [rb.size_to_n(entry, b) for b in sizes_bytes]
    # First occurrence of the max, so a tie deterministically picks the
    # earliest point rather than depending on list-scan order elsewhere.
    largest_idx = ns.index(max(ns))

    devices, mapping = resolve_devices(args.mount_glob)
    if devices:
        print(f"Resolved {len(mapping)} mount(s) -> {len(devices)} device(s): "
              f"{', '.join(devices)}", flush=True)
    else:
        print(f"  !!! no block devices resolved from {args.mount_glob!r} — disk "
              "panels will be EMPTY (tmpfs dev box? wrong glob?). CPU still "
              "recorded.", flush=True)

    rb.make(entry["target"])
    binary = os.path.join(rb.BINDIR, os.path.basename(entry["target"]))
    flags = args.ssd_args.split() if args.ssd_args else []

    # One stamp for the whole sweep, so every point's trace dir and the sweep
    # summary land together under the same results/<stamp>/ directory.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_dir = os.path.join(rb.REPO_ROOT, args.outdir, stamp)
    sweep_rows = []
    # Rows for the example's own performance comparison (the CSV,... line it
    # already prints — n vs. each implementation's time/throughput; this is
    # what run_benches.py's normal sweep would plot). io_trace.py runs the
    # binary anyway, so this is free — no separate run_benches.py pass needed.
    perf_rows = []
    # Raw (ser, markers, t0, outdir, n, desc) for the traced point that
    # turns out largest (normally just THE traced point, since only
    # largest_idx is sampled by default); per-algorithm breakdown plots are
    # generated for this one only, after the loop — the largest point is the
    # one where per-algorithm IO/CPU behavior is most informative anyway.
    largest = None

    for point_idx, point_label in enumerate(point_labels):
        size_bytes = sizes_bytes[point_idx]
        n = ns[point_idx]
        trace_this_point = args.trace_all or point_idx == largest_idx
        desc = _desc(point_label, n, n_mode)
        print(f"\n--- {args.example}  {desc} "
              f"[{point_idx + 1}/{len(point_labels)}] "
              f"{'[traced]' if trace_this_point else '[perf only]'} ---", flush=True)

        # ParseGlobalArguments consumes only *leading* --flags, then positionals
        # shift down; so order must be [--flags] [n] [example-specific positionals].
        # entry["extra_argv"] (e.g. bellman_ford_*'s case filter) are fixed
        # positionals the registry entry itself needs; passthrough is whatever
        # the user added after "--" on io_trace.py's own command line.
        bin_args = flags + [n] + entry.get("extra_argv", []) + passthrough

        if not trace_this_point:
            # No Sampler, no PLAID_TRACE — just the CSV performance row, at
            # the actual run cost (sampling itself is cheap; skipping it here
            # doesn't speed up the example, but it does skip the per-point
            # trace.csv/trace.png work, and — the main point — lets the
            # smaller points be swept at all without paying for a trace
            # nobody was going to look at).
            fields, problem = rb.run_binary(binary, bin_args, fatal=False)
            if problem:
                print(f"  !!! {problem}", flush=True)
            if fields is not None and len(fields) == len(entry["cols"]):
                perf_row = dict(zip(entry["cols"], fields))
                perf_row["input_bytes"] = str(size_bytes)
                perf_rows.append(perf_row)
            elif fields is not None:
                print(f"  !!! no (or malformed) CSV, line for {desc}; "
                      "excluded from the performance comparison", flush=True)
            rb.clear_bench_data(args.mount_glob, not args.no_clean)
            continue

        samples, markers, stdout = run_traced(binary, bin_args, devices, args.interval)
        if len(samples) < 2:
            print(f"  !!! too few samples for {desc} (run too short for "
                  "--interval); skipping this point", flush=True)
            continue

        # Same CSV,... parsing run_benches.run_binary uses, so this row is a
        # drop-in match for rb.write_csv/rb.plot_example's expected shape.
        csv_fields = None
        for line in stdout.splitlines():
            if line.startswith("CSV,"):
                csv_fields = line[len("CSV,"):].split(",")
        if csv_fields is not None and len(csv_fields) == len(entry["cols"]):
            perf_row = dict(zip(entry["cols"], csv_fields))
            perf_row["input_bytes"] = str(size_bytes)
            perf_rows.append(perf_row)
        else:
            print(f"  !!! no (or malformed) CSV, line for {desc}; "
                  "excluded from the performance comparison", flush=True)

        ser = compute_series(samples, devices)
        t0 = samples[0][0]

        # point_idx suffix avoids collisions when the same point appears twice
        # in one sweep (e.g. --n "8192 8192" to check run-to-run variance).
        outdir = os.path.join(sweep_dir, f"trace_{entry['name']}_{point_label}_{point_idx}")
        os.makedirs(outdir, exist_ok=True)
        print(f"Trace directory: {outdir}", flush=True)

        write_trace_csv(os.path.join(outdir, "trace.csv"), ser, devices, t0)
        plot_trace(ser, markers, devices, t0, os.path.join(outdir, "trace.png"))

        # tidy the drives before the next point, matching run_benches' hygiene
        rb.clear_bench_data(args.mount_glob, not args.no_clean)

        ph = ", ".join(f"{lbl}@{mono - t0:.2f}s" for lbl, mono in markers) or "none"
        print(f"Done. Phases: {ph}\nResults in {outdir}", flush=True)

        if largest is None or n > largest["n"]:
            largest = {"ser": ser, "markers": markers, "t0": t0, "outdir": outdir,
                      "n": n, "desc": desc}

        # Raw values (not pre-formatted): plot_sweep needs floats to plot;
        # write_sweep_csv formats at write time.
        row = {"size": point_label, "size_bytes": size_bytes, "n": n}
        for label, kinds in pair_windows(markers).items():
            for kind, (start_mono, end_mono) in kinds.items():
                stats = window_series_stats(ser, t0, start_mono, end_mono)
                for stat_name, val in stats.items():
                    row[f"{label}_{kind}_{stat_name}"] = val
        sweep_rows.append(row)

    # Per-algorithm breakdown (same 3-file style as the combined trace:
    # aggregate throughput / CPU-disk utilization / per-drive heatmap, each
    # its own PNG) for the largest point only, one set of plots per label
    # found in its markers (e.g. peter/direct/primitives for
    # samplesort_three_way). Each is that algorithm's own build_start..op_end
    # window sliced out and plotted exactly as if it had been traced alone.
    if largest is not None:
        windows = pair_windows(largest["markers"])
        if len(windows) >= 2:
            print(f"\nPer-algorithm breakdown for the largest point "
                  f"({largest['desc']}):", flush=True)
            for label, kinds in windows.items():
                lo = min(w[0] for w in kinds.values())
                hi = max(w[1] for w in kinds.values())
                sub_ser = slice_ser(largest["ser"], devices, lo, hi)
                if len(sub_ser["t"]) < 2:
                    print(f"  {label}: too few samples in its window, skipping",
                          flush=True)
                    continue
                sub_markers = [(lbl, m) for lbl, m in largest["markers"] if lo <= m <= hi]
                write_trace_csv(os.path.join(largest["outdir"], f"trace_{label}.csv"),
                                sub_ser, devices, lo)
                plot_trace(sub_ser, sub_markers, devices, lo,
                          os.path.join(largest["outdir"], f"trace_{label}.png"))
        elif windows:
            print(f"\n(only one labeled phase ({next(iter(windows))}) found; "
                  "nothing to split into a per-algorithm breakdown)", flush=True)

    # The example's own headline comparison (n vs. each implementation's time/
    # throughput — what run_benches.py's normal sweep would produce), reusing
    # its write_csv/plot_example directly so this matches every other
    # *_scale.csv/.png in the repo exactly, not a separate reimplementation.
    # Every point contributes here regardless of whether it was traced.
    if len(point_labels) > 1 and perf_rows:
        os.makedirs(sweep_dir, exist_ok=True)
        rb.write_csv(os.path.join(sweep_dir, f"{entry['name']}_scale.csv"),
                    ["input_bytes"] + entry["cols"], perf_rows)
        try:
            rb.plot_example(perf_rows, entry,
                            os.path.join(sweep_dir, f"{entry['name']}_scale.png"))
        except Exception as exc:
            print(f"  !!! {entry['name']}_scale.png plotting failed ({exc}); "
                  "CSV was written", flush=True)

    # The multi-point IO/CPU trend needs 2+ TRACED points, not just 2+ sizes
    # requested — with the default (only the largest size traced), that's
    # normally exactly one, so there's nothing to plot a trend across; this
    # is expected, not a failure, so it's a note rather than a warning.
    # (--trace-all restores one traced point per size, as before.)
    if len(sweep_rows) > 1:
        os.makedirs(sweep_dir, exist_ok=True)
        write_sweep_csv(os.path.join(sweep_dir, "io_sweep.csv"), sweep_rows)
        plot_sweep(sweep_rows, os.path.join(sweep_dir, "io_sweep.png"),
                  f"{args.example}: IO/CPU sweep across {len(sweep_rows)} sizes")
        print(f"\nSweep done. Summary in {sweep_dir}", flush=True)
    elif len(point_labels) > 1 and not args.trace_all:
        print(f"\n(only the largest point was traced, as usual — pass --trace-all "
              "for a multi-point io_sweep.csv/.png)", flush=True)
    elif len(point_labels) > 1:
        print("\n  !!! every traced point failed; no io_sweep summary written", flush=True)


if __name__ == "__main__":
    main()
