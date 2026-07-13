#!/usr/bin/env python3
"""Single-run IO/CPU profiler for the examples — time-series, not a sweep.

Where run_benches.py sweeps `n` and records one aggregate number per point,
this runs ONE example at ONE `n` and samples what the machine is actually doing
over the course of the run: per-SSD read/write throughput + device utilization
(from /proc/diskstats) and CPU (from /proc/stat), every `--interval` seconds.
The example is launched with PLAID_TRACE=1 so it emits `TRACE,<phase>,<mono>`
markers (see utils/trace_marker.h); those land on the same CLOCK_MONOTONIC
timeline the sampler uses, so the build vs. timed-op phases are drawn as
labelled bands on the plot.

It answers, for a given algorithm:
  * IO-bound vs CPU-bound — device %util pinned near 100 with CPU headroom is
    IO-bound; the inverse is CPU-bound.
  * read/write asymmetry — aggregate read MB/s vs write MB/s over time.
  * per-drive access pattern — a device x time heatmap of read+write MB/s.

MEANINGFUL ONLY ON REAL BLOCK DEVICES.  On the tmpfs dev box the "SSDs" are
RAM-backed and generate no /proc/diskstats traffic, so the disk panels come out
empty (the script warns and still records CPU).  Run it on the 30-SSD machine.

  usage:
    python3 benchmarks/io_trace.py <example> [--size 1GiB] [--interval 0.1]
        [--mount-glob /mnt/ssd*] [--outdir results] [--ssd-args '...']
        [-- <extra args passed through to the example binary>]

Reuses run_benches.py by import: the EXAMPLES registry, parse_bytes, size_to_n,
make(), clear_bench_data(), REPO_ROOT/BINDIR.  Compilation stays in the Makefile.
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


PHASE_STYLE = {"build_start": ("tab:orange", "build"),
               "build_end": ("tab:gray", "quiesce"),
               "op_start": ("tab:green", "op"),
               "op_end": ("tab:red", "end")}


def _overlay_phases(ax, markers, t0):
    for label, mono in markers:
        color, _ = PHASE_STYLE.get(label, ("k", label))
        ax.axvline(mono - t0, color=color, linestyle="--", linewidth=1.0, alpha=0.8)


def plot_trace(ser, markers, devices, t0, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [t - t0 for t in ser["t"]]
    fig, (ax_bw, ax_bn, ax_hm) = plt.subplots(
        3, 1, figsize=(12, 12), constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 1, 1.3]})

    # A — aggregate throughput (read/write asymmetry).
    ax_bw.plot(xs, ser["agg_read"], "-", color="tab:blue", label="read")
    ax_bw.plot(xs, ser["agg_write"], "-", color="tab:red", label="write")
    ax_bw.set_ylabel("aggregate MB/s")
    ax_bw.set_title("Aggregate throughput across all drives (read vs write)")
    ax_bw.grid(True, linestyle=":", linewidth=0.5)
    _overlay_phases(ax_bw, markers, t0)
    ax_bw.legend(loc="upper right")

    # B — bottleneck (io-bound vs cpu-bound).
    ax_bn.plot(xs, ser["mean_util"], "-", color="tab:purple", label="mean drive %util")
    ax_bn.plot(xs, ser["cpu"], "-", color="tab:green", label="CPU %")
    ax_bn.plot(xs, ser["iowait"], "-", color="tab:orange", label="iowait %")
    ax_bn.set_ylabel("percent")
    ax_bn.set_ylim(0, 105)
    ax_bn.set_title("Bottleneck: drive utilization vs CPU")
    ax_bn.grid(True, linestyle=":", linewidth=0.5)
    _overlay_phases(ax_bn, markers, t0)
    ax_bn.legend(loc="upper right")

    # C — per-drive access pattern heatmap (read+write MB/s).
    if devices and xs:
        import numpy as np
        mat = np.array([[ser["dev_read"][d][i] + ser["dev_write"][d][i]
                         for i in range(len(xs))] for d in devices])
        im = ax_hm.imshow(mat, aspect="auto", origin="lower", cmap="viridis",
                          extent=[xs[0], xs[-1], -0.5, len(devices) - 0.5],
                          interpolation="nearest")
        ax_hm.set_yticks(range(len(devices)))
        ax_hm.set_yticklabels(devices, fontsize=6)
        fig.colorbar(im, ax=ax_hm, label="read+write MB/s")
    else:
        ax_hm.text(0.5, 0.5, "no block-device traffic\n(tmpfs? wrong --mount-glob?)",
                   ha="center", va="center", transform=ax_hm.transAxes)
    ax_hm.set_title("Per-drive access pattern (read+write MB/s)")
    ax_hm.set_xlabel("seconds since first sample")
    _overlay_phases(ax_hm, markers, t0)

    # phase legend across the top
    seen = {}
    for label, _ in markers:
        color, name = PHASE_STYLE.get(label, ("k", label))
        seen[name] = color
    if seen:
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color=c, linestyle="--", label=n)
                   for n, c in seen.items()]
        # Below the figure so it never collides with the suptitle / panel A
        # title; bbox_inches="tight" keeps it in frame.
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   bbox_to_anchor=(0.5, -0.05), title="phase boundaries")
    fig.suptitle(title)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


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
    ap.add_argument("--size", default="512MiB",
                    help="input size (e.g. 1GiB, 512MiB); converted per example to "
                         "an element count (see run_benches.size_to_n)")
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
    args = ap.parse_args(argv)

    entry = next((e for e in rb.EXAMPLES if e["name"] == args.example), None)
    if entry is None:
        ap.error(f"unknown example {args.example!r}; choices: "
                 + ", ".join(e["name"] for e in rb.EXAMPLES))
    size = rb.parse_bytes(args.size)
    n = rb.size_to_n(entry, size)

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

    # ParseGlobalArguments consumes only *leading* --flags, then positionals
    # shift down; so order must be [--flags] [n] [example-specific positionals].
    flags = args.ssd_args.split() if args.ssd_args else []
    bin_args = flags + [n] + passthrough

    samples, markers, _ = run_traced(binary, bin_args, devices, args.interval)
    if len(samples) < 2:
        sys.exit("too few samples (run too short for --interval); nothing to plot")

    ser = compute_series(samples, devices)
    t0 = samples[0][0]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(rb.REPO_ROOT, args.outdir, stamp,
                          f"trace_{entry['name']}_{args.size}")
    os.makedirs(outdir, exist_ok=True)
    print(f"Trace directory: {outdir}", flush=True)

    write_trace_csv(os.path.join(outdir, "trace.csv"), ser, devices, t0)
    title = (f"{entry['name']}  size={args.size} (n={n})  "
             f"({len(devices)} drives, {args.interval}s samples)")
    plot_trace(ser, markers, devices, t0, os.path.join(outdir, "trace.png"), title)

    # tidy the drives, matching run_benches' between-point hygiene
    rb.clear_bench_data(args.mount_glob, not args.no_clean)

    ph = ", ".join(f"{lbl}@{mono - t0:.2f}s" for lbl, mono in markers) or "none"
    print(f"\nDone. Phases: {ph}\nResults in {outdir}", flush=True)


if __name__ == "__main__":
    main()
