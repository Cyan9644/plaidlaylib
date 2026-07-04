#!/usr/bin/env python3
"""Run the ExternalChunkSequence example benchmarks and plot them.

Single-file driver, trimmed from the predecessor project's run_benches.py to the
*examples sweep* (this tree carries the External* primitives and their example
programs, not the delayed/chunk-size substrate benchmarks).  It shells out to
`make` to build each example binary (all compilation stays in the Makefile),
sweeps it over a range of n, parses the `CSV,` line each binary prints, and
writes PNGs + raw CSVs into a timestamped run directory.

Each example is a dual-purpose binary (bin/<name>Example) that prints a
`CSV,<cols...>` line and -- for sizes within its RAM budget -- also times the
in-memory parlaylib baseline and cross-checks the result, exiting non-zero on a
mismatch.  Like the KMP/Rabin-Karp sweeps, a problem here (mismatch, crash) is
*warned* about (immediately and again at the end of the run) but does NOT stop
the sweep; a crashed point that printed no CSV line is dropped.

To keep data off the drives, the script deletes each example's data files
between every sweep point and after the run.  It also best-effort `fstrim`s the
mounts once at startup (a no-op on a tmpfs dev box).  --no-clean / --no-fstrim
disable each.

Defaults are sized for a ~5 GiB tmpfs dev box; override via flags or env.
"""

import argparse
import glob
import math
import os
import re
import subprocess
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINDIR = os.path.join(REPO_ROOT, "bin")

DEFAULT_EXAMPLE_N_VALUES = "2^20 2^21 2^22 2^23 2^24 2^25 2^26"  # examples sweep (dev/tmpfs)
DEFAULT_FSTRIM_GLOB = "/mnt/ssd*"

# ── examples registry ───────────────────────────────────────────────────────
# Each example is a dual-purpose binary (bin/<name>Example) that prints a
# `CSV,<cols...>` line the sweep greps.  `cols` names those fields in order;
# `time_col`/`inmem_col` pick the plotted out-of-core / in-memory series;
# `data_globs` lists the per-mount globs of files the example leaves on the
# drives (cleared between sweep points).  Add a new example by appending one
# entry (and dropping its .cpp in ChunkSequence/Examples/external/).
EXAMPLES = [
    # kth_smallestExample sweeps n with k at the median (n/2); the plotted time
    # is the selection pass only (input build excluded).  Its recursion leaves
    # id_/flags_/next_ intermediates in addition to the kth_in input.
    {"name": "kth_smallest", "target": "bin/kth_smallestExample",
     "cols": ["n", "k", "build_s", "select_s", "inmem_select_s", "result",
              "throughput_gb_s"],
     "time_col": "select_s", "inmem_col": "inmem_select_s",
     "xlabel": "n (number of keys)",
     "title": "kth-smallest: out-of-core (ChunkSequenceOps) vs in-mem parlaylib",
     "data_globs": ["kth_in*", "id_*", "flags_*", "next_*"]},
]


# ── size parsing ────────────────────────────────────────────────────────────
def parse_count(s):
    """Element count: `2^k`/`2**k`, decimal suffixes K=1e3/M=1e6/G=1e9, or raw."""
    s = s.strip()
    m = re.fullmatch(r"2(?:\^|\*\*)(\d+)", s)     # 2^30, 2**30
    if m:
        return 2 ** int(m.group(1))
    m = re.fullmatch(r"(\d+)([kKmMgG]?)", s)
    if not m:
        raise ValueError(f"bad count {s!r}")
    mult = {"": 1, "k": 10**3, "m": 10**6, "g": 10**9}[m.group(2).lower()]
    return int(m.group(1)) * mult


def _f(s):
    """Parse a CSV time field; blank (skipped in-memory line) -> None."""
    s = s.strip()
    return float(s) if s else None


# ── device maintenance ──────────────────────────────────────────────────────
# File-name globs (per mount) of every prefix the examples write, flattened from
# each example's `data_globs`.  Used to clear the drives between points so
# nothing accumulates.
BENCH_FILE_GLOBS = tuple(g for e in EXAMPLES for g in e["data_globs"])


def clear_bench_data(glob_pat, enabled):
    """Best-effort unlink of leftover bench files under every mount in glob_pat."""
    if not enabled:
        return
    removed = 0
    for m in sorted(glob.glob(glob_pat)):
        for pat in BENCH_FILE_GLOBS:
            for f in glob.glob(os.path.join(m, pat)):
                try:
                    os.unlink(f)
                    removed += 1
                except OSError:
                    pass
    if removed:
        print(f"  cleared {removed} leftover bench files", flush=True)


def fstrim_mounts(glob_pat, enabled):
    """Best-effort `fstrim` of every mount matching glob_pat.

    Announces itself (fstrim can be slow) but returns the outcome as a note
    string for the end-of-run summary rather than printing it here, where it
    would scroll away.  Returns None when disabled or no mounts match.
    """
    if not enabled:
        return None
    mounts = sorted(glob.glob(glob_pat))
    if not mounts:
        return None
    print(f"  fstrim {len(mounts)} mount(s) matching {glob_pat} ...", flush=True)
    ok, failed = 0, None
    for m in mounts:
        r = subprocess.run(["fstrim", m], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        if r.returncode == 0:
            ok += 1
        else:
            failed = r.stdout.strip() or f"fstrim {m} exit {r.returncode}"
    note = f"fstrim: ok on {ok} of {len(mounts)} mount(s)"
    if failed:
        note += f"; skipped/unsupported on the rest: {failed}"
    return note


# ── running binaries ────────────────────────────────────────────────────────
def make(target):
    print(f"  $ make {target}", flush=True)
    r = subprocess.run(["make", target], cwd=REPO_ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout)
        sys.exit(f"make {target} failed (exit {r.returncode})")


def run_binary(path, args):
    """Run an example binary, echo its output, return (csv_fields, problem).

    `problem` is None on a clean run, else a short description (crash,
    verification mismatch, missing CSV line).  It is returned (not fatal) so the
    caller can warn and keep sweeping.  `csv_fields` is None if the binary
    printed no CSV line (e.g. it crashed).
    """
    cmd = [path] + [str(a) for a in args]
    print(f"  $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=REPO_ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout, end="", flush=True)
    csv = None
    for line in r.stdout.splitlines():
        if line.startswith("CSV,"):
            csv = line[len("CSV,"):].split(",")
    problem = None
    if r.returncode != 0:
        why = ("killed by a signal (crash)" if r.returncode < 0
               else "correctness mismatch or error")
        problem = f"exited {r.returncode} ({why})"
    elif csv is None:
        problem = "no CSV line in output"
    return csv, problem


# ── examples sweep ──────────────────────────────────────────────────────────
def run_example(entry, n_values, extra_args, clear_glob, clear_enabled, warnings):
    """Sweep one example over n_values; return parsed rows.

    Correctness is checked inside the binary (against the in-memory parlaylib
    baseline when it fits in RAM).  A problem here does NOT stop the sweep: it is
    appended to `warnings` (echoed again at the end of the run) and the sweep
    continues -- a crashed point that printed no CSV line is dropped.
    """
    make(entry["target"])
    binary = os.path.join(BINDIR, os.path.basename(entry["target"]))
    rows = []
    for n in n_values:
        print(f"\n=== example {entry['name']}: n={n} ===", flush=True)
        fields, problem = run_binary(binary, [n] + extra_args)
        if problem:
            w = (f"example {entry['name']} at n={n}: {problem}"
                 + ("" if fields else " — point dropped"))
            print(f"  !!! {w}", flush=True)
            warnings.append(w)
        if fields:
            rows.append(dict(zip(entry["cols"], fields)))
        clear_bench_data(clear_glob, clear_enabled)   # don't leave output on the drives
    return rows


# ── output: raw CSV + plots ─────────────────────────────────────────────────
def write_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r[c] for c in header) + "\n")
    print(f"  wrote {path}", flush=True)


def _pow2_fmt(val, _):
    if val <= 0:
        return ""
    return f"$2^{{{round(math.log2(val))}}}$"


def _series(rows, name):
    return [_f(r[name]) for r in rows]


def _runtime(xs, times):
    """(xs, ys) of the non-blank, positive points only."""
    out_x, out_y = [], []
    for x, t in zip(xs, times):
        if t is not None and t > 0.0:
            out_x.append(x)
            out_y.append(t)
    return out_x, out_y


def _draw_panel(ax, xs, lines, xlabel, title, xfmt=None):
    import matplotlib.ticker as ticker
    for label, times, style in lines:
        px, py = _runtime(xs, times)
        if px:
            ax.plot(px, py, style, label=label, markersize=5)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_pow2_fmt))
    if xfmt is not None:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(xfmt))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("operation time (s)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend()


def plot_example(rows, entry, path):
    """Single-panel log-log plot of an example's runtime vs n.

    Two series: the in-memory parlaylib baseline (which stops at the RAM cliff --
    blank CSV fields are dropped) and the out-of-core chunk implementation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [int(r["n"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    _draw_panel(ax, xs, [
        ("in-mem parlaylib (DRAM)", _series(rows, entry["inmem_col"]), "o-"),
        ("out-of-core (chunk)", _series(rows, entry.get("time_col", "time_s")), "s-"),
    ], entry["xlabel"],
       entry["title"] + "\n(in-mem line stops where the input exceeds the RAM budget)",
       xfmt=_pow2_fmt)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", action="store_true",
                    help="run the examples sweep (currently the only sweep)")
    ap.add_argument("--outdir", default=os.environ.get("BENCH_OUTDIR", "results"),
                    help="parent dir for the timestamped run (default: results)")
    ap.add_argument("--example-n-values",
                    default=os.environ.get("BENCH_EXAMPLE_N_VALUES", DEFAULT_EXAMPLE_N_VALUES),
                    help="examples n sweep (space-separated, e.g. '2^20 2^24 2^26')")
    ap.add_argument("--ssd-args", default=os.environ.get("BENCH_SSD_ARGS", ""),
                    help="extra global flags passed to each binary (e.g. '--num_ssd=4')")
    ap.add_argument("--fstrim-glob",
                    default=os.environ.get("BENCH_FSTRIM_GLOB", DEFAULT_FSTRIM_GLOB),
                    help="glob of mounts to fstrim once at startup (default: /mnt/ssd*)")
    ap.add_argument("--no-fstrim", action="store_true",
                    help="disable the startup fstrim")
    ap.add_argument("--no-clean", action="store_true",
                    help="leave bench data files on the mounts (default: clear between points)")
    args = ap.parse_args()

    if not args.examples:
        ap.error("nothing to run: pass --examples")

    extra = args.ssd_args.split() if args.ssd_args else []
    example_n_values = [parse_count(x) for x in args.example_n_values.split()]
    fstrim_enabled = not args.no_fstrim
    clear_enabled = not args.no_clean

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(REPO_ROOT, args.outdir, stamp)
    os.makedirs(outdir, exist_ok=True)
    print(f"Run directory: {outdir}\n")

    # Start from clean drives, then trim once up front; both no-ops on tmpfs.
    clear_bench_data(args.fstrim_glob, clear_enabled)
    fstrim_note = fstrim_mounts(args.fstrim_glob, fstrim_enabled)

    warnings = []
    for entry in EXAMPLES:
        print(f"\n######## example: {entry['name']} ########")
        rows = run_example(entry, example_n_values, extra,
                           args.fstrim_glob, clear_enabled, warnings)
        write_csv(os.path.join(outdir, f"{entry['name']}_scale.csv"),
                  entry["cols"], rows)
        if rows:
            # The CSV above is the substantive output; plotting is a convenience
            # that needs matplotlib (from shell.nix / the system).  If it's not
            # installed, warn and keep going rather than throwing away a good run.
            try:
                plot_example(rows, entry, os.path.join(outdir, f"{entry['name']}_scale.png"))
            except ImportError as e:
                w = (f"plot for {entry['name']} skipped: {e} "
                     "(install matplotlib to get PNGs; the CSV was still written)")
                print(f"  !!! {w}", flush=True)
                warnings.append(w)

    # ── end-of-run summary (warnings persisted next to the results).
    print("\n======== run summary ========")
    if fstrim_note:
        print(f"  {fstrim_note}")
    if warnings:
        print(f"  !!! {len(warnings)} example warning(s) (sweep continued past them):")
        for w in warnings:
            print(f"  !!!   {w}")
        wpath = os.path.join(outdir, "warnings.txt")
        with open(wpath, "w") as f:
            f.write("\n".join(warnings) + "\n")
        print(f"  !!! (also written to {wpath})")
    else:
        print("  no example warnings")
    print(f"\nDone. Results in {outdir}")


if __name__ == "__main__":
    main()
