#!/usr/bin/env python3
"""Run the chunk-sequence benchmarks and plot them — single-file driver.

Replaces the predecessor project's two shell drivers (bench_delayed_scale.sh,
bench_chunk_size.sh) with one script that both *runs* the sweeps and *plots*
them.  It shells out to `make` to build the C++ binaries (all compilation stays
in the Makefile), runs them across a parameter sweep, parses the `CSV,` line
each binary prints, and writes PNGs + raw CSVs into a timestamped run directory.

Two benchmarks:

  * delayed scale  (benchmarks/delayed_compare.cpp -> bin/delayedCompare)
      Fixed chunk size, sweep n.  Compares in-mem delayed / chunk-eager /
      chunk-delayed for map|reduce, map|map|reduce, and force(map|map).

  * chunk size     (benchmarks/chunk_size_compare.cpp -> bin/chunkSizeCompare_<bytes>)
      Fixed n, sweep CHUNK_SIZE (one binary compiled per size via
      -DCHUNK_SIZE_BYTES).  Compares chunk-eager vs chunk-delayed.

The written CSVs use the same descriptive column names / units as the
predecessor's standalone plotters, and the plots follow the same two-panel
(read-bound map|map|reduce vs write-bound force(map|map)) base-2 log-log style.

To keep data off the drives, the script deletes the benchmarks' data files
(`perm<drive>` inputs + `bw_*` intermediates) between every sweep point and after
the run, so nothing accumulates (the C++ binaries clean their own intermediates,
but delayed_compare leaves its `perm` input behind).  It also best-effort
`fstrim`s the mounts once at startup (fstrim can be slow on real SSDs, so it is
not repeated between points; a no-op on a tmpfs dev box, where FITRIM is
unsupported — skipped quietly).  `--no-clean` / `--no-fstrim` disable each.

Every run carries a cross-substrate correctness check (agree=1).  If any point
reports agree=0 or a binary exits non-zero, this script prints the offending
output and exits non-zero — so `make bench` doubles as a differential test.

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

# Defaults tuned for a ~5 GiB tmpfs (peak disk use in a step is ~3*n*8 bytes).
DEFAULT_N_VALUES = "1M 2M 4M 8M 16M 32M 64M"          # element counts
DEFAULT_CHUNK_SIZES = "512KiB 1MiB 2MiB 4MiB 8MiB 16MiB"
DEFAULT_CHUNK_N = "32M"                                # fixed n for chunk sweep
DEFAULT_FSTRIM_GLOB = "/mnt/ssd*"


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


def parse_bytes(s):
    """Byte size: binary suffixes KiB/MiB/GiB (or K/M/G = *1024^x), or raw."""
    s = s.strip()
    m = re.fullmatch(r"(\d+)\s*([kKmMgG]?)(i?[bB]?)", s)
    if not m:
        raise ValueError(f"bad byte size {s!r}")
    mult = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3}[m.group(2).lower()]
    return int(m.group(1)) * mult


def _f(s):
    """Parse a CSV time field; blank (skipped in-memory line) -> None."""
    s = s.strip()
    return float(s) if s else None


# ── device maintenance ──────────────────────────────────────────────────────
_fstrim_warned = False

# File-name globs (per mount) of every prefix the benchmarks write: the shared
# `perm<drive>` input plus each pipeline's intermediates.  Used to clear the
# drives between points so nothing accumulates — the C++ binaries clean their
# own intermediates, but delayed_compare leaves its `perm` input behind.
BENCH_FILE_GLOBS = ("perm[0-9]*", "bw_dl_*", "bw_cs_*")


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

    On a tmpfs dev box FITRIM is unsupported and fstrim exits non-zero; we warn
    once and keep going.  (fstrim may also need privileges on real SSDs.)
    """
    global _fstrim_warned
    if not enabled:
        return
    mounts = sorted(glob.glob(glob_pat))
    if not mounts:
        return
    failed = None
    for m in mounts:
        r = subprocess.run(["fstrim", m], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        if r.returncode != 0:
            failed = r.stdout.strip() or f"fstrim {m} exit {r.returncode}"
    if failed and not _fstrim_warned:
        print(f"  (fstrim skipped/unsupported: {failed})", flush=True)
        _fstrim_warned = True


# ── running binaries ────────────────────────────────────────────────────────
def make(target):
    print(f"  $ make {target}", flush=True)
    r = subprocess.run(["make", target], cwd=REPO_ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout)
        sys.exit(f"make {target} failed (exit {r.returncode})")


def run_binary(path, args):
    """Run a benchmark binary, echo its output, return the parsed CSV fields.

    Exits non-zero on binary failure or a agree=0 correctness mismatch.
    """
    cmd = [path] + [str(a) for a in args]
    print(f"  $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=REPO_ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout, end="", flush=True)
    if r.returncode != 0:
        sys.exit(f"\n*** {os.path.basename(path)} exited {r.returncode} "
                 f"(likely a correctness mismatch) — aborting ***")
    csv = None
    for line in r.stdout.splitlines():
        if line.startswith("CSV,"):
            csv = line[len("CSV,"):].split(",")
    if csv is None:
        sys.exit(f"\n*** no CSV line from {os.path.basename(path)} — aborting ***")
    return csv


# ── delayed scale sweep ─────────────────────────────────────────────────────
# CSV cols emitted by delayed_compare.cpp, in order.
DELAYED_COLS = ["n", "raw_read_s", "eager_mr_s", "delayed_mr_s", "inmem_mr_s",
                "eager_mmr_s", "delayed_mmr_s", "inmem_mmr_s",
                "eager_fmm_s", "delayed_fmm_s", "inmem_fmm_s", "agree"]


def run_delayed(n_values, extra_args, clear_glob, clear_enabled):
    make("bin/delayedCompare")
    binary = os.path.join(BINDIR, "delayedCompare")
    rows = []
    for n in n_values:
        print(f"\n=== delayed scale: n={n} ===", flush=True)
        fields = run_binary(binary, [n] + extra_args)
        row = dict(zip(DELAYED_COLS, fields))
        if row["agree"].strip() != "1":
            sys.exit(f"\n*** agree={row['agree']} at n={n} — aborting ***")
        rows.append(row)
        clear_bench_data(clear_glob, clear_enabled)   # don't leave input on the drives
    return rows


# ── chunk size sweep ────────────────────────────────────────────────────────
# CSV cols emitted by chunk_size_compare.cpp, in order.
CHUNK_COLS = ["chunk_size_bytes", "n", "raw_s", "eager_mr_s", "delayed_mr_s",
              "eager_mmr_s", "delayed_mmr_s", "eager_fmm_s", "delayed_fmm_s", "agree"]


def run_chunk_size(chunk_sizes, n, extra_args, clear_glob, clear_enabled):
    rows = []
    for cs in chunk_sizes:
        make(f"bin/chunkSizeCompare_{cs}")
        binary = os.path.join(BINDIR, f"chunkSizeCompare_{cs}")
        print(f"\n=== chunk size: {cs} bytes, n={n} ===", flush=True)
        fields = run_binary(binary, [n] + extra_args)
        row = dict(zip(CHUNK_COLS, fields))
        if row["agree"].strip() != "1":
            sys.exit(f"\n*** agree={row['agree']} at chunk_bytes={cs} — aborting ***")
        rows.append(row)
        clear_bench_data(clear_glob, clear_enabled)   # don't leave input on the drives
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


def _bytes_fmt(val, _):
    """Human-readable power-of-two byte size, e.g. 256 KiB / 1 MiB / 16 MiB."""
    if val <= 0:
        return ""
    for factor, unit in ((1024**3, "GiB"), (1024**2, "MiB"), (1024, "KiB")):
        if val >= factor:
            q = val / factor
            s = str(int(round(q))) if abs(q - round(q)) < 1e-6 else f"{q:g}"
            return f"{s} {unit}"
    return f"{int(round(val))} B"


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


def plot_delayed(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [int(r["n"]) for r in rows]
    raw = _series(rows, "raw_read_s")

    fig, (ax_r, ax_w) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    _draw_panel(ax_r, xs, [
        ("in-mem delayed (DRAM)", _series(rows, "inmem_mmr_s"), "o-"),
        ("chunk-delayed",         _series(rows, "delayed_mmr_s"), "s-"),
        ("chunk-eager",           _series(rows, "eager_mmr_s"), "^-"),
        ("raw read (1 pass)",     raw, "k:"),
    ], "n (elements)", "map(x+1) | map(2x) | reduce(sum)  — read-bound")
    _draw_panel(ax_w, xs, [
        ("in-mem materialize (DRAM)", _series(rows, "inmem_fmm_s"), "o-"),
        ("chunk-delayed",            _series(rows, "delayed_fmm_s"), "s-"),
        ("chunk-eager",              _series(rows, "eager_fmm_s"), "^-"),
        ("raw read (1 pass)",        raw, "k:"),
    ], "n (elements)", "force(map(x+1) | map(2x))  — write-bound")

    fig.suptitle("Delayed sequences: in-memory vs chunk-eager vs chunk-delayed — scaling\n"
                 "(in-memory line stops where the input exceeds the RAM budget)")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


def plot_chunk_size(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [int(r["chunk_size_bytes"]) for r in rows]
    raw = _series(rows, "raw_s")
    n = rows[0]["n"] if rows else "?"

    fig, (ax_r, ax_w) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    _draw_panel(ax_r, xs, [
        ("chunk-delayed",     _series(rows, "delayed_mmr_s"), "s-"),
        ("chunk-eager",       _series(rows, "eager_mmr_s"), "^-"),
        ("raw read (1 pass)", raw, "k:"),
    ], "chunk size", "map(x+1) | map(2x) | reduce(sum)  — read-bound", xfmt=_bytes_fmt)
    _draw_panel(ax_w, xs, [
        ("chunk-delayed",     _series(rows, "delayed_fmm_s"), "s-"),
        ("chunk-eager",       _series(rows, "eager_fmm_s"), "^-"),
        ("raw read (1 pass)", raw, "k:"),
    ], "chunk size", "force(map(x+1) | map(2x))  — write-bound", xfmt=_bytes_fmt)

    fig.suptitle(f"Chunk-size sensitivity: chunk-eager vs chunk-delayed  (n={n})\n"
                 "(read-bound map|map|reduce vs write-bound force(map|map))")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="run both benchmarks")
    ap.add_argument("--delayed", action="store_true", help="run the delayed-scale sweep")
    ap.add_argument("--chunk-size", action="store_true", help="run the chunk-size sweep")
    ap.add_argument("--outdir", default=os.environ.get("BENCH_OUTDIR", "results"),
                    help="parent dir for the timestamped run (default: results)")
    ap.add_argument("--n-values", default=os.environ.get("BENCH_N_VALUES", DEFAULT_N_VALUES),
                    help="delayed-scale n sweep (space-separated, e.g. '1M 8M 64M')")
    ap.add_argument("--chunk-sizes",
                    default=os.environ.get("BENCH_CHUNK_SIZES", DEFAULT_CHUNK_SIZES),
                    help="chunk-size sweep (space-separated, e.g. '512KiB 4MiB')")
    ap.add_argument("--n", default=os.environ.get("BENCH_CHUNK_N", DEFAULT_CHUNK_N),
                    help="fixed n for the chunk-size sweep (default: 32M)")
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

    do_delayed = args.all or args.delayed
    do_chunk = args.all or args.chunk_size
    if not (do_delayed or do_chunk):
        ap.error("nothing to run: pass --all, --delayed, and/or --chunk-size")

    extra = args.ssd_args.split() if args.ssd_args else []
    n_values = [parse_count(x) for x in args.n_values.split()]
    chunk_sizes = [parse_bytes(x) for x in args.chunk_sizes.split()]
    chunk_n = parse_count(args.n)
    fstrim_enabled = not args.no_fstrim
    clear_enabled = not args.no_clean

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(REPO_ROOT, args.outdir, stamp)
    os.makedirs(outdir, exist_ok=True)
    print(f"Run directory: {outdir}\n")

    # Start from clean drives, then trim once up front (fstrim can be slow on
    # real SSDs, so we don't repeat it between points); both no-ops on tmpfs.
    clear_bench_data(args.fstrim_glob, clear_enabled)
    fstrim_mounts(args.fstrim_glob, fstrim_enabled)

    if do_delayed:
        print("######## delayed scale ########")
        rows = run_delayed(n_values, extra, args.fstrim_glob, clear_enabled)
        write_csv(os.path.join(outdir, "delayed_scale.csv"), DELAYED_COLS, rows)
        plot_delayed(rows, os.path.join(outdir, "delayed_scale.png"))

    if do_chunk:
        print("\n######## chunk size ########")
        rows = run_chunk_size(chunk_sizes, chunk_n, extra, args.fstrim_glob, clear_enabled)
        write_csv(os.path.join(outdir, "chunk_size.csv"), CHUNK_COLS, rows)
        plot_chunk_size(rows, os.path.join(outdir, "chunk_size.png"))

    print(f"\nDone. Results in {outdir}")


if __name__ == "__main__":
    main()
