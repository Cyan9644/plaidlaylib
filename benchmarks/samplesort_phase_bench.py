#!/usr/bin/env python3
"""Per-phase benchmark for ChunkSequenceOps::sample_sort (external_samplesort.h).

Runs the head-to-head driver (bin/external_samplesort_vs_peterExample) across a
sweep of input sizes with SS_PHASE_TIMING enabled, so each run prints both:

  * a machine-readable ``SSPHASE,0,<phase>=<s>,...,total=<s>`` line
    (emitted by SsPhaseTimer in external_samplesort.h), and
  * the driver's ``CSV,n,ext_build_s,ext_sort_s,peter_build_s,peter_sort_s,...``
    line (so Peter's out-of-core sort is a same-run reference).

It collects them into ``samplesort_phases.csv`` and prints a per-phase table plus
a normalized ASCII stacked bar, so you can see *which phase* dominates and how it
scales -- exactly the "where is it going wrong" view.  If matplotlib happens to
be installed it also writes a stacked-bar PNG; otherwise that step is skipped
(the CSV + ASCII table are the primary artifacts on a headless dev box).

Usage:
  python3 benchmarks/samplesort_phase_bench.py                  # default sweep
  python3 benchmarks/samplesort_phase_bench.py --n-values "2^23 2^25 2^27"
  python3 benchmarks/samplesort_phase_bench.py --reps 3         # min-of-reps
  python3 benchmarks/samplesort_phase_bench.py --no-build

n must be a multiple of 512 (the driver's O_DIRECT constraint); values are
rounded down to satisfy it.
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "bin", "external_samplesort_vs_peterExample")

# Phases in the order SsPhaseTimer marks them (see external_samplesort.h).
PHASES = ["size/params", "sample:probe", "sample:pivots/heap",
          "count_sort", "bucket_sort", "flatten"]


def parse_n(tok):
    tok = tok.strip()
    if "^" in tok:
        b, e = tok.split("^")
        v = int(b) ** int(e)
    elif tok[-1:].upper() in ("K", "M", "G"):
        mult = {"K": 10**3, "M": 10**6, "G": 10**9}[tok[-1].upper()]
        v = int(float(tok[:-1]) * mult)
    else:
        v = int(tok)
    return (v // 512) * 512  # honor the driver's n % 512 == 0 requirement


def run_once(n, env):
    out = subprocess.run([BIN, str(n)], cwd=ROOT, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True)
    phases, ext_sort, peter_sort = {}, None, None
    for line in out.stdout.splitlines():
        if line.startswith("SSPHASE,"):
            for tok in line.split(",")[2:]:
                if "=" in tok:
                    k, val = tok.split("=", 1)
                    phases[k] = float(val)
        elif line.startswith("CSV,"):
            f = line.split(",")
            # CSV,n,ext_build,ext_sort,peter_build,peter_sort,ext_gb,peter_gb
            ext_sort, peter_sort = float(f[2]), float(f[4])
    if out.returncode != 0 or ext_sort is None:
        sys.stderr.write("  !! run failed (rc=%d) for n=%d\n" % (out.returncode, n))
        sys.stderr.write("\n".join(out.stdout.splitlines()[-15:]) + "\n")
    return phases, ext_sort, peter_sort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-values", default="2^22 2^23 2^24 2^25 2^26 2^27",
                    help="space-separated n's (accepts 2^k, 8M, plain ints)")
    ap.add_argument("--reps", type=int, default=1,
                    help="repetitions per n; the per-phase minimum is kept "
                         "(min = least page-cache/disk noise)")
    ap.add_argument("--outdir", default=ROOT)
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    if not args.no_build:
        sys.stderr.write("Building driver...\n")
        r = subprocess.run(["make", "bin/external_samplesort_vs_peterExample"],
                           cwd=ROOT)
        if r.returncode != 0:
            sys.exit("build failed")

    ns = [parse_n(t) for t in args.n_values.split()]
    env = dict(os.environ)
    env["SS_PHASE_TIMING"] = "1"
    env["EXAMPLE_INMEM_BUDGET_BYTES"] = "1"  # skip the RAM cross-check, just time

    rows = []
    for n in ns:
        best = None
        for _ in range(max(1, args.reps)):
            phases, ext_sort, peter_sort = run_once(n, env)
            if ext_sort is None:
                continue
            key = phases.get("total", ext_sort)
            if best is None or key < best[0]:
                best = (key, phases, ext_sort, peter_sort)
        if best is None:
            continue
        _, phases, ext_sort, peter_sort = best
        rows.append((n, phases, ext_sort, peter_sort))
        sys.stderr.write("n=%d  total=%.3fs  peter=%.3fs\n"
                         % (n, phases.get("total", ext_sort), peter_sort))

    # ---- CSV ----
    csv_path = os.path.join(args.outdir, "samplesort_phases.csv")
    with open(csv_path, "w") as f:
        f.write("n," + ",".join(PHASES) + ",our_total,peter_sort\n")
        for n, ph, ext_sort, peter_sort in rows:
            vals = [ph.get(p, 0.0) for p in PHASES]
            f.write("%d,%s,%.6f,%.6f\n" % (
                n, ",".join("%.6f" % v for v in vals),
                ph.get("total", ext_sort), peter_sort))
    sys.stderr.write("wrote %s\n" % csv_path)

    # ---- ASCII table + normalized stacked bar ----
    print("\n=== sample_sort per-phase breakdown (seconds) ===")
    hdr = "%12s " % "n" + "".join("%10s" % p.split(":")[-1][:10] for p in PHASES)
    hdr += "%10s%10s%8s" % ("our_tot", "peter", "spd")
    print(hdr)
    for n, ph, ext_sort, peter_sort in rows:
        tot = ph.get("total", ext_sort)
        line = "%12d " % n + "".join("%10.3f" % ph.get(p, 0.0) for p in PHASES)
        spd = (peter_sort / tot) if tot else 0.0
        line += "%10.3f%10.3f%7.2fx" % (tot, peter_sort, spd)
        print(line)

    print("\n=== phase share (each bar = 40 cols = our_total) ===")
    marks = " .:-=+*#%@"  # ramp used cyclically per phase
    for n, ph, ext_sort, peter_sort in rows:
        tot = ph.get("total", ext_sort) or 1.0
        bar = ""
        for i, p in enumerate(PHASES):
            w = int(round(40 * ph.get(p, 0.0) / tot))
            bar += (marks[(i % (len(marks) - 1)) + 1]) * w
        print("%12d |%-40s| %.3fs" % (n, bar[:40], tot))
    legend = "  ".join("%s=%s" % (marks[(i % (len(marks) - 1)) + 1], PHASES[i])
                       for i in range(len(PHASES)))
    print("legend: " + legend)

    # ---- optional PNG ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        xs = [str(n) for n, _, _, _ in rows]
        bottoms = np.zeros(len(rows))
        fig, ax = plt.subplots(figsize=(10, 6))
        for p in PHASES:
            vals = np.array([ph.get(p, 0.0) for _, ph, _, _ in rows])
            ax.bar(xs, vals, bottom=bottoms, label=p)
            bottoms += vals
        ax.plot(xs, [ps for _, _, _, ps in rows], "k--o", label="peter_sort (total)")
        ax.set_ylabel("seconds")
        ax.set_xlabel("n (elements)")
        ax.set_title("external sample_sort: per-phase time vs Peter's total")
        ax.legend()
        png = os.path.join(args.outdir, "samplesort_phases.png")
        fig.tight_layout()
        fig.savefig(png, dpi=120)
        sys.stderr.write("wrote %s\n" % png)
    except Exception as e:  # noqa
        sys.stderr.write("(matplotlib unavailable, skipped PNG: %s)\n" % e)


if __name__ == "__main__":
    main()
