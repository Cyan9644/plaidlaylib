#!/usr/bin/env python3
"""Benchmark and plot the external prime sieve (primes.h) against the in-memory
sieve across input sizes 2**30 .. 2**40.

Each (variant, size) is run as its own process via the `primesScaling` binary
(built from primes_scaling_benchmark.cpp), so a size that exhausts memory only
kills that one run -- it gets recorded as OOM and the sweep continues. The
in-memory sieve is expected to die on the larger sizes; the external one should
survive further since it streams flags through SSDs instead of RAM.

Outputs (written next to this script by default):
  * primes_benchmark_results.csv -- raw seconds + peak RSS per (variant, size)
  * primes_benchmark.png         -- seconds vs size, and peak memory vs size

In-memory is drawn in yellow, the external algorithm in blue.

Data collection and plotting are decoupled so the benchmark can run on a remote
machine that has no matplotlib: the sweep always writes the CSV, and plotting is
attempted only as a best-effort final step (skipped with a hint if matplotlib is
missing). Copy the CSV to a machine that has matplotlib and re-render with
`--plot-only`.

Usage:
    python3 plot_primes_benchmark.py                 # full 2**30 .. 2**40 sweep
    python3 plot_primes_benchmark.py --min-exp 28 --max-exp 34 --repeats 3
    python3 plot_primes_benchmark.py --no-plot       # data only (no matplotlib)
    python3 plot_primes_benchmark.py --plot-only \
        --csv primes_benchmark_results.csv           # plot a CSV from elsewhere
"""

import argparse
import csv
import json
import os
import subprocess
import sys

# .../Plaidlay/examples/external/plot_primes_benchmark.py -> repo root is 4 up.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
BINARY = os.path.join(REPO_ROOT, "bazel-bin", "primesScaling")

VARIANTS = ("mem", "ext")
LABELS = {"mem": "in-memory", "ext": "external"}
COLORS = {"mem": "gold", "ext": "tab:blue"}  # yellow vs blue, as requested


def ensure_binary():
    """Build bazel-bin/primesScaling via make if it isn't there already."""
    if os.path.exists(BINARY):
        return
    print("primesScaling not found; building with make...", file=sys.stderr)
    subprocess.run(["make", "bazel-bin/primesScaling"], cwd=REPO_ROOT, check=True)


def run_one(variant, n, timeout):
    """Run one measurement. Returns (seconds, rss_kb, count) or None on
    OOM / crash / timeout."""
    try:
        proc = subprocess.run(
            [BINARY, variant, str(n)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"  {LABELS[variant]:>9} n={n}: TIMEOUT (>{timeout}s)", file=sys.stderr)
        return None

    if proc.returncode != 0:
        # Negative return code == killed by signal (e.g. SIGKILL from OOM killer).
        reason = f"signal {-proc.returncode}" if proc.returncode < 0 else f"exit {proc.returncode}"
        print(f"  {LABELS[variant]:>9} n={n}: FAILED ({reason}) -- treating as OOM",
              file=sys.stderr)
        if proc.stderr.strip():
            print(f"      stderr: {proc.stderr.strip()}", file=sys.stderr)
        return None

    line = proc.stdout.strip().splitlines()[-1]
    _, _, seconds, rss_kb, count = line.split(",")
    seconds, rss_kb, count = float(seconds), int(rss_kb), int(count)
    print(f"  {LABELS[variant]:>9} n={n}: {seconds:8.3f} s   "
          f"{rss_kb / (1024 * 1024):7.2f} GiB peak   ({count} primes)")
    return seconds, rss_kb, count


def sweep(exponents, repeats, timeout):
    """Return dict: variant -> list aligned with `exponents`, each entry a
    (best_seconds, rss_kb) tuple or None for OOM."""
    results = {v: [] for v in VARIANTS}
    for e in exponents:
        n = 1 << e
        print(f"n = 2**{e} = {n}")
        for v in VARIANTS:
            best = None
            for _ in range(repeats):
                r = run_one(v, n, timeout)
                if r is None:
                    best = None
                    break  # OOM/timeout won't get better on a retry
                if best is None or r[0] < best[0]:
                    best = (r[0], r[1])
            results[v].append(best)
        print()
    return results


def write_csv(path, exponents, results):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exponent", "n", "variant", "seconds", "peak_rss_kb"])
        for i, e in enumerate(exponents):
            for v in VARIANTS:
                r = results[v][i]
                if r is None:
                    w.writerow([e, 1 << e, v, "OOM", "OOM"])
                else:
                    w.writerow([e, 1 << e, v, f"{r[0]:.6f}", r[1]])
    print(f"wrote {path}")


def write_json(path, exponents, results):
    """Self-contained, trivially reloadable form of the sweep results. OOM /
    timeout entries (None) serialize as JSON null."""
    payload = {"exponents": list(exponents), "results": results}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {path}")


def read_json(path):
    """Inverse of write_json. Returns (exponents, results) in the same shape the
    sweep produces: results[variant] is a list aligned with exponents, each entry
    a (seconds, rss_kb) tuple or None."""
    with open(path) as f:
        payload = json.load(f)
    exponents = payload["exponents"]
    results = {
        v: [tuple(r) if r is not None else None for r in entries]
        for v, entries in payload["results"].items()
    }
    return exponents, results


def plot(path, exponents, results):
    import matplotlib
    matplotlib.use("Agg")  # headless: just write a file
    import matplotlib.pyplot as plt

    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(14, 6))

    for v in VARIANTS:
        xs_t, ys_t, xs_m, ys_m = [], [], [], []
        for i, e in enumerate(exponents):
            r = results[v][i]
            if r is None:
                continue
            xs_t.append(e)
            ys_t.append(r[0])
            xs_m.append(e)
            ys_m.append(r[1] / (1024 * 1024))  # KiB -> GiB
        ax_time.plot(xs_t, ys_t, marker="o", color=COLORS[v], label=LABELS[v])
        ax_mem.plot(xs_m, ys_m, marker="o", color=COLORS[v], label=LABELS[v])

    # Mark the sizes where the in-memory sieve ran out of memory.
    oom_exps = [e for i, e in enumerate(exponents) if results["mem"][i] is None]
    for ax in (ax_time, ax_mem):
        for e in oom_exps:
            ax.axvline(e, color="red", linestyle=":", alpha=0.4)
        if oom_exps:
            ax.axvline(oom_exps[0], color="red", linestyle=":", alpha=0.4,
                       label="in-memory OOM")

    ticks = list(exponents)
    ticklabels = [f"$2^{{{e}}}$" for e in exponents]

    ax_time.set_title("Prime sieve runtime vs input size")
    ax_time.set_xlabel("input size n")
    ax_time.set_ylabel("time (seconds)")
    ax_time.set_yscale("log")
    ax_time.set_xticks(ticks)
    ax_time.set_xticklabels(ticklabels)
    ax_time.grid(True, which="both", alpha=0.3)
    ax_time.legend()

    ax_mem.set_title("Peak memory vs input size")
    ax_mem.set_xlabel("input size n")
    ax_mem.set_ylabel("peak RSS (GiB)")
    ax_mem.set_yscale("log")
    ax_mem.set_xticks(ticks)
    ax_mem.set_xticklabels(ticklabels)
    ax_mem.grid(True, which="both", alpha=0.3)
    ax_mem.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-exp", type=int, default=30, help="smallest size = 2**min_exp")
    ap.add_argument("--max-exp", type=int, default=36, help="largest size = 2**max_exp")
    ap.add_argument("--repeats", type=int, default=1, help="timed repeats per point (keeps best)")
    ap.add_argument("--timeout", type=float, default=3600.0, help="per-run timeout in seconds")
    ap.add_argument("--csv", default=os.path.join(SCRIPT_DIR, "primes_benchmark_results.csv"))
    ap.add_argument("--json", default=os.path.join(SCRIPT_DIR, "primes_benchmark_results.json"))
    ap.add_argument("--png", default=os.path.join(SCRIPT_DIR, "primes_benchmark.png"))
    ap.add_argument("--no-plot", action="store_true",
                    help="skip plotting (e.g. on a remote machine without matplotlib)")
    ap.add_argument("--plot-only", action="store_true",
                    help="skip the sweep; render the plot from an existing --json data file")
    args = ap.parse_args()

    if args.plot_only:
        exponents, results = read_json(args.json)
        plot(args.png, exponents, results)
        return

    exponents = list(range(args.min_exp, args.max_exp + 1))
    ensure_binary()
    results = sweep(exponents, args.repeats, args.timeout)
    write_csv(args.csv, exponents, results)
    write_json(args.json, exponents, results)

    if args.no_plot:
        return
    try:
        plot(args.png, exponents, results)
    except ImportError:
        print(f"matplotlib not available; data written to {args.json}.\n"
              f"Re-render on a machine with matplotlib via:\n"
              f"  python3 {os.path.basename(__file__)} --plot-only "
              f"--json {args.json} --png {args.png}", file=sys.stderr)


if __name__ == "__main__":
    main()
