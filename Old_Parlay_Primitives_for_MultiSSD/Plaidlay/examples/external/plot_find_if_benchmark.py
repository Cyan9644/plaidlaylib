#!/usr/bin/env python3
"""Benchmark and plot the external find_if (extern_find_if.h) against the
in-memory find_if (examples/in_memory/find_if.h) across input sizes
2**30 .. 2**36.

This drives the `find_if_benchmark` binary (built from
examples/external/find_if_benchmark.cpp). The benchmark itself can sweep a whole
range in one process, but this script invokes it once per exponent
(min_exp == max_exp == e). That isolation is deliberate: if one size crashes or
the in-memory side exhausts RAM and the OOM killer fires, only that single point
is lost and the sweep continues. (The default upper bound is 2**36 because that
is the largest sequence the in-memory side can hold in DRAM.)

Outputs (written next to this script by default):
  * find_if_benchmark_results.csv  -- seconds per (variant, size) + external GiB/s
  * find_if_benchmark_results.json -- same data, trivially reloadable
  * find_if_benchmark.png          -- runtime vs size, and external throughput vs size

In-memory is drawn in yellow, the external algorithm in blue.

Data collection and plotting are decoupled so the benchmark can run on a remote
machine that has no matplotlib: the sweep always writes the CSV/JSON, and
plotting is attempted only as a best-effort final step (skipped with a hint if
matplotlib is missing). Copy the JSON to a machine that has matplotlib and
re-render with `--plot-only`.

If the find_if_benchmark binary is missing, this script builds it on demand via
`make bazel-bin/find_if_benchmark` (run from the repo root), the same way
plot_primes_benchmark.py builds primesScaling. You can also build it by hand:
    make bazel-bin/find_if_benchmark
then point this script at it with --binary if it is not at
bazel-bin/find_if_benchmark.

Usage:
    python3 plot_find_if_benchmark.py                 # full 2**30 .. 2**36 sweep
    python3 plot_find_if_benchmark.py --min-exp 30 --max-exp 34 --repeats 3
    python3 plot_find_if_benchmark.py --pos-frac 0.5  # match halfway through
    python3 plot_find_if_benchmark.py --no-plot       # data only (no matplotlib)
    python3 plot_find_if_benchmark.py --plot-only \
        --json find_if_benchmark_results.json         # plot a JSON from elsewhere
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys

# .../Plaidlay/examples/external/plot_find_if_benchmark.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAIDLAY_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
REPO_ROOT = os.path.dirname(PLAIDLAY_DIR)
DEFAULT_BINARY = os.path.join(REPO_ROOT, "bazel-bin", "find_if_benchmark")

VARIANTS = ("mem", "ext")
LABELS = {"mem": "in-memory", "ext": "external"}
COLORS = {"mem": "gold", "ext": "tab:blue"}  # yellow vs blue, as requested

# Parses the per-size detail line printed by find_if_benchmark, e.g.
#   "    in-memory: 1.2345 s (lazy)   external: 0.6789 s   (external read 12.34 GiB/s)"
DETAIL_RE = re.compile(
    r"in-memory:\s+([0-9.]+)\s+s(?:\s+\(lazy\))?\s+"
    r"external:\s+([0-9.]+)\s+s\s+\(external read\s+([0-9.]+)\s+GiB/s\)"
)


def ensure_binary(binary):
    """Build bazel-bin/find_if_benchmark via make if it isn't there already.
    Only auto-builds the default target; a custom --binary path is left to the
    caller to provide."""
    if os.path.exists(binary):
        return
    if os.path.abspath(binary) != os.path.abspath(DEFAULT_BINARY):
        return  # not our target to build; let the existence check report it
    print("find_if_benchmark not found; building with make...", file=sys.stderr)
    subprocess.run(["make", "bazel-bin/find_if_benchmark"], cwd=REPO_ROOT, check=True)


def run_one(binary, exp, repeats, pos_frac, ssd_base, timeout):
    """Run find_if_benchmark for a single exponent. Returns a dict
    {mem, ext, gibs} of floats, or None on OOM / crash / timeout / parse error."""
    n = 1 << exp
    cmd = [binary, str(exp), str(exp), str(repeats), str(pos_frac), ssd_base]
    try:
        proc = subprocess.run(
            cmd, cwd=PLAIDLAY_DIR, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        print(f"  2**{exp} (n={n}): TIMEOUT (>{timeout}s)", file=sys.stderr)
        return None

    if proc.returncode != 0:
        # Negative return code == killed by signal (e.g. SIGKILL from OOM killer).
        reason = f"signal {-proc.returncode}" if proc.returncode < 0 else f"exit {proc.returncode}"
        print(f"  2**{exp} (n={n}): FAILED ({reason}) -- treating as OOM/crash",
              file=sys.stderr)
        if proc.stderr.strip():
            print(f"      stderr: {proc.stderr.strip()}", file=sys.stderr)
        return None

    m = None
    for line in proc.stdout.splitlines():
        hit = DETAIL_RE.search(line)
        if hit:
            m = hit
            break
    if m is None:
        print(f"  2**{exp} (n={n}): could not parse timing from output", file=sys.stderr)
        if proc.stdout.strip():
            print(f"      last line: {proc.stdout.strip().splitlines()[-1]}", file=sys.stderr)
        return None

    mem_s, ext_s, gibs = float(m.group(1)), float(m.group(2)), float(m.group(3))
    print(f"  2**{exp} (n={n}): in-memory {mem_s:8.4f} s   "
          f"external {ext_s:8.4f} s   ({gibs:6.2f} GiB/s)")
    return {"mem": mem_s, "ext": ext_s, "gibs": gibs}


def sweep(binary, exponents, repeats, pos_frac, ssd_base, timeout):
    """Return dict: variant -> list aligned with `exponents`, each entry seconds
    or None; plus 'gibs' -> list of external read throughput or None."""
    results = {v: [] for v in VARIANTS}
    results["gibs"] = []
    for e in exponents:
        print(f"n = 2**{e} = {1 << e}")
        r = run_one(binary, e, repeats, pos_frac, ssd_base, timeout)
        results["mem"].append(None if r is None else r["mem"])
        results["ext"].append(None if r is None else r["ext"])
        results["gibs"].append(None if r is None else r["gibs"])
        print()
    return results


def write_csv(path, exponents, results):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exponent", "n", "variant", "seconds", "external_gibs"])
        for i, e in enumerate(exponents):
            for v in VARIANTS:
                s = results[v][i]
                gibs = results["gibs"][i] if v == "ext" else ""
                w.writerow([e, 1 << e, v,
                            "OOM" if s is None else f"{s:.6f}",
                            "" if gibs in ("", None) else f"{gibs:.4f}"])
    print(f"wrote {path}")


def write_json(path, exponents, results):
    payload = {"exponents": list(exponents), "results": results}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {path}")


def read_json(path):
    with open(path) as f:
        payload = json.load(f)
    return payload["exponents"], payload["results"]


def plot(path, exponents, results):
    import matplotlib
    matplotlib.use("Agg")  # headless: just write a file
    import matplotlib.pyplot as plt

    fig, (ax_time, ax_tput) = plt.subplots(1, 2, figsize=(14, 6))

    # --- runtime vs size --------------------------------------------------
    for v in VARIANTS:
        xs, ys = [], []
        for i, e in enumerate(exponents):
            s = results[v][i]
            if s is None:
                continue
            xs.append(e)
            ys.append(s)
        ax_time.plot(xs, ys, marker="o", color=COLORS[v], label=LABELS[v])

    # Mark sizes where the in-memory search failed (OOM/crash).
    oom_exps = [e for i, e in enumerate(exponents) if results["mem"][i] is None]
    for e in oom_exps:
        ax_time.axvline(e, color="red", linestyle=":", alpha=0.4)
    if oom_exps:
        ax_time.axvline(oom_exps[0], color="red", linestyle=":", alpha=0.4,
                        label="in-memory OOM/crash")

    ticks = list(exponents)
    ticklabels = [f"$2^{{{e}}}$" for e in exponents]

    ax_time.set_title("find_if runtime vs input size")
    ax_time.set_xlabel("input size n")
    ax_time.set_ylabel("time (seconds)")
    ax_time.set_yscale("log")
    ax_time.set_xticks(ticks)
    ax_time.set_xticklabels(ticklabels)
    ax_time.grid(True, which="both", alpha=0.3)
    ax_time.legend()

    # --- external read throughput vs size ---------------------------------
    xs, ys = [], []
    for i, e in enumerate(exponents):
        g = results["gibs"][i]
        if g is None:
            continue
        xs.append(e)
        ys.append(g)
    ax_tput.plot(xs, ys, marker="o", color=COLORS["ext"], label="external")

    ax_tput.set_title("External find_if read throughput vs input size")
    ax_tput.set_xlabel("input size n")
    ax_tput.set_ylabel("throughput (GiB/s)")
    ax_tput.set_xticks(ticks)
    ax_tput.set_xticklabels(ticklabels)
    ax_tput.grid(True, which="both", alpha=0.3)
    ax_tput.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-exp", type=int, default=30, help="smallest size = 2**min_exp")
    ap.add_argument("--max-exp", type=int, default=36, help="largest size = 2**max_exp")
    ap.add_argument("--repeats", type=int, default=1, help="timed repeats per point (keeps best)")
    ap.add_argument("--pos-frac", type=float, default=1.0,
                    help="match position as a fraction of n (1.0 = last element, worst case)")
    ap.add_argument("--ssd-base", default="",
                    help="SSD mount prefix; files go to <ssd-base>0/, <ssd-base>1/, ... one "
                         "per SSD (e.g. --ssd-base /mnt/ssd). Default \"\" writes flat scratch "
                         "files in the cwd (no root-owned mounts needed), like the primes benchmark.")
    ap.add_argument("--binary", default=DEFAULT_BINARY,
                    help="path to the compiled find_if_benchmark binary")
    ap.add_argument("--timeout", type=float, default=3600.0, help="per-run timeout in seconds")
    ap.add_argument("--csv", default=os.path.join(SCRIPT_DIR, "find_if_benchmark_results.csv"))
    ap.add_argument("--json", default=os.path.join(SCRIPT_DIR, "find_if_benchmark_results.json"))
    ap.add_argument("--png", default=os.path.join(SCRIPT_DIR, "find_if_benchmark.png"))
    ap.add_argument("--no-plot", action="store_true",
                    help="skip plotting (e.g. on a remote machine without matplotlib)")
    ap.add_argument("--plot-only", action="store_true",
                    help="skip the sweep; render the plot from an existing --json data file")
    args = ap.parse_args()

    if args.plot_only:
        exponents, results = read_json(args.json)
        plot(args.png, exponents, results)
        return

    ensure_binary(args.binary)
    if not os.path.exists(args.binary):
        sys.exit(
            f"find_if_benchmark binary not found at {args.binary}.\n"
            f"Build it from the repo root with:\n"
            f"  make bazel-bin/find_if_benchmark\n"
            f"then re-run, or pass --binary <path>."
        )

    exponents = list(range(args.min_exp, args.max_exp + 1))
    results = sweep(args.binary, exponents, args.repeats, args.pos_frac,
                    args.ssd_base, args.timeout)
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
