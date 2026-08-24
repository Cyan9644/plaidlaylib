#!/usr/bin/env python3
"""Combined relative-performance bar chart: 21 primitives/examples, each run
ONCE at (approximately) the largest input size where its own in-memory
parlaylib baseline still fits DRAM, plotted against a pinned in-mem
reference at 1.0 -- in the spirit of parlaylib's own "ParlayLib vs
ParallelSTL" figure (see benchresults/example_figure/), but comparing this
library's out-of-core primitives/examples against their in-memory parlaylib
counterparts instead.

Reuses run_benches.py by import (the EXAMPLES registry, size_to_n, make(),
run_binary(), clear_bench_data(), REPO_ROOT/BINDIR) -- the same
`import run_benches as rb` precedent io_trace.py / csv_from_log.py /
work_exponent_bench.py already use.  A SEPARATE script rather than a mode
bolted onto run_benches.py: that driver's whole shape (run_example,
plot_example, --example-sizes) is built around sweeping one entry across many
sizes and log-log line-plotting it, which is structurally different from
"one point per entry, cross-entry categorical bar plot" -- this keeps
run_benches.py, and every existing consumer of it, completely unchanged.

Sizing: each of the 20 "generic" entries (every requested primitive/example
except bellman_ford, which has its own two-part budget check -- see below)
already independently decides, IN C++, whether its own in-memory baseline
fits (n <= budget/mult, reading EXAMPLE_INMEM_BUDGET_BYTES or falling back to
sysconf). Rather than re-deriving that gate a second time here and risking
drift, this script predicts a target n close to the cliff (using the
budget_mult/budget_base recorded on each registry entry, verified against
each driver's own source -- see run_benches.py's EXAMPLES comments), runs the
binary ONCE, and treats the binary's own internal gate as authoritative: if
the CSV's inmem_col field comes back blank (the driver's own check decided
the baseline didn't fit -- e.g. rounding put this script's estimate a hair
over the true cliff), the target is shrunk and retried (up to MAX_ATTEMPTS)
before the entry is dropped from the figure with a warning -- never silently
omitted without explanation, never fabricated.

bellman_ford is special-cased: its budget gate is a two-part check (n <=
inmem_max_n, AND ext_bytes(n)+inmem_bytes(n) <= budget) read from its OWN
env var (BELLMAN_FORD_BUILD_BUDGET_BYTES, not EXAMPLE_INMEM_BUDGET_BYTES),
verified directly against bellman_ford.cpp's source. Its per-round streaming
cost can make even a "fits DRAM" point take far longer than every other
entry's point combined -- pass --bellman-ford-n to bypass the predictor
entirely with an explicit, non-cliff-anchored n if the predicted point proves
impractical.

  usage:
    python3 benchmarks/summary_figure.py --outdir results
    python3 benchmarks/summary_figure.py --only "reduce,tabulate,zip"
    python3 benchmarks/summary_figure.py --bellman-ford-n 4096
"""

import argparse
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benches as rb  # noqa: E402  (sibling module; import-safe)

SAFETY = 0.97           # margin below the predicted cliff (RSS accounting is a simplification)
MAX_ATTEMPTS = 3
SHRINK = 0.85            # per-retry shrink factor when a prediction overshoots

# Per-entry OOM fallback: if the DRAM-cliff predictor's normal shrink-retry
# loop exhausts MAX_ATTEMPTS without ever landing a point, retry once more at
# this fixed byte size before giving up on the entry entirely.  kth_smallest's
# recursive out-of-core path has been observed to OOM repeatedly under the
# predictor's own estimate; 23232774144 bytes (n=2904031232 at its 8-byte
# elements, chunk-grid rounded by rb.size_to_n) is a size confirmed to run
# cleanly, so it's registered here as that entry's fallback.  Add more
# entries as other predictors prove unreliable on a given box.
OOM_FALLBACK_SIZE_BYTES = {
    "kth_smallest": 23232774144,
}

# (display_label, EXAMPLES registry name) pairs for the 21 bars: 12 primitives
# + 9 examples, "sort"/"samplesort" collapsed to the single samplesort entry
# and "bellman_ford" mapping to the bellman_ford_sparse registry entry (the
# sparse RMAT case; see predict_n_bellman_ford_sparse below for why it needs
# its own budget math). Already alphabetical by display_label; asserted below
# so a future edit can't silently desync the claim from the actual order.
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
    ("zip", "zip"),
]
assert sorted(SUMMARY_ENTRIES) == SUMMARY_ENTRIES, \
    "SUMMARY_ENTRIES must stay alphabetical by display_label"

REGISTRY = {e["name"]: e for e in rb.EXAMPLES}
for _label, _name in SUMMARY_ENTRIES:
    assert _name in REGISTRY, f"SUMMARY_ENTRIES: {_name!r} not in rb.EXAMPLES"


def phys_bytes():
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def avail_bytes():
    """Approximate utils/file_utils.h's AvailablePhysicalMemoryBytes(): the
    kernel's own free+reclaimable estimate (/proc/meminfo's MemAvailable),
    not raw installed RAM -- an example's own C++ RAM gate uses the real
    thing (and re-reads it at run time), so predicting off total RAM here can
    target a size the C++ gate would itself reject (or, worse, accept with no
    real headroom behind it).  Falls back to phys_bytes() if unreadable (e.g.
    non-Linux), same fallback the C++ helper uses.  Doesn't chase the C++
    side's cgroup-headroom min -- the existing SAFETY margin and shrink-retry
    loop already absorb estimation slop of this size.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return phys_bytes()


def get_budget(budget_base, env_var="EXAMPLE_INMEM_BUDGET_BYTES"):
    override = os.environ.get(env_var)
    if override:
        return int(override)
    base = avail_bytes() if budget_base.startswith("avail") else phys_bytes()
    return base // 2 if budget_base.endswith("/2") else base


def predict_n(entry, scale):
    """Target n for a 'generic' entry: budget/mult, chunk-grid rounded.

    Some entries hard-cap their in-mem baseline below whatever the RAM
    budget would otherwise allow (e.g. convex_hull's n<2^31 upstream int
    index limit) -- "max_n", if present, clamps the prediction to that.
    """
    budget = get_budget(entry["budget_base"])
    n_target = max(1, int(budget / entry["budget_mult"] * SAFETY * scale))
    if "max_n" in entry:
        n_target = min(n_target, entry["max_n"])
    size_bytes = n_target * entry["elem_bytes"] * entry["input_seqs"]
    return rb.size_to_n(entry, size_bytes)


def predict_n_bellman_ford_sparse(scale):
    """bellman_ford's two-part gate, sparse case (avg_degree=2), inverted.

    From bellman_ford.cpp: n_req is the requested vertex count (== n here,
    the sparse case does no separate scaling); ext_bytes = 8*(n+1) + 32*n,
    inmem_bytes = 48*n + 64*avg_degree*n_req = 48*n + 128*n (avg_degree=2,
    n_req~=n).  Condition: ext_bytes + inmem_bytes <= budget (its own
    BELLMAN_FORD_BUILD_BUDGET_BYTES env var, default phys/2), AND
    n <= inmem_max_n (BELLMAN_FORD_INMEM_MAX_N, default 2^30).
    """
    budget = get_budget("phys/2", env_var="BELLMAN_FORD_BUILD_BUDGET_BYTES")
    inmem_max_n = int(os.environ.get("BELLMAN_FORD_INMEM_MAX_N", 1 << 30))
    # ext_bytes(n) + inmem_bytes(n) ~= (8+32)*n + (48+128)*n = 216*n (+8 const)
    n_target = max(1, int(budget / 216 * SAFETY * scale))
    return min(n_target, inmem_max_n)


def run_once(entry, n, extra_argv, extra_ssd_args):
    rb.make(entry["target"])
    binary = os.path.join(rb.BINDIR, os.path.basename(entry["target"]))
    argv = (entry.get("pre_argv", []) + [n] + entry.get("extra_argv", [])
            + list(extra_argv) + list(extra_ssd_args))
    fields, problem = rb.run_binary(binary, argv, fatal=False)
    return fields, problem


def run_summary(bellman_ford_n, extra_ssd_args, clear_glob, clear_enabled, warnings,
                n_override=None):
    """Run every SUMMARY_ENTRIES point once; return a list of result rows.

    Each row: {label, name, n, time_s, inmem_time_s, ratio}. An entry whose
    baseline never fits within MAX_ATTEMPTS retries falls back, if it's
    registered in OOM_FALLBACK_SIZE_BYTES, to one more attempt at that fixed
    byte size; only after that also fails (or for entries with no fallback
    registered) is it omitted (and warned about), not fabricated.

    n_override, if given, replaces the per-entry DRAM-cliff prediction with
    one fixed n for every entry and disables the shrink-retry loop (one
    attempt only) -- the user asked for an exact n, so silently resizing it
    away would defeat the point.
    """
    max_attempts = 1 if n_override is not None else MAX_ATTEMPTS
    rows = []
    for label, name in SUMMARY_ENTRIES:
        entry = REGISTRY[name]
        time_col = entry.get("time_col", "time_s")
        inmem_col = entry["inmem_col"]
        print(f"\n######## summary: {label} ({name}) ########", flush=True)

        # fft quantizes its problem size to the largest power of two <= n
        # (choose_dims), so the default SHRINK=0.85 per-retry factor can
        # leave every retry inside the same power-of-two bucket -- identical
        # A/B dims, identical failure. Halving guarantees each retry crosses
        # into a strictly smaller bucket.
        shrink = 0.5 if name == "fft" else SHRINK
        scale = 1.0
        got_row = None
        for attempt in range(1, max_attempts + 1):
            if n_override is not None:
                n = n_override
            elif name == "bellman_ford_sparse" and bellman_ford_n is not None:
                n = bellman_ford_n
            elif name == "bellman_ford_sparse":
                n = predict_n_bellman_ford_sparse(scale)
            else:
                n = predict_n(entry, scale)

            print(f"  attempt {attempt}: n={n}", flush=True)
            fields, problem = run_once(entry, n, [], extra_ssd_args)
            if problem:
                w = f"summary {label} ({name}) at n={n}: {problem}"
                print(f"  !!! {w}", flush=True)
                if attempt < max_attempts:
                    warnings.append(w)
                rb.clear_bench_data(clear_glob, clear_enabled)
                # A crash (no CSV) is often the same RAM-cliff-adjacent
                # problem as a blank inmem field (e.g. real peak RSS
                # slightly exceeds a budget estimate) -- shrink and retry
                # the same as any other cliff miss, rather than giving up
                # on the first attempt.
                scale *= shrink
                continue

            row = dict(zip(entry["cols"], fields))
            rb.clear_bench_data(clear_glob, clear_enabled)
            if not row.get(inmem_col, "").strip():
                retry_note = "not retrying, --n was explicit" if n_override is not None else "retrying smaller"
                w = (f"summary {label} ({name}): in-mem baseline skipped at "
                     f"n={n} ({retry_note})")
                print(f"  !!! {w}", flush=True)
                if attempt < max_attempts:
                    warnings.append(w)
                scale *= shrink
                continue

            got_row = row
            got_row["n"] = str(n)
            break

        if got_row is None and n_override is None and name in OOM_FALLBACK_SIZE_BYTES:
            fallback_n = rb.size_to_n(entry, OOM_FALLBACK_SIZE_BYTES[name])
            print(f"  predictor exhausted; OOM fallback attempt: n={fallback_n} "
                  f"(fixed {OOM_FALLBACK_SIZE_BYTES[name]} bytes)", flush=True)
            fields, problem = run_once(entry, fallback_n, [], extra_ssd_args)
            if problem:
                w = f"summary {label} ({name}) OOM fallback at n={fallback_n}: {problem}"
                print(f"  !!! {w}", flush=True)
                warnings.append(w)
                rb.clear_bench_data(clear_glob, clear_enabled)
            else:
                row = dict(zip(entry["cols"], fields))
                rb.clear_bench_data(clear_glob, clear_enabled)
                if not row.get(inmem_col, "").strip():
                    w = (f"summary {label} ({name}): in-mem baseline also skipped at "
                         f"OOM fallback n={fallback_n}")
                    print(f"  !!! {w}", flush=True)
                    warnings.append(w)
                else:
                    got_row = row
                    got_row["n"] = str(fallback_n)

        if got_row is None:
            w = f"summary {label} ({name}): gave up after {max_attempts} attempt(s), no bar"
            print(f"  !!! {w}", flush=True)
            warnings.append(w)
            continue

        t = float(got_row[time_col])
        t_inmem = float(got_row[inmem_col])
        ratio = t / t_inmem if t_inmem > 0 else float("nan")
        rows.append({"label": label, "name": name, "n": got_row["n"],
                     "time_s": f"{t:.9g}", "inmem_time_s": f"{t_inmem:.9g}",
                     "ratio": f"{ratio:.9g}"})
        print(f"  {label}: n={got_row['n']}  out-of-core={t:.4g}s  "
              f"in-mem={t_inmem:.4g}s  ratio={ratio:.4g}x", flush=True)

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
    # two-column \textwidth, short enough not to dominate the page. This
    # figure is much shorter than plot_style's other consumers are tuned
    # for (they run ~5.5-6.5in tall), so left at plot_style's absolute
    # point sizes the text would dwarf a 2.5in-tall plot area. \includegraphics
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
        fig, ax = plt.subplots(figsize=(fig_width, 2.5), constrained_layout=True)
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
                    help="parent dir for the timestamped run (default: results)")
    ap.add_argument("--only", default="",
                    help="run only these display label(s) (comma/space-separated, "
                         f"e.g. 'reduce,tabulate,zip'); choices: "
                         f"{', '.join(l for l, _ in SUMMARY_ENTRIES)}")
    ap.add_argument("--bellman-ford-n", type=int, default=None,
                    help="bypass the RAM-cliff predictor for bellman_ford and use "
                         "this n directly (see the file-level docstring)")
    ap.add_argument("--n", default=None,
                    help="run every entry at exactly this element count instead of "
                         "predicting a per-entry DRAM-cliff n (e.g. '2^20' or "
                         "'1048576', same format as run_benches.py's --n-values). "
                         "With this set, each entry runs once, no shrink-retries: if "
                         "the in-mem baseline doesn't fit at this n, that entry is "
                         "dropped with a warning rather than silently resized. "
                         "Omit to keep the default largest-n-that-fits-DRAM behavior.")
    ap.add_argument("--ssd-args", default=os.environ.get("BENCH_SSD_ARGS", ""),
                    help="extra global flags passed to each binary (e.g. '--num_ssd=4')")
    ap.add_argument("--fstrim-glob", default=os.environ.get("BENCH_FSTRIM_GLOB", rb.DEFAULT_FSTRIM_GLOB),
                    help="glob of mounts to fstrim once at startup (default: /mnt/ssd*)")
    ap.add_argument("--no-fstrim", action="store_true", help="disable the startup fstrim")
    ap.add_argument("--no-clean", action="store_true",
                    help="leave bench data files on the mounts (default: clear between points)")
    args = ap.parse_args()

    selected = [x for x in re.split(r"[,\s]+", args.only) if x]
    known = {l for l, _ in SUMMARY_ENTRIES}
    for name in selected:
        if name not in known:
            ap.error(f"unknown --only label {name!r}; choices: {', '.join(sorted(known))}")
    if selected:
        SUMMARY_ENTRIES = [(l, n) for l, n in SUMMARY_ENTRIES if l in selected]

    extra_ssd_args = args.ssd_args.split() if args.ssd_args else []
    clear_enabled = not args.no_clean
    fstrim_enabled = not args.no_fstrim

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(rb.REPO_ROOT, args.outdir, stamp)
    os.makedirs(outdir, exist_ok=True)
    print(f"Run directory: {outdir}\n")

    rb.clear_bench_data(args.fstrim_glob, clear_enabled)
    fstrim_note = rb.fstrim_mounts(args.fstrim_glob, fstrim_enabled)

    n_override = rb.parse_count(args.n) if args.n else None

    warnings = []
    rows = run_summary(args.bellman_ford_n, extra_ssd_args, args.fstrim_glob,
                       clear_enabled, warnings, n_override=n_override)

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
    if fstrim_note:
        print(f"  {fstrim_note}")
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
