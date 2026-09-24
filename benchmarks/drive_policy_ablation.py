#!/usr/bin/env python3
"""One-off sweep: how much does chunk->drive placement matter?

TEMPORARY / EXPERIMENTAL (branch `sequence-ablation`).  Deliberately NOT wired
into any make target -- it is a data-gathering script for one question, not part
of the benchmark suite.

The library normally scatters a chunk_seq's chunks pseudo-randomly across the
drives (balls-in-bins).  `utils/drive_policy.h` adds a runtime switch,
PLAID_DRIVE_POLICY, with two alternatives; this script runs bigint_add across
an input-size ladder under each, so the three can be plotted against one
another:

  random       balls-in-bins (the status quo, and the default)
  round_robin  chunk i -> drive i % D
  blocked      the first N/D of the sequence on drive 0, the next on drive 1,
               ...  Proportional, so it exercises all D drives without needing
               an input large enough to physically fill them.

What each example exercises (see the drive_policy.h call sites):

  bigint_add   `add_s`       plaid::tabulate (input build) + delayed::force
               `eager_add_s` the same plus ChunkEmitter/ExternalTransform,
                             which the delayed path never touches -- so each
                             bigint point yields two independent measurements
                             of the same policy.  The eager pass also
                             bit-compares against the delayed result, which is
                             this sweep's only end-to-end correctness check
                             (the DRAM baselines are switched off below).
  samplesort   `sort_s`      plaid::tabulate + BucketWriter (via
                             group_by_index).  NOT swept by default any more --
                             see EXAMPLE_NAMES.  If you re-enable it, note that
                             BucketWriter's status quo is already round-robin
                             over bucket index, so for the bucket files it is
                             `round_robin` that reproduces today's layout and
                             `random` that is the new arm -- unlike every other
                             placement site.

In-memory baselines are switched off (EXAMPLE_INMEM_BUDGET_BYTES=0) for every
run: the comparison here is policy-vs-policy, the baselines do not depend on
placement at all, and at the top of the ladder they would be skipped anyway.

Everything is reused from run_benches.py -- the size->n conversion, the `make`
wrapper, the CSV-line parser, the between-point drive cleanup -- so this script
cannot drift from how the real sweeps size and run the same binaries.

Between every run the mounts are purged of ALL files (not just the prefixes
the runner knows about), synced, and fstrimmed -- in that order, since fstrim
only reclaims blocks of already-deleted files.  Per-point rather than once at
startup: drive state that accumulates across points would otherwise show up as
a difference between policy arms.

Usage:
    python3 benchmarks/drive_policy_ablation.py                  # full ladder
    python3 benchmarks/drive_policy_ablation.py --sizes 1GiB \
        --policies random,blocked                                # smoke test
"""

import argparse
import glob
import os
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benches as rb   # noqa: E402  (path must be set first)

def hb(n):
    """Human-readable byte size, via run_benches' own matplotlib formatter (it
    ignores its second argument), so sweep output labels sizes identically."""
    return rb._bytes_fmt(n, None)


POLICIES = ("random", "round_robin", "blocked")
# samplesort is deliberately NOT swept by default: its DRAM appetite scales
# with machine size (process_inplace_budgeted sizes waves off available
# RAM/4, sample_sort picks a bucket count to fit DRAM), which made it the
# prime suspect for taking the bench box down.  Still reachable with
# `--examples samplesort` if you want it back.
EXAMPLE_NAMES = ("bigint_add",)

# x4 ladder, matching `make bench-examples-full`'s shape so points line up with
# sweeps collected there.
DEFAULT_SIZES = "1GiB 4GiB 16GiB 64GiB 256GiB 1TiB"

# Which column each example's plot follows: (csv_column, legend suffix).  A
# bigint point carries two independent measurements; samplesort one.
PLOT_COLS = {
    "bigint_add": [("add_s", "delayed"), ("eager_add_s", "eager")],
    "samplesort": [("sort_s", "")],
}


def child_env(policy):
    """Child environment for one run: the policy, and no DRAM baseline."""
    env = dict(os.environ)
    env["PLAID_DRIVE_POLICY"] = policy
    # Policy-vs-policy comparison; the DRAM baseline is placement-independent
    # dead weight at the small end and skipped at the large end.
    env["EXAMPLE_INMEM_BUDGET_BYTES"] = "0"
    return env


def purge_mounts(glob_pat, enabled):
    """Unlink every data file under each mount, then confirm none remain.

    Two passes.  First run_benches' known-glob sweep (it owns the
    chmod-and-retry path for files a crashed or sudo run left unwritable).
    Then a catch-all for anything those globs miss -- a prefix no `data_globs`
    entry covers, or debris from a run that died before it could clean up.
    The catch-all prints every file it removes, since by definition those are
    files nothing expected to be there.

    Returns the number of unexpected files removed.  Directories (lost+found)
    are never touched.
    """
    if not enabled:
        return 0
    rb.clear_bench_data(glob_pat, True)
    stray = []
    for m in sorted(glob.glob(glob_pat)):
        try:
            entries = list(os.scandir(m))
        except OSError as e:
            print(f"  !!! cannot scan {m}: {e}", flush=True)
            continue
        for de in entries:
            if not de.is_file(follow_symlinks=False):
                continue
            try:
                os.unlink(de.path)
                stray.append(de.path)
            except OSError:
                try:
                    os.chmod(de.path, 0o644)
                    os.unlink(de.path)
                    stray.append(de.path)
                except OSError as e:
                    print(f"  !!! could NOT remove {de.path}: {e}", flush=True)
    if stray:
        print(f"  purged {len(stray)} file(s) the known globs missed:",
              flush=True)
        for f in stray[:10]:
            print(f"      {f}", flush=True)
        if len(stray) > 10:
            print(f"      ... and {len(stray) - 10} more", flush=True)
    return len(stray)


def reclaim(glob_pat, clear_enabled, trim_enabled, label):
    """Unlink everything, commit the unlinks, then fstrim -- in that order.

    fstrim only reclaims blocks belonging to *deleted* files, so a trim that
    runs while data is still linked (or while the unlinks are still only in
    the journal) frees nothing.  Hence: purge -> sync -> trim.

    Unlike run_benches, this trims after every point rather than once at
    startup: the whole reason this sweep exists is to compare policies against
    each other, so any drive state that accumulates across points shows up as
    a difference between arms.
    """
    purge_mounts(glob_pat, clear_enabled)
    if not trim_enabled:
        return
    os.sync()          # unlinks must be on-device before FITRIM sees the blocks
    mounts = sorted(glob.glob(glob_pat))
    if not mounts:
        return
    ok, err = 0, None
    for m in mounts:
        r = subprocess.run(["fstrim", m], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        if r.returncode == 0:
            ok += 1
        elif err is None:
            err = r.stdout.strip() or f"fstrim {m} exit {r.returncode}"
    msg = f"  fstrim after {label}: {ok}/{len(mounts)} mount(s)"
    if err:
        msg += f" (rest skipped/unsupported: {err})"
    print(msg, flush=True)


def sweep(entry, policies, sizes, extra_args, clear_glob, clear_enabled,
          trim_enabled, warnings, timeout):
    """Run every (policy, size) point for one example, INTERLEAVED.

    The loop is size-major with the policy order rotated one step per size, so
    that consecutive points alternate policies and each policy takes a
    different position within the size group as the ladder advances.

    This ordering is the whole point.  An earlier version ran one policy's
    entire ladder before starting the next, which confounded policy with
    elapsed time: the first arm got freshly-fstrimmed drives and every later
    arm inherited the accumulated write/delete state of the ones before it.
    That produced a monotone-in-run-order result (and made round_robin and
    blocked -- maximally different layouts -- converge to within 0.2% of each
    other at the top of the ladder, which no placement mechanism explains).
    Interleaving spreads device drift across the arms instead of aligning it
    with them.  Drift is not eliminated, only decorrelated from the variable
    under test; `fstrim` still runs once at startup, not per point.

    Soft-failure discipline matches run_benches' examples sweep: warn, drop the
    point, keep going.  A timeout additionally skips that policy's larger sizes
    -- they can only be slower -- without disturbing the other policies.
    """
    binary = os.path.join(rb.BINDIR, os.path.basename(entry["target"]))
    rows_by_policy = {p: [] for p in policies}
    timed_out = set()

    for si, size in enumerate(sizes):
        n = rb.size_to_n(entry, size)
        rot = si % len(policies)
        order = policies[rot:] + policies[:rot]
        for policy in order:
            if policy in timed_out:
                continue
            print(f"\n=== {entry['name']} [{policy}]: {hb(size)} "
                  f"(n={n}) ===", flush=True)
            fields, problem = rb.run_binary(
                binary, entry.get("pre_argv", []) + [n] +
                entry.get("extra_argv", []) + extra_args,
                fatal=False, env=child_env(policy), timeout=timeout)

            if problem is not None:
                w = f"{entry['name']} [{policy}] at {hb(size)}: {problem}"
                warnings.append(w)
                print(f"  !!! {w}", flush=True)
                reclaim(clear_glob, clear_enabled, trim_enabled,
                        f"{entry['name']} [{policy}] {hb(size)} (failed)")
                if "timed out" in problem:
                    timed_out.add(policy)
                    rest = [hb(s) for s in sizes[si + 1:]]
                    if rest:
                        w = (f"{entry['name']} [{policy}]: skipping larger "
                             f"sizes {', '.join(rest)} after the timeout above")
                        warnings.append(w)
                        print(f"  !!! {w}", flush=True)
                continue

            row = dict(zip(entry["cols"], fields))
            row["policy"] = policy
            row["input_bytes"] = str(size)
            rows_by_policy[policy].append(row)
            reclaim(clear_glob, clear_enabled, trim_enabled,
                    f"{entry['name']} [{policy}] {hb(size)}")
    return rows_by_policy


def plot(rows_by_policy, entry, path):
    """One log-log panel per example: time vs input size, a line per policy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    import plot_style
    plot_style.apply()

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    colors = plot_style.PALETTE
    order = ["blue", "orange", "aqua"]
    styles = {"delayed": "-o", "eager": "--s", "": "-o"}

    plotted = False
    for pi, policy in enumerate(POLICIES):
        rows = rows_by_policy.get(policy, [])
        if not rows:
            continue
        color = colors[order[pi % len(order)]]
        for col, suffix in PLOT_COLS[entry["name"]]:
            pts = [(int(r["input_bytes"]), rb._f(r[col])) for r in rows
                   if r.get(col, "").strip()]
            if not pts:
                continue
            xs, ys = zip(*pts)
            # plot_style's cmr10 renders "_" as a dot accent, so hyphenate.
            label = policy.replace("_", "-") + (f" ({suffix})" if suffix else "")
            ax.plot(xs, ys, styles[suffix], color=color, label=label,
                    markersize=4, linewidth=1.4)
            plotted = True
    if not plotted:
        raise RuntimeError("no usable points to plot")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("input size (bytes)")
    ax.set_ylabel("time (s)")
    # Likewise "_" and ">" both render badly under cmr10 -- keep both out.
    ax.set_title(f"{entry['name'].replace('_', '-')}: drive placement policy")
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: hb(int(v))))
    ax.grid(True, which="major", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Drive-placement policy ablation (one-off).")
    ap.add_argument("--sizes", default=DEFAULT_SIZES,
                    help=f"input sizes, space/comma separated (default: {DEFAULT_SIZES})")
    ap.add_argument("--policies", default=",".join(POLICIES),
                    help="policies to sweep (default: all three)")
    ap.add_argument("--examples", default=",".join(EXAMPLE_NAMES),
                    help="examples to sweep (default: bigint_add)")
    ap.add_argument("--outdir", default=os.environ.get("BENCH_OUTDIR", "results"))
    ap.add_argument("--timeout-min", type=float, default=30.0,
                    help="SIGKILL any single run over this many minutes (0 = no limit)")
    ap.add_argument("--ssd-args", default=os.environ.get("BENCH_SSD_ARGS", ""),
                    help="extra global flags passed to every binary")
    ap.add_argument("--fstrim-glob", default=rb.DEFAULT_FSTRIM_GLOB)
    ap.add_argument("--no-fstrim", action="store_true")
    ap.add_argument("--no-clean", action="store_true")
    args = ap.parse_args()

    sizes = [rb.parse_bytes(s) for s in args.sizes.replace(",", " ").split()]
    policies = [p for p in args.policies.replace(",", " ").split()]
    for p in policies:
        if p not in POLICIES:
            sys.exit(f"unknown policy {p!r}; choose from {', '.join(POLICIES)}")
    names = [e for e in args.examples.replace(",", " ").split()]
    by_name = {e["name"]: e for e in rb.EXAMPLES}
    for nm in names:
        if nm not in by_name:
            sys.exit(f"unknown example {nm!r}")
    entries = [by_name[nm] for nm in names]
    extra_args = args.ssd_args.split()
    clear_enabled = not args.no_clean
    trim_enabled = not args.no_fstrim
    timeout = args.timeout_min * 60 or None

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(rb.REPO_ROOT, args.outdir, f"{stamp}-drive-policy")
    os.makedirs(outdir, exist_ok=True)
    print(f"Drive-placement ablation -> {outdir}")
    print(f"  policies: {', '.join(policies)}")
    print(f"  examples: {', '.join(names)}")
    print(f"  sizes:    {', '.join(hb(s) for s in sizes)}")
    tmo = f"{args.timeout_min:g} min/run" if timeout else "none"
    print(f"  {len(policies) * len(entries) * len(sizes)} runs, "
          f"in-memory baselines OFF, timeout: {tmo}")
    print("  order:    interleaved (size-major, policy order rotated per size) "
          "so device drift does not align with the policy under test")
    print(f"  cleanup:  purge every file from the mounts, sync, then "
          f"{'fstrim' if trim_enabled else 'NO fstrim (--no-fstrim)'} "
          f"after every run")

    # The policy is a runtime switch, so one build serves every arm.
    for entry in entries:
        rb.make(entry["target"])

    print("\nclearing and trimming the mounts before the first run ...",
          flush=True)
    reclaim(args.fstrim_glob, clear_enabled, trim_enabled, "startup")

    warnings = []
    all_rows = []
    for entry in entries:
        rows_by_policy = sweep(entry, policies, sizes, extra_args,
                               args.fstrim_glob, clear_enabled, trim_enabled,
                               warnings, timeout)

        flat = [r for p in policies for r in rows_by_policy[p]]
        header = ["policy", "input_bytes"] + entry["cols"]
        rb.write_csv(os.path.join(outdir, f"drive_policy_{entry['name']}.csv"),
                     header, flat)
        for r in flat:
            all_rows.append(dict(r, example=entry["name"]))

        # The CSV is the result; a plotting failure must not discard timings we
        # just spent hours of I/O collecting (same rule as run_benches).
        try:
            plot(rows_by_policy, entry,
                 os.path.join(outdir, f"drive_policy_{entry['name']}.png"))
        except Exception as exc:
            warnings.append(f"{entry['name']}: plotting failed ({exc}); "
                            "CSV was written")
            print(f"  !!! plotting failed ({exc}); CSV was written", flush=True)

    # Combined long-format CSV: one row per (example, policy, size), with the
    # union of both examples' columns so it loads as a single table.
    if all_rows:
        cols = ["example", "policy", "input_bytes"]
        for e in entries:
            for c in e["cols"]:
                if c not in cols:
                    cols.append(c)
        for r in all_rows:
            for c in cols:
                r.setdefault(c, "")
        rb.write_csv(os.path.join(outdir, "drive_policy_all.csv"), cols, all_rows)

    print("\n======== run summary ========")
    if warnings:
        print(f"  !!! {len(warnings)} warning(s) (the sweep continued past them):")
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
