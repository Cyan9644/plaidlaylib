#!/usr/bin/env python3
"""Clean up files left behind by the benchmark suite.

Two independent categories, both opt-out/opt-in via flags:

  1. Per-drive data files (default ON) -- the `iota<drive>` inputs and every
     pipeline's intermediates (`bw_*`, `csrt_in*`, `zip_a*`, ...) that the
     benchmarks write under each mount in --mount-glob (default /mnt/ssd*).
     Reuses run_benches.py's own BENCH_FILE_GLOBS / clear_bench_data() --
     the same list it already sweeps between points -- so this can never
     drift out of sync with what the benchmarks actually produce.
  2. The results/ output tree (default OFF, --results) -- every timestamped
     results/<stamp>/ directory (CSVs, PNGs, warnings.txt from run_benches.py,
     summary_figure.py, io_trace.py). Off by default since those are the
     actual deliverables (figures/CSVs), not transient working data.

--dry-run lists what would be removed without deleting anything.

  usage:
    python3 benchmarks/clean_bench_data.py                  # drive data only
    python3 benchmarks/clean_bench_data.py --results         # + results/
    python3 benchmarks/clean_bench_data.py --dry-run
    python3 benchmarks/clean_bench_data.py --no-drives --results
"""

import argparse
import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_benches as rb  # noqa: E402  (sibling module; import-safe)


def dry_run_drives(mount_glob):
    total = 0
    for m in sorted(glob.glob(mount_glob)):
        for pat in rb.BENCH_FILE_GLOBS:
            for f in glob.glob(os.path.join(m, pat)):
                print(f"  would remove {f}")
                total += 1
    print(f"  {total} file(s) under {mount_glob!r} would be removed")


def dry_run_results(outdir):
    path = os.path.join(rb.REPO_ROOT, outdir)
    if not os.path.isdir(path):
        print(f"  {path} does not exist, nothing to remove")
        return
    stamps = sorted(os.listdir(path))
    for s in stamps:
        print(f"  would remove {os.path.join(path, s)}")
    print(f"  {len(stamps)} results run(s) under {path!r} would be removed")


def clean_results(outdir):
    path = os.path.join(rb.REPO_ROOT, outdir)
    if not os.path.isdir(path):
        print(f"  {path} does not exist, nothing to remove")
        return
    n = len(os.listdir(path))
    shutil.rmtree(path)
    print(f"  removed {path} ({n} run(s))")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mount-glob", default=os.environ.get("BENCH_FSTRIM_GLOB", rb.DEFAULT_FSTRIM_GLOB),
                    help="glob of mount points to clear per-drive data from (default: /mnt/ssd*)")
    ap.add_argument("--no-drives", action="store_true",
                    help="skip clearing per-drive data files")
    ap.add_argument("--results", action="store_true",
                    help="also remove the results/ output tree (figures/CSVs -- off by default)")
    ap.add_argument("--outdir", default=os.environ.get("BENCH_OUTDIR", "results"),
                    help="results directory to remove with --results (default: results)")
    ap.add_argument("--fstrim", action="store_true",
                    help="fstrim the mounts after clearing drive data")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be removed without deleting anything")
    args = ap.parse_args()

    if not args.no_drives:
        print(f"Clearing per-drive bench data under {args.mount_glob!r}...")
        if args.dry_run:
            dry_run_drives(args.mount_glob)
        else:
            rb.clear_bench_data(args.mount_glob, True)
            if args.fstrim:
                note = rb.fstrim_mounts(args.mount_glob, True)
                if note:
                    print(f"  {note}")
    else:
        print("Skipping per-drive data (--no-drives).")

    if args.results:
        print(f"\nRemoving results tree ({args.outdir!r})...")
        if args.dry_run:
            dry_run_results(args.outdir)
        else:
            clean_results(args.outdir)
    else:
        print("\nLeaving results/ alone (pass --results to remove it too).")


if __name__ == "__main__":
    main()
