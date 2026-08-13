#!/usr/bin/env bash
# Clean up files left behind by the benchmark suite.
#
# Two independent categories, both opt-out/opt-in via flags:
#
#   1. Per-drive data files (default ON) -- the iota<drive> inputs and every
#      pipeline's intermediates (bw_*, csrt_in*, zip_a*, ...) that the
#      benchmarks write under each mount in --mount-glob (default /mnt/ssd*).
#      The pattern list is pulled live from run_benches.py's own
#      BENCH_FILE_GLOBS (the same list it sweeps between points), via a
#      one-line python3 call -- so this can never drift out of sync with
#      what the benchmarks actually produce. python3 is already a hard
#      requirement of this repo's benchmark suite (shell.nix), so this adds
#      no new dependency.
#   2. The results/ output tree (default OFF, --results) -- every timestamped
#      results/<stamp>/ directory (CSVs, PNGs, warnings.txt from
#      run_benches.py, summary_figure.py, io_trace.py). Off by default since
#      those are the actual deliverables (figures/CSVs), not transient
#      working data.
#
# --dry-run lists what would be removed without deleting anything.
#
#   usage:
#     ./benchmarks/clean_bench_data.sh                  # drive data only
#     ./benchmarks/clean_bench_data.sh --results         # + results/
#     ./benchmarks/clean_bench_data.sh --dry-run
#     ./benchmarks/clean_bench_data.sh --no-drives --results

set -euo pipefail
shopt -s nullglob

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MOUNT_GLOB="${BENCH_FSTRIM_GLOB:-/mnt/ssd*}"
OUTDIR="${BENCH_OUTDIR:-results}"
NO_DRIVES=0
DO_RESULTS=0
DO_FSTRIM=0
DRY_RUN=0

usage() {
    sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mount-glob) MOUNT_GLOB="$2"; shift 2 ;;
        --mount-glob=*) MOUNT_GLOB="${1#*=}"; shift ;;
        --no-drives) NO_DRIVES=1; shift ;;
        --results) DO_RESULTS=1; shift ;;
        --outdir) OUTDIR="$2"; shift 2 ;;
        --outdir=*) OUTDIR="${1#*=}"; shift ;;
        --fstrim) DO_FSTRIM=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ $NO_DRIVES -eq 0 ]]; then
    echo "Clearing per-drive bench data under '$MOUNT_GLOB'..."

    mapfile -t patterns < <(python3 -c "
import sys, os
sys.path.insert(0, os.path.join('$REPO_ROOT', 'benchmarks'))
import run_benches as rb
print('\n'.join(rb.BENCH_FILE_GLOBS))
")

    removed=0
    for mount in $MOUNT_GLOB; do
        [[ -d "$mount" ]] || continue
        for pat in "${patterns[@]}"; do
            for f in "$mount"/$pat; do
                if [[ $DRY_RUN -eq 1 ]]; then
                    echo "  would remove $f"
                else
                    rm -f "$f"
                fi
                removed=$((removed + 1))
            done
        done
    done
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  $removed file(s) under '$MOUNT_GLOB' would be removed"
    else
        echo "  removed $removed file(s)"
        if [[ $DO_FSTRIM -eq 1 ]]; then
            for mount in $MOUNT_GLOB; do
                [[ -d "$mount" ]] || continue
                if fstrim "$mount" 2>/dev/null; then
                    echo "  fstrim ok on $mount"
                else
                    echo "  fstrim skipped/unsupported on $mount"
                fi
            done
        fi
    fi
else
    echo "Skipping per-drive data (--no-drives)."
fi

if [[ $DO_RESULTS -eq 1 ]]; then
    results_path="$REPO_ROOT/$OUTDIR"
    echo
    echo "Removing results tree ('$OUTDIR')..."
    if [[ -d "$results_path" ]]; then
        n=$(find "$results_path" -mindepth 1 -maxdepth 1 | wc -l)
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "  would remove $results_path ($n run(s))"
        else
            rm -rf "$results_path"
            echo "  removed $results_path ($n run(s))"
        fi
    else
        echo "  $results_path does not exist, nothing to remove"
    fi
else
    echo
    echo "Leaving results/ alone (pass --results to remove it too)."
fi
