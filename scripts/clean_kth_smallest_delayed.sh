#!/usr/bin/env bash
# Remove the files left behind by a cancelled kth_smallest_delayedExample run.
#
# The benchmark (ChunkSequence/examples/external/kth_smallest_delayed.cpp)
# cleans up after itself between contestants via bench_drives::clear_drives(),
# but that only runs if the process exits normally — Ctrl-C mid-selection
# leaves that contestant's input and recursion intermediates sitting on every
# drive. This script does the same prefix sweep by hand.
#
# Prefixes must stay in sync with kth_smallest_delayed.cpp's kFastPrefixes /
# kDelayedPrefixes (kth_next_ is kth_smallest_fast's own ChunkPartition
# scratch prefix, from ExternalKthSmallest.h; kdl_next_ is
# kth_smallest_delayed's force() scratch prefix, from kth_smallest_delayed.h).
#
# Usage:
#   scripts/clean_kth_smallest_delayed.sh              # delete
#   scripts/clean_kth_smallest_delayed.sh --dry-run     # list what would be deleted
#   SSD_ROOT_DIR=/mnt/ssd scripts/clean_kth_smallest_delayed.sh   # override root

set -euo pipefail

PREFIXES=(kthd_fast_in kth_next_ kthd_delayed_in kdl_next_)

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

# configs.h's SSD_ROOT is a printf template like "/mnt/ssd%lu" or
# "/home/acmyr/ssd_dev/ssd%lu"; turn it into a glob prefix (strip the "%lu").
detect_root() {
    local configured
    configured=$(grep -oP 'SSD_ROOT\s*=\s*"\K[^"]+' "$(dirname "$0")/../configs.h" 2>/dev/null || true)
    configured="${configured%%%*}"
    if [[ -n "$configured" && -d "${configured}0" ]]; then
        echo "$configured"
        return
    fi
    # Fall back to the dev-box path this repo has used before, then /mnt/ssd.
    if [[ -d "$HOME/ssd_dev/ssd0" ]]; then
        echo "$HOME/ssd_dev/ssd"
        return
    fi
    echo "/mnt/ssd"
}

ROOT="${SSD_ROOT_DIR:-$(detect_root)}"
SSD_COUNT="${SSD_COUNT:-30}"

echo "Scanning ${ROOT}0..$((SSD_COUNT - 1)) for kth_smallest_delayed files..."

found=0
for ((i = 0; i < SSD_COUNT; i++)); do
    dir="${ROOT}${i}"
    [[ -d "$dir" ]] || continue
    for f in "$dir"/*; do
        [[ -e "$f" ]] || continue
        base="$(basename -- "$f")"
        for prefix in "${PREFIXES[@]}"; do
            if [[ "$base" == "${prefix}"* ]]; then
                found=1
                if [[ "$DRY_RUN" -eq 1 ]]; then
                    echo "would remove: $f"
                else
                    rm -f -- "$f"
                    echo "removed: $f"
                fi
                break
            fi
        done
    done
done

if [[ "$found" -eq 0 ]]; then
    echo "nothing to clean."
elif [[ "$DRY_RUN" -eq 0 ]]; then
    echo "done."
fi
