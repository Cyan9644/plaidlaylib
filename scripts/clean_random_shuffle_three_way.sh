#!/usr/bin/env bash
# Remove the files left behind by a cancelled random_shuffle_three_wayExample
# run.
#
# The benchmark (benchmarks/benchmark_files/random_shuffle_three_way.cpp)
# cleans up after itself between contestants via bench_drives::clear_drives(),
# but that only runs if the process exits normally — Ctrl-C mid-shuffle leaves
# that contestant's input, intermediates, and output sitting on every drive.
# This script does the same prefix sweep by hand.
#
# Prefixes must stay in sync with random_shuffle_three_way.cpp's
# kPrimPrefixes / kDirectPrefixes.
#
# Usage:
#   scripts/clean_random_shuffle_three_way.sh              # delete
#   scripts/clean_random_shuffle_three_way.sh --dry-run     # list what would be deleted
#   SSD_ROOT_DIR=/mnt/ssd scripts/clean_random_shuffle_three_way.sh   # override root

set -euo pipefail

PREFIXES=(rs3_in ss_bucket_ drs3_in drs3)

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

echo "Scanning ${ROOT}0..$((SSD_COUNT - 1)) for random_shuffle_three_way files..."

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
