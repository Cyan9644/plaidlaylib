#!/usr/bin/env bash
# Remove the files left behind by a cancelled/crashed convexHullTest or
# convex_hullExample/convex_hull_lazy_filterExample run.
#
# convexHullTest exercises both UpperHull (ChunkPartition-based,
# ChunkSequence/examples/chunk_convex_hull.h) and UpperHullLazyFilter
# (delayed::lazy_filter-based, ChunkSequence/examples/chunk_convex_hull_lazy_filter.h)
# per case, under prefixes "cht_in" (point cloud), "cht_s" (UpperHull's
# scratch), and "cht_lazy" (UpperHullLazyFilter's scratch).
# convex_hull_lazy_filterExample uses the same two algorithms under "ch_in",
# "ch_scratch", and "ch_lazy_scratch" (plain convex_hullExample only ever
# writes "ch_in"/"ch_scratch", a subset of this). Both UpperHull and
# UpperHullLazyFilter clean up their own recursion scratch (the "p"+counter /
# "a"+counter / "L"/"R" suffixes each level appends to its base prefix) before
# returning normally, but a crash or Ctrl-C mid-recursion leaves those
# suffixed files sitting on every drive. This script does the same prefix
# sweep by hand; a plain prefix match catches every such suffix since they all
# extend the base string.
#
# Prefixes must stay in sync with convex_hull.cpp/convex_hull_lazy_filter.cpp's
# in_prefix and chunk_convex_hull(_lazy_filter).h's scratch_prefix defaults,
# and convex_hull_test.cpp's in_prefix/scratch-prefix arguments.
#
# Usage:
#   scripts/clean_convex_hull.sh              # delete
#   scripts/clean_convex_hull.sh --dry-run    # list what would be deleted
#   SSD_ROOT_DIR=/mnt/ssd scripts/clean_convex_hull.sh   # override root

set -euo pipefail

PREFIXES=(cht_in cht_s cht_lazy ch_in ch_scratch ch_lazy_scratch)

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

echo "Scanning ${ROOT}0..$((SSD_COUNT - 1)) for convex hull files..."

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
