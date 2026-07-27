#!/usr/bin/env bash
# Remove the files left behind by a cancelled/crashed bellman_fordExample run.
#
# The benchmark (ChunkSequence/examples/external/bellman_ford.cpp) cleans up
# after itself between cases via cleanup_prefix(), but that only runs if
# run_case() returns normally -- Ctrl-C or a crash mid-run (e.g. the process
# getting OOM-killed while building a very large graph) leaves that case's
# edge_prefix files ("bf_edges_" + sparse/balanced/dense) sitting on every
# drive. This script does the same prefix sweep by hand.
#
# Prefixes must stay in sync with bellman_ford.cpp's edge_prefix construction
# (edge_prefix = "bf_edges_" + label, for label in sparse/balanced/dense).
#
# Usage:
#   scripts/clean_bellman_ford.sh              # delete
#   scripts/clean_bellman_ford.sh --dry-run     # list what would be deleted
#   SSD_ROOT_DIR=/mnt/ssd scripts/clean_bellman_ford.sh   # override root

set -euo pipefail

PREFIXES=(bf_edges_sparse bf_edges_balanced bf_edges_dense)

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

echo "Scanning ${ROOT}0..$((SSD_COUNT - 1)) for bellman_ford files..."

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
