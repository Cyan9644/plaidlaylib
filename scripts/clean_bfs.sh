#!/usr/bin/env bash
# Remove the files left behind by a cancelled/crashed bfsExample run.
#
# The benchmark (ChunkSequence/examples/external_TODO/bfs.cpp) cleans up after
# itself between cases via cleanup_prefix() (edge files) and
# bench_drives::clear_drives() (frontier files), but that only runs if
# run_case() returns normally -- Ctrl-C or a crash mid-run (e.g. the process
# getting OOM-killed while building a very large graph) leaves that case's
# edge_prefix files ("bfs_edges_" + sparse/balanced/dense, plus their
# external_rmat.h "_gen"/"_srt" intermediates, all caught by the same prefix)
# and/or frontier files ("bfs_frontier" + round number, no case suffix)
# sitting on every drive. This script does the same prefix sweep by hand.
#
# Prefixes must stay in sync with bfs.cpp's edge_prefix construction
# (edge_prefix = "bfs_edges_" + label, for label in sparse/balanced/dense) and
# external_bfs.h's BFS_simple frontier naming ("bfs_frontier" + round).
#
# Usage:
#   scripts/clean_bfs.sh              # delete
#   scripts/clean_bfs.sh --dry-run    # list what would be deleted
#   SSD_ROOT_DIR=/mnt/ssd scripts/clean_bfs.sh   # override root

set -euo pipefail

PREFIXES=(bfs_edges_sparse bfs_edges_balanced bfs_edges_dense bfs_frontier)

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

echo "Scanning ${ROOT}0..$((SSD_COUNT - 1)) for bfs files..."

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
