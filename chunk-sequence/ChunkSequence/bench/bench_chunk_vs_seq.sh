#!/usr/bin/env bash
#
# Scaling benchmark: ChunkSequence vs file-based sequence_algorithms.
#
# Runs the bwCompare workload for n = 2^e over a range of exponents and records
# the *operation-only* times (data generation is excluded — see bw_compare.cpp)
# into a CSV that plot_chunk_bw.py turns into a log-log graph.
#
# Usage:
#   bench_chunk_vs_seq.sh [min_exp] [max_exp] [reps]
#     min_exp, max_exp  inclusive range of exponents; n = 2^exp  (default 20 26)
#     reps              runs per size; the MIN time per column is kept (default 1)
#
# Disk: each run needs ~32 * n bytes on the SSD mounts (chunk input + seq input +
# both map outputs).  On this dev box the 30 "SSDs" share a single 5 GiB tmpfs,
# so max_exp ~26 (~2 GiB) is the practical ceiling; real 30-SSD hardware goes
# much higher.
set -euo pipefail

MIN_EXP="${1:-20}"
MAX_EXP="${2:-26}"
REPS="${3:-1}"

# Resolve repo root from this script's location so it runs from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
CSV="$RESULTS_DIR/chunk_bw.csv"
BIN="$REPO_ROOT/bin/bwCompare"

mkdir -p "$RESULTS_DIR"

echo "Building bwCompare..."
# Force a fresh build: the Makefile tracks no header dependencies, so an edit to
# a header (e.g. chunk_map.h, map.h) would otherwise leave a stale binary and
# silently benchmark old code.
rm -f "$BIN"
make -C "$REPO_ROOT" bin/bwCompare >/dev/null

echo "n,map_s,chunkmap_s,reduce_s,chunkreduce_s" > "$CSV"

for (( e = MIN_EXP; e <= MAX_EXP; e++ )); do
    n=$(( 1 << e ))
    echo "=== exponent $e  (n = $n) ==="

    best=""
    for (( r = 0; r < REPS; r++ )); do
        # Free prior run's files so the shared tmpfs doesn't fill up.
        find /mnt -path "/mnt/ssd*/*" -delete 2>/dev/null || true

        line="$("$BIN" "$n" | grep '^CSV,' | head -1)"
        if [ -z "$line" ]; then
            echo "  WARNING: no CSV line for n=$n (run $r); skipping" >&2
            continue
        fi
        row="${line#CSV,}"   # strip the "CSV," prefix -> n,map,chunkmap,reduce,chunkreduce

        if [ -z "$best" ]; then
            best="$row"
        else
            # Keep the per-column minimum across reps (best-case, least noise).
            best="$(awk -F, -v a="$best" -v b="$row" 'BEGIN {
                split(a, x, ","); split(b, y, ",");
                printf "%s", x[1];                       # n (same for both)
                for (i = 2; i <= 5; i++)
                    printf ",%s", (y[i] < x[i] ? y[i] : x[i]);
                printf "\n";
            }')"
        fi
    done

    if [ -n "$best" ]; then
        echo "$best" >> "$CSV"
        echo "  -> $best"
    fi
done

# Leave the tmpfs clean.
find /mnt -path "/mnt/ssd*/*" -delete 2>/dev/null || true

echo
echo "Wrote $CSV"
echo "Plot with: python3 $SCRIPT_DIR/plot_chunk_bw.py"
