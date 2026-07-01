#!/usr/bin/env bash
#
# Scaling harness: in-memory delayed vs chunk-eager vs chunk-delayed.
#
# Runs the bwDelayed program for n = 2^e over a range of exponents and records
# the operation-only times (data generation is excluded — see delayed_compare.cpp)
# into a CSV that plot_delayed_scale.py turns into a two-panel throughput figure.
#
# Usage:
#   bench_delayed_scale.sh [min_exp] [max_exp] [reps]
#     min_exp, max_exp  inclusive range of exponents; n = 2^exp  (default 24 32)
#     reps              runs per size; the per-column MIN time is kept (default 1)
#
# The in-memory baseline (parlay::delayed) is skipped automatically once n exceeds
# a RAM budget (default half of physical RAM; override with the env var
# DELAYED_INMEM_BUDGET_BYTES, which is inherited by the binary).  Past that cliff
# the inmem_* CSV columns are blank and the plotter stops drawing those lines.
#
# Disk: each run needs ~16 * n bytes on the SSD mounts (chunk input + eager
# intermediates + force output, freed between sizes).  A storage-rich multi-SSD
# box is assumed; the point of the sweep is to cross the DRAM boundary, where the
# in-memory baseline can no longer run.  On a tmpfs dev box (SSDs = RAM) you only
# get the in-DRAM-overhead regime and never see the cliff.
set -euo pipefail

MIN_EXP="${1:-24}"
MAX_EXP="${2:-32}"
REPS="${3:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
CSV="$RESULTS_DIR/delayed_scale.csv"
BIN="$REPO_ROOT/bin/bwDelayed"

mkdir -p "$RESULTS_DIR"

echo "Building bwDelayed..."
# Force a fresh build: the Makefile tracks no header dependencies, so an edit to a
# header (e.g. chunk_delayed.h) would otherwise leave a stale binary.
rm -f "$BIN"
make -C "$REPO_ROOT" bin/bwDelayed >/dev/null

echo "n,raw_read_s,eager_mr_s,delayed_mr_s,inmem_mr_s,eager_mmr_s,delayed_mmr_s,inmem_mmr_s,eager_fmm_s,delayed_fmm_s,inmem_fmm_s,agree" > "$CSV"

for (( e = MIN_EXP; e <= MAX_EXP; e++ )); do
    n=$(( 1 << e ))
    echo "=== exponent $e  (n = $n) ==="

    best=""
    for (( r = 0; r < REPS; r++ )); do
        # Free prior run's files so the SSD mounts don't fill up.
        find /mnt -path "/mnt/ssd*/*" -delete 2>/dev/null || true

        line="$("$BIN" "$n" | grep '^CSV,' | head -1)"
        if [ -z "$line" ]; then
            echo "  WARNING: no CSV line for n=$n (run $r); skipping" >&2
            continue
        fi
        row="${line#CSV,}"   # strip the "CSV," prefix

        if [ -z "$best" ]; then
            best="$row"
        else
            # Per-column minimum across reps (best-case, least noise).  Blank
            # in-memory fields stay blank; 'agree' takes the min so any rep's
            # mismatch (0) is preserved.
            best="$(awk -F, -v a="$best" -v b="$row" 'BEGIN {
                na = split(a, x, ","); nb = split(b, y, ",");
                printf "%s", x[1];                       # n (same for both)
                for (i = 2; i <= na; i++) {
                    if (x[i] == "" && y[i] == "")      printf ",";
                    else if (x[i] == "")               printf ",%s", y[i];
                    else if (y[i] == "")               printf ",%s", x[i];
                    else                               printf ",%s", (y[i] < x[i] ? y[i] : x[i]);
                }
                printf "\n";
            }')"
        fi
    done

    if [ -n "$best" ]; then
        echo "$best" >> "$CSV"
        echo "  -> $best"
        # Flag any correctness mismatch loudly (last column is 'agree').
        case "$best" in
            *,0) echo "  *** WARNING: agree=0 for n=$n (substrate mismatch) ***" >&2 ;;
        esac
    fi
done

# Leave the SSD mounts clean.
find /mnt -path "/mnt/ssd*/*" -delete 2>/dev/null || true

echo
echo "Wrote $CSV"
echo "Plot with: python3 $SCRIPT_DIR/plot_delayed_scale.py   (needs matplotlib)"
