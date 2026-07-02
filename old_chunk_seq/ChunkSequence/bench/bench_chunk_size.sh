#!/usr/bin/env bash
#
# Chunk-size sensitivity benchmark: eager vs chunk-delayed across varying CHUNK_SIZE.
#
# For a fixed problem size n, compiles and runs one binary per chunk size and
# records the operation-only times (data generation is excluded) into a CSV that
# plot_chunk_size.py turns into a two-panel figure.
#
# Usage:
#   bench_chunk_size.sh [n] [reps]
#     n     problem size in elements  (default 2^24 = 16,777,216 ≈ 128 MB)
#     reps  runs per chunk size; the per-column MIN time is kept (default 1)
#
# Chunk sizes swept (bytes): 262144 524288 1048576 2097152 4194304 8388608 16777216
#   (256 KB → 16 MB, all powers of two, all multiples of 4096)
#
# Disk: each run needs roughly 16 * n * sizeof(uint64_t) bytes on the SSD mounts
# (input perm + eager intermediates + force output).  On the dev box (5 GiB tmpfs)
# the default n=2^24 requires ~2 GB — safely within budget.
set -euo pipefail

N="${1:-$((1 << 24))}"
REPS="${2:-1}"

CHUNK_SIZES="262144 524288 1048576 2097152 4194304 8388608 16777216"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
CSV="$RESULTS_DIR/chunk_size.csv"

mkdir -p "$RESULTS_DIR"

echo "chunk_size_bytes,n,raw_s,eager_mr_s,delayed_mr_s,eager_mmr_s,delayed_mmr_s,eager_fmm_s,delayed_fmm_s,agree" > "$CSV"

for cs in $CHUNK_SIZES; do
    BIN="$REPO_ROOT/bin/bwChunkSize_$cs"
    echo "=== CHUNK_SIZE=$cs bytes ($(( cs / 1024 )) KB) ==="

    echo "  Building bwChunkSize_$cs..."
    # Force rebuild: the Makefile tracks no header deps, so a stale binary would
    # silently benchmark old code.
    rm -f "$BIN"
    make -C "$REPO_ROOT" "bin/bwChunkSize_$cs" >/dev/null

    best=""
    for (( r = 0; r < REPS; r++ )); do
        # Clear SSD mounts so the previous run's files don't count toward disk
        # pressure or confuse file-discovery logic.
        find /mnt -path "/mnt/ssd*/*" -delete 2>/dev/null || true

        line="$("$BIN" "$N" | grep '^CSV,' | head -1)"
        if [ -z "$line" ]; then
            echo "  WARNING: no CSV line for chunk_size=$cs (run $r); skipping" >&2
            continue
        fi
        row="${line#CSV,}"   # strip "CSV," prefix

        if [ -z "$best" ]; then
            best="$row"
        else
            # Keep per-column minimum across reps (best-case, least noise).
            # 'agree' takes the min so any rep's mismatch (0) is preserved.
            best="$(awk -F, -v a="$best" -v b="$row" 'BEGIN {
                na = split(a, x, ","); split(b, y, ",");
                # columns 1-2 are chunk_size_bytes,n — same for both rows
                printf "%s,%s", x[1], x[2];
                for (i = 3; i <= na; i++)
                    printf ",%s", (y[i] < x[i] ? y[i] : x[i]);
                printf "\n";
            }')"
        fi
    done

    if [ -n "$best" ]; then
        echo "$best" >> "$CSV"
        echo "  -> $best"
        case "$best" in
            *,0) echo "  *** WARNING: agree=0 for chunk_size=$cs (substrate mismatch) ***" >&2 ;;
        esac
    fi
done

# Leave the SSD mounts clean.
find /mnt -path "/mnt/ssd*/*" -delete 2>/dev/null || true

echo
echo "Wrote $CSV"
echo "Plot with: python3 $SCRIPT_DIR/plot_chunk_size.py"
