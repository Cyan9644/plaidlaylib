#!/usr/bin/env bash
# Rerun the summary-figure sweep (`make bench-examples-full`'s command, capped at
# 64 GiB -- the summary bars only use rows <= 64 GiB) REPS more times, then
# aggregate mean/median against the existing summary_figure.csv (run1).
#
#   usage:  bash benchresults/Sep24-FinalFigures/run_summary_repeats.sh [REPS]
#   env:    SIZES="1GiB 4GiB 16GiB 64GiB"  TAG=<name>  DRY_RUN=1 (echo only)
#
# Output: results/summary-repeats-$TAG/
#   rep<i>/<ts>/            run_benches.py sweep (*_scale.csv, warnings.txt, ...)
#   rep<i>/summary/<ts>/    summary_figure.py output for that rep
#   summary_run<i+1>.csv    that rep's bars (run1 = the existing figure data)
#   summary_stats.csv, summary_figure_{mean,median}.csv
set -euo pipefail

REPS="${1:-2}"
SIZES="${SIZES:-1GiB 4GiB 16GiB 64GiB}"
TAG="${TAG:-$(date +%Y%m%d-%H%M%S)}"
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"

FIGDIR="benchresults/Sep24-FinalFigures"
BASE_CSV="$FIGDIR/summary_figure.csv"
OUT="results/summary-repeats-$TAG"

run() {
  echo "+ $*"
  [[ -n "${DRY_RUN:-}" ]] || "$@"
}

ENTRIES="$(python3 benchmarks/summary_figure.py --list)"
echo "entries: $ENTRIES"
echo "sizes:   $SIZES"
echo "output:  $OUT"
[[ -n "${DRY_RUN:-}" ]] || mkdir -p "$OUT"

csvs=("$BASE_CSV")
failed=()
for i in $(seq 1 "$REPS"); do
  echo
  echo "======== rep $i / $REPS  ($(date)) ========"
  rep="$OUT/rep$i"
  [[ -n "${DRY_RUN:-}" ]] || mkdir -p "$rep"
  set +e
  run python3 benchmarks/run_benches.py --inmem-uncapped --timeout-min 30 \
      --outdir "$rep" --example "$ENTRIES" --example-sizes "$SIZES"
  rc=$?
  set -e
  [[ $rc -eq 0 ]] || { echo "!!! rep $i: run_benches.py exited $rc"; failed+=("rep$i:rc=$rc"); }
  [[ -n "${DRY_RUN:-}" ]] && continue

  sweep="$(find "$rep" -mindepth 1 -maxdepth 1 -type d ! -name summary | sort | tail -n1)"
  [[ -n "$sweep" ]] || { echo "!!! rep $i: no sweep directory under $rep"; failed+=("rep$i:nodir"); continue; }
  [[ -s "$sweep/warnings.txt" ]] && { echo "--- rep $i warnings ($sweep/warnings.txt):"; cat "$sweep/warnings.txt"; }

  run python3 benchmarks/summary_figure.py --dir "$sweep" --at-size 64GiB \
      --outdir "$rep/summary"
  fig="$(ls -1d "$rep"/summary/*/ | sort | tail -n1)summary_figure.csv"
  dst="$OUT/summary_run$((i + 1)).csv"
  cp "$fig" "$dst"
  echo "rep $i bars -> $dst"
  csvs+=("$dst")
done

echo
echo "======== aggregate (${#csvs[@]} runs) ========"
if [[ -n "${DRY_RUN:-}" ]]; then
  echo "+ python3 $FIGDIR/aggregate_summary_runs.py ${csvs[*]} $OUT/summary_run{2..$((REPS + 1))}.csv --out $OUT"
else
  python3 "$FIGDIR/aggregate_summary_runs.py" "${csvs[@]}" --out "$OUT" \
    | tee "$OUT/summary_stats.md"
fi

if ((${#failed[@]})); then
  echo "!!! problems: ${failed[*]}  (check rep*/*/warnings.txt)"
  exit 1
fi
