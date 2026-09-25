#!/usr/bin/env bash
# Rerun the paper's 17 core summary entries (everything but fft / bellman_ford /
# convex_hull) at 64 GiB only, REPS times, then aggregate mean/median against
# the existing summary_figure.csv (run1).  Same conditions as
# `make bench-examples-full`: uncapped in-mem baselines, 30-min timeout.
#
#   usage:  bash benchresults/Sep24-FinalFigures/run_core_64g_repeats.sh [REPS]
#   env:    TAG=<name>  DRY_RUN=1 (echo only)  ENTRIES=a,b (subset rerun)
#
# Output: results/core64-repeats-$TAG/
#   rep<i>/<ts>/            run_benches.py sweep (*_scale.csv, warnings.txt, ...)
#   rep<i>/summary/<ts>/    summary_figure.py output for that rep
#   summary_run<i+1>.csv    that rep's bars (run1 = the existing figure data)
#   summary_stats.{md,csv}, summary_figure_{mean,median}.csv
set -euo pipefail

REPS="${1:-2}"
SIZE="64GiB"
TAG="${TAG:-$(date +%Y%m%d-%H%M%S)}"
ENTRIES="${ENTRIES:-map,reduce,filter,scan,tabulate,group_by_index,histogram_by_index,pack,random_shuffle,reverse,bigint_add,linefit,kmp,rabin_karp,primes,samplesort,kth_smallest}"
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"

FIGDIR="benchresults/Sep24-FinalFigures"
BASE_CSV="$FIGDIR/summary_figure.csv"
OUT="results/core64-repeats-$TAG"

run() {
  echo "+ $*"
  [[ -n "${DRY_RUN:-}" ]] || "$@"
}

echo "entries: $ENTRIES"
echo "size:    $SIZE"
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
      --outdir "$rep" --example "$ENTRIES" --example-sizes "$SIZE"
  rc=$?
  set -e
  [[ $rc -eq 0 ]] || { echo "!!! rep $i: run_benches.py exited $rc"; failed+=("rep$i:rc=$rc"); }
  [[ -n "${DRY_RUN:-}" ]] && continue

  sweep="$(find "$rep" -mindepth 1 -maxdepth 1 -type d ! -name summary | sort | tail -n1)"
  [[ -n "$sweep" ]] || { echo "!!! rep $i: no sweep directory under $rep"; failed+=("rep$i:nodir"); continue; }
  [[ -s "$sweep/warnings.txt" ]] && { echo "--- rep $i warnings ($sweep/warnings.txt):"; cat "$sweep/warnings.txt"; }

  # No --only: "sort / samplesort" has a space and --only splits on whitespace.
  # The three entries not swept (fft, bellman_ford, convex_hull) are skipped
  # with a "no ..._scale.csv" warning; hide that expected noise.
  python3 benchmarks/summary_figure.py --dir "$sweep" --at-size "$SIZE" \
      --outdir "$rep/summary" \
    | grep -v -E '(FFT|Bellman|Convex|fft|bellman_ford|convex_hull).*no .*_scale\.csv found' || true
  fig="$(ls -1d "$rep"/summary/*/ | sort | tail -n1)summary_figure.csv"
  dst="$OUT/summary_run$((i + 1)).csv"
  cp "$fig" "$dst"
  echo "rep $i bars -> $dst"
  csvs+=("$dst")
done

echo
echo "======== aggregate (${#csvs[@]} runs) ========"
if [[ -n "${DRY_RUN:-}" ]]; then
  echo "+ python3 $FIGDIR/aggregate_summary_runs.py $BASE_CSV $OUT/summary_run{2..$((REPS + 1))}.csv --entries $ENTRIES --out $OUT"
else
  python3 "$FIGDIR/aggregate_summary_runs.py" "${csvs[@]}" \
      --entries "$ENTRIES" --out "$OUT" | tee "$OUT/summary_stats.md"
fi

if ((${#failed[@]})); then
  echo "!!! problems: ${failed[*]}  (check rep*/*/warnings.txt)"
  exit 1
fi
