#!/usr/bin/bash

bazel build -c opt "//:all"
bin="./bazel-bin"

if [ "$1" == "i" ]; then
  echo "In-memory test"
  for data_size in $(seq 29 34); do
    printf "%s," "$data_size" >> "result.txt"
    for _ in $(seq 1 5); do
      "$bin/speed_test" "permutation_in_memory" "$data_size" > "temp.txt" 2>&1
      grep "DONE" "temp.txt" | sed -e "s/^.*DONE: \([0-9.]\+\)$/\1,/" | tr -d '\n' >> "result.txt"
    done
    echo "" >> "result.txt"
  done
elif [ "$1" == "e" ]; then
  echo "External memory test"
  for data_size in $(seq 39 40); do
    printf "%s," "$data_size" >> "result.txt"
    ./scripts/clear.sh
    "$bin/sample_sort" "gen" "$data_size" "numbers" 0 0
    loop_end=5
    if [ "$data_size" -ge 34 ]; then
      loop_end=1
    fi
    for _ in $(seq 1 "$loop_end"); do
      ./scripts/reset.sh
      sudo ./scripts/fstrim.sh
      sleep 60
      "$bin/permutation" "run" "numbers" "result" > "temp.txt" 2>&1
      grep "DONE" "temp.txt" | sed -e "s/^.*DONE: \([0-9.]\+\)$/\1,/" | tr -d '\n' >> "result.txt"
    done
    echo "" >> "result.txt"
  done
else
  echo "Usage: $0 [i|e]"
  exit 1
fi

rm "temp.txt"
cat "result.txt"
