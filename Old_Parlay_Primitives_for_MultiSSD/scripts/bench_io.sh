#!/bin/bash

[ $# -ne 3 ] && echo "Usage ${0} start end data_size" && exit 1

bazel build -c opt "//:speed_test"

executable="./bazel-bin/speed_test"
file_prefix="files"
data_size="$3"
workload="read_only"

for ssd_count in $(seq "$1" "$2"); do
  echo "Testing with ${ssd_count} random SSDs"
  # Only need to reset SSDs and perform fstrim when we are writing
  if [[ $workload == *"write"* ]]; then
#    bash scripts/clear.sh
    sudo bash scripts/fstrim.sh
    sleep 30
  fi
#  num_threads=$(( ("$ssd_count" + 2) / 3 ))
  num_threads="$ssd_count"
  ssds=$(python scripts/ssd_assignment.py "$ssd_count")
  if [[ $workload == *"read"* ]]; then
    # No need to specify data size for reads
    LD_PRELOAD=/usr/lib64/libmimalloc.so.2 "${executable}" "--ssd=$ssds" "$workload" "$file_prefix" 4 "$num_threads" > "temp.txt" 2>&1
  else
    "${executable}" "--ssd=$ssds" "$workload" "$file_prefix" "$data_size" 4 "$num_threads" > "temp.txt" 2>&1
  fi
  printf "%s," "${ssd_count}" >> "result.txt"
  grep "Throughput" "temp.txt" | sed -e "s/^.*Throughput: \([0-9.]\+\)$/\1,/" >> "result.txt"
done
rm "temp.txt"
cat "result.txt"
