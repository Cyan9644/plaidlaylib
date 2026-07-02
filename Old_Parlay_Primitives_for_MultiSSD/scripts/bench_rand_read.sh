#!/bin/bash

bazel build -c opt "//:speed_test"

executable="./bazel-bin/speed_test"
file_prefix="files"
workload="rand_read"

for ssd_count in $(seq 1 30); do
  echo "Testing with ${ssd_count} random SSDs"
  num_samples=$(( "$ssd_count" * 30000000 ))
  ssds=$(python scripts/ssd_assignment.py "$ssd_count")
  # No need to specify data size for reads
  LD_PRELOAD=/usr/lib64/libmimalloc.so.2 "${executable}" "--ssd=$ssds" "$workload" "$file_prefix" $num_samples > "temp.txt" 2>&1
  printf "%s," "${ssd_count}" >> "result.txt"
  grep "Throughput" "temp.txt" | sed -e "s/^.*Throughput: \([0-9.]\+\)$/\1,/" >> "result.txt"
done
rm "temp.txt"
cat "result.txt"
