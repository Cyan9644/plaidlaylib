#! /usr/bin/bash

bazel build -c opt "//:speed_test"

executable="./bazel-bin/speed_test"
workload="scatter_gather_nop"

for num_buckets in 4 8 16 32 64 128 256 512 1024 2048 4096; do
  ./scripts/reset.sh
  sudo ./scripts/fstrim.sh
  sleep 30
  "${executable}" "$workload" numbers "$num_buckets" > "temp.txt" 2>&1
  printf "%s," "$num_buckets" >> "result.txt"
  grep "Throughput1" "temp.txt" | sed -e "s/^.*Throughput1: \([0-9.GB]\+\)$/\1,/" >> "result.txt"
done
rm "temp.txt"
cat "result.txt"