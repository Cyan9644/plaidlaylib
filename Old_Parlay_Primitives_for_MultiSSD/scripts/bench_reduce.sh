#!/usr/bin/bash

bazel build -c opt "//:all"
bin="./bazel-bin"

if [ "$1" == "i" ]; then
    echo "In-memory test"
    for data_size in $(seq 30 34); do
        printf "%s," "$data_size" >> "result.txt"
        "$bin/speed_test" "reduce_in_memory" "$data_size" > "temp.txt" 2>&1
        grep "DONE" "temp.txt" | sed -e "s/^.*DONE: \([0-9.]\+\)$/\1,/" | tr -d '\n' >> "result.txt"
        grep "Throughput" "temp.txt" | sed -e "s/^.*Throughput: \([0-9.GB]\+\)$/\1,/" | tr -d '\n' >> "result.txt"
        echo "" >> "result.txt"
    done
elif [ "$1" == "e" ]; then
    echo "External memory test"
    for data_size in $(seq 30 40); do
        printf "%s," "$data_size" >> "result.txt"
        ./scripts/clear.sh
        "$bin/sample_sort" "gen" "$data_size" "numbers" 0 0
        "$bin/sequence" "reduce" "numbers" > "temp.txt" 2>&1
        grep "Time" "temp.txt" | sed -e "s/^.*Time: \([0-9.GB]\+\)$/\1,/" | tr -d '\n' >> "result.txt"
        grep "Throughput" "temp.txt" | sed -e "s/^.*Throughput: \([0-9.GB]\+\)$/\1,/" | tr -d '\n' >> "result.txt"
        echo "" >> "result.txt"
    done
else
    echo "Usage: $0 [i|e]"
    exit 1
fi
rm "temp.txt"
cat "result.txt"
