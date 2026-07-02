#!/bin/bash

bazel build -c opt "//:speed_test"

executable="./bazel-bin/speed_test"
for type in 0 2 3; do
    echo "RNG type: $type" >> "result.txt"
    data_size="34"

    if [ "$type" == 0 ]; then
        params=("10" "1000" "100000" "10000000" "1000000000")
    elif [ "$type" == 2 ]; then
        params=("0.6" "0.8" "1" "1.2" "1.5")
    elif [ "$type" == 3 ]; then
        params=("0.00001" "0.00002" "0.00005" "0.00007" "0.0001")
    fi
    for param in "${params[@]}"; do
        echo "Param $param" >> "result.txt"
        for _ in {1..5}; do
            "${executable}" "sorting_in_memory" "$data_size" "$type" "$param" > "temp.txt"
            grep "DONE" "temp.txt" | sed -e "s/^.*DONE: \([0-9.]\+\)$/\1,/" | tr -d '\n' >> "result.txt"
        done
        echo "" >> "result.txt"
    done
done
cat "result.txt"
