#!/bin/bash

# Sorting with input data size scaling

bazel build -c opt "//:all"

for type in 0 2 3; do
    echo "RNG type: $type" >> "result.txt"

    if [ "$type" == 0 ]; then
        param=0 # unbounded RNG
    elif [ "$type" == 2 ]; then
        param=1
    elif [ "$type" == 3 ]; then
        param=0.00005
    fi
    if [ "$1" == "i" ]; then
        executable="./bazel-bin/speed_test"
        for data_size in $(seq 30 34); do
            printf "%s," "$data_size" >> "result.txt"
            "${executable}" "sorting_in_memory" "$data_size" "$type" "$param" > "temp.txt"
            grep "DONE" "temp.txt" | sed -e "s/^.*DONE: \([0-9.]\+\)$/\1,/" | tr -d '\n' >> "result.txt"
            echo "" >> "result.txt"
        done
    elif [ "$1" == "e" ]; then
        executable="./bazel-bin/sample_sort"
        input_prefix="numbers"
        output_prefix="result"
        for data_size in {30..40}; do
            bash scripts/clear.sh
            sudo ./scripts/fstrim.sh
            sleep 10
            "${executable}" gen "$data_size" "${input_prefix}" "$type" "$param"
            printf "%s," "$data_size" >> "result.txt"
            loop_end=3
            sleep_time=5
            if [ "$data_size" -ge 34 ]; then
                loop_end=1
                sleep_time=60
            fi
            for _ in $(seq 1 "$loop_end"); do
                bash scripts/reset.sh
                sudo ./scripts/fstrim.sh
                sleep "$sleep_time"
                "${executable}" "run" "${input_prefix}" "${output_prefix}" > "temp.txt"
                grep "DONE" "temp.txt" | sed -e "s/^.*DONE: \([0-9.]\+\)$/\1,/" | tr -d '\n' >> "result.txt"
            done
            echo "" >> "result.txt"
        done
    else
        echo "Usage: $0 [i|e]"
        exit 1
    fi
done
rm "temp.txt"
cat "result.txt"
