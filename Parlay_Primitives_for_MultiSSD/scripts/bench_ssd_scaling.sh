#!/bin/bash

bazel build -c opt "//:sample_sort"

executable="./bazel-bin/sample_sort"
input_prefix="numbers"
output_prefix="result"
data_size="34"

for type in 0 2 3; do
    echo "Type: $type" >> "result.txt"

    if [ "$type" == 0 ]; then
      params=("10" "1000" "100000" "10000000" "1000000000")
    elif [ "$type" == 2 ]; then
      params=("0.6" "0.8" "1" "1.2" "1.5")
    elif [ "$type" == 3 ]; then
      params=("0.00001" "0.00002" "0.00005" "0.00007" "0.0001")
    fi

    for param in "${params[@]}"; do
        echo "Param: $param" >> "result.txt"
        for ssd_count in 1 2 4 8 16 30; do
            echo "Testing with ${ssd_count} random SSDs"
            bash scripts/clear.sh
            sudo bash scripts/fstrim.sh
            ssd_numbers=$(python scripts/ssd_assignment.py "$ssd_count")
            "${executable}" "--ssd=${ssd_numbers}" "gen" "${data_size}" "${input_prefix}" "$type" "$param" > "temp.txt" 2>&1
            echo "$ssd_numbers"
            printf "%s," "${ssd_count}" >> "result.txt"
            for _ in {1..3}; do
                bash scripts/reset.sh
                sudo bash scripts/fstrim.sh
                if [ "$ssd_count" -le 4 ]; then
                    # Wait for SLC cache flush
                    sleep 60
                fi
                "${executable}" "--ssd=${ssd_numbers}" "run" "${input_prefix}" "${output_prefix}" > "temp.txt"
                grep "DONE" "temp.txt" | sed -e "s/^.*DONE: \([0-9.]\+\)$/\1,/" | tr -d '\n' >> "result.txt"
            done
            echo "" >> "result.txt"
        done
    done
done
rm "temp.txt"
cat "result.txt"
