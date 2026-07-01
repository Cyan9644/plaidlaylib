#! /usr/bin/bash
for i in {1..30}; do
    python3 experiment.py -j 4 -n $i -r >> results/ssd_scaling.txt
done

for i in {1..30}; do
    python3 experiment.py -j 4 -n $i >> results/ssd_scaling_root_complex.txt
done
