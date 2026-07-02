#! /usr/bin/bash
for i in {1..30}; do
    python3 experiment.py -j 4 -n $i -r --fs >> results/fs_scaling.txt
done

