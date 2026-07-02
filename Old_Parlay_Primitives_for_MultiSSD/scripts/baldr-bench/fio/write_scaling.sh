#! /usr/bin/bash
for i in {1..30}; do
    sleep 5
    bash ../utils/blkdiscard.sh
    sleep 45
    python3 experiment.py --rw=randwrite -j 4 -n "$i" -r --runtime=10 >> results/write_scaling.txt
    echo "Experiment $i done"
done