#! /usr/bin/bash
for i in {0..29}; do
    target="/mnt/ssd${i}"
    fstrim "$target" -v &
done
wait

