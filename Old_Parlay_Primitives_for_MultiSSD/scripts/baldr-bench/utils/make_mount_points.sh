#! /usr/bin/bash
for i in {0..29}; do
    target="/mnt/ssd${i}"
    mkdir -p $target
    chmod -R 777 $target
done
