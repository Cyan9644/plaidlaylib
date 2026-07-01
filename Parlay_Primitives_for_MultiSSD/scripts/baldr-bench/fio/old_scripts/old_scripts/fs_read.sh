#! /usr/bin/bash
for i in {1..30}; do 
    python3 fs_read.py -j 3 -n $i -r
done
