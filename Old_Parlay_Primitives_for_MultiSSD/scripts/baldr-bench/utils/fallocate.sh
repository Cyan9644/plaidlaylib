#! /usr/bin/bash

[ $# -ne 2 ] && echo "Need 2 arguments: name of file to allocate and allocation size" && exit 0

for i in {0..29}; do
    target="/mnt/ssd${i}/$1"
    fallocate -l "$2" "$target"
done