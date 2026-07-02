#!/usr/bin/bash

source disks.sh

for disk in "${disks[@]}"; do 
    parted="$disk-part1"
    xfs_repair "$parted"
done