#!/usr/bin/bash
# WARNING: the name of block devices in /dev is not persistent. See https://wiki.archlinux.org/title/Persistent_block_device_naming
# Verify that the SSD that is skipped corresponds to the OS install, otherwise the machine might be bricked.

[ $# -ne 1 ] && echo "Need one argument: part for partition and unpart for unpartition" && exit 0

source disks.sh

for disk in "${disks[@]}"; do 
    source="$disk"
    parted="$disk-part1"
    if [ "$1" = "part" ]; then
        echo "Partitioning $source into $parted"
        parted "$source" mklabel gpt mkpart P1 xfs 0% 100% -s
        sudo mkfs.xfs "$parted"
    elif [ "$1" = "unpart" ]; then
        echo "Destorying the partition at $source"
        parted "$source" rm 1
    else
        echo "$1 not recognized: need to be part or unpart"
        exit 1
    fi
done
