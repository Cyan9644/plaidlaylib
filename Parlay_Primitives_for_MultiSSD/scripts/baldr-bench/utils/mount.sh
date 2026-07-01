#! /usr/bin/bash

[ $# -ne 1 ] && echo "Need one argument: mount for mount and umount for unmount" && exit 0

source disks.sh
i=0
for disk in "${disks[@]}"; do
    ORIGINAL="$disk-part1"
    TARGET="/mnt/ssd${i}"
    if [ "$1" = "mount" ]; then
        mount -t xfs "$ORIGINAL" "$TARGET"
        chmod 777 "$TARGET"
    elif [ "$1" = "umount" ]; then
        umount "$TARGET"
    else
        echo "Invalid argument"
    fi
    ((i++))
done
