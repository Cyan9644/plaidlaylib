#!/usr/bin/bash
source disks.sh
device_name=raid0
if [ "$1" = "create" ]; then
    sudo mdadm --create --verbose "$device_name" --level=0 --raid-devices=${#disks[@]} "${disks[@]}"
elif [ "$1" = "stop" ]; then
    sudo mdadm --stop "/dev/md/${device_name}"
    for disk in "${disks[@]}"; do
        sudo mdadm --zero-superblock "$disk"
    done
else
    echo "Invalid argument"
fi