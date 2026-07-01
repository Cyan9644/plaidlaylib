#! /usr/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/disks.sh"
for disk in "${disks[@]}"; do
    blkdiscard "$disk" -v -f &
done
wait
