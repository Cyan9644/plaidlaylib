#!/bin/bash

[ $# -ne 3 ] && echo Usage $0 numjobs /dev/DEVICENAME1:/dev/DEVICENAME2:/dev/DEVICENAME3:/dev/DEVICENAME4 BLOCKSIZE && exit 1

fio --readonly --name=four_ssd_read \
    --filename=$2 \
    --filesize=100g --rw=randread --bs=$3 --direct=1 --overwrite=0 \
    --numjobs=$1 --iodepth=32 --time_based=1 --runtime=3600 \
    --ioengine=io_uring --registerfiles --fixedbufs \
    --gtod_reduce=1 --group_reporting
