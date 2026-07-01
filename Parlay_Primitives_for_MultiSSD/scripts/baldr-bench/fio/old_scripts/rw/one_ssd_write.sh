#!/bin/bash

[ $# -ne 3 ] && echo Usage $0 numjobs FILENAME BLOCKSIZE && exit 1
#--sqthread_poll
fio --name=one_ssd_rw \
    --filename=$2 \
    --filesize=10g --rw=rw --bs=$3 --direct=1 --overwrite=0 \
    --numjobs=$1 --iodepth=128 --time_based=1 --runtime=3600 \
    --ioengine=io_uring --registerfiles \
    --gtod_reduce=1 --group_reporting
