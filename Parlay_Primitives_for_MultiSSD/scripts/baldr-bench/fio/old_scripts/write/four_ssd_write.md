Command: `./four_ssd_write.sh 4 /mnt/ssd2/fio:/mnt/ssd3/fio:/mnt/ssd4/fio:/mnt/ssd5/fio 4M`

Result:
```
four_ssd_write: (groupid=0, jobs=4): err= 0: pid=1652958: Thu Apr 11 16:41:49 2024
  write: IOPS=6120, BW=23.9GiB/s (25.7GB/s)(288GiB/12063msec); 0 zone resets
   bw (  MiB/s): min=16639, max=29120, per=99.84%, avg=24442.37, stdev=830.92, samples=96
   iops        : min= 4159, max= 7280, avg=6110.54, stdev=207.76, samples=96
  cpu          : usr=8.18%, sys=14.35%, ctx=58646, majf=0, minf=26
  IO depths    : 1=0.1%, 2=0.1%, 4=0.1%, 8=0.1%, 16=0.2%, 32=0.4%, >=64=99.2%
     submit    : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%
     complete  : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.1%
     issued rwts: total=0,73833,0,0 short=0,0,0,0 dropped=0,0,0,0
     latency   : target=0, window=0, percentile=100.00%, depth=128

Run status group 0 (all jobs):
  WRITE: bw=23.9GiB/s (25.7GB/s), 23.9GiB/s-23.9GiB/s (25.7GB/s-25.7GB/s), io=288GiB (310GB), run=12063-12063msec

Disk stats (read/write):
  nvme2n1: ios=0/584542, merge=0/168, ticks=0/48605421, in_queue=48605422, util=97.54%
  nvme3n1: ios=0/585877, merge=0/167, ticks=0/34335902, in_queue=34335904, util=97.48%
  nvme4n1: ios=0/587557, merge=0/169, ticks=0/9047608, in_queue=9047609, util=97.29%
  nvme5n1: ios=0/586800, merge=0/165, ticks=0/11003005, in_queue=11003006, util=97.63%
```
