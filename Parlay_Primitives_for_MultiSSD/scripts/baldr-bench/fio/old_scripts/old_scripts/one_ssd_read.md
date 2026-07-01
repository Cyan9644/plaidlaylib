All testing done with `aiolib` instead of `io_uring`. Waiting for RHEL 9.3 to add `io_uring` support. 

# Larger reads

Command: `sudo ./one_ssd_read.sh 1 /dev/nvme0n1 4M`

Note that when we perform larger reads, the CPU is no longer the bottleneck. At about 15% utilization of a single CPU core, we can achieve 7GB/s for reading. 

Switching from `libaio` to `io_uring` does not improve performance for this benchmark. 

```
onessd: (groupid=0, jobs=1): err= 0: pid=52502: Fri Oct 20 18:06:01 2023
  read: IOPS=1733, BW=6935MiB/s (7272MB/s)(343GiB/50644msec)
   bw (  MiB/s): min= 6592, max= 6960, per=100.00%, avg=6938.30, stdev=35.75, samples=101
   iops        : min= 1648, max= 1740, avg=1734.57, stdev= 8.94, samples=101
  cpu          : usr=0.07%, sys=14.80%, ctx=75954, majf=0, minf=581
  IO depths    : 1=0.1%, 2=0.1%, 4=0.1%, 8=0.1%, 16=0.1%, 32=100.0%, >=64=0.0%
     submit    : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%
     complete  : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.1%, 64=0.0%, >=64=0.0%
     issued rwts: total=87806,0,0,0 short=0,0,0,0 dropped=0,0,0,0
     latency   : target=0, window=0, percentile=100.00%, depth=32

Run status group 0 (all jobs):
   READ: bw=6935MiB/s (7272MB/s), 6935MiB/s-6935MiB/s (7272MB/s-7272MB/s), io=343GiB (368GB), run=50644-50644msec

Disk stats (read/write):
  nvme0n1: ios=1400797/0, merge=0/0, ticks=25022961/0, in_queue=25022961, util=99.85%
```

# Smaller reads

Command: `sudo ./one_ssd_read.sh 4 /dev/nvme0n1 4K`

For smaller reads, we get much better IOPS, but issuing this too many reads require 4 processes to overcome the CPU bottleneck. We are also getting smaller read speeds. 

```
onessd: (groupid=0, jobs=4): err= 0: pid=54154: Fri Oct 20 18:13:19 2023
  read: IOPS=1548k, BW=6046MiB/s (6339MB/s)(590GiB/99921msec)
   bw (  MiB/s): min= 4240, max= 6447, per=100.00%, avg=6048.07, stdev=50.17, samples=796
   iops        : min=1085510, max=1650616, avg=1548306.54, stdev=12844.27, samples=796
  cpu          : usr=23.63%, sys=63.27%, ctx=22068173, majf=0, minf=162
  IO depths    : 1=0.1%, 2=0.1%, 4=0.1%, 8=0.1%, 16=0.1%, 32=100.0%, >=64=0.0%
     submit    : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%
     complete  : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.1%, 64=0.0%, >=64=0.0%
     issued rwts: total=154648533,0,0,0 short=0,0,0,0 dropped=0,0,0,0
     latency   : target=0, window=0, percentile=100.00%, depth=32

Run status group 0 (all jobs):
   READ: bw=6046MiB/s (6339MB/s), 6046MiB/s-6046MiB/s (6339MB/s-6339MB/s), io=590GiB (633GB), run=99921-99921msec

Disk stats (read/write):
  nvme0n1: ios=154608504/0, merge=0/0, ticks=11817333/0, in_queue=11817333, util=99.94%
```