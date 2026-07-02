
Command: `sudo ./four_ssd_read.sh 4 "/dev/nvme0n1:/dev/nvme1n1:/dev/nvme2n1:/dev/nvme3n1" 4M`

Some performance degradation observed for this test case. 26.9GiB/s = 6.725GiB/s. A bit lower than a single SSD.

Changing jobs from 4 to 16 does not change performance. 

Changing block size from 4M to 64M does not change performance.

Switching from `libaio` to `io_uring` does not affect performance. 

```
four_ssd: (groupid=0, jobs=4): err= 0: pid=58039: Fri Oct 20 18:32:13 2023
  read: IOPS=6877, BW=26.9GiB/s (28.8GB/s)(1735GiB/64568msec)
   bw (  MiB/s): min=26176, max=29864, per=100.00%, avg=27561.81, stdev=140.68, samples=512
   iops        : min= 6544, max= 7466, avg=6890.45, stdev=35.17, samples=512
  cpu          : usr=0.13%, sys=18.04%, ctx=393092, majf=0, minf=2346
  IO depths    : 1=0.1%, 2=0.1%, 4=0.1%, 8=0.1%, 16=0.1%, 32=100.0%, >=64=0.0%
     submit    : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%
     complete  : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.1%, 64=0.0%, >=64=0.0%
     issued rwts: total=444066,0,0,0 short=0,0,0,0 dropped=0,0,0,0
     latency   : target=0, window=0, percentile=100.00%, depth=32

Run status group 0 (all jobs):
   READ: bw=26.9GiB/s (28.8GB/s), 26.9GiB/s-26.9GiB/s (28.8GB/s-28.8GB/s), io=1735GiB (1863GB), run=64568-64568msec

Disk stats (read/write):
  nvme0n1: ios=1773548/0, merge=755/0, ticks=2384148/0, in_queue=2384148, util=99.85%
  nvme1n1: ios=1773620/0, merge=816/0, ticks=3808080/0, in_queue=3808080, util=99.91%
  nvme2n1: ios=1771651/0, merge=835/0, ticks=90764930/0, in_queue=90764930, util=99.93%
  nvme3n1: ios=1773630/0, merge=880/0, ticks=30572111/0, in_queue=30572111, util=99.94%
```