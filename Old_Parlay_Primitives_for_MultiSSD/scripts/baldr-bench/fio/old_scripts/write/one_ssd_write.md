Command: `./one_ssd_write.sh 2 /mnt/ssd6/fio 4M`

```
one_ssd_write: (groupid=0, jobs=2): err= 0: pid=230609: Tue Apr 16 15:26:55 2024
  write: IOPS=1547, BW=6191MiB/s (6492MB/s)(74.2GiB/12269msec); 0 zone resets
   bw (  MiB/s): min= 3680, max= 6336, per=99.04%, avg=6131.33, stdev=261.42, samples=48
   iops        : min=  920, max= 1584, avg=1532.83, stdev=65.36, samples=48
  cpu          : usr=3.75%, sys=6.12%, ctx=75391, majf=0, minf=9
  IO depths    : 1=0.1%, 2=0.1%, 4=0.1%, 8=0.1%, 16=0.2%, 32=0.3%, >=64=99.3%
     submit    : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%
     complete  : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.1%
     issued rwts: total=0,18989,0,0 short=0,0,0,0 dropped=0,0,0,0
     latency   : target=0, window=0, percentile=100.00%, depth=128

Run status group 0 (all jobs):
  WRITE: bw=6191MiB/s (6492MB/s), 6191MiB/s-6191MiB/s (6492MB/s-6492MB/s), io=74.2GiB (79.6GB), run=12269-12269msec
```