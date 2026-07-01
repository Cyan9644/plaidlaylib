# New script

Usage: `sudo python3 all_ssd_read.py <options>`

Use `-h` to see help message and read below for usage examples. 

Note that the number of threads and block size should not affect the result, as long as there are large enough blocks and large enough threads so that SSD read speeds are the bottleneck. 

## Performance scaling

6804MiB/s (7134MB/s)
13.3GiB/s (14.3GB/s)
20.0GiB/s (21.5GB/s)
26.6GiB/s (28.6GB/s)
33.4GiB/s (35.9GB/s)
40.1GiB/s (43.1GB/s)
46.6GiB/s (50.0GB/s)
53.6GiB/s (57.6GB/s)
60.1GiB/s (64.5GB/s)
66.8GiB/s (71.7GB/s)
69.2GiB/s (74.3GB/s)
75.6GiB/s (81.1GB/s)
81.9GiB/s (87.9GB/s)
87.9GiB/s (94.4GB/s)
94.2GiB/s (101GB/s)
101GiB/s (108GB/s)
107GiB/s (115GB/s)
112GiB/s (120GB/s)
117GiB/s (126GB/s)
111GiB/s (120GB/s)
114GiB/s (122GB/s)
119GiB/s (127GB/s)
109GiB/s (117GB/s)
102GiB/s (110GB/s)
106GiB/s (114GB/s)
110GiB/s (118GB/s)
113GiB/s (122GB/s)
117GiB/s (125GB/s)
120GiB/s (128GB/s)
122GiB/s (131GB/s)

## Current Maximum throughput

In previous 28-SSD runs, root complex `[0000:20]` only has 4 SSDs on it, so we have not saturated its bandwidth. `nvme18n1` has the OS, but `nvme17n1` and `nvme16n1` are both unused. Wiping the file system from these devices, we benchmark again wtih 

`sudo python3 all_ssd_read.py -j 4 -s 0-15 16-17 19-22 23-30`

16 and 17 are the newly added devices. 

```
all_ssd: (groupid=0, jobs=4): err= 0: pid=5142: Tue Feb  6 16:37:39 2024
  read: IOPS=31.3k, BW=122GiB/s (131GB/s)(2301GiB/18834msec)
   bw (  MiB/s): min=123861, max=126712, per=100.00%, avg=125326.56, stdev=150.73, samples=148
   iops        : min=30965, max=31678, avg=31331.57, stdev=37.70, samples=148
  cpu          : usr=0.34%, sys=4.22%, ctx=548733, majf=0, minf=149
  IO depths    : 1=0.1%, 2=0.1%, 4=0.1%, 8=0.1%, 16=0.1%, 32=100.0%, >=64=0.0%
     submit    : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%
     complete  : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.1%, 64=0.0%, >=64=0.0%
     issued rwts: total=589084,0,0,0 short=0,0,0,0 dropped=0,0,0,0
     latency   : target=0, window=0, percentile=100.00%, depth=32

Run status group 0 (all jobs):
   READ: bw=122GiB/s (131GB/s), 122GiB/s-122GiB/s (131GB/s-131GB/s), io=2301GiB (2471GB), run=18834-18834msec

Disk stats (read/write):
  nvme0n1: ios=627048/0, merge=0/0, ticks=709372/0, in_queue=709372, util=95.20%
  nvme1n1: ios=627040/0, merge=0/0, ticks=689232/0, in_queue=689232, util=95.46%
  nvme2n1: ios=627040/0, merge=0/0, ticks=531413/0, in_queue=531413, util=95.60%
  nvme3n1: ios=627040/0, merge=0/0, ticks=487519/0, in_queue=487519, util=95.63%
  nvme4n1: ios=627032/0, merge=0/0, ticks=482434/0, in_queue=482434, util=95.86%
  nvme5n1: ios=627038/0, merge=0/0, ticks=481867/0, in_queue=481867, util=95.99%
  nvme6n1: ios=627034/0, merge=0/0, ticks=425242/0, in_queue=425242, util=96.05%
  nvme7n1: ios=627034/0, merge=0/0, ticks=435048/0, in_queue=435048, util=96.29%
  nvme8n1: ios=626964/0, merge=0/0, ticks=3257090/0, in_queue=3257090, util=96.47%
  nvme9n1: ios=626930/0, merge=0/0, ticks=1127525/0, in_queue=1127525, util=96.55%
  nvme10n1: ios=626968/0, merge=0/0, ticks=3518024/0, in_queue=3518024, util=96.69%
  nvme11n1: ios=626940/0, merge=0/0, ticks=3601457/0, in_queue=3601457, util=96.75%
  nvme12n1: ios=626965/0, merge=0/0, ticks=4306778/0, in_queue=4306778, util=97.01%
  nvme13n1: ios=626918/0, merge=0/0, ticks=1072887/0, in_queue=1072887, util=97.19%
  nvme14n1: ios=626899/0, merge=0/0, ticks=1486426/0, in_queue=1486426, util=97.29%
  nvme15n1: ios=626295/0, merge=0/0, ticks=10440346/0, in_queue=10440346, util=97.30%
  nvme16n1: ios=627021/0, merge=0/0, ticks=547707/0, in_queue=547707, util=97.47%
  nvme17n1: ios=626923/0, merge=0/0, ticks=1249126/0, in_queue=1249126, util=97.59%
  nvme19n1: ios=627032/0, merge=0/0, ticks=199629/0, in_queue=199629, util=97.31%
  nvme20n1: ios=627031/0, merge=0/0, ticks=281339/0, in_queue=281339, util=97.59%
  nvme21n1: ios=627012/0, merge=0/0, ticks=231194/0, in_queue=231194, util=98.03%
  nvme22n1: ios=626999/0, merge=0/0, ticks=228653/0, in_queue=228653, util=98.23%
  nvme23n1: ios=626997/0, merge=0/0, ticks=230595/0, in_queue=230595, util=98.25%
  nvme24n1: ios=626997/0, merge=0/0, ticks=227605/0, in_queue=227605, util=98.53%
  nvme25n1: ios=626448/0, merge=0/0, ticks=10526243/0, in_queue=10526243, util=98.83%
  nvme26n1: ios=626453/0, merge=0/0, ticks=8196091/0, in_queue=8196091, util=98.94%
  nvme27n1: ios=626582/0, merge=0/0, ticks=8988093/0, in_queue=8988093, util=98.96%
  nvme28n1: ios=626976/0, merge=0/0, ticks=638284/0, in_queue=638284, util=99.16%
  nvme29n1: ios=626962/0, merge=0/0, ticks=564899/0, in_queue=564899, util=99.30%
  nvme30n1: ios=626962/0, merge=0/0, ticks=520927/0, in_queue=520927, util=99.54%
```

## More info on anamalous result

Turns out that https://tanelpoder.com/posts/11m-iops-with-10-ssds-on-amd-threadripper-pro-workstation/#multi-disk-test already has an explanation. Using `lspci -tv`, we see that baldr has 4 root complexes and that the 28 SSDs are distributed fairly evenly among them, 8 on `[0000:60]`, `[0000:40]`, and `[0000:00]`, and 7 (4 on the adapters and 3 directly plugged in to M.2 slots) on `[0000:20]`. The blogger can move 4 SSDs to another root complex, but our root complexes are all saturated with 8 SSDs (+ the ones directly plugged into the motherboard). Further testing confirms that limited root complex bandwidth is the culprit. 

```
-+-[0000:60]-+-00.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse Root Complex
 |           +-00.2  Advanced Micro Devices, Inc. [AMD] Starship/Matisse IOMMU
 |           +-01.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
 |           +-01.1-[61]----00.0  Solidigm P44 Pro NVMe SSD
 |           +-01.2-[62]----00.0  Solidigm P44 Pro NVMe SSD
 |           +-01.3-[63]----00.0  Solidigm P44 Pro NVMe SSD
 |           +-01.4-[64]----00.0  Solidigm P44 Pro NVMe SSD
 |           +-02.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
 |           +-03.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
 |           +-03.1-[65]----00.0  Solidigm P44 Pro NVMe SSD
 |           +-03.2-[66]----00.0  Solidigm P44 Pro NVMe SSD
 |           +-03.3-[67]----00.0  Solidigm P44 Pro NVMe SSD
 |           +-03.4-[68]----00.0  Solidigm P44 Pro NVMe SSD
 |           +-04.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
 |           +-05.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
 |           +-07.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
 |           +-07.1-[69]----00.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Function
 |           +-08.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
 |           \-08.1-[6a]----00.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse Reserved SPP
 +-[0000:40]-+-00.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse Root Complex
 |           +-00.2  Advanced Micro Devices, Inc. [AMD] Starship/Matisse IOMMU
 |           +-01.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
 |           +-01.1-[41]----00.0  Solidigm P44 Pro NVMe SSD
 |           +-01.2-[42]----00.0  Solidigm P44 Pro NVMe SSD
 |           +-01.3-[43]----00.0  Solidigm P44 Pro NVMe SSD
 |           +-01.4-[44]----00.0  Solidigm P44 Pro NVMe SSD
 |           +-02.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
 |           +-03.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
 |           +-03.1-[45]----00.0  Solidigm P44 Pro NVMe SSD
 |           +-03.2-[46]----00.0  Solidigm P44 Pro NVMe SSD
 |           +-03.3-[47]----00.0  Solidigm P44 Pro NVMe SSD
 |           +-03.4-[48]----00.0  Solidigm P44 Pro NVMe SSD
 |           +-04.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
 |           +-05.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
 |           +-07.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
 |           +-07.1-[49]----00.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Function
 |           +-08.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
 |           \-08.1-[4a]----00.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse Reserved SPP
 +-[0000:20]-+-00.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse Root Complex
 |           +-00.2  Advanced Micro Devices, Inc. [AMD] Starship/Matisse IOMMU
 |           +-01.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
 |           +-01.1-[21-29]----00.0-[22-29]--+-00.0-[23]----00.0  Solidigm P44 Pro NVMe SSD
 |           |                               +-01.0-[24]----00.0  ASMedia Technology Inc. ASM3242 USB 3.2 Host Controller
 |           |                               +-02.0-[25]--+-00.0  Intel Corporation Ethernet Controller X550
 |           |                               |            \-00.1  Intel Corporation Ethernet Controller X550
 |           |                               +-03.0-[26]----00.0  Intel Corporation Wi-Fi 6 AX200
 |           |                               +-04.0-[27]----00.0  ASMedia Technology Inc. ASM1062 Serial ATA Controller
 |           |                               +-05.0-[28]----00.0  ASMedia Technology Inc. ASM1062 Serial ATA Controller
 |           |                               \-08.0-[29]--+-00.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse Reserved SPP
 |           |                                            +-00.1  Advanced Micro Devices, Inc. [AMD] Matisse USB 3.0 Host Controller
 |           |                                            \-00.3  Advanced Micro Devices, Inc. [AMD] Matisse USB 3.0 Host Controller
 |           +-01.2-[2a]----00.0  Solidigm P44 Pro NVMe SSD
 |           +-01.3-[2b]----00.0  Solidigm P44 Pro NVMe SSD
 |           +-02.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
 |           +-03.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
 |           +-03.1-[2c]----00.0  Solidigm P44 Pro NVMe SSD
 |           +-03.2-[2d]----00.0  Solidigm P44 Pro NVMe SSD
 |           +-03.3-[2e]----00.0  Solidigm P44 Pro NVMe SSD
 |           +-03.4-[2f]----00.0  Solidigm P44 Pro NVMe SSD
 |           +-04.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
 |           +-05.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
 |           +-07.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
 |           +-07.1-[30]----00.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Function
 |           +-08.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
 |           \-08.1-[31]--+-00.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse Reserved SPP
 |                        +-00.1  Advanced Micro Devices, Inc. [AMD] Starship/Matisse Cryptographic Coprocessor PSPCPP
 |                        +-00.3  Advanced Micro Devices, Inc. [AMD] Starship USB 3.0 Host Controller
 |                        \-00.4  Advanced Micro Devices, Inc. [AMD] Starship/Matisse HD Audio Controller
 \-[0000:00]-+-00.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse Root Complex
             +-00.2  Advanced Micro Devices, Inc. [AMD] Starship/Matisse IOMMU
             +-01.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
             +-01.1-[01]----00.0  Solidigm P44 Pro NVMe SSD
             +-01.2-[02]----00.0  Solidigm P44 Pro NVMe SSD
             +-01.3-[03]----00.0  Solidigm P44 Pro NVMe SSD
             +-01.4-[04]----00.0  Solidigm P44 Pro NVMe SSD
             +-02.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
             +-03.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
             +-03.1-[05]----00.0  Solidigm P44 Pro NVMe SSD
             +-03.2-[06]----00.0  Solidigm P44 Pro NVMe SSD
             +-03.3-[07]----00.0  Solidigm P44 Pro NVMe SSD
             +-03.4-[08]----00.0  Solidigm P44 Pro NVMe SSD
             +-04.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
             +-05.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
             +-07.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
             +-07.1-[09]----00.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Function
             +-08.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse PCIe Dummy Host Bridge
             +-08.1-[0a]--+-00.0  Advanced Micro Devices, Inc. [AMD] Starship/Matisse Reserved SPP
             |            \-00.3  Advanced Micro Devices, Inc. [AMD] Starship USB 3.0 Host Controller
             +-14.0  Advanced Micro Devices, Inc. [AMD] FCH SMBus Controller
             +-14.3  Advanced Micro Devices, Inc. [AMD] FCH LPC Bridge
             +-18.0  Advanced Micro Devices, Inc. [AMD] Starship Device 24; Function 0
             +-18.1  Advanced Micro Devices, Inc. [AMD] Starship Device 24; Function 1
             +-18.2  Advanced Micro Devices, Inc. [AMD] Starship Device 24; Function 2
             +-18.3  Advanced Micro Devices, Inc. [AMD] Starship Device 24; Function 3
             +-18.4  Advanced Micro Devices, Inc. [AMD] Starship Device 24; Function 4
             +-18.5  Advanced Micro Devices, Inc. [AMD] Starship Device 24; Function 5
             +-18.6  Advanced Micro Devices, Inc. [AMD] Starship Device 24; Function 6
             \-18.7  Advanced Micro Devices, Inc. [AMD] Starship Device 24; Function 7
```

`sudo python3 all_ssd_read.py -j 4 -s 0 1 2 3 4 5 6 7` and `sudo python3 all_ssd_read.py -j 4 -s 8 9 10 11 12 13 14 15` gave similar results. 

```
all_ssd: (groupid=0, jobs=4): err= 0: pid=451532: Fri Feb  2 18:43:44 2024
  read: IOPS=10.2k, BW=39.7GiB/s (42.6GB/s)(495GiB/12459msec)
   bw (  MiB/s): min=26112, max=51496, per=100.00%, avg=40783.77, stdev=1903.94, samples=96
   iops        : min= 6528, max=12874, avg=10195.92, stdev=475.98, samples=96
  cpu          : usr=0.03%, sys=1.40%, ctx=124513, majf=0, minf=42
  IO depths    : 1=0.1%, 2=0.1%, 4=0.1%, 8=0.1%, 16=0.1%, 32=99.9%, >=64=0.0%
     submit    : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%
     complete  : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.1%, 64=0.0%, >=64=0.0%
     issued rwts: total=126605,0,0,0 short=0,0,0,0 dropped=0,0,0,0
     latency   : target=0, window=0, percentile=100.00%, depth=32

Run status group 0 (all jobs):
   READ: bw=39.7GiB/s (42.6GB/s), 39.7GiB/s-39.7GiB/s (42.6GB/s-42.6GB/s), io=495GiB (531GB), run=12459-12459msec

Disk stats (read/write):
  nvme0n1: ios=504428/0, merge=0/0, ticks=24703196/0, in_queue=24703196, util=97.57%
  nvme1n1: ios=505648/0, merge=0/0, ticks=7064353/0, in_queue=7064353, util=97.93%
  nvme2n1: ios=505681/0, merge=0/0, ticks=6650573/0, in_queue=6650573, util=98.11%
  nvme3n1: ios=506201/0, merge=0/0, ticks=297134/0, in_queue=297134, util=97.81%
  nvme4n1: ios=505503/0, merge=0/0, ticks=7373272/0, in_queue=7373272, util=98.50%
  nvme5n1: ios=506150/0, merge=0/0, ticks=271000/0, in_queue=271000, util=98.27%
  nvme6n1: ios=506118/0, merge=0/0, ticks=241329/0, in_queue=241329, util=98.34%
  nvme7n1: ios=506114/0, merge=0/0, ticks=261663/0, in_queue=261663, util=98.77%
```

`sudo python3 all_ssd_read.py -j 4 -s 0 1 2 3 8 9 10 11` and `sudo python3 all_ssd_read.py -j 4 -s 4 5 6 7 8 9 10 11` gave similar results. 

```
all_ssd: (groupid=0, jobs=4): err= 0: pid=451738: Fri Feb  2 18:44:02 2024
  read: IOPS=13.6k, BW=53.2GiB/s (57.1GB/s)(546GiB/10260msec)
   bw (  MiB/s): min=42498, max=68024, per=100.00%, avg=54805.79, stdev=1594.68, samples=80
   iops        : min=10624, max=17006, avg=13701.40, stdev=398.69, samples=80
  cpu          : usr=0.07%, sys=1.86%, ctx=136501, majf=0, minf=46
  IO depths    : 1=0.1%, 2=0.1%, 4=0.1%, 8=0.1%, 16=0.1%, 32=99.9%, >=64=0.0%
     submit    : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%
     complete  : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.1%, 64=0.0%, >=64=0.0%
     issued rwts: total=139733,0,0,0 short=0,0,0,0 dropped=0,0,0,0
     latency   : target=0, window=0, percentile=100.00%, depth=32

Run status group 0 (all jobs):
   READ: bw=53.2GiB/s (57.1GB/s), 53.2GiB/s-53.2GiB/s (57.1GB/s-57.1GB/s), io=546GiB (586GB), run=10260-10260msec

Disk stats (read/write):
  nvme0n1: ios=558870/0, merge=0/0, ticks=17913503/0, in_queue=17913503, util=97.41%
  nvme1n1: ios=556485/0, merge=0/0, ticks=16497171/0, in_queue=16497171, util=97.87%
  nvme2n1: ios=558976/0, merge=0/0, ticks=608890/0, in_queue=608890, util=98.02%
  nvme3n1: ios=558944/0, merge=0/0, ticks=492993/0, in_queue=492993, util=97.96%
  nvme8n1: ios=558912/0, merge=0/0, ticks=537263/0, in_queue=537263, util=98.24%
  nvme9n1: ios=558912/0, merge=0/0, ticks=537015/0, in_queue=537015, util=98.41%
  nvme10n1: ios=558912/0, merge=0/0, ticks=472684/0, in_queue=472684, util=98.67%
  nvme11n1: ios=558848/0, merge=0/0, ticks=925261/0, in_queue=925261, util=98.79%
```

Given the limitations imposed by root complexes, it seems that 16 SSDs can achieve near-maximal throughput and throwing 12 additional SSDs into the system provides marginal improvement. 

```
all_ssd: (groupid=0, jobs=4): err= 0: pid=478827: Tue Feb  6 11:38:28 2024
  read: IOPS=27.0k, BW=106GiB/s (113GB/s)(2680GiB/25370msec)
   bw (  MiB/s): min=82000, max=138680, per=100.00%, avg=108421.60, stdev=3426.38, samples=200
   iops        : min=20500, max=34670, avg=27105.44, stdev=856.59, samples=200
  cpu          : usr=0.30%, sys=3.61%, ctx=653619, majf=0, minf=86
  IO depths    : 1=0.1%, 2=0.1%, 4=0.1%, 8=0.1%, 16=0.1%, 32=100.0%, >=64=0.0%
     submit    : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%
     complete  : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.1%, 64=0.0%, >=64=0.0%
     issued rwts: total=686119,0,0,0 short=0,0,0,0 dropped=0,0,0,0
     latency   : target=0, window=0, percentile=100.00%, depth=32

Run status group 0 (all jobs):
   READ: bw=106GiB/s (113GB/s), 106GiB/s-106GiB/s (113GB/s-113GB/s), io=2680GiB (2878GB), run=25370-25370msec

Disk stats (read/write):
  nvme0n1: ios=1364927/0, merge=0/0, ticks=11040414/0, in_queue=11040414, util=98.18%
  nvme1n1: ios=1363670/0, merge=0/0, ticks=35392914/0, in_queue=35392914, util=98.35%
  nvme2n1: ios=1364578/0, merge=0/0, ticks=26824167/0, in_queue=26824167, util=98.44%
  nvme3n1: ios=1365369/0, merge=0/0, ticks=939596/0, in_queue=939596, util=98.47%
  nvme8n1: ios=1365358/0, merge=0/0, ticks=1335525/0, in_queue=1335525, util=98.59%
  nvme9n1: ios=1365356/0, merge=0/0, ticks=1163349/0, in_queue=1163349, util=98.65%
  nvme10n1: ios=1365347/0, merge=0/0, ticks=980675/0, in_queue=980675, util=98.79%
  nvme11n1: ios=1365338/0, merge=0/0, ticks=1189278/0, in_queue=1189278, util=98.83%
  nvme19n1: ios=1365334/0, merge=0/0, ticks=1000552/0, in_queue=1000552, util=99.00%
  nvme20n1: ios=1365303/0, merge=0/0, ticks=1288113/0, in_queue=1288113, util=99.01%
  nvme21n1: ios=1365323/0, merge=0/0, ticks=1008311/0, in_queue=1008311, util=99.19%
  nvme22n1: ios=1365312/0, merge=0/0, ticks=1268936/0, in_queue=1268936, util=99.33%
  nvme27n1: ios=1365312/0, merge=0/0, ticks=1317286/0, in_queue=1317286, util=99.35%
  nvme28n1: ios=1365300/0, merge=0/0, ticks=1668896/0, in_queue=1668896, util=99.49%
  nvme29n1: ios=1365290/0, merge=0/0, ticks=1594535/0, in_queue=1594535, util=99.59%
  nvme30n1: ios=1365254/0, merge=0/0, ticks=1718082/0, in_queue=1718082, util=99.78%
```

## Resolution of anomalous results

Turns out that the anomaly might be due to the way SSDs are selected. If we always select the first X SSDs. All of them reside on the first X/4 PCIE to M.2 adapters. If we use round robin (i.e. select a SSD from each adapter and repeat if we need more), the load is evenly distributed and we get good performance again. However, it is still a mystery why reading SSDs 0-3 at the same time gives good performance even though they are all on the same adapter, and when we try to use the first two adapters at the same time, we see significant performance degradation. 

Command: `sudo python3 all_ssd_read.py 4 4M 8 1`

```
all_ssd: (groupid=0, jobs=4): err= 0: pid=22728: Thu Nov  2 19:16:07 2023
  read: IOPS=13.7k, BW=53.5GiB/s (57.5GB/s)(1424GiB/26606msec)
   bw (  MiB/s): min=52296, max=58008, per=100.00%, avg=54996.68, stdev=267.68, samples=212
   iops        : min=13074, max=14502, avg=13749.17, stdev=66.92, samples=212
  cpu          : usr=0.24%, sys=30.08%, ctx=297453, majf=0, minf=2354
  IO depths    : 1=0.1%, 2=0.1%, 4=0.1%, 8=0.1%, 16=0.1%, 32=100.0%, >=64=0.0%
     submit    : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%
     complete  : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.1%, 64=0.0%, >=64=0.0%
     issued rwts: total=364513,0,0,0 short=0,0,0,0 dropped=0,0,0,0
     latency   : target=0, window=0, percentile=100.00%, depth=32

Run status group 0 (all jobs):
   READ: bw=53.5GiB/s (57.5GB/s), 53.5GiB/s-53.5GiB/s (57.5GB/s-57.5GB/s), io=1424GiB (1529GB), run=26606-26606msec

Disk stats (read/write):
  nvme0n1: ios=725194/0, merge=1360/0, ticks=1057683/0, in_queue=1057683, util=98.82%
  nvme4n1: ios=725186/0, merge=1396/0, ticks=691955/0, in_queue=691955, util=99.01%
  nvme8n1: ios=725177/0, merge=1345/0, ticks=2337371/0, in_queue=2337371, util=99.14%
  nvme12n1: ios=725160/0, merge=1351/0, ticks=3648007/0, in_queue=3648007, util=99.33%
  nvme19n1: ios=725161/0, merge=1339/0, ticks=813067/0, in_queue=813067, util=99.51%
  nvme23n1: ios=725154/0, merge=1434/0, ticks=1441551/0, in_queue=1441551, util=99.57%
  nvme27n1: ios=725145/0, merge=1384/0, ticks=1035864/0, in_queue=1035864, util=99.58%
  nvme1n1: ios=723337/0, merge=1416/0, ticks=40524301/0, in_queue=40524301, util=99.73%
```

## Anomalous results

Anomalous results after all 28 SSDs are installed. Compare `sudo python3 all_ssd.py 4 4M 6` and `sudo python3 all_ssd.py 4 4M 8`. The throughput of the former should be 3/4 of the latter, but they have near identical throughput.

```
all_ssd: (groupid=0, jobs=4): err= 0: pid=18931: Thu Nov  2 18:48:21 2023
  read: IOPS=8852, BW=34.6GiB/s (37.1GB/s)(597GiB/17262msec)
   bw (  MiB/s): min=34312, max=36368, per=100.00%, avg=35474.59, stdev=90.66, samples=136
   iops        : min= 8578, max= 9092, avg=8868.65, stdev=22.67, samples=136
  cpu          : usr=0.20%, sys=22.58%, ctx=130737, majf=0, minf=2354
  IO depths    : 1=0.1%, 2=0.1%, 4=0.1%, 8=0.1%, 16=0.1%, 32=99.9%, >=64=0.0%
     submit    : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%
     complete  : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.1%, 64=0.0%, >=64=0.0%
     issued rwts: total=152808,0,0,0 short=0,0,0,0 dropped=0,0,0,0
     latency   : target=0, window=0, percentile=100.00%, depth=32

Run status group 0 (all jobs):
   READ: bw=34.6GiB/s (37.1GB/s), 34.6GiB/s-34.6GiB/s (37.1GB/s-37.1GB/s), io=597GiB (641GB), run=17262-17262msec

Disk stats (read/write):
  nvme0n1: ios=407297/0, merge=0/0, ticks=7399837/0, in_queue=7399837, util=98.65%
  nvme1n1: ios=406986/0, merge=0/0, ticks=9719319/0, in_queue=9719319, util=98.89%
  nvme2n1: ios=407252/0, merge=0/0, ticks=7826798/0, in_queue=7826798, util=99.03%
  nvme3n1: ios=407121/0, merge=0/0, ticks=8883908/0, in_queue=8883908, util=99.07%
  nvme4n1: ios=407472/0, merge=0/0, ticks=171122/0, in_queue=171122, util=99.31%
  nvme5n1: ios=407456/0, merge=0/0, ticks=172457/0, in_queue=172457, util=99.47%
```

```
all_ssd: (groupid=0, jobs=4): err= 0: pid=19229: Thu Nov  2 18:50:21 2023
  read: IOPS=8972, BW=35.0GiB/s (37.6GB/s)(2804GiB/80017msec)
   bw (  MiB/s): min=34800, max=36768, per=100.00%, avg=35942.24, stdev=46.65, samples=636
   iops        : min= 8700, max= 9192, avg=8985.56, stdev=11.66, samples=636
  cpu          : usr=0.21%, sys=23.31%, ctx=595660, majf=0, minf=2354
  IO depths    : 1=0.1%, 2=0.1%, 4=0.1%, 8=0.1%, 16=0.1%, 32=100.0%, >=64=0.0%
     submit    : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%
     complete  : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.1%, 64=0.0%, >=64=0.0%
     issued rwts: total=717941,0,0,0 short=0,0,0,0 dropped=0,0,0,0
     latency   : target=0, window=0, percentile=100.00%, depth=32

Run status group 0 (all jobs):
   READ: bw=35.0GiB/s (37.6GB/s), 35.0GiB/s-35.0GiB/s (37.6GB/s-37.6GB/s), io=2804GiB (3011GB), run=80017-80017msec

Disk stats (read/write):
  nvme0n1: ios=1436965/0, merge=0/0, ticks=37352280/0, in_queue=37352280, util=99.74%
  nvme1n1: ios=1437081/0, merge=0/0, ticks=30530501/0, in_queue=30530501, util=99.80%
  nvme2n1: ios=1436993/0, merge=0/0, ticks=41857809/0, in_queue=41857809, util=99.83%
  nvme3n1: ios=1436953/0, merge=0/0, ticks=40554180/0, in_queue=40554180, util=99.84%
  nvme4n1: ios=1437426/0, merge=0/0, ticks=1669674/0, in_queue=1669674, util=99.89%
  nvme5n1: ios=1437313/0, merge=0/0, ticks=1687627/0, in_queue=1687627, util=99.93%
  nvme6n1: ios=1437378/0, merge=0/0, ticks=1840467/0, in_queue=1840467, util=99.94%
  nvme7n1: ios=1437285/0, merge=0/0, ticks=1628614/0, in_queue=1628614, util=100.00%
```

Note that using 4 SSDs or all 28 SSDs give identical performance to the previous script. However, using 17 SSDs will give worse performance than before. The 6GiB/s average drops to 4GiB/s

Command: `sudo python3 all_ssd_read.py 10 4M 17`

```
all_ssd: (groupid=0, jobs=10): err= 0: pid=19979: Thu Nov  2 18:52:56 2023
  read: IOPS=18.8k, BW=73.5GiB/s (78.9GB/s)(1166GiB/15864msec)
   bw (  MiB/s): min=60600, max=89496, per=100.00%, avg=75427.69, stdev=718.90, samples=310
   iops        : min=15150, max=22374, avg=18856.77, stdev=179.72, samples=310
  cpu          : usr=0.18%, sys=17.02%, ctx=251827, majf=0, minf=926
  IO depths    : 1=0.1%, 2=0.1%, 4=0.1%, 8=0.1%, 16=0.1%, 32=99.9%, >=64=0.0%
     submit    : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%
     complete  : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.1%, 64=0.0%, >=64=0.0%
     issued rwts: total=298521,0,0,0 short=0,0,0,0 dropped=0,0,0,0
     latency   : target=0, window=0, percentile=100.00%, depth=32

Run status group 0 (all jobs):
   READ: bw=73.5GiB/s (78.9GB/s), 73.5GiB/s-73.5GiB/s (78.9GB/s-78.9GB/s), io=1166GiB (1252GB), run=15864-15864msec

Disk stats (read/write):
  nvme0n1: ios=278334/0, merge=0/0, ticks=4351900/0, in_queue=4351900, util=96.70%
  nvme1n1: ios=278082/0, merge=0/0, ticks=6496778/0, in_queue=6496778, util=96.96%
  nvme2n1: ios=278125/0, merge=0/0, ticks=7166290/0, in_queue=7166290, util=97.11%
  nvme3n1: ios=278422/0, merge=0/0, ticks=3970432/0, in_queue=3970432, util=97.14%
  nvme4n1: ios=278418/0, merge=0/0, ticks=901773/0, in_queue=901773, util=97.40%
  nvme5n1: ios=278407/0, merge=0/0, ticks=1461173/0, in_queue=1461173, util=97.53%
  nvme6n1: ios=278404/0, merge=0/0, ticks=1332440/0, in_queue=1332440, util=97.62%
  nvme7n1: ios=278402/0, merge=0/0, ticks=1713156/0, in_queue=1713156, util=97.87%
  nvme8n1: ios=278394/0, merge=0/0, ticks=1546380/0, in_queue=1546380, util=98.07%
  nvme9n1: ios=278390/0, merge=0/0, ticks=1161699/0, in_queue=1161699, util=98.23%
  nvme10n1: ios=278387/0, merge=0/0, ticks=1437810/0, in_queue=1437810, util=98.40%
  nvme11n1: ios=278384/0, merge=0/0, ticks=1474472/0, in_queue=1474472, util=98.48%
  nvme12n1: ios=277472/0, merge=0/0, ticks=10753676/0, in_queue=10753676, util=98.81%
  nvme13n1: ios=277410/0, merge=0/0, ticks=10548448/0, in_queue=10548448, util=99.03%
  nvme14n1: ios=277463/0, merge=0/0, ticks=10325627/0, in_queue=10325627, util=99.13%
  nvme15n1: ios=277403/0, merge=0/0, ticks=11281415/0, in_queue=11281415, util=99.13%
  nvme19n1: ios=278368/0, merge=0/0, ticks=118038/0, in_queue=118038, util=97.53%
```

# Old script

## 28 SSDs

Command: `sudo python3 all_ssd.py 28 4M`

Unknown bottleneck that results in 4.2GiB/s/SSD. 

Increasing the number of processes does not seem to aid performance. 

Switching from `libaio` to `io_uring` does not affect performance. In fact, `BW` went from 118GiB/s to 115GiB/s, but that might be random fluctuation. 

```
all_ssd: (groupid=0, jobs=28): err= 0: pid=3835: Tue Oct 31 18:03:40 2023
  read: IOPS=30.2k, BW=118GiB/s (126GB/s)(11.9TiB/103337msec)
   bw (  MiB/s): min=50354, max=213360, per=100.00%, avg=120922.31, stdev=1546.90, samples=5768
   iops        : min=12587, max=53340, avg=30229.86, stdev=386.72, samples=5768
  cpu          : usr=0.13%, sys=11.41%, ctx=2599674, majf=0, minf=2976
  IO depths    : 1=0.1%, 2=0.1%, 4=0.1%, 8=0.1%, 16=0.1%, 32=100.0%, >=64=0.0%
     submit    : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%
     complete  : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.1%, 64=0.0%, >=64=0.0%
     issued rwts: total=3116292,0,0,0 short=0,0,0,0 dropped=0,0,0,0
     latency   : target=0, window=0, percentile=100.00%, depth=32

Run status group 0 (all jobs):
   READ: bw=118GiB/s (126GB/s), 118GiB/s-118GiB/s (126GB/s-126GB/s), io=11.9TiB (13.1TB), run=103337-103337msec

Disk stats (read/write):
  nvme0n1: ios=1780912/0, merge=0/0, ticks=4110951/0, in_queue=4110951, util=99.06%
  nvme1n1: ios=1780912/0, merge=0/0, ticks=3691235/0, in_queue=3691235, util=99.12%
  nvme2n1: ios=1780896/0, merge=0/0, ticks=4112668/0, in_queue=4112668, util=99.04%
  nvme3n1: ios=1780896/0, merge=0/0, ticks=4334567/0, in_queue=4334567, util=98.79%
  nvme4n1: ios=1780896/0, merge=0/0, ticks=2989717/0, in_queue=2989717, util=99.08%
  nvme5n1: ios=1780880/0, merge=0/0, ticks=3071300/0, in_queue=3071300, util=99.09%
  nvme6n1: ios=1780848/0, merge=0/0, ticks=3227915/0, in_queue=3227915, util=98.99%
  nvme7n1: ios=1780848/0, merge=0/0, ticks=3124216/0, in_queue=3124216, util=98.85%
  nvme8n1: ios=1780832/0, merge=0/0, ticks=48525849/0, in_queue=48525849, util=99.51%
  nvme9n1: ios=1780816/0, merge=0/0, ticks=55672243/0, in_queue=55672243, util=99.52%
  nvme10n1: ios=1780800/0, merge=0/0, ticks=46025456/0, in_queue=46025456, util=99.55%
  nvme11n1: ios=1780784/0, merge=0/0, ticks=47981222/0, in_queue=47981222, util=99.56%
  nvme12n1: ios=1780752/0, merge=0/0, ticks=3979027/0, in_queue=3979027, util=99.38%
  nvme13n1: ios=1780752/0, merge=0/0, ticks=3599901/0, in_queue=3599901, util=99.51%
  nvme14n1: ios=1780736/0, merge=0/0, ticks=3321940/0, in_queue=3321940, util=99.53%
  nvme15n1: ios=1780736/0, merge=0/0, ticks=3764183/0, in_queue=3764183, util=99.50%
  nvme19n1: ios=1780720/0, merge=0/0, ticks=846049/0, in_queue=846049, util=96.37%
  nvme20n1: ios=1780720/0, merge=0/0, ticks=809931/0, in_queue=809931, util=96.57%
  nvme21n1: ios=1780720/0, merge=0/0, ticks=795667/0, in_queue=795667, util=96.44%
  nvme22n1: ios=1780720/0, merge=0/0, ticks=802010/0, in_queue=802010, util=96.22%
  nvme23n1: ios=1777587/0, merge=0/0, ticks=324828058/0, in_queue=324828058, util=99.84%
  nvme24n1: ios=1778468/0, merge=0/0, ticks=266943198/0, in_queue=266943198, util=99.90%
  nvme25n1: ios=1778171/0, merge=0/0, ticks=290105622/0, in_queue=290105622, util=99.94%
  nvme26n1: ios=1778364/0, merge=0/0, ticks=270478909/0, in_queue=270478909, util=99.96%
  nvme27n1: ios=1780464/0, merge=0/0, ticks=3313343/0, in_queue=3313343, util=99.31%
  nvme28n1: ios=1780464/0, merge=0/0, ticks=2720899/0, in_queue=2720899, util=99.61%
  nvme29n1: ios=1780464/0, merge=0/0, ticks=2554823/0, in_queue=2554823, util=99.68%
  nvme30n1: ios=1780464/0, merge=0/0, ticks=2481770/0, in_queue=2481770, util=99.67%
```

## 17 SSDs

Command: `sudo python3 all_ssd.py 17 4M`

Speed per SSD is 106GiB/17=6.23GiB/s. A minor decrease over single SSD performance. Note that fio seems to be creating 17 processes each randomly selecting a SSD to perform reads, so maybe this is a contributing factor to decreased throughout as we can have multiple processes hitting the same SSD. However, increasing the number of processes does not seem to make a difference. 

```
all_ssd: (groupid=0, jobs=17): err= 0: pid=71816: Fri Oct 20 19:27:03 2023
  read: IOPS=27.2k, BW=106GiB/s (114GB/s)(1848GiB/17419msec)
   bw (  MiB/s): min=64656, max=160112, per=100.00%, avg=109047.40, stdev=1782.61, samples=578
   iops        : min=16164, max=40028, avg=27261.06, stdev=445.65, samples=578
  cpu          : usr=0.19%, sys=16.07%, ctx=367838, majf=0, minf=1585
  IO depths    : 1=0.1%, 2=0.1%, 4=0.1%, 8=0.1%, 16=0.1%, 32=99.9%, >=64=0.0%
     submit    : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%
     complete  : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.1%, 64=0.0%, >=64=0.0%
     issued rwts: total=472985,0,0,0 short=0,0,0,0 dropped=0,0,0,0
     latency   : target=0, window=0, percentile=100.00%, depth=32

Run status group 0 (all jobs):
   READ: bw=106GiB/s (114GB/s), 106GiB/s-106GiB/s (114GB/s-114GB/s), io=1848GiB (1984GB), run=17419-17419msec

Disk stats (read/write):
  nvme0n1: ios=445248/0, merge=0/0, ticks=592210/0, in_queue=592210, util=97.72%
  nvme1n1: ios=445248/0, merge=0/0, ticks=523014/0, in_queue=523014, util=97.94%
  nvme2n1: ios=445248/0, merge=0/0, ticks=765416/0, in_queue=765416, util=97.99%
  nvme3n1: ios=445232/0, merge=0/0, ticks=533423/0, in_queue=533423, util=98.04%
  nvme4n1: ios=445216/0, merge=0/0, ticks=397014/0, in_queue=397014, util=98.17%
  nvme5n1: ios=445216/0, merge=0/0, ticks=403727/0, in_queue=403727, util=98.29%
  nvme6n1: ios=445200/0, merge=0/0, ticks=396439/0, in_queue=396439, util=98.31%
  nvme7n1: ios=445200/0, merge=0/0, ticks=399093/0, in_queue=399093, util=98.49%
  nvme10n1: ios=445200/0, merge=0/0, ticks=348150/0, in_queue=348150, util=98.61%
  nvme11n1: ios=443535/0, merge=0/0, ticks=34028328/0, in_queue=34028328, util=98.76%
  nvme12n1: ios=443595/0, merge=0/0, ticks=34259373/0, in_queue=34259373, util=98.96%
  nvme13n1: ios=443455/0, merge=0/0, ticks=35079329/0, in_queue=35079329, util=99.10%
  nvme14n1: ios=443544/0, merge=0/0, ticks=34154481/0, in_queue=34154481, util=99.18%
  nvme15n1: ios=445056/0, merge=0/0, ticks=536161/0, in_queue=536161, util=99.08%
  nvme16n1: ios=445024/0, merge=0/0, ticks=449225/0, in_queue=449225, util=99.23%
  nvme17n1: ios=445008/0, merge=0/0, ticks=446278/0, in_queue=446278, util=99.32%
  nvme18n1: ios=445008/0, merge=0/0, ticks=478048/0, in_queue=478048, util=99.34%
  ```