Command: `likwid-bench -t stream_mem_avx_fma -w M0:1GB`

MByte/s: 148545.59 / 1024 = 145GB/s

```
Cycles: 297168964650
CPU Clock: 2694290379
Cycle Clock: 2694290379
Time: 1.102958e+02 sec
Iterations: 2097152
Iterations per thread: 16384
Inner loop executions: 20345
Size (Byte): 999997440
Size per thread: 7812480
Number of Flops: 1365329838080
MFlops/s: 12378.80
Data volume (Byte): 16383958056960
MByte/s: 148545.59
Cycles per update: 0.435307
Cycles per cacheline: 3.482458
Loads per update: 2
Stores per update: 1
Load bytes per element: 16
Store bytes per elem.: 8
Load/store ratio: 2.00
Instructions: 639998361617
UOPs: 938664263680
```

Command: `likwid-bench -t store_mem_avx -w S0:100GB`

```
Cycles:                 117546867819
CPU Clock:              2691869643
Cycle Clock:            2691869643
Time:                   4.366737e+01 sec
Iterations:             8192
Iterations per thread:  64
Inner loop executions:  6103515
Size (Byte):            99999989760
Size per thread:        781249920
Number of Flops:        0
MFlops/s:               0.00
Data volume (Byte):     6399999344640
MByte/s:                146562.51
Cycles per update:      0.146934
Cycles per cacheline:   1.175469
Loads per update:       0
Stores per update:      1
Load bytes per element: 0
Store bytes per elem.:  8
Instructions:           349999964180
UOPs:                   499999948800
```

Command: `likwid-bench -t load_avx -w S0:100GB`

```
Cycles:                 57865662495
CPU Clock:              2694889386
Cycle Clock:            2694889386
Time:                   2.147237e+01 sec
Iterations:             4096
Iterations per thread:  32
Inner loop executions:  6103515
Size (Byte):            99999989760
Size per thread:        781249920
Number of Flops:        0
MFlops/s:               0.00
Data volume (Byte):     3199999672320
MByte/s:                149028.71
Cycles per update:      0.144664
Cycles per cacheline:   1.157313
Loads per update:       1
Stores per update:      0
Load bytes per element: 8
Store bytes per elem.:  0
Instructions:           174999982096
UOPs:                   149999984640
```
