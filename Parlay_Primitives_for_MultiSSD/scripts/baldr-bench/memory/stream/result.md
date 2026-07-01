See also: [this superuser post](https://superuser.com/questions/1815148/expected-results-of-a-stream-memory-bandwidth-benchmark)

# AOCC results

Command: `./run.sh 16 10000000000`

16 seems to give the best performance.

```
Function    Best Rate MB/s  Avg time     Min time     Max time
Copy:          163585.1     0.979134     0.978084     0.980080
Scale:         102983.4     1.554170     1.553648     1.555248
Add:           113216.5     2.120998     2.119832     2.123423
Triad:         113144.9     2.122118     2.121173     2.123545
```

Command: `./run.sh 32 10000000`

When array size is small and thread count is large, we get strange results, possibly because the 3995WX has 256MB of L3 cache, which is larger than the size of the array. However, these L3 cache are distributed on multiple CCXs, so we need more threads to utilize more L3 cache. 

```
Function    Best Rate MB/s  Avg time     Min time     Max time
Copy:         1288078.0     0.000137     0.000124     0.000158
Scale:        1261444.8     0.000140     0.000127     0.000160
Add:          1319309.3     0.000204     0.000182     0.000244
Triad:        1394228.5     0.000203     0.000172     0.000238
```

# Old results (using gcc)

Command: `./run.sh 16 1000000000`

Weird result. Theoretical limit is twice the current amount. 

16 threads gives comparatively good performance as having too much or too little threads running the benchmark decreases performance.

```
Function    Best Rate MB/s  Avg time     Min time     Max time
Copy:          103566.6     0.154768     0.154490     0.155157
Scale:         103404.6     0.155270     0.154732     0.157047
Add:           112979.4     0.212911     0.212428     0.213778
Triad:         113143.5     0.212315     0.212120     0.212602
```