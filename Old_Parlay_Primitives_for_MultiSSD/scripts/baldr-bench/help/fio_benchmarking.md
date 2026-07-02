## Sudden bandwidth drops

After about two months of use, rerunning `fio` benchmarks gives results inconsistent with the ones before. In particular, some SSDs experience sudden drops in read performance. SSD27 (`/dev/disk/by-path/pci-0000:66:00.0-nvme-1`) is one such example. Its read bandwidth drops to 2600M/s on `dstat` at 8-9 seconds after `fio` is started and then rebounds to normal values (6900M/s). This phenomenon is not observed in other drives such as SSD 26 and SSD 28. Running `blkdiscard` on this SSD resolved this issue. Read performance is again consistently at peak. Read bandwidth is back to peak as well.

## Performance drop after switching to Fedora

Performance drops are observed after switching to Fedora. In particular, CPU usage became much higher, so simple tasks (e.g. performing 4MB reads) would require multiple jobs for good performance. 6 jobs are required for peak read performance BW=113GiB/s (121GB/s). There are also unexplained performance drops when all 30 SSDs are used (100GB/s read).

Disabling iommu with `amd_iommu=off` in the grub seems to help: 3 jobs can achieve performance of 116GiB/s (124GB/s).

Disabling exploit mitigations with `mitigations=off` in the grub further reduces CPU usage from ~5.5% to ~2.8%. Results also seem to be consistently higher than before.

Another anomaly that has been observed is the fact that performance fluctuates depending which drives we include in a `fio` benchmark. Sometimes including less drives gives better performance. For example, in `fio/read/raw_results/mitigations_off.txt` (hopefully the file is not renamed in the future), having 30 SSDs gives 127GB/s, but 28 SSDs gives 131GB/s and 24 gives 132GB/s. This behavior is consistent across runs (see also `mitigations_off_2.txt`).

## Impact of file system

Before partitioning: 1 fio job achieves 105GB/s with 30 SSDs. 4 fio jobs achieve the result recorded in `mitigations_off.txt`.

After partitioning: 
- For `xfs`, 1 fio job achieves 61.3GB/s with 30 SSDs. Max throughput is decreased slightly.
