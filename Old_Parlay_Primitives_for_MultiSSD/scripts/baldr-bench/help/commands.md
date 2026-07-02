Some useful commands.

# SSD info

## smartctl

Example: `sudo smartctl -a /dev/nvme0n1`

Gives information about a particular nvme drive. The `Data Units Written` field is of particular importance since that tells us how worn out the drive is in terms of writes. 

## nvme

Example `sudo nvme list`

Lists all SSDs and their serial numbers. 

## lsblk

Example: `lsblk`

Lists all drives. Useful for manually determining which SSD has a file system on it so that we can avoid writing directly to it. `experiment.py` has a script that automatically calls `lsblk` on a single drive to determine if it's safe to write to it. 

## lspci

Example: `sudo lspci -tv`

Lists PCI buses and devices on the system. `-v` is verbose mode. Adding `-t` display the tree-diagram of the topology. 

## hdparm

Example: `hdparm /dev/nvme0n1`

Check and toggle status for readonly mode and readahead mode. 

# Partitioning

## wipefs

Example: `sudo wipefs --all --backup --force /dev/nvme17n1`

Wipes the file system? Maybe? `lsblk` still shows the same thing, but after `partprobe` and a reboot, the partition is gone. I did some extra things as well, so not sure if wipefs is the one that wiped off the file system. 

## partprobe

Example: `sudo partprobe /dev/nvme17n1`

Updates the partition information on a device (sometimes `lsblk` still uses the old information). Some updates cannot be done and this command will ask the user to reboot.

# Performance monitoring

## dstat

Example: `dstat -pcmrd`

Gives system resource usage information. It's particularly helpful for monitoring disk IO. 

## sensors

Gives information about the CPU temperature.

`CPUTIN` is the CPU temperature reported by a sensor on the motherboard. 

`k10temp-pci-00c3` is the CPU temperature reported by the CPU itself. 

# Grub updates

Use `sudo grubby --info=ALL` to see the grub information on all kernels.

Use `sudo grubby --args="amd_iommu=off" --update-kernel /boot/vmlinuz-<kernel version>.fc<fedora version>.x86_64` to turn off iommu on next boot. Similarly, have `mitigations=off` to turn off mitigations. 

These changes made with `grubby` do not seem to persist through kernel updates, so it's necessary to double check grub configurations if there ever is an unexpected slowdown.
