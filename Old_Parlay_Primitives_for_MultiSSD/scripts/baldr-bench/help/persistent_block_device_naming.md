See [https://wiki.archlinux.org/title/Persistent_block_device_naming]

Essentially, the device names that appear in `/dev` have no guarantees in terms of order. Some nvme devices' file name may change across boots. For example, the below commands reveal that `/dev/nvme30n1`, `/dev/nvme7n1`, and `/dev/nvme8n1` are loaded out of the order they are installed on the motherboard. Of course, this assumes that the device with similar `pci` labelling reside on the same root complex and are adjacent to each other on the motherboard.

```
peter@baldr /dev/disk/by-path$ ls -l
total 0
lrwxrwxrwx. 1 root root 14 Apr 26 16:47 pci-0000:01:00.0-nvme-1 -> ../../nvme22n1
lrwxrwxrwx. 1 root root 16 Apr 26 18:41 pci-0000:01:00.0-nvme-1-part1 -> ../../nvme22n1p1
lrwxrwxrwx. 1 root root 14 Apr 26 16:47 pci-0000:02:00.0-nvme-1 -> ../../nvme23n1
lrwxrwxrwx. 1 root root 16 Apr 26 18:41 pci-0000:02:00.0-nvme-1-part1 -> ../../nvme23n1p1
lrwxrwxrwx. 1 root root 14 Apr 26 16:47 pci-0000:03:00.0-nvme-1 -> ../../nvme24n1
lrwxrwxrwx. 1 root root 16 Apr 26 18:41 pci-0000:03:00.0-nvme-1-part1 -> ../../nvme24n1p1
lrwxrwxrwx. 1 root root 14 Apr 26 16:47 pci-0000:04:00.0-nvme-1 -> ../../nvme25n1
lrwxrwxrwx. 1 root root 16 Apr 26 18:41 pci-0000:04:00.0-nvme-1-part1 -> ../../nvme25n1p1
lrwxrwxrwx. 1 root root 14 Apr 26 16:47 pci-0000:05:00.0-nvme-1 -> ../../nvme26n1
lrwxrwxrwx. 1 root root 16 Apr 26 18:41 pci-0000:05:00.0-nvme-1-part1 -> ../../nvme26n1p1
lrwxrwxrwx. 1 root root 14 Apr 26 16:47 pci-0000:06:00.0-nvme-1 -> ../../nvme27n1
lrwxrwxrwx. 1 root root 16 Apr 26 18:41 pci-0000:06:00.0-nvme-1-part1 -> ../../nvme27n1p1
lrwxrwxrwx. 1 root root 14 Apr 26 16:47 pci-0000:07:00.0-nvme-1 -> ../../nvme28n1
lrwxrwxrwx. 1 root root 16 Apr 26 18:41 pci-0000:07:00.0-nvme-1-part1 -> ../../nvme28n1p1
lrwxrwxrwx. 1 root root 14 Apr 26 16:47 pci-0000:08:00.0-nvme-1 -> ../../nvme29n1
lrwxrwxrwx. 1 root root 16 Apr 26 16:47 pci-0000:08:00.0-nvme-1-part1 -> ../../nvme29n1p1
lrwxrwxrwx. 1 root root 16 Apr 26 16:47 pci-0000:08:00.0-nvme-1-part2 -> ../../nvme29n1p2
lrwxrwxrwx. 1 root root 16 Apr 26 16:47 pci-0000:08:00.0-nvme-1-part3 -> ../../nvme29n1p3
lrwxrwxrwx. 1 root root 14 Apr 26 16:47 pci-0000:23:00.0-nvme-1 -> ../../nvme16n1
lrwxrwxrwx. 1 root root 16 Apr 26 18:41 pci-0000:23:00.0-nvme-1-part1 -> ../../nvme16n1p1
lrwxrwxrwx. 1 root root 14 Apr 26 16:47 pci-0000:2a:00.0-nvme-1 -> ../../nvme17n1
lrwxrwxrwx. 1 root root 16 Apr 26 18:41 pci-0000:2a:00.0-nvme-1-part1 -> ../../nvme17n1p1
lrwxrwxrwx. 1 root root 14 Apr 26 16:47 pci-0000:2b:00.0-nvme-1 -> ../../nvme18n1
lrwxrwxrwx. 1 root root 16 Apr 26 18:41 pci-0000:2b:00.0-nvme-1-part1 -> ../../nvme18n1p1
lrwxrwxrwx. 1 root root 14 Apr 26 16:47 pci-0000:2c:00.0-nvme-1 -> ../../nvme19n1
lrwxrwxrwx. 1 root root 16 Apr 26 18:41 pci-0000:2c:00.0-nvme-1-part1 -> ../../nvme19n1p1
lrwxrwxrwx. 1 root root 14 Apr 26 16:47 pci-0000:2d:00.0-nvme-1 -> ../../nvme20n1
lrwxrwxrwx. 1 root root 16 Apr 26 18:41 pci-0000:2d:00.0-nvme-1-part1 -> ../../nvme20n1p1
lrwxrwxrwx. 1 root root 14 Apr 26 16:47 pci-0000:2e:00.0-nvme-1 -> ../../nvme30n1
lrwxrwxrwx. 1 root root 16 Apr 26 18:41 pci-0000:2e:00.0-nvme-1-part1 -> ../../nvme30n1p1
lrwxrwxrwx. 1 root root 14 Apr 26 16:47 pci-0000:2f:00.0-nvme-1 -> ../../nvme21n1
lrwxrwxrwx. 1 root root 16 Apr 26 18:41 pci-0000:2f:00.0-nvme-1-part1 -> ../../nvme21n1p1
lrwxrwxrwx. 1 root root 13 Apr 26 16:47 pci-0000:41:00.0-nvme-1 -> ../../nvme7n1
lrwxrwxrwx. 1 root root 15 Apr 26 18:40 pci-0000:41:00.0-nvme-1-part1 -> ../../nvme7n1p1
lrwxrwxrwx. 1 root root 13 Apr 26 16:47 pci-0000:42:00.0-nvme-1 -> ../../nvme9n1
lrwxrwxrwx. 1 root root 15 Apr 26 18:40 pci-0000:42:00.0-nvme-1-part1 -> ../../nvme9n1p1
lrwxrwxrwx. 1 root root 14 Apr 26 16:47 pci-0000:43:00.0-nvme-1 -> ../../nvme10n1
lrwxrwxrwx. 1 root root 16 Apr 26 18:40 pci-0000:43:00.0-nvme-1-part1 -> ../../nvme10n1p1
lrwxrwxrwx. 1 root root 14 Apr 26 16:47 pci-0000:44:00.0-nvme-1 -> ../../nvme11n1
lrwxrwxrwx. 1 root root 16 Apr 26 18:40 pci-0000:44:00.0-nvme-1-part1 -> ../../nvme11n1p1
lrwxrwxrwx. 1 root root 14 Apr 26 16:47 pci-0000:45:00.0-nvme-1 -> ../../nvme12n1
lrwxrwxrwx. 1 root root 16 Apr 26 18:40 pci-0000:45:00.0-nvme-1-part1 -> ../../nvme12n1p1
lrwxrwxrwx. 1 root root 14 Apr 26 16:47 pci-0000:46:00.0-nvme-1 -> ../../nvme15n1
lrwxrwxrwx. 1 root root 16 Apr 26 18:40 pci-0000:46:00.0-nvme-1-part1 -> ../../nvme15n1p1
lrwxrwxrwx. 1 root root 14 Apr 26 16:47 pci-0000:47:00.0-nvme-1 -> ../../nvme13n1
lrwxrwxrwx. 1 root root 16 Apr 26 18:40 pci-0000:47:00.0-nvme-1-part1 -> ../../nvme13n1p1
lrwxrwxrwx. 1 root root 14 Apr 26 16:47 pci-0000:48:00.0-nvme-1 -> ../../nvme14n1
lrwxrwxrwx. 1 root root 16 Apr 26 18:40 pci-0000:48:00.0-nvme-1-part1 -> ../../nvme14n1p1
lrwxrwxrwx. 1 root root 13 Apr 26 16:47 pci-0000:61:00.0-nvme-1 -> ../../nvme0n1
lrwxrwxrwx. 1 root root 15 Apr 26 18:40 pci-0000:61:00.0-nvme-1-part1 -> ../../nvme0n1p1
lrwxrwxrwx. 1 root root 13 Apr 26 16:47 pci-0000:62:00.0-nvme-1 -> ../../nvme1n1
lrwxrwxrwx. 1 root root 15 Apr 26 18:40 pci-0000:62:00.0-nvme-1-part1 -> ../../nvme1n1p1
lrwxrwxrwx. 1 root root 13 Apr 26 16:47 pci-0000:63:00.0-nvme-1 -> ../../nvme2n1
lrwxrwxrwx. 1 root root 15 Apr 26 18:40 pci-0000:63:00.0-nvme-1-part1 -> ../../nvme2n1p1
lrwxrwxrwx. 1 root root 13 Apr 26 16:47 pci-0000:64:00.0-nvme-1 -> ../../nvme3n1
lrwxrwxrwx. 1 root root 15 Apr 26 18:40 pci-0000:64:00.0-nvme-1-part1 -> ../../nvme3n1p1
lrwxrwxrwx. 1 root root 13 Apr 26 16:47 pci-0000:65:00.0-nvme-1 -> ../../nvme4n1
lrwxrwxrwx. 1 root root 15 Apr 26 18:40 pci-0000:65:00.0-nvme-1-part1 -> ../../nvme4n1p1
lrwxrwxrwx. 1 root root 13 Apr 26 16:47 pci-0000:66:00.0-nvme-1 -> ../../nvme5n1
lrwxrwxrwx. 1 root root 15 Apr 26 18:40 pci-0000:66:00.0-nvme-1-part1 -> ../../nvme5n1p1
lrwxrwxrwx. 1 root root 13 Apr 26 16:47 pci-0000:67:00.0-nvme-1 -> ../../nvme6n1
lrwxrwxrwx. 1 root root 15 Apr 26 18:40 pci-0000:67:00.0-nvme-1-part1 -> ../../nvme6n1p1
lrwxrwxrwx. 1 root root 13 Apr 26 16:47 pci-0000:68:00.0-nvme-1 -> ../../nvme8n1
lrwxrwxrwx. 1 root root 15 Apr 26 18:40 pci-0000:68:00.0-nvme-1-part1 -> ../../nvme8n1p1
```

The UUID of a device does not change unless a new file system is created and is therefore used by `fstab` to mount devices. Running `cat /etc/fstab`, we get
```
UUID=82ed51b7-30f8-465f-94ec-1033845250ee /                       xfs     defaults        0 0
UUID=2180ba9d-a163-4f4b-821d-d000610739e6 /boot                   xfs     defaults        0 0
UUID=CB9D-7B90          /boot/efi               vfat    umask=0077,shortname=winnt 0 2
```
(FAT file systems do not support UUID, hence the shorter UID for `/boot/efi`.)

This ensures that the SSD on which the operating system is installed can always be recognized by the OS across boots.

For our purposes, we care mostly about the physical location on the motherboard/PCIe splitter each device is mounted on because they affect the performance: we want to divide work evenly among root complexes. Therefore, when we mount these devices, it's better to mount them in the order they appeared in `/dev/disk/by-path` instead of following their volatile numbering in `/dev/nvme<number>n1`. Another consideration is that in case we need to replace a failed drive, going by the `pci` device path will involve minimal changes on our end: the UUID might change because it's a new SSD with a new file system, but the physical `pci` port does not.
