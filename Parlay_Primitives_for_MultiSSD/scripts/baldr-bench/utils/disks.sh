#!/usr/bin/bash

declare -a disks=(
    "/dev/disk/by-path/pci-0000:01:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:02:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:03:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:04:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:05:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:06:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:07:00.0-nvme-1"
    # this is where the OS resides; hopefully we don't switch around the PCIE splitters
    # /dev/disk/by-path/pci-0000:08:00.0-nvme-1
    # /dev/disk/by-path/pci-0000:08:00.0-nvme-1-part2
    # /dev/disk/by-path/pci-0000:08:00.0-nvme-1-part3
    # "/dev/disk/by-path/pci-0000:23:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:2a:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:2b:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:2c:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:2d:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:2e:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:2f:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:41:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:42:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:43:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:44:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:45:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:46:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:47:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:48:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:61:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:62:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:63:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:64:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:65:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:66:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:67:00.0-nvme-1"
    "/dev/disk/by-path/pci-0000:68:00.0-nvme-1"
)
export disks
