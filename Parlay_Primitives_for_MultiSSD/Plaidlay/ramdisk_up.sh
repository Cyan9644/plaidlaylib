# #!/bin/bash
# set -e
# sudo modprobe brd rd_nr=30 rd_size=$((2*1024*1024))   # 30 devices, 2 GiB cap each
# for i in $(seq 0 29); do
#   sudo mkfs.ext4 -qF /dev/ram$i
#   sudo mkdir -p /mnt/ssd$i
#   sudo mount -o noatime /dev/ram$i /mnt/ssd$i
#   sudo chown "$USER" /mnt/ssd$i
# done
# echo "30 ramdisks mounted at /mnt/ssd0..29"

#!/bin/bash
set -e

# RAM-backed pool to hold the backing files (only allocates pages as used)
sudo mkdir -p /mnt/rampool
sudo mount -t tmpfs -o size=64G tmpfs /mnt/rampool

for i in $(seq 0 29); do
  truncate -s 2G /mnt/rampool/disk$i.img
  dev=$(sudo losetup -f --show /mnt/rampool/disk$i.img)   # auto-creates loop dev
  sudo mkfs.ext4 -qF "$dev"
  sudo mkdir -p /mnt/ssd$i
  sudo mount -o noatime "$dev" /mnt/ssd$i
  sudo chown "$USER" /mnt/ssd$i
done
echo "30 ext4 ramdisks mounted at /mnt/ssd0..29"
