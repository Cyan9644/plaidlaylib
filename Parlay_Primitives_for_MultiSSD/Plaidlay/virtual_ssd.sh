#one of the great mysteries of the universe
cat > ramdisk_up.sh << 'EOF'
#!/bin/bash
set -e
sudo modprobe brd rd_nr=30 rd_size=$((2*1024*1024))   # 30 devices, 2 GiB cap each
for i in $(seq 0 29); do
  sudo mkfs.ext4 -qF /dev/ram$i
  sudo mkdir -p /mnt/ssd$i
  sudo mount -o noatime /dev/ram$i /mnt/ssd$i
  sudo chown "$USER" /mnt/ssd$i
done
echo "30 ramdisks mounted at /mnt/ssd0..29"
EOF
chmod +x ramdisk_up.sh