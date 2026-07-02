cat > ramdisk_down.sh << 'EOF'
#!/bin/bash
for i in $(seq 0 29); do sudo umount /mnt/ssd$i 2>/dev/null; done
sudo rmmod brd
echo "ramdisks removed"
EOF
chmod +x ramdisk_down.sh