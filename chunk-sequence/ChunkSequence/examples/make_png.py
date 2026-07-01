#!/usr/bin/env python3
"""Convert chunk_ray_meta.txt + SSD chunk files → render.png.

Usage:
    python3 Plaidlay/fun_examples/make_png.py [meta_file] [out_png]

Defaults: chunk_ray_meta.txt, render.png
Requires: numpy, Pillow  (pip install numpy pillow)
"""

import sys
import numpy as np
from PIL import Image

# Pillow's default decompression bomb limit (~178 MP) is below our max render
# size (~1.3 GP at 5 GB SSD budget). Disable it — we control the input.
Image.MAX_IMAGE_PIXELS = None

meta_path = sys.argv[1] if len(sys.argv) > 1 else "chunk_ray_meta.txt"
out_path  = sys.argv[2] if len(sys.argv) > 2 else "render.png"

with open(meta_path) as f:
    lines = [l.rstrip("\n") for l in f if l.strip()]

W, H = map(int, lines[0].split())

chunks = []
for line in lines[1:]:
    parts = line.split()
    idx   = int(parts[0])
    fname = parts[1]
    addr  = int(parts[2])
    used  = int(parts[3])
    chunks.append((idx, fname, addr, used))

chunks.sort()  # ascending chunk index = logical pixel order

total_pixels = sum(c[3] // 4 for c in chunks)
assert total_pixels == W * H, f"pixel count mismatch: {total_pixels} != {W*H}"

rgb = np.empty(W * H * 3, dtype=np.uint8)
out_off = 0
for (_, fname, addr, used) in chunks:
    n = used // 4   # number of uint32_t pixels in this chunk
    with open(fname, "rb") as f:
        f.seek(addr)
        raw = f.read(used)
    # Zero-copy view as uint32, then vectorised channel extraction.
    packed = np.frombuffer(raw, dtype=np.uint32)[:n]
    r = ((packed >> 16) & 0xFF).astype(np.uint8)
    g = ((packed >>  8) & 0xFF).astype(np.uint8)
    b = ( packed        & 0xFF).astype(np.uint8)
    rgb[out_off:out_off + 3*n] = np.stack([r, g, b], axis=1).ravel()
    out_off += 3 * n

arr = rgb.reshape(H, W, 3)
Image.fromarray(arr, "RGB").save(out_path)
print(f"Saved {out_path} ({W}x{H})")
