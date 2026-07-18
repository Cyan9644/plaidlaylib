#!/usr/bin/env python3
"""Conceptual diagram: Peter's sequential disk striping vs. chunk_seq's hashed
chunk placement.  Not data-driven (no CSV input, no CLI sweep) -- it renders
synthetic layout data purely to illustrate the two placement strategies.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# Light-mode chart palette (see the dataviz skill's reference palette).
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
BASELINE = "#c3c2b7"

# Sequential blue ramp (steps 100->700): color encodes a chunk/range's original
# position in the logical sequence, consistently across both panels.
_BLUE_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", _BLUE_STEPS)

N_DISKS = 6
CHUNKS_PER_DISK = 5
N_CHUNKS = N_DISKS * CHUNKS_PER_DISK

COL_GAP = 0.10
COL_WIDTH = (1.0 - COL_GAP * (N_DISKS - 1)) / N_DISKS


def col_x0(d):
    return d * (COL_WIDTH + COL_GAP)


def group_width():
    return N_DISKS * (COL_WIDTH + COL_GAP) - COL_GAP


def draw_disk_frame(ax, x0, label):
    frame = mpatches.FancyBboxPatch(
        (x0, 0), COL_WIDTH, 1,
        boxstyle="round,pad=0.0,rounding_size=0.05",
        linewidth=1.2, edgecolor=BASELINE, facecolor="none",
        zorder=5, mutation_aspect=1,
    )
    ax.add_patch(frame)
    ax.text(x0 + COL_WIDTH / 2, -0.06, label, ha="center", va="top",
            fontsize=9.5, color=INK_PRIMARY)


def style_panel(ax, caption):
    ax.set_xlim(-0.05, group_width() + 0.05)
    ax.set_ylim(-0.85, 1.34)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(BASELINE)
        spine.set_linewidth(1.3)
    ax.text(group_width() / 2, 1.30, caption, ha="center", va="top",
            fontsize=9.5, color=INK_PRIMARY, wrap=True)


def draw_peter_panel(ax):
    ax.set_title("Sequential Striping", fontsize=13.5,
                 color=INK_PRIMARY, pad=16, fontweight="bold")
    rows = 200
    for d in range(N_DISKS):
        x0 = col_x0(d)
        # This disk holds the d-th contiguous 1/N_DISKS slice of the logical
        # sequence, so its column continues exactly where the previous disk's
        # column left off -- position order == disk order.
        seg = np.linspace(d / N_DISKS, (d + 1) / N_DISKS, rows).reshape(rows, 1)
        ax.imshow(seg, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto",
                  extent=(x0, x0 + COL_WIDTH, 0, 1), origin="upper", zorder=1)
        draw_disk_frame(ax, x0, f"Disk {d}")
    style_panel(
        ax,
        "Contiguous Within Disk, Ordered by Sequence Position",
    )


def draw_chunkseq_panel(ax):
    ax.set_title("Hashed Chunking", fontsize=13.5,
                 color=INK_PRIMARY, pad=16, fontweight="bold")

    # Simulates parlay::hash64(slot) % num_drives: chunk order is scrambled by
    # a deterministic hash, then dealt round-robin so drive load stays
    # balanced (balls-in-bins) while original-position order is destroyed.
    order = np.random.default_rng(7).permutation(N_CHUNKS)
    disks = [[] for _ in range(N_DISKS)]
    for i, chunk_idx in enumerate(order):
        disks[i % N_DISKS].append(chunk_idx / N_CHUNKS)

    tile_gap = 0.025
    tile_h = (1.0 - tile_gap * (CHUNKS_PER_DISK - 1)) / CHUNKS_PER_DISK
    for d in range(N_DISKS):
        x0 = col_x0(d)
        for t, val in enumerate(disks[d]):
            y0 = 1 - (t + 1) * tile_h - t * tile_gap
            ax.add_patch(mpatches.Rectangle(
                (x0, y0), COL_WIDTH, tile_h,
                facecolor=SEQ_CMAP(val), edgecolor=SURFACE, linewidth=0.8,
                zorder=1,
            ))
        draw_disk_frame(ax, x0, f"Disk {d}")
    style_panel(
        ax,
        "Non-contiguous Across Disks, Contiguous Chunks"
        "",
    )
    draw_chunks_array(ax)


def draw_chunks_array(ax):
    # Depicts chunk_seq.chunks: the in-memory, index-ordered accessor -- drawn
    # as a plain array so it visually contrasts with the scrambled disks above
    # (same chunks, same colors, but back in original index order: chunks[i]
    # is a direct, no-I/O lookup regardless of which physical disk holds it).
    ax.text(group_width() / 2, -0.18,
            "chunk_seq.chunks (std::vector<chunk>)",
            ha="center", va="top", fontsize=10, color=INK_PRIMARY,
            fontweight="bold")

    array_top, array_bot = -0.30, -0.48
    cell_w = group_width() / N_CHUNKS
    for i in range(N_CHUNKS):
        x0 = i * cell_w
        ax.add_patch(mpatches.Rectangle(
            (x0, array_bot), cell_w, array_top - array_bot,
            facecolor=SEQ_CMAP(i / N_CHUNKS), edgecolor=SURFACE,
            linewidth=0.8, zorder=2,
        ))
    ax.add_patch(mpatches.Rectangle(
        (0, array_bot), group_width(), array_top - array_bot,
        facecolor="none", edgecolor=BASELINE, linewidth=1.2, zorder=3,
    ))

    for i in (0, N_CHUNKS // 2, N_CHUNKS - 1):
        ax.text(i * cell_w + cell_w / 2, array_bot - 0.03, str(i),
                ha="center", va="top", fontsize=8, color=INK_PRIMARY)

    ax.text(
        group_width() / 2, -0.58,
        "Allows for efficient cut, flatten, and other operations without external access\n",
        ha="center", va="top", fontsize=9, color=INK_PRIMARY,
    )


def draw_colorbar(ax):
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, cmap=SEQ_CMAP, aspect="auto",
              extent=(0, 1, 0, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["sequence start", "sequence end"])
    ax.tick_params(colors=INK_MUTED, length=0, labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, 1.9, "",
            ha="center", va="bottom", fontsize=9.5, color=INK_SECONDARY,
            transform=ax.transAxes)


def build_figure():
    fig = plt.figure(figsize=(13, 6.8), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.10], hspace=0.05, wspace=0.14)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[1, :])
    cax.set_facecolor(SURFACE)

    draw_peter_panel(ax1)
    draw_chunkseq_panel(ax2)
    draw_colorbar(cax)

    fig.suptitle(
        "Sequence Representation",
        fontsize=15, color=INK_PRIMARY, fontweight="bold",
    )

    # Vertical divider between the two panels, placed at the actual midpoint
    # of their figure-coordinate bounding boxes once constrained_layout settles.
    fig.canvas.draw()
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    mid_x = (pos1.x1 + pos2.x0) / 2
    fig.add_artist(Line2D(
        [mid_x, mid_x], [pos2.y0 - 0.02, pos1.y1 + 0.01],
        transform=fig.transFigure, color=INK_MUTED, linewidth=1.4,
        linestyle=(0, (4, 3)), zorder=10,
    ))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="disk_layout_diagram.png",
                         help="output PNG path (default: %(default)s)")
    args = parser.parse_args()

    fig = build_figure()
    fig.savefig(args.out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
