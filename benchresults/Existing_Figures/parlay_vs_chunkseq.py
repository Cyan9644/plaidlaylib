#!/usr/bin/env python3
"""Conceptual diagram: parlaylib's in-memory parlay::sequence<T> vs. this
project's out-of-core chunk_seq.  Not data-driven (no CSV input, no CLI
sweep) -- it renders synthetic layout data purely to illustrate the two
representations.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# Light-mode chart palette (see the dataviz skill's reference palette;
# matches figures/disk_layout_diagram.py so the two figures read as one set).
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
BASELINE = "#c3c2b7"

# Sequential blue ramp (steps 100->700): color encodes an element/chunk's
# original position in the logical sequence, consistently across both panels.
_BLUE_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", _BLUE_STEPS)

N_DISKS = 5
CHUNKS_PER_DISK = 4
N_CHUNKS = N_DISKS * CHUNKS_PER_DISK

COL_GAP = 0.12
COL_WIDTH = (1.0 - COL_GAP * (N_DISKS - 1)) / N_DISKS


def group_width():
    return N_DISKS * (COL_WIDTH + COL_GAP) - COL_GAP


def dram_box(ax, x0, y0, w, h, label):
    box = mpatches.FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.03",
        linewidth=1.3, edgecolor=INK_SECONDARY, facecolor="none",
        linestyle=(0, (4, 2.5)), zorder=6, mutation_aspect=1,
    )
    ax.add_patch(box)
    ax.text(x0 + w / 2, y0 + h + 0.05, label, ha="center", va="bottom",
            fontsize=10.5, color=INK_SECONDARY, fontweight="bold")


def style_panel(ax, caption):
    ax.set_xlim(-0.05, group_width() + 0.05)
    ax.set_ylim(-1.05, 1.34)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(BASELINE)
        spine.set_linewidth(1.3)
    ax.text(group_width() / 2, 1.30, caption, ha="center", va="top",
            fontsize=9.5, color=INK_PRIMARY, wrap=True)


def draw_parlay_panel(ax):
    ax.set_title("parlay::sequence<T>", fontsize=13.5,
                 color=INK_PRIMARY, pad=16, fontweight="bold")

    box_x0, box_y0, box_w, box_h = 0.06, -0.05, group_width() - 0.12, 0.62
    dram_box(ax, box_x0, box_y0, box_w, box_h, "DRAM")

    # One long contiguous row: every element is resident, in position order.
    n_cells = 48
    cell_w = (box_w - 0.06) / n_cells
    y0, cell_h = box_y0 + 0.10, box_h - 0.20
    for i in range(n_cells):
        x0 = box_x0 + 0.03 + i * cell_w
        ax.add_patch(mpatches.Rectangle(
            (x0, y0), cell_w, cell_h,
            facecolor=SEQ_CMAP(i / n_cells), edgecolor=SURFACE,
            linewidth=0.4, zorder=2,
        ))
    ax.add_patch(mpatches.Rectangle(
        (box_x0 + 0.03, y0), box_w - 0.06, cell_h,
        facecolor="none", edgecolor=INK_MUTED, linewidth=1.0, zorder=3,
    ))

    ax.text(group_width() / 2, box_y0 - 0.14,
            "T elements[n] -- contiguous, fully resident",
            ha="center", va="top", fontsize=10, color=INK_PRIMARY,
            fontweight="bold")
    ax.text(group_width() / 2, box_y0 - 0.34,
            "Whole sequence must fit in RAM --\ncapacity bounded by physical DRAM",
            ha="center", va="top", fontsize=9, color=INK_PRIMARY)

    style_panel(ax, "Sequence Lives Entirely in DRAM")


def draw_chunkseq_panel(ax):
    ax.set_title("chunk_seq", fontsize=13.5,
                 color=INK_PRIMARY, pad=16, fontweight="bold")

    # -- DRAM accessor: chunk_seq.chunks, one small cell per chunk header. --
    box_x0, box_y0 = 0.10, 0.62
    box_w = group_width() - 0.20
    box_h = 0.30
    dram_box(ax, box_x0, box_y0, box_w, box_h, "DRAM")

    idx_gap = 0.02
    idx_w = (box_w - 0.06 - idx_gap * (N_CHUNKS - 1)) / N_CHUNKS
    idx_y0, idx_h = box_y0 + 0.06, box_h - 0.12
    for i in range(N_CHUNKS):
        x0 = box_x0 + 0.03 + i * (idx_w + idx_gap)
        ax.add_patch(mpatches.Rectangle(
            (x0, idx_y0), idx_w, idx_h,
            facecolor=SEQ_CMAP(i / N_CHUNKS), edgecolor=SURFACE,
            linewidth=0.5, zorder=4,
        ))
    ax.add_patch(mpatches.Rectangle(
        (box_x0 + 0.03, idx_y0), box_w - 0.06, idx_h,
        facecolor="none", edgecolor=INK_MUTED, linewidth=1.0, zorder=5,
    ))
    ax.text(group_width() / 2, box_y0 - 0.09,
            "chunk_seq.chunks -- in-memory accessor: {filename, begin_addr,"
            " used, index} per chunk",
            ha="center", va="top", fontsize=8.5, color=INK_SECONDARY)

    # One arrow standing in for the accessor -> data indirection (drawn before
    # the storage block/label below so its head lands well clear of both).
    arrow_top, arrow_bot = 0.42, 0.19
    ax.annotate(
        "", xy=(group_width() / 2, arrow_bot),
        xytext=(group_width() / 2, arrow_top),
        arrowprops=dict(arrowstyle="-|>", color=INK_MUTED, linewidth=1.4,
                        shrinkA=0, shrinkB=0),
        zorder=6,
    )

    ax.text(group_width() / 2, 0.15,
            "SSD storage (out-of-core)", ha="center", va="top",
            fontsize=10.5, color=INK_PRIMARY, fontweight="bold")

    # -- SSD storage: one out-of-core store, deliberately drawn contiguous
    # (not fragmented) -- the placement-strategy detail lives in
    # disk_layout_diagram.py, not here.
    store_x0, store_y0 = 0.10, -0.48
    store_w = group_width() - 0.20
    store_h = 0.52

    n_cells = 40
    cell_w = store_w / n_cells
    for i in range(n_cells):
        x0 = store_x0 + i * cell_w
        ax.add_patch(mpatches.Rectangle(
            (x0, store_y0), cell_w, store_h,
            facecolor=SEQ_CMAP(i / n_cells), edgecolor=SURFACE,
            linewidth=0.4, zorder=2,
        ))
    ax.add_patch(mpatches.FancyBboxPatch(
        (store_x0, store_y0), store_w, store_h,
        boxstyle="round,pad=0.0,rounding_size=0.03",
        linewidth=1.2, edgecolor=INK_MUTED, facecolor="none", zorder=3,
    ))

    ax.text(group_width() / 2, store_y0 - 0.08,
            "Element data lives out-of-core on SSD, not in DRAM --\n"
            "total size scales past RAM capacity",
            ha="center", va="top", fontsize=9, color=INK_PRIMARY)

    style_panel(ax, "Accessor in DRAM, Data Lives Out-of-Core on SSD")


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


def build_figure():
    fig = plt.figure(figsize=(13, 7.4), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.09], hspace=0.05, wspace=0.14)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[1, :])
    cax.set_facecolor(SURFACE)

    draw_parlay_panel(ax1)
    draw_chunkseq_panel(ax2)
    draw_colorbar(cax)

    fig.suptitle(
        "In-Memory vs. Out-of-Core Sequence Representation",
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
    parser.add_argument("--out", default="parlay_vs_chunkseq.png",
                         help="output PNG path (default: %(default)s)")
    args = parser.parse_args()

    fig = build_figure()
    fig.savefig(args.out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
