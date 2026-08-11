#!/usr/bin/env python3
"""Conceptual diagram: Peter's sequential disk striping vs. chunk_seq's hashed
chunk placement.  Not data-driven (no CSV input, no CLI sweep) -- it renders
synthetic layout data purely to illustrate the two placement strategies.

Untitled variant of disk_layout_diagram.py for contexts that supply their own
heading (e.g. a slide with its own title placeholder): identical diagrams,
but the figure-level `fig.suptitle` ("Sequential Striping" / "Hashed
Chunking") is dropped. The in-panel descriptive caption (e.g. "Contiguous
Within Disk, Ordered by Sequence Position") is kept -- that text is part of
the diagram itself (it lives inside the axes, next to the colorbar legend
and the in-memory-array sub-panel), not a headline, so it stays even though
the outer title goes.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, to_rgba
from matplotlib.lines import Line2D

# Light-mode chart palette (see the dataviz skill's reference palette).
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
BASELINE = "#c3c2b7"

# Color encodes a chunk/range's original position in the logical sequence,
# consistently across both images: a discrete monochrome blue ramp, light to
# dark, so lighter == earlier and darker == later -- a genuinely
# sequential/ordinal encoding rather than a decorative multi-hue one,
# matching the blue steps in the dataviz skill's reference palette.
DISCRETE_STEPS = [
    "#86b6ef",  # step 250 -- lightest (earliest position)
    "#5598e7",  # step 350
    "#2a78d6",  # step 450
    "#1c5cab",  # step 550
    "#104281",  # step 650
    "#0d366b",  # step 700 -- darkest (latest position)
]
N_DISCRETE = len(DISCRETE_STEPS)


def seq_color(i, n):
    """Discrete academic-palette color for element i of n, bucketed into
    N_DISCRETE equal-width buckets of the logical sequence (monotonic, so
    elements far apart in i land in different/ordered buckets)."""
    bucket = min(int(i / n * N_DISCRETE), N_DISCRETE - 1)
    return DISCRETE_STEPS[bucket]

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


def style_panel(ax, caption, ylim=(-0.85, 1.34)):
    ax.set_xlim(-0.05, group_width() + 0.05)
    ax.set_ylim(*ylim)
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
    rows = 200
    # Lighter near the top of a disk, darker near the bottom -- an HSV "V"
    # ramp layered on top of the disk's flat hue, so the within-disk gradient
    # reads clearly. Scaling only V leaves hue/saturation -- i.e. the
    # red-to-blue color scheme -- untouched.
    depth = np.linspace(0, 1, rows)
    v_scale = (1.25 - 0.55 * depth).reshape(rows, 1)
    for d in range(N_DISKS):
        x0 = col_x0(d)
        # This disk holds the d-th contiguous 1/N_DISKS slice of the logical
        # sequence, so its column continues exactly where the previous disk's
        # column left off -- position order == disk order. N_DISKS ==
        # N_DISCRETE, so each disk gets exactly one discrete hue.
        rgba = np.tile(np.array(to_rgba(DISCRETE_STEPS[d])), (rows, 1))
        hsv = rgb_to_hsv(rgba[:, :3])
        hsv[:, 2:3] = np.clip(hsv[:, 2:3] * v_scale, 0, 1)
        rgba[:, :3] = hsv_to_rgb(hsv)
        img = rgba.reshape(rows, 1, 4)
        ax.imshow(img, aspect="auto",
                  extent=(x0, x0 + COL_WIDTH, 0, 1), origin="upper", zorder=1)
        draw_disk_frame(ax, x0, f"Disk {d}")
    style_panel(
        ax,
        "Contiguous Within Disk, Ordered by Sequence Position",
        ylim=(-0.18, 1.34),
    )


def draw_chunkseq_panel(ax):
    # Simulates parlay::hash64(slot) % num_drives: chunk order is scrambled by
    # a deterministic hash, then dealt round-robin so drive load stays
    # balanced (balls-in-bins) while original-position order is destroyed.
    order = np.random.default_rng(7).permutation(N_CHUNKS)
    disks = [[] for _ in range(N_DISKS)]
    for i, chunk_idx in enumerate(order):
        disks[i % N_DISKS].append(chunk_idx)

    tile_gap = 0.025
    tile_h = (1.0 - tile_gap * (CHUNKS_PER_DISK - 1)) / CHUNKS_PER_DISK
    for d in range(N_DISKS):
        x0 = col_x0(d)
        for t, chunk_idx in enumerate(disks[d]):
            y0 = 1 - (t + 1) * tile_h - t * tile_gap
            ax.add_patch(mpatches.Rectangle(
                (x0, y0), COL_WIDTH, tile_h,
                facecolor=seq_color(chunk_idx, N_CHUNKS), edgecolor=SURFACE,
                linewidth=0.8,
                zorder=1,
            ))
        draw_disk_frame(ax, x0, f"Disk {d}")
    style_panel(
        ax,
        "Non-contiguous Across Disks, Contiguous Chunks"
        "",
    )
    draw_chunks_array(ax)

    # Dashed divider between the external/on-disk tiles above and the
    # in-memory chunk_seq.chunks array below.
    ax.add_line(Line2D(
        [0, group_width()], [-0.12, -0.12],
        color=INK_MUTED, linewidth=1.4, linestyle=(0, (4, 3)), zorder=10,
    ))


def draw_chunks_array(ax):
    # Depicts chunk_seq.chunks: the in-memory, index-ordered accessor -- drawn
    # as a plain array so it visually contrasts with the scrambled disks above
    # (same chunks, same colors, but back in original index order: chunks[i]
    # is a direct, no-I/O lookup regardless of which physical disk holds it).
    ax.text(group_width() / 2, -0.18,
            "In-Memory Representation",
            ha="center", va="top", fontsize=10, color=INK_PRIMARY,
            fontweight="bold")

    array_top, array_bot = -0.30, -0.48
    cell_w = group_width() / N_CHUNKS
    for i in range(N_CHUNKS):
        x0 = i * cell_w
        ax.add_patch(mpatches.Rectangle(
            (x0, array_bot), cell_w, array_top - array_bot,
            facecolor=seq_color(i, N_CHUNKS), edgecolor=SURFACE,
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
    step_w = 1.0 / N_DISCRETE
    for k, color in enumerate(DISCRETE_STEPS):
        ax.add_patch(mpatches.Rectangle(
            (k * step_w, 0), step_w, 1,
            facecolor=color, edgecolor=SURFACE, linewidth=1.5,
        ))
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


def build_single_panel_figure(draw_panel, figsize=(7.5, 7)):
    # No fig.suptitle here (see module docstring) -- the titled original
    # passed a `title` straight to fig.suptitle; this variant drops that
    # param entirely rather than passing an empty string, so
    # constrained_layout doesn't reserve title space at all.
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    # Colorbar first (top row) so the legend is seen before the panel it
    # explains; the panel itself is the larger bottom row.
    gs = fig.add_gridspec(2, 1, height_ratios=[0.14, 1], hspace=0.05)

    cax = fig.add_subplot(gs[0])
    cax.set_facecolor(SURFACE)
    ax = fig.add_subplot(gs[1])

    draw_colorbar(cax)
    draw_panel(ax)

    return fig


def build_sequential_figure():
    return build_single_panel_figure(draw_peter_panel, figsize=(7.5, 4.9))


def build_hashed_figure():
    return build_single_panel_figure(draw_chunkseq_panel, figsize=(7.5, 6.5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-prefix", default="disk_layout_diagram",
        help="output PNG basename prefix; writes "
             "<prefix>_sequential_notitle.png and "
             "<prefix>_hashed_notitle.png (default: %(default)s)")
    args = parser.parse_args()

    for suffix, build in (
        ("sequential", build_sequential_figure),
        ("hashed", build_hashed_figure),
    ):
        out = f"{args.out_prefix}_{suffix}_notitle.png"
        fig = build()
        fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
