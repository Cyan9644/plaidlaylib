#!/usr/bin/env python3
"""Conceptual diagrams: parallel collection operations over an in-memory
parlay::sequence<T>.  Not data-driven (no CSV input, no CLI sweep) -- it
renders synthetic layout data purely to illustrate the primitives.  Writes two
independent single-panel PNGs (sequence, map); shares its palette and
single-panel layout with figures/disk_layout_diagram.py so the figures read
as one set.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Light-mode chart palette (see the dataviz skill's reference palette;
# matches figures/disk_layout_diagram.py).
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
BASELINE = "#c3c2b7"

# Color encodes an element's original position in the logical sequence,
# consistently across panels. A red-to-blue ramp (through dusty rose, mauve,
# plum, and indigo) shifts noticeably from end to end, but stays low-chroma
# throughout -- no saturated/neon stop anywhere on the ramp -- so it stays
# easy on the eyes.
_RED_BLUE_STEPS = [
    "#e6c2c0", "#d8a6a6", "#c78d94", "#b17587", "#976480",
    "#7c5678", "#614d6e", "#4a4869", "#3a4568", "#2f4166",
    "#293c60", "#243655", "#1f2f48",
]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_red_blue", _RED_BLUE_STEPS)


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


def style_panel(ax, caption, ylim):
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(BASELINE)
        spine.set_linewidth(1.3)
    ax.text(0.5, ylim[1] - 0.04, caption, ha="center", va="top",
            fontsize=10.5, color=INK_PRIMARY, fontweight="bold", wrap=True)


def draw_parlay_panel(ax):
    ax.set_title("parlay::sequence<T>", fontsize=13.5,
                 color=INK_PRIMARY, pad=16, fontweight="bold")

    box_x0, box_y0, box_w, box_h = 0.08, -0.30, 0.84, 0.60
    dram_box(ax, box_x0, box_y0, box_w, box_h, "DRAM")

    n_cells = 48
    cell_w = (box_w - 0.06) / n_cells
    y0, cell_h = box_y0 + 0.08, box_h - 0.16
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

    ax.text(0.5, box_y0 - 0.14,
            "Physically Contiguous",
            ha="center", va="top", fontsize=10, color=INK_PRIMARY,
            fontweight="bold")

    style_panel(ax, "Sequence lives in DRAM", ylim=(-0.85, 0.60))


def draw_map_panel(ax):
    ax.set_title("parlay::map", fontsize=13, color=INK_PRIMARY, pad=12,
                 fontweight="bold")

    n = 12
    x0s, cell_w = 0.08, 0.84 / n
    in_y, out_y, cell_h = 0.46, -0.18, 0.20

    for i in range(n):
        x0 = x0s + i * cell_w
        ax.add_patch(mpatches.Rectangle(
            (x0, in_y), cell_w * 0.9, cell_h,
            facecolor=SEQ_CMAP(i / n), edgecolor=SURFACE,
            linewidth=0.5, zorder=2,
        ))
        ax.add_patch(mpatches.Rectangle(
            (x0, out_y), cell_w * 0.9, cell_h,
            facecolor=SEQ_CMAP(i / n), edgecolor=SURFACE,
            linewidth=0.5, zorder=2,
        ))
    ax.text(0.03, in_y + cell_h / 2, "in", ha="right", va="center",
            fontsize=8.5, color=INK_MUTED)
    ax.text(0.03, out_y + cell_h / 2, "out", ha="right", va="center",
            fontsize=8.5, color=INK_MUTED)

    # A few representative arrows (not one per cell) stand in for n
    # independent, parallel applications of f.
    for i in (0, n // 2, n - 1):
        cx = x0s + (i + 0.45) * cell_w
        ax.annotate(
            "", xy=(cx, out_y + cell_h + 0.02), xytext=(cx, in_y - 0.02),
            arrowprops=dict(arrowstyle="-|>", color=INK_MUTED, linewidth=1.2,
                            shrinkA=0, shrinkB=0),
            zorder=4,
        )
        ax.text(cx + 0.02, (in_y + out_y + cell_h) / 2, "f",
                ha="left", va="center", fontsize=9, color=INK_SECONDARY,
                fontstyle="italic")

    ax.text(0.5, in_y + cell_h + 0.10,
            "Apply a function f in parallel to each element",
            ha="center", va="bottom", fontsize=9.5, color=INK_PRIMARY,
            fontweight="bold")

    style_panel(
        ax, "map(sequence) --> sequence",
        ylim=(-0.45, 0.95),
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


def build_single_panel_figure(title, draw_panel, figsize=(7.5, 7)):
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

    fig.suptitle(title, fontsize=15, color=INK_PRIMARY, fontweight="bold")
    return fig


def build_sequence_figure():
    return build_single_panel_figure(
        "Sequence Representation", draw_parlay_panel, figsize=(7.5, 4.6))


def build_map_figure():
    return build_single_panel_figure(
        "The Map Primitive", draw_map_panel, figsize=(7.5, 4.4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-prefix", default="parlay_operations",
        help="output PNG basename prefix; writes "
             "<prefix>_sequence.png and <prefix>_map.png "
             "(default: %(default)s)")
    args = parser.parse_args()

    for suffix, build in (
        ("sequence", build_sequence_figure),
        ("map", build_map_figure),
    ):
        out = f"{args.out_prefix}_{suffix}.png"
        fig = build()
        fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
