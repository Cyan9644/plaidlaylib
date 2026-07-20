#!/usr/bin/env python3
"""Conceptual diagram: parallel collection operations (map, reduce) over an
in-memory parlay::sequence<T>.  Not data-driven (no CSV input, no CLI sweep)
-- it renders synthetic layout data purely to illustrate the two primitives.
Sibling to parlay_vs_chunkseq.py / disk_layout_diagram.py; shares their
palette so the three figures read as one set.
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
# matches figures/parlay_vs_chunkseq.py / disk_layout_diagram.py).
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
BASELINE = "#c3c2b7"

# Sequential blue ramp (steps 100->700): color encodes an element's original
# position in the logical sequence, consistently across panels.
_BLUE_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", _BLUE_STEPS)


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
    ax.text(0.5, box_y0 - 0.34,
            "Stored in DRAM",
            ha="center", va="top", fontsize=9.5, color=INK_PRIMARY,
            fontweight="bold")

    style_panel(ax, "Sequence lives in DRAM", ylim=(-2.05, 1.35))


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
        ylim=(-0.62, 0.95),
    )


def draw_reduce_panel(ax):
    ax.set_title("parlay::reduce", fontsize=13, color=INK_PRIMARY, pad=12,
                 fontweight="bold")

    n_leaves = 8
    x0s, span = 0.08, 0.84
    leaf_x = [x0s + (i + 0.5) * span / n_leaves for i in range(n_leaves)]
    levels_y = [0.55, 0.28, 0.01, -0.30]  # leaves, 4, 2, 1 (root)

    level_x = [leaf_x]
    for lvl in range(1, 4):
        prev = level_x[-1]
        level_x.append([(prev[2 * k] + prev[2 * k + 1]) / 2
                         for k in range(len(prev) // 2)])

    # Connectors, drawn before nodes so node markers sit on top.
    for lvl in range(1, 4):
        y_child, y_parent = levels_y[lvl - 1], levels_y[lvl]
        prev = level_x[lvl - 1]
        for k, px in enumerate(level_x[lvl]):
            for cx in (prev[2 * k], prev[2 * k + 1]):
                ax.add_line(Line2D(
                    [cx, px], [y_child, y_parent],
                    color=INK_MUTED, linewidth=1.1, zorder=1,
                ))

    # Leaves: small colored squares (position-encoded, same ramp as elsewhere).
    leaf_w, leaf_h = 0.045, 0.16
    for i, cx in enumerate(leaf_x):
        ax.add_patch(mpatches.Rectangle(
            (cx - leaf_w / 2, levels_y[0] - leaf_h / 2), leaf_w, leaf_h,
            facecolor=SEQ_CMAP(i / n_leaves), edgecolor=SURFACE,
            linewidth=0.5, zorder=3,
        ))

    # Internal combine nodes: neutral ink tone -- these are aggregate
    # values, not one original position.
    for lvl in (1, 2):
        for px in level_x[lvl]:
            ax.add_patch(mpatches.Circle(
                (px, levels_y[lvl]), 0.028,
                facecolor=INK_SECONDARY, edgecolor=SURFACE,
                linewidth=0.6, zorder=3,
            ))

    # Root result.
    root_x, root_y = level_x[3][0], levels_y[3]
    ax.add_patch(mpatches.Circle(
        (root_x, root_y), 0.055,
        facecolor=INK_PRIMARY, edgecolor=SURFACE, linewidth=0.8, zorder=4,
    ))
    ax.text(root_x, root_y, "r", ha="center", va="center",
            fontsize=10, color=SURFACE, fontweight="bold", zorder=5)

    # ax.text(0.5, levels_y[0] + leaf_h / 2 + 0.06,
    #         "monoid-combine pairs at each level, in parallel",
    #         ha="center", va="bottom", fontsize=8.5, color=INK_SECONDARY)

    style_panel(
        ax, "reduce(sequence, monoid) --> value",
        ylim=(-0.55, 0.92),
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


def build_figure():
    fig = plt.figure(figsize=(13, 8.4), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.08],
                           hspace=0.22, wspace=0.14)

    ax_left = fig.add_subplot(gs[0:2, 0])
    ax_map = fig.add_subplot(gs[0, 1])
    ax_reduce = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[2, :])
    cax.set_facecolor(SURFACE)

    draw_parlay_panel(ax_left)
    draw_map_panel(ax_map)
    draw_reduce_panel(ax_reduce)
    draw_colorbar(cax)

    fig.suptitle(
        "Objectives and Primitive Operations",
        fontsize=15, color=INK_PRIMARY, fontweight="bold",
    )

    fig.canvas.draw()
    pos_left = ax_left.get_position()
    pos_map = ax_map.get_position()
    pos_reduce = ax_reduce.get_position()

    mid_x = (pos_left.x1 + pos_map.x0) / 2
    fig.add_artist(Line2D(
        [mid_x, mid_x], [pos_reduce.y0 - 0.01, pos_left.y1 + 0.01],
        transform=fig.transFigure, color=INK_MUTED, linewidth=1.4,
        linestyle=(0, (4, 3)), zorder=10,
    ))
    mid_y = (pos_map.y0 + pos_reduce.y1) / 2
    fig.add_artist(Line2D(
        [mid_x, pos_map.x1], [mid_y, mid_y],
        transform=fig.transFigure, color=INK_MUTED, linewidth=1.1,
        linestyle=(0, (4, 3)), zorder=10,
    ))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="parlay_operations.png",
                         help="output PNG path (default: %(default)s)")
    args = parser.parse_args()

    fig = build_figure()
    fig.savefig(args.out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
