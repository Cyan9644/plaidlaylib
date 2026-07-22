#!/usr/bin/env python3
"""Conceptual diagram: overlapping reads, compute, and writes in
ExternalTransform (ChunkSequence/external_engine.h). Not data-driven (no CSV
input, no CLI sweep) -- it renders a synthetic pipeline snapshot purely to
illustrate the mechanism. Shares its palette and single-panel conventions
with figures/disk_layout_diagram.py and figures/parlay_operations.py so the
figures read as one set.

DRAM is drawn as a single flat vector of cells (literally one row), divided
into a read-staging zone, a compute zone, and a write-staging zone. Several
chunks are shown simultaneously at different pipeline stages -- idle at an
input SSD, mid-read, landed in read-staging, mid-compute, landed in
write-staging, mid-write, landed at an output SSD -- each keeping one
identity color throughout, which is what conveys "overlap" in a single
static frame: multiple stages are occupied at once, not one chunk moving
through an otherwise-empty pipeline.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# Light-mode chart palette (see the dataviz skill's reference palette;
# matches figures/disk_layout_diagram.py and figures/parlay_operations.py).
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
BASELINE = "#c3c2b7"

# Color encodes a chunk's entry order into the pipeline, consistently across
# every stage it passes through. Same low-chroma red-to-blue ramp as the
# other figures in this set.
_RED_BLUE_STEPS = [
    "#e6c2c0", "#d8a6a6", "#c78d94", "#b17587", "#976480",
    "#7c5678", "#614d6e", "#4a4869", "#3a4568", "#2f4166",
    "#293c60", "#243655", "#1f2f48",
]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_red_blue", _RED_BLUE_STEPS)

N_IN_DRIVES = 5
N_OUT_DRIVES = 5
N_CELLS = 21  # 7 per zone: read-staging | compute | write-staging
CELLS_PER_ZONE = N_CELLS // 3

BAR_X0, BAR_X1 = 0.145, 0.855
BAR_Y0, BAR_Y1 = 0.10, 0.42
CELL_GAP = 0.055
CELL_W = ((BAR_X1 - BAR_X0) - CELL_GAP * (N_CELLS - 1)) / N_CELLS

DRIVE_W, DRIVE_H = 0.085, 0.075


def cell_x0(i):
    return BAR_X0 + i * (CELL_W + CELL_GAP)


def zone_bounds(zone):
    # zone in {0: read-staging, 1: compute, 2: write-staging}
    lo = cell_x0(zone * CELLS_PER_ZONE)
    hi = cell_x0(zone * CELLS_PER_ZONE + CELLS_PER_ZONE - 1) + CELL_W
    return lo, hi


def draw_drive(ax, x0, y0, label, filled=False, color=None):
    box = mpatches.FancyBboxPatch(
        (x0, y0), DRIVE_W, DRIVE_H,
        boxstyle="round,pad=0.0,rounding_size=0.012",
        linewidth=1.2, edgecolor=BASELINE,
        facecolor=(color if filled else "none"),
        zorder=5, mutation_aspect=1,
    )
    ax.add_patch(box)
    # A short platter line reads as "disk" at a glance, echoing
    # disk_layout_diagram.py's per-drive framing at a much smaller size.
    ax.plot([x0 + 0.012, x0 + DRIVE_W - 0.012],
            [y0 + DRIVE_H * 0.42, y0 + DRIVE_H * 0.42],
            color=BASELINE, linewidth=1.0, zorder=6)
    ax.text(x0 + DRIVE_W / 2, y0 - 0.018, label, ha="center", va="top",
            fontsize=7.3, color=INK_MUTED)


def draw_drive_column(ax, x0, n, y_top, y_bot, prefix):
    ys = np.linspace(y_top, y_bot, n)
    for i, y in enumerate(ys):
        draw_drive(ax, x0, y, f"{prefix}{i}")
    return ys


def curved_arrow(ax, xy_from, xy_to, color, rad=0.25, style="-|>",
                  lw=1.4, alpha=1.0, ls="solid"):
    ax.annotate(
        "", xy=xy_to, xytext=xy_from,
        arrowprops=dict(
            arrowstyle=style, color=color, linewidth=lw, alpha=alpha,
            linestyle=ls, shrinkA=2, shrinkB=2,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=4,
    )


def draw_dram_bar(ax):
    # The DRAM bar itself: a single flat vector, dashed border, matching
    # parlay_operations.py's dram_box styling.
    box = mpatches.FancyBboxPatch(
        (BAR_X0 - 0.012, BAR_Y0 - 0.05), (BAR_X1 - BAR_X0) + 0.024,
        (BAR_Y1 - BAR_Y0) + 0.10,
        boxstyle="round,pad=0.0,rounding_size=0.02",
        linewidth=1.3, edgecolor=INK_SECONDARY, facecolor="none",
        linestyle=(0, (4, 2.5)), zorder=6, mutation_aspect=1,
    )
    ax.add_patch(box)
    ax.text((BAR_X0 + BAR_X1) / 2, BAR_Y1 + 0.075, "DRAM",
             ha="center", va="bottom", fontsize=11.5, color=INK_SECONDARY,
             fontweight="bold")


def draw_cells(ax, occupants):
    """occupants: dict cell_index -> (color or None, hatched bool)."""
    for i in range(N_CELLS):
        x0 = cell_x0(i)
        color, hatched = occupants.get(i, (None, False))
        if color is not None:
            ax.add_patch(mpatches.Rectangle(
                (x0, BAR_Y0), CELL_W, BAR_Y1 - BAR_Y0,
                facecolor=color, edgecolor=SURFACE, linewidth=1.0, zorder=2,
            ))
        else:
            ax.add_patch(mpatches.Rectangle(
                (x0, BAR_Y0), CELL_W, BAR_Y1 - BAR_Y0,
                facecolor=SURFACE, edgecolor=BASELINE, linewidth=0.9,
                hatch="//" if hatched else None, zorder=2,
                alpha=0.6 if hatched else 1.0,
            ))


def draw_zone_brackets(ax):
    labels = ["read-staging\n(bounded read pool)",
              "compute\n(parlay workers)",
              "write-staging\n(bounded write pool)"]
    y = BAR_Y0 - 0.045
    for z, label in enumerate(labels):
        lo, hi = zone_bounds(z)
        ax.add_line(Line2D([lo, hi], [y, y], color=INK_MUTED, linewidth=1.2,
                            zorder=3))
        ax.add_line(Line2D([lo, lo], [y, y + 0.012], color=INK_MUTED,
                            linewidth=1.2, zorder=3))
        ax.add_line(Line2D([hi, hi], [y, y + 0.012], color=INK_MUTED,
                            linewidth=1.2, zorder=3))
        ax.text((lo + hi) / 2, y - 0.012, label, ha="center", va="top",
                fontsize=8.3, color=INK_SECONDARY)
    # internal dividers between zones, distinguishing them within the one
    # flat vector rather than drawing three separate boxes
    for z in (1, 2):
        x = cell_x0(z * CELLS_PER_ZONE) - CELL_GAP / 2
        ax.add_line(Line2D([x, x], [BAR_Y0 - 0.01, BAR_Y1 + 0.01],
                            color=BASELINE, linewidth=1.3,
                            linestyle=(0, (3, 2)), zorder=3))


def draw_compute_marks(ax, compute_cells):
    cell_mid_y = (BAR_Y0 + BAR_Y1) / 2
    for i in compute_cells:
        cx = cell_x0(i) + CELL_W / 2
        ax.text(cx, cell_mid_y, "f", ha="center", va="center",
                fontsize=11, color=SURFACE, fontstyle="italic",
                fontweight="bold", zorder=7)


def build_panel(ax):
    n_chunks = 9
    colors = [SEQ_CMAP(i / (n_chunks - 1)) for i in range(n_chunks)]

    in_y = draw_drive_column(ax, 0.02, N_IN_DRIVES, 0.62, 0.10, "in")
    out_y = draw_drive_column(ax, 1.0 - 0.02 - DRIVE_W, N_OUT_DRIVES,
                               0.62, 0.10, "out")

    draw_dram_bar(ax)

    read_lo, _ = zone_bounds(0)
    _, write_hi = zone_bounds(2)
    cell_mid_y = (BAR_Y0 + BAR_Y1) / 2

    # Pipeline snapshot: chunk c0 already landed at an output drive; c1 is
    # mid-write; c2..c3 sit in write-staging; c4 is mid-compute; c5..c6 sit
    # in read-staging; c7 is mid-read; c8 is still idle at an input drive.
    # Every stage of the pipeline is occupied at once -- that simultaneity
    # is the point. Every unoccupied read-/write-staging cell is hatched to
    # read consistently as spare capacity in a *bounded* pool, not blank
    # space; the compute zone isn't a buffer pool, so its idle cells (spare
    # worker slots) stay plain.
    occupants = {}
    for i in range(N_CELLS):
        zone = i // CELLS_PER_ZONE
        occupants[i] = (None, zone in (0, 2))

    # write-staging cells (zone 2): indices 14..20
    occupants[14] = (colors[3], False)
    occupants[16] = (colors[2], False)

    # compute cells (zone 1): indices 7..13
    occupants[10] = (colors[4], False)

    # read-staging cells (zone 0): indices 0..6
    occupants[3] = (colors[6], False)
    occupants[5] = (colors[5], False)

    draw_cells(ax, occupants)
    draw_compute_marks(ax, [10])
    draw_zone_brackets(ax)

    # c0: landed at output drive 0
    draw_drive(ax, 1.0 - 0.02 - DRIVE_W, out_y[0], "out0",
               filled=True, color=colors[0])
    # c1: mid-write, write-staging -> output drive 2 (hashed placement, not
    # the nearest drive -- echoes disk_layout_diagram.py's hashed panel)
    curved_arrow(ax, (write_hi, cell_mid_y),
                 (1.0 - 0.02, out_y[2] + DRIVE_H / 2), colors[1], rad=-0.15)

    # c7: mid-read, input drive 1 -> read-staging
    curved_arrow(ax, (0.02 + DRIVE_W, in_y[1] + DRIVE_H / 2),
                 (read_lo, cell_mid_y), colors[7], rad=-0.15)
    # c8: idle at an input drive, not yet requested
    draw_drive(ax, 0.02, in_y[3], "in3", filled=True, color=colors[8])

    ax.text(
        (BAR_X0 + BAR_X1) / 2, 0.955,
        "Overlapping Reads, Compute, and Writes",
        ha="center", va="top", fontsize=15.5, color=INK_PRIMARY,
        fontweight="bold",
    )
    ax.text(
        (BAR_X0 + BAR_X1) / 2, 0.895,
        "ExternalTransform: every stage is busy with a different chunk at once",
        ha="center", va="top", fontsize=10.5, color=INK_SECONDARY,
    )
    ax.text(
        (BAR_X0 + BAR_X1) / 2, -0.10,
        "~16 reads in flight × 5 reader threads over a shared, growing "
        "buffer pool  ·  write staging bounded at 64 × 4MB ≈ 256MB\n"
        "output placement hashed across drives (balls-in-bins), independent "
        "of input placement",
        ha="center", va="top", fontsize=8.6, color=INK_MUTED,
    )

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.22, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_colorbar(ax, n_chunks=9):
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, cmap=SEQ_CMAP, aspect="auto", extent=(0, 1, 0, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["chunk enters pipeline", "chunk exits pipeline"])
    ax.tick_params(colors=INK_MUTED, length=0, labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_figure():
    fig = plt.figure(figsize=(11.5, 6.3), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.05, 1], hspace=0.02)

    cax = fig.add_subplot(gs[0])
    cax.set_facecolor(SURFACE)
    ax = fig.add_subplot(gs[1])

    draw_colorbar(cax)
    build_panel(ax)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-prefix", default="io_compute_overlap",
        help="output PNG basename prefix; writes <prefix>.png "
             "(default: %(default)s)")
    args = parser.parse_args()

    out = f"{args.out_prefix}.png"
    fig = build_figure()
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
