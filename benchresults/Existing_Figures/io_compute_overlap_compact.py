#!/usr/bin/env python3
"""Conceptual diagram: overlapping reads, compute, and writes in
ExternalTransform (ChunkSequence/external_engine.h). Compact/narrow variant
of io_compute_overlap.py for slide layouts that can't fit the wide
left-SSDs/DRAM/right-SSDs original: the DRAM content is identical (same two
zones, same compute arrow, same role colors), but the two SSD stacks move
from flanking the DRAM box to sitting **above** it -- read-source drives over
the read-buffer zone, write-destination drives over the write-buffer zone --
with the read/write arrows running vertically instead of horizontally. That
trades width for height, which is the right trade when the constraint is
slide width rather than slide height. See io_compute_overlap.py's docstring
for the underlying conceptual rationale (two zones not three, why the
compute arrow stays inside the dashed DRAM box, source references into
external_engine.h/chunk_seq_reader.h/unordered_file_writer.h) -- none of
that changes here, only the SSD placement and the figure's aspect ratio.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Light-mode chart palette (see the dataviz skill's reference palette;
# matches figures/disk_layout_diagram.py and figures/parlay_operations.py).
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
BASELINE = "#c3c2b7"

# Fixed role colors -- not an ordered ramp, since this figure (unlike the
# other two) has no "position" to encode. Both are steps of the same blue
# sequential ramp used as DISCRETE_STEPS there, so the figure set stays one
# family.
STAGING_COLOR = "#86b6ef"  # step 250 -- light blue, both buffer pools
COMPUTE_COLOR = "#1c5cab"  # step 550 -- stronger blue, actively being read

BAR_X0, BAR_X1 = 0.14, 0.86
BAR_Y0, BAR_Y1 = 0.10, 0.34
CELL_GAP = 0.014

# Two zones, not three: read-buffer pool | (compute arrow) | write-buffer
# pool. ZONE_GAP is the visual gap between them where the compute arrow
# lives -- there is no third zone of cells there.
READ_CELLS, WRITE_CELLS = 6, 6
ZONE_GAP = 0.09
ZONE_CELLS_W = (BAR_X1 - BAR_X0) - ZONE_GAP
READ_ZONE_X0 = BAR_X0
READ_ZONE_W = ZONE_CELLS_W * READ_CELLS / (READ_CELLS + WRITE_CELLS)
WRITE_ZONE_X0 = READ_ZONE_X0 + READ_ZONE_W + ZONE_GAP
WRITE_ZONE_W = ZONE_CELLS_W * WRITE_CELLS / (READ_CELLS + WRITE_CELLS)
READ_CELL_W = (READ_ZONE_W - CELL_GAP * (READ_CELLS - 1)) / READ_CELLS
WRITE_CELL_W = (WRITE_ZONE_W - CELL_GAP * (WRITE_CELLS - 1)) / WRITE_CELLS

# SSDs sit above the DRAM bar instead of flanking it -- a short row of
# drives over each zone (read sources over the read zone, write
# destinations over the write zone) rather than a tall column to either
# side. This is the one structural change from io_compute_overlap.py: it
# frees the width the side columns used to take, at the cost of height,
# which is the trade slides want.
SSD_PER_SIDE = 3
SSD_STACK_GAP = 0.018
SSD_H = 0.11
READ_ZONE_HI = READ_ZONE_X0 + READ_ZONE_W
WRITE_ZONE_HI = WRITE_ZONE_X0 + WRITE_ZONE_W
READ_SSD_W = (READ_ZONE_W - SSD_STACK_GAP * (SSD_PER_SIDE - 1)) / SSD_PER_SIDE
WRITE_SSD_W = (WRITE_ZONE_W - SSD_STACK_GAP * (SSD_PER_SIDE - 1)) / SSD_PER_SIDE
SSD_Y0 = 0.62
SSD_LABEL_Y = SSD_Y0 + SSD_H + 0.03


def read_cell_x0(i):
    return READ_ZONE_X0 + i * (READ_CELL_W + CELL_GAP)


def write_cell_x0(i):
    return WRITE_ZONE_X0 + i * (WRITE_CELL_W + CELL_GAP)


def read_ssd_x0(i):
    return READ_ZONE_X0 + i * (READ_SSD_W + SSD_STACK_GAP)


def write_ssd_x0(i):
    return WRITE_ZONE_X0 + i * (WRITE_SSD_W + SSD_STACK_GAP)


def draw_drive(ax, x0, y0, w, h, label=None):
    box = mpatches.FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.018",
        linewidth=1.4, edgecolor=INK_SECONDARY, facecolor="none",
        zorder=5, mutation_aspect=1,
    )
    ax.add_patch(box)
    # A short platter line reads as "disk" at a glance, echoing
    # disk_layout_diagram.py's per-drive framing.
    ax.plot([x0 + w * 0.12, x0 + w - w * 0.12],
            [y0 + h * 0.42, y0 + h * 0.42],
            color=INK_SECONDARY, linewidth=1.3, zorder=6)
    if label:
        ax.text(x0 + w / 2, y0 - 0.022, label, ha="center", va="top",
                fontsize=10, color=INK_SECONDARY, fontweight="bold")


def curved_arrow(ax, xy_from, xy_to, color, rad=0.25):
    ax.annotate(
        "", xy=xy_to, xytext=xy_from,
        arrowprops=dict(
            arrowstyle="-|>", color=color, linewidth=1.6,
            shrinkA=2, shrinkB=2, connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=4,
    )


def draw_dram_bar(ax):
    box = mpatches.FancyBboxPatch(
        (BAR_X0 - 0.015, BAR_Y0 - 0.03), (BAR_X1 - BAR_X0) + 0.03,
        (BAR_Y1 - BAR_Y0) + 0.06,
        boxstyle="round,pad=0.0,rounding_size=0.02",
        linewidth=1.3, edgecolor=INK_SECONDARY, facecolor="none",
        linestyle=(0, (4, 2.5)), zorder=6, mutation_aspect=1,
    )
    ax.add_patch(box)
    ax.text((BAR_X0 + BAR_X1) / 2, BAR_Y1 + 0.09, "DRAM",
             ha="center", va="bottom", fontsize=12.5, color=INK_SECONDARY,
             fontweight="bold")


IDLE_ALPHA = 0.28  # pale tint of the zone's role color -- never pure white


def draw_cells(ax, cell_x0_fn, cell_w, cells):
    """cells: list of (color, active bool), one per cell index."""
    for i, (color, active) in enumerate(cells):
        ax.add_patch(mpatches.Rectangle(
            (cell_x0_fn(i), BAR_Y0), cell_w, BAR_Y1 - BAR_Y0,
            facecolor=color, edgecolor=SURFACE, linewidth=1.0,
            alpha=1.0 if active else IDLE_ALPHA, zorder=2,
        ))


def draw_zone_bracket(ax, lo, hi, label):
    # Just the role name -- no implementation-detail subtext (pool sizes,
    # thread counts); the figure is meant to read at a glance. No divider
    # line is needed between zones any more -- the real ZONE_GAP (and the
    # compute arrow crossing it) already marks the boundary.
    y = BAR_Y0 - 0.04
    ax.add_line(Line2D([lo, hi], [y, y], color=INK_MUTED, linewidth=1.2,
                        zorder=3))
    ax.add_line(Line2D([lo, lo], [y, y + 0.012], color=INK_MUTED,
                        linewidth=1.2, zorder=3))
    ax.add_line(Line2D([hi, hi], [y, y + 0.012], color=INK_MUTED,
                        linewidth=1.2, zorder=3))
    ax.text((lo + hi) / 2, y - 0.014, label, ha="center", va="top",
            fontsize=10, color=INK_SECONDARY, fontweight="bold")


def draw_compute_marks(ax, cell_x0_fn, cell_w, cells):
    # "f" marks a read-buffer cell a worker is actively reading right now,
    # copying/transforming its data into a freshly emitted output buffer
    # (the compute arrow) -- not a separate compute zone (see module
    # docstring).
    cell_mid_y = (BAR_Y0 + BAR_Y1) / 2
    for i in cells:
        cx = cell_x0_fn(i) + cell_w / 2
        ax.text(cx, cell_mid_y, "f", ha="center", va="center",
                fontsize=12, color=SURFACE, fontstyle="italic",
                fontweight="bold", zorder=7)


def build_panel(ax):
    # Two zones, not three: a read-buffer pool (idle/available cells, cells
    # holding data that landed but isn't picked up yet, and a couple being
    # actively read by a worker right now -- marked "f") and a write-buffer
    # pool (idle/available or occupied, queued to be written), joined by a
    # compute arrow rather than a third same-kind zone. See the module
    # docstring for why -- this is what ExternalTransform's body() actually
    # does, not just a simplification of it. All cells stay tinted with
    # their zone's role color even when idle (alpha only) -- never hatched,
    # never pure white.
    read_cells = [
        (STAGING_COLOR, True),   # landed from disk, not yet picked up
        (STAGING_COLOR, False),  # idle: available in the pool
        (STAGING_COLOR, True),
        (STAGING_COLOR, False),
        (COMPUTE_COLOR, True),   # being read by a worker right now ("f")
        (COMPUTE_COLOR, True),
    ]
    write_cells = [
        (STAGING_COLOR, False),
        (STAGING_COLOR, True),   # queued, waiting to be written
        (STAGING_COLOR, False),
        (STAGING_COLOR, True),
        (STAGING_COLOR, False),
        (STAGING_COLOR, True),
    ]

    draw_dram_bar(ax)
    draw_cells(ax, read_cell_x0, READ_CELL_W, read_cells)
    draw_cells(ax, write_cell_x0, WRITE_CELL_W, write_cells)
    draw_compute_marks(ax, read_cell_x0, READ_CELL_W, [4, 5])

    read_zone_hi = read_cell_x0(READ_CELLS - 1) + READ_CELL_W
    write_zone_lo = write_cell_x0(0)
    draw_zone_bracket(ax, read_cell_x0(0), read_zone_hi, "read buffers")
    draw_zone_bracket(ax, write_zone_lo,
                       write_cell_x0(WRITE_CELLS - 1) + WRITE_CELL_W,
                       "write buffers")

    # Compute arrow: body() copies from a read buffer into a freshly
    # emitted output buffer (ChunkEmitter::alloc()) -- this arrow *is* that
    # copy, not travel through a third memory zone. It stays inside the
    # dashed DRAM box (unlike the read/write arrows below, which cross it)
    # since the transform never touches disk.
    compute_y = (BAR_Y0 + BAR_Y1) / 2
    curved_arrow(ax, (read_zone_hi, compute_y), (write_zone_lo, compute_y),
                 INK_SECONDARY, rad=0.0)
    ax.text((read_zone_hi + write_zone_lo) / 2, compute_y + 0.035, "compute",
            ha="center", va="bottom", fontsize=9.5, color=INK_SECONDARY,
            fontstyle="italic")

    # A short row of SSDs sits above each zone -- read sources above the
    # read-buffer zone, write destinations above the write-buffer zone --
    # instead of a tall column flanking the DRAM bar. This is the one
    # structural difference from io_compute_overlap.py: it turns the
    # left-DRAM-right triptych into a top-DRAM stack, trading width for
    # height. Per-drive arrows run straight down into the read zone's top
    # edge / up from the write zone's top edge; one shared label per row
    # (not per drive) keeps the caption from repeating three times.
    dram_top = BAR_Y1 + 0.03
    read_to = (READ_ZONE_X0 + READ_ZONE_W / 2, dram_top)
    write_from = (WRITE_ZONE_X0 + WRITE_ZONE_W / 2, dram_top)

    for i in range(SSD_PER_SIDE):
        rx0 = read_ssd_x0(i)
        wx0 = write_ssd_x0(i)
        mid_x_r = rx0 + READ_SSD_W / 2
        mid_x_w = wx0 + WRITE_SSD_W / 2
        draw_drive(ax, rx0, SSD_Y0, READ_SSD_W, SSD_H)
        draw_drive(ax, wx0, SSD_Y0, WRITE_SSD_W, SSD_H)
        curved_arrow(ax, (mid_x_r, SSD_Y0), read_to, INK_SECONDARY, rad=0.0)
        curved_arrow(ax, write_from, (mid_x_w, SSD_Y0), INK_SECONDARY,
                     rad=0.0)

    ax.text(READ_ZONE_X0 + READ_ZONE_W / 2, SSD_LABEL_Y, "SSDs",
            ha="center", va="bottom", fontsize=10, color=INK_SECONDARY,
            fontweight="bold")
    ax.text(WRITE_ZONE_X0 + WRITE_ZONE_W / 2, SSD_LABEL_Y, "SSDs",
            ha="center", va="bottom", fontsize=10, color=INK_SECONDARY,
            fontweight="bold")

    arrow_mid_y = (SSD_Y0 + dram_top) / 2
    ax.text(READ_ZONE_X0 - 0.05, arrow_mid_y, "read",
            ha="center", va="center", fontsize=9.5, color=INK_SECONDARY,
            fontstyle="italic")
    ax.text(WRITE_ZONE_HI + 0.05, arrow_mid_y, "write",
            ha="center", va="center", fontsize=9.5, color=INK_SECONDARY,
            fontstyle="italic")

    ax.text(
        0.5, 0.99,
        "Overlapping I/O and\nComputation to Hide Latency",
        ha="center", va="top", fontsize=15.5, color=INK_PRIMARY,
        fontweight="bold",
    )
    ax.text(
        0.5, 0.855,
        "Separate Read/Computation Area and\nWrite Staging Section",
        ha="center", va="top", fontsize=10.5, color=INK_SECONDARY,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_figure():
    fig = plt.figure(figsize=(7.6, 7.4), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_subplot(111)
    build_panel(ax)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-prefix", default="io_compute_overlap_compact",
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
