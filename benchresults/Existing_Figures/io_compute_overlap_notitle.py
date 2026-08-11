#!/usr/bin/env python3
"""Conceptual diagram: overlapping reads, compute, and writes in
ExternalTransform (ChunkSequence/external_engine.h). Untitled variant of
io_compute_overlap.py for contexts that supply their own caption/heading
(e.g. a slide with its own title placeholder) -- the diagram content is
identical (same two zones, same compute arrow, same role colors, same
flanking-SSD layout), but the in-image title/subtitle text is dropped and
the "DRAM" label is pulled in to sit right above the dashed box instead of
floating with the same gap the title used to occupy. The figure's y-extent
and figsize height are both cropped/rescaled to match (no dead space where
the title used to be) while keeping the same x-scale (in/data-unit) as the
original, so the drawn shapes keep their proportions. See
io_compute_overlap.py's docstring for the full conceptual rationale (two
zones not three, why the compute arrow stays inside the dashed DRAM box,
source references into external_engine.h/chunk_seq_reader.h/
unordered_file_writer.h) -- none of that changes here.
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

BAR_X0, BAR_X1 = 0.195, 0.805
BAR_Y0, BAR_Y1 = 0.46, 0.70
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

# SSDs flank the DRAM bar left/right (read sources on the left, write
# destinations on the right) rather than sitting below it, so the figure
# reads as one left-to-right pipeline instead of two arrows crossing beneath
# a single drive. Each side is a small vertical stack of drives (standing in
# for the real machine's many SSDs) instead of one, with per-drive arrows
# fanning into/out of the DRAM bar's midline.
SSD_PER_SIDE = 3
SSD_STACK_GAP = 0.02
SSD_W, SSD_H = 0.11, (BAR_Y1 - BAR_Y0) + 0.06  # matches the DRAM box's outer height
SSD_ITEM_H = (SSD_H - SSD_STACK_GAP * (SSD_PER_SIDE - 1)) / SSD_PER_SIDE
SSD_MID_Y = (BAR_Y0 + BAR_Y1) / 2
SSD_Y0 = SSD_MID_Y - SSD_H / 2
SSD_LEFT_X0 = 0.02
SSD_RIGHT_X0 = 1 - 0.02 - SSD_W


def ssd_item_ys():
    """Bottom y of each stacked drive, bottom-to-top."""
    return [SSD_Y0 + i * (SSD_ITEM_H + SSD_STACK_GAP) for i in range(SSD_PER_SIDE)]


def read_cell_x0(i):
    return READ_ZONE_X0 + i * (READ_CELL_W + CELL_GAP)


def write_cell_x0(i):
    return WRITE_ZONE_X0 + i * (WRITE_CELL_W + CELL_GAP)


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
    # Pulled in from the titled original's BAR_Y1 + 0.09 (a gap sized to
    # clear the in-image title/subtitle) to BAR_Y1 + 0.045 -- about half the
    # gap -- now that there's no title text above it to clear.
    ax.text((BAR_X0 + BAR_X1) / 2, BAR_Y1 + 0.045, "DRAM",
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
    curved_arrow(ax, (read_zone_hi, SSD_MID_Y), (write_zone_lo, SSD_MID_Y),
                 INK_SECONDARY, rad=0.0)
    ax.text((read_zone_hi + write_zone_lo) / 2, SSD_MID_Y + 0.035, "compute",
            ha="center", va="bottom", fontsize=9.5, color=INK_SECONDARY,
            fontstyle="italic")

    # A small stack of SSDs flanks each side of the DRAM bar -- read sources
    # on the left feeding the read-buffer pool, write destinations on the
    # right fed by the write-buffer pool -- so the whole figure reads as a
    # single left-to-right pipeline (many drives in, one DRAM region, many
    # drives out) instead of two arrows crossing beneath one drive. Per-drive
    # arrows fan into/out of the DRAM bar's midline; one shared label per
    # stack (not per drive) keeps the caption from repeating three times.
    dram_left = BAR_X0 - 0.015
    dram_right = BAR_X1 + 0.015
    read_to = (dram_left, SSD_MID_Y)
    write_from = (dram_right, SSD_MID_Y)

    for y0 in ssd_item_ys():
        mid_y = y0 + SSD_ITEM_H / 2
        draw_drive(ax, SSD_LEFT_X0, y0, SSD_W, SSD_ITEM_H)
        draw_drive(ax, SSD_RIGHT_X0, y0, SSD_W, SSD_ITEM_H)
        curved_arrow(ax, (SSD_LEFT_X0 + SSD_W, mid_y), read_to,
                     INK_SECONDARY, rad=0.0)
        curved_arrow(ax, write_from, (SSD_RIGHT_X0, mid_y),
                     INK_SECONDARY, rad=0.0)

    ax.text(SSD_LEFT_X0 + SSD_W / 2, SSD_Y0 - 0.022, "SSDs",
            ha="center", va="top", fontsize=10, color=INK_SECONDARY,
            fontweight="bold")
    ax.text(SSD_RIGHT_X0 + SSD_W / 2, SSD_Y0 - 0.022, "SSDs",
            ha="center", va="top", fontsize=10, color=INK_SECONDARY,
            fontweight="bold")

    read_mid_x = (SSD_LEFT_X0 + SSD_W + dram_left) / 2
    write_mid_x = (dram_right + SSD_RIGHT_X0) / 2
    ax.text(read_mid_x, SSD_MID_Y + 0.10, "read",
            ha="center", va="bottom", fontsize=9.5, color=INK_SECONDARY,
            fontstyle="italic")
    ax.text(write_mid_x, SSD_MID_Y + 0.10, "write",
            ha="center", va="bottom", fontsize=9.5, color=INK_SECONDARY,
            fontstyle="italic")

    # No title/subtitle text (see module docstring) -- so the y-extent is
    # cropped to just clear the DRAM label instead of the titled original's
    # 1.02 (sized to fit two lines of title text above the box).
    ax.set_xlim(0, 1)
    ax.set_ylim(0.32, 0.80)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_figure():
    # Height rescaled from the titled original's 4.3 to keep the same
    # in/data-unit scale on both axes (x: 12.4 in over an xlim range of 1.0;
    # y: previously 4.3 in over a 0.70 ylim range, now 0.48) now that the
    # cropped ylim above holds less vertical data range.
    fig = plt.figure(figsize=(12.4, 2.95), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_subplot(111)
    build_panel(ax)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-prefix", default="io_compute_overlap_notitle",
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
