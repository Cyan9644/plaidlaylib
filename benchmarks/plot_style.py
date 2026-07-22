"""Shared matplotlib style for the benchmark/trace PNGs — a paper-style look
(serif figure font, boxed legend, validated categorical palette) applied
entirely through rcParams, so callers change how a chart renders, never what
it plots.

Import and call `apply()` once per plotting function, right after the local
`import matplotlib.pyplot as plt` each of them already does.
"""

import matplotlib

# Colorblind-validated categorical palette, fixed order (never reassigned per
# chart/filter) — see the project's dataviz skill reference palette.
PALETTE = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
}
CATEGORICAL_ORDER = [
    "blue", "orange", "aqua", "yellow", "magenta", "green", "violet", "red",
]


def apply():
    """Update matplotlib's rcParams in place. Safe to call repeatedly."""
    matplotlib.rcParams.update({
        # Computer-Modern-style serif look, no LaTeX install required —
        # matplotlib bundles cmr10 itself. Listing font names directly under
        # font.family (rather than font.family="serif" + font.serif=[...])
        # is what makes matplotlib's per-glyph fallback kick in, so DejaVu
        # Serif fills in glyphs cmr10 doesn't have (e.g. em-dash).
        "font.family": ["cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        # cmr10 has no bundled bold variant — forcing "bold" here just makes
        # matplotlib warn and silently fall back to regular weight.
        "axes.formatter.use_mathtext": True,
        "axes.prop_cycle": matplotlib.cycler(
            color=[PALETTE[k] for k in CATEGORICAL_ORDER]),
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "figure.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.linewidth": 1.1,
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.7,
        "lines.linewidth": 1.6,
        "lines.markersize": 7,
        "lines.markeredgewidth": 1.0,
        "legend.frameon": True,
        "legend.edgecolor": "#333333",
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
