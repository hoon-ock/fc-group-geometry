#!/usr/bin/env python3
"""Manuscript Figure 4: probe accuracy against depth, and the geometry at two of those depths.

    python visuals/figure/fig4_depth_structure.py [--out-dir DIR]

  f4_depth_structure.pdf    (a) probe accuracy vs layer; (b, c) Base clustermaps at L8 and L24

Panel (a) is the probe curve, five checkpoints. Panels (b) and (c) are the base model's
between-class matrix at the two layers marked on it -- layer 8, on the low plateau, and layer
24, after the rise. The pairing is the figure's point: the jump is a change in how the groups
sit relative to one another, not only in what a classifier can score off them.

Both clustermaps carry ONE leaf order, the base model's at layer 31 -- the same order Figure 2
uses, so a cell is the same group pair in (b), in (c) and in Figure 2. Ordering each layer by
its own dendrogram would be wrong here rather than merely different: clustering always returns
some order, so layer 8 would be drawn with blocks it does not have.

PDF, not PNG, like the other scripts here: the destination is a LaTeX manuscript whose figures
are vector. It is not one of the 70 release figures and `run_local.sh` does not build it.

Composed from the existing generators rather than reimplemented. `probe_panel` was lifted out
of `make_visuals_probe.probe_figure` for this figure, and `clustermap_panel` is shared with
`fig2_clustermaps.py` so the two manuscript figures cannot disagree on the colour limit.

Reads `analysis/data/layer_curves.csv`, `surface_baseline.csv` and `between_class_matrix.csv`
through those generators -- never `results/`, which is gitignored.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
sys.path.insert(0, HERE)

from _common import BASE, FIGURE_LAYER, VISUALS  # noqa: E402
from make_visuals import leaf_order, read_matrices  # noqa: E402
import make_visuals_probe as probe  # noqa: E402
import fig2_clustermaps as f2  # noqa: E402

DEFAULT_OUT = os.path.join(VISUALS, 'figure')

WIDTH, HEIGHT = 6.5, 6.2      # included at \textwidth

# The two depths, and the panel each one becomes. Layer 8 sits on the probe's low plateau and
# layer 24 after its rise, which is what makes the pair worth drawing.
CUTS = [(8, '(b)'), (24, '(c)')]

# Where the two floor annotations sit, moved off layer 8 because this figure draws a guide
# line there. Far enough right that 'Character $n$-gram baseline' still ends well before the
# layer-24 guide.
REF_X = 10

GUIDE = dict(color='0.45', linewidth=0.8, linestyle=(0, (4, 2)), zorder=1.5)

PANEL_LAB = dict(fontsize=10, fontweight='bold', va='top', ha='left')

TICK_FS = 5.6         # the 19 group labels, panel (b) only
CUTLAB_FS = 7.5       # the '(b)' / '(c)' tags on the guide lines
MAPTITLE_FS = 8.0
CB_LABEL_FS = 6.5
CB_TICK_FS = 6.0


def retune():
    """Rebind the generators' sizes for composite scale.

    Module globals read when the drawing functions run, so this changes only this figure --
    `make_visuals_probe`'s own released output is unaffected.
    """
    # The released curve is 7.2 x 4.2in; panel (a) here is about 6.0 x 2.3in, so the type comes
    # down and the marks come down further -- there is one on every a one of the 32 layers.
    probe.PROBE_LABEL = 8.5
    probe.PROBE_TICK = 7.5
    probe.PROBE_LEGEND = 7.5
    probe.PROBE_REF = 6.8
    probe.PROBE_MARK = 2.8
    probe.PROBE_LW = 1.1
    probe.PROBE_REF_X = REF_X


def probe_block(fig, gs_cell, curves, surface):
    """(a) -- the probe curve, with a guide at each layer the clustermaps are taken from."""
    ax = fig.add_subplot(gs_cell)
    probe.probe_panel(ax, curves, surface)
    for layer, tag in CUTS:
        ax.axvline(layer, **GUIDE)
        # Tagged in the empty band between the random-guess line and the n-gram baseline --
        # about a third of the axes height with nothing in it. Neither end of the plot works:
        # the five curves converge on 1.0, so a tag at the top lands on them at layer 24, and
        # the strip below the random-guess line is only ~9pt tall at this panel height. x in
        # data coordinates so the tag tracks the guide, y in axes coordinates so it does not
        # move if the limits change.
        ax.annotate(tag, xy=(layer, 0.10), xycoords=('data', 'axes fraction'),
                    textcoords='offset points', xytext=(2, 0), ha='left', va='bottom',
                    fontsize=CUTLAB_FS, fontweight='bold', color='0.25')
    return ax


def map_block(fig, gs_cell, groups, z_by, order):
    """(b) and (c) -- the base model at the two marked depths, sharing one colourbar."""
    sub = gs_cell.subgridspec(1, 2, wspace=0.08)
    labels = [groups[i] for i in order]
    axes, im = [], None
    for k, (layer, _) in enumerate(CUTS):
        ax = fig.add_subplot(sub[0, k])
        im = f2.clustermap_panel(ax, z_by[(BASE, layer)], order)
        ax.set_title(f'Layer {layer}', pad=3, fontsize=MAPTITLE_FS)
        ax.set_xticks([])
        # Rows and columns carry the same order, so one set of labels serves both panels and
        # a second copy would only cost width.
        if k == 0:
            ax.set_yticks(range(len(groups)))
            ax.set_yticklabels(labels, fontsize=TICK_FS)
        else:
            ax.set_yticks([])
        axes.append(ax)
    return axes, im


def _fig_bbox(fig, artist):
    """An artist's drawn extent in figure coordinates."""
    bb = artist.get_window_extent(fig.canvas.get_renderer())
    return bb.transformed(fig.transFigure.inverted())


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    p.add_argument('--order-layer', type=int, default=FIGURE_LAYER,
                   help=f'layer whose base-model dendrogram orders both maps. Default '
                        f'{FIGURE_LAYER}, matching Figure 2.')
    args = p.parse_args()

    # Figure 2's style, not the probe module's: this figure draws clustermaps and has to match
    # Figure 2's vector output settings exactly.
    f2.style()
    retune()

    curves, surface = probe.read_curves()
    groups, z_by = read_matrices()
    missing = [L for L, _ in CUTS if (BASE, L) not in z_by]
    if missing or (BASE, args.order_layer) not in z_by:
        raise SystemExit(
            f'no base-model matrix at layer(s) {missing or [args.order_layer]} in '
            'analysis/data/between_class_matrix.csv -- re-run analysis/analyze_taxonomy.py')
    order = leaf_order(z_by[(BASE, args.order_layer)])

    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.28],
                          left=0.135, right=0.895, top=0.900, bottom=0.030,
                          hspace=0.30)
    ax_a = probe_block(fig, gs[0, 0], curves, surface)
    map_axes, im = map_block(fig, gs[1, 0], groups, z_by, order)

    # Placed from measured extents rather than figure-fraction constants, which on Figures 3
    # and 5 were the single largest source of retune churn: every one of them silently
    # misplaces itself the moment WIDTH or HEIGHT changes.
    fig.canvas.draw()
    a_pos = ax_a.get_position()
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, a_pos.y1 + 0.004),
               ncol=5, fontsize=probe.PROBE_LEGEND, handlelength=2.4, columnspacing=1.2,
               frameon=False)

    # The colourbar spans the two maps' shared height, in the margin `right` leaves for it.
    m0, m1 = map_axes[0].get_position(), map_axes[-1].get_position()
    cax = fig.add_axes([m1.x1 + 0.016, m0.y0, 0.016, m0.height])
    cb = fig.colorbar(im, cax=cax, extend='both')
    cb.set_label('$Z$-Cosine Similarity', fontsize=CB_LABEL_FS)
    cb.ax.tick_params(labelsize=CB_TICK_FS)
    cb.outline.set_visible(False)

    lab = dict(PANEL_LAB, va='bottom')
    fig.text(0.004, 0.992, '(a)', **PANEL_LAB)
    fig.text(0.004, _fig_bbox(fig, map_axes[0].title).y1 + 0.004, '(b)', **lab)
    fig.text(m1.x0 - 0.030, _fig_bbox(fig, map_axes[1].title).y1 + 0.004, '(c)', **lab)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, 'f4_depth_structure.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
