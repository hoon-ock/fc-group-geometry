#!/usr/bin/env python3
"""Manuscript Figure 4: probe accuracy against depth, with the geometry at two of those depths.

    python visuals/figure/fig4_depth_structure.py [--out-dir DIR]

  f4_depth_structure.pdf    probe accuracy vs layer, with two clustermap insets

One plot, five checkpoints, carrying its own evidence: two insets show the base model's
between-class matrix at layer 8 and at layer 24, each joined by a leader to the point on the
base curve it is taken from. Layer 8 sits on the low plateau and layer 24 after the rise, and
the insets say what that rise is -- the groups stop being speckle and fall into blocks. The
claim is that the jump is a change in the geometry, not only in what a classifier scores off it.

Both insets carry ONE leaf order, the base model's at layer 31 -- the same order Figure 2 uses,
so a cell is the same group pair in both insets and in Figure 2. Ordering each layer by its own
dendrogram would be wrong here rather than merely different: clustering always returns some
order, so layer 8 would be drawn with blocks it does not have.

The insets are deliberately unlabelled. At this size each of the 19 cells is about 3pt, so
group names are not an option; the insets carry texture, and Figure 2 is the labelled reference.

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

WIDTH, HEIGHT = 6.5, 5.0      # included at \textwidth

# The two depths drawn as insets. Layer 8 sits on the probe's low plateau and layer 24 after
# its rise, which is what makes the pair worth drawing.
CUTS = (8, 24)

# Inset geometry, in axes fractions.
#
# Both insets sit LOW, and that is forced by the data rather than chosen. At layer 8 the base
# model is the LOWEST of the five curves and at layer 24 the HIGHEST, so a leader reaching
# layer 8 from above would cross all four other checkpoints; from below it crosses nothing but
# the n-gram baseline rule. The band between the two reference rules -- random guess at 0.20,
# baseline at 0.475 -- is empty across the full width, and these two corners hold no curve
# point at all.
INSET_W, INSET_H = 0.185, 0.240
INSET_Y = 0.078
INSET_X = (0.030, 0.712)

# Vertical colourbar immediately right of the layer-24 inset. This is what sets that inset's
# x position: the bar, its tick labels and its rotated label together need about 0.09 of the
# axes width between the inset's right edge and the frame.
CB_X, CB_W = 0.912, 0.020

LEADER = dict(arrowstyle='-', color='0.45', linewidth=0.8, shrinkA=1.5, shrinkB=2.5)

INSETLAB_FS = 9.0
CB_LABEL_FS = 8.0
CB_TICK_FS = 7.5


def retune():
    """Rebind the generators' sizes for composite scale.

    Module globals read when the drawing functions run, so this changes only this figure --
    `make_visuals_probe`'s own released output is unaffected.
    """
    # The released curve is 7.2 x 4.2in and this one is about 5.8 x 4.2in -- nearly the same
    # plotting area, so the type barely comes down at all. The first draft scaled it as if this
    # were a small sub-panel, which left everything unreadably small at print size.
    probe.PROBE_LABEL = 11.5
    probe.PROBE_TICK = 10.5
    probe.PROBE_LEGEND = 10.0
    probe.PROBE_REF = 9.0
    probe.PROBE_MARK = 3.6
    probe.PROBE_LW = 1.4
    # Left at its default 8: the floor labels run rightward from there, which lands them in the
    # gap between the two insets. The earlier guide-line draft had to move them off layer 8.
    probe.PROBE_REF_X = 8


def inset_block(ax, curves, z_by, order):
    """The two clustermap insets, each joined to its point on the base curve."""
    im = None
    for layer, x0 in zip(CUTS, INSET_X):
        cell = ax.inset_axes([x0, INSET_Y, INSET_W, INSET_H])
        im = f2.clustermap_panel(cell, z_by[(BASE, layer)], order)
        cell.set_xticks([])
        cell.set_yticks([])
        cell.set_title(f'Layer {layer}', pad=2, fontsize=INSETLAB_FS)

        value = curves[BASE][layer]
        # Leader from the top CORNER nearest the anchor, not the top centre: the title sits at
        # the centre, and a leader leaving from there runs straight through it -- which is
        # exactly what the first render did to 'Layer 24'. Which corner is derived from where
        # the anchor actually falls, so it stays right if the insets move.
        anchor_x = ax.transLimits.transform((layer, value))[0]
        x_start = x0 + INSET_W if anchor_x > x0 + INSET_W / 2 else x0
        # zorder BELOW the curves: at layer 24 base is the top of a converged bundle with no
        # room above it, so the leader has to cross the others and should pass under them.
        ax.annotate('', xy=(layer, value), xycoords='data',
                    xytext=(x_start, INSET_Y + INSET_H), textcoords='axes fraction',
                    arrowprops=LEADER, zorder=1.5)
        # A ring on the anchor, so the leader lands on something the eye can find among five
        # overlapping curves.
        ax.plot([layer], [value], linestyle='none', marker='o', markersize=8.5,
                markerfacecolor='none', markeredgecolor='0.25', markeredgewidth=1.0,
                zorder=8)
    return im


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    p.add_argument('--order-layer', type=int, default=FIGURE_LAYER,
                   help=f'layer whose base-model dendrogram orders both insets. Default '
                        f'{FIGURE_LAYER}, matching Figure 2.')
    args = p.parse_args()

    # Figure 2's style, not the probe module's: this figure draws clustermaps and has to match
    # Figure 2's vector output settings exactly.
    f2.style()
    retune()

    curves, surface = probe.read_curves()
    groups, z_by = read_matrices()
    missing = [L for L in CUTS if (BASE, L) not in z_by]
    if missing or (BASE, args.order_layer) not in z_by:
        raise SystemExit(
            f'no base-model matrix at layer(s) {missing or [args.order_layer]} in '
            'analysis/data/between_class_matrix.csv -- re-run analysis/analyze_taxonomy.py')
    order = leaf_order(z_by[(BASE, args.order_layer)])

    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    ax = fig.add_axes([0.095, 0.092, 0.890, 0.835])
    probe.probe_panel(ax, curves, surface)
    im = inset_block(ax, curves, z_by, order)

    cax = ax.inset_axes([CB_X, INSET_Y, CB_W, INSET_H])
    cb = fig.colorbar(im, cax=cax, orientation='vertical', extend='both')
    cb.set_label('$Z$-Cosine Similarity', fontsize=CB_LABEL_FS, labelpad=4)
    cb.ax.tick_params(labelsize=CB_TICK_FS, length=2, pad=1)
    cb.outline.set_visible(False)

    # Measured, not a figure-fraction constant: on Figures 3 and 5 those were the single
    # largest source of retune churn, silently misplacing themselves whenever the canvas moved.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, ax.get_position().y1 + 0.006), ncol=5,
               fontsize=probe.PROBE_LEGEND, handlelength=2.4, columnspacing=1.2,
               frameon=False)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, 'f4_depth_structure.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
