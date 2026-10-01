#!/usr/bin/env python3
"""Supplementary Figure S2: between-class clustermaps, every checkpoint at every depth.

    python visuals/figure/figs2_clustermap_grid.py [--out-dir DIR]

  fs2_clustermap_grid.pdf    5 checkpoints x 5 layers

The full version of what Figure 2 shows at one depth and Figure 4 shows for one model at two:
rows are checkpoints, columns are the five depths the snapshot carries, and the block structure
can be followed across both at once.

All 25 panels carry ONE leaf order, the base model's at layer 31 -- the order Figures 2 and 4
use -- so a given cell is the same group pair everywhere and a difference between panels is a
difference in the numbers, never a re-clustering. Ordering each panel by its own dendrogram
would be wrong rather than merely different: clustering always returns some order, so the early
layers would be drawn with blocks they do not have.

The group names are deliberately absent. Their pitch is set by panel HEIGHT and `imshow` holds
the cells square, so at this width 19 names would land on 3.9pt of pitch; the grid carries
texture and Figure 2 is the labelled reference.

PDF, not PNG, like the other scripts here: the destination is a LaTeX manuscript whose figures
are vector. It is not one of the 70 release figures and `run_local.sh` does not build it.

Reads `analysis/data/between_class_matrix.csv` -- never `results/`, which is gitignored.
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

from _common import BASE, BEHAVIOURAL, FIGURE_LAYER, LAYERS, VISUALS  # noqa: E402
from make_visuals import leaf_order, read_matrices  # noqa: E402
from make_visuals_probe import DISPLAY5  # noqa: E402
import fig2_clustermaps as f2  # noqa: E402

DEFAULT_OUT = os.path.join(VISUALS, 'figure')

WIDTH, HEIGHT = 6.5, 6.1      # included at \textwidth

ROWLAB_FS = 8.0
COLTITLE_FS = 8.5
CB_LABEL_FS = 7.0
CB_TICK_FS = 6.5


def grid(groups, z_by, order):
    """Five checkpoint rows by five depth columns, sharing one colour limit."""
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    gs = fig.add_gridspec(len(BEHAVIOURAL), len(LAYERS),
                          left=0.072, right=0.905, top=0.962, bottom=0.012,
                          wspace=0.07, hspace=0.07)
    im = None
    for i, model in enumerate(BEHAVIOURAL):
        for j, layer in enumerate(LAYERS):
            ax = fig.add_subplot(gs[i, j])
            # The same panel Figures 2 and 4 draw, so all three agree on the colour limit and
            # on which cells saturate.
            im = f2.clustermap_panel(ax, z_by[(model, layer)], order)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f'Layer {layer}', pad=3, fontsize=COLTITLE_FS)
            if j == 0:
                ax.set_ylabel(DISPLAY5[model], labelpad=4, fontsize=ROWLAB_FS)
    return fig, im


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    p.add_argument('--order-layer', type=int, default=FIGURE_LAYER,
                   help=f'layer whose base-model dendrogram orders every panel. Default '
                        f'{FIGURE_LAYER}, matching Figures 2 and 4.')
    args = p.parse_args()
    f2.style()

    groups, z_by = read_matrices()
    missing = [f'{m} L{L}' for m in BEHAVIOURAL for L in LAYERS if (m, L) not in z_by]
    if missing or (BASE, args.order_layer) not in z_by:
        raise SystemExit('no matrix for ' + ', '.join(missing or [f'base L{args.order_layer}'])
                         + ' in analysis/data/between_class_matrix.csv -- re-run '
                           'analysis/analyze_taxonomy.py')
    order = leaf_order(z_by[(BASE, args.order_layer)])

    fig, im = grid(groups, z_by, order)
    # One colourbar for all 25 panels, spanning the grid's full height.
    cax = fig.add_axes([0.920, 0.012, 0.016, 0.950])
    cb = fig.colorbar(im, cax=cax, extend='both')
    cb.set_label('$Z$-Cosine Similarity', fontsize=CB_LABEL_FS)
    cb.ax.tick_params(labelsize=CB_TICK_FS)
    cb.outline.set_visible(False)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, 'fs2_clustermap_grid.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
