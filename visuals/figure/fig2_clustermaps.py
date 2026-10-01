#!/usr/bin/env python3
"""Manuscript Figure 2: five models' between-class clustermaps at the final layer.

    python visuals/figure/fig2_clustermaps.py [--out-dir DIR]

  f2_clustermaps_l31.pdf    3 + 2 panels, one per checkpoint, at layer 31

A cross-model comparison at fixed depth. The release's own clustermap topic draws one
checkpoint at a time (`visuals/clustermap/<model>/heat_L31.png`); this puts all five side by
side so the block structure can be read across them.

PDF, not PNG, unlike everything under `visuals/`: this figure's destination is a LaTeX
manuscript whose other figures are vector. It is therefore NOT part of the 70-PNG release set
and `run_local.sh --stages figures` does not build it.

Three conventions come from `make_visuals.py`, whose `read_matrices` and `leaf_order` this
imports rather than reimplements. They are what make five separate panels comparable rather
than five unrelated pictures:

  * the off-diagonal cosines are z-scored WITHIN each model, because raw cosine levels differ
    between checkpoints for anisotropy reasons that carry no chemistry;
  * ONE leaf order for every panel, from the base model's dendrogram at this layer, so a
    difference between two panels is a difference in the numbers and never a re-clustering;
  * ONE colour limit across all five, so the single colourbar is valid for each of them.

Reads `analysis/data/between_class_matrix.csv` -- never `results/`, which is gitignored.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# This script sits one level below the other generators, so `visuals/` has to go on the path
# before `_common` and its siblings can be imported. `_common` then resolves the repo root
# for `paths` itself.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))

from _common import BASE, BEHAVIOURAL, CHEMDFM, FIGURE_LAYER, VISUALS  # noqa: E402
from make_visuals import leaf_order, read_matrices  # noqa: E402
from make_visuals_probe import DISPLAY5  # noqa: E402

DEFAULT_OUT = os.path.join(VISUALS, 'figure')

WIDTH = 5.5           # the width paper_acs includes its figures at
HEIGHT = 3.80
TICK_FS = 5.2         # the 19 group labels, leftmost panel of each row
TITLE_FS = 7.5
CB_LABEL_FS = 6.5
CB_TICK_FS = 6.0

# Not the maximum. |z| has median 0.69 and 95th percentile 1.96 but reaches 4.07 in a handful
# of cells, so scaling to the maximum spends the whole colour range on 0.5% of the data and
# leaves every block pale -- which is what the first render looked like. 2.5 is the 98th
# percentile; the 1.8% of cells beyond it saturate, and the colourbar is drawn with arrowheads
# so that is visible rather than hidden.
LIM = 2.5


def style():
    """Shared styling, matching the other generators but at a vector-appropriate save."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
        'axes.linewidth': 0.7, 'lines.linewidth': 1.4,
        'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
        'legend.frameon': False, 'figure.dpi': 200, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02, 'pdf.fonttype': 42,
    })


def figure(groups, z_by, order, layer):
    """3 + 2, with the vacant sixth cell carrying the colourbar.

    One row of five would put the 19 group labels near 4pt at this width; three columns keeps
    them at 5.2pt, the size the paper's other clustermap figures already use. Rows and columns
    carry the same order, so only the leftmost panel of each row is labelled and no panel
    carries x labels -- 19 rotated labels under each of five panels would cost more height
    than the matrices themselves.
    """
    labels = [groups[i] for i in order]
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, HEIGHT),
                             gridspec_kw=dict(wspace=0.12, hspace=0.20))
    flat = axes.ravel()
    im = None
    for k, model in enumerate(BEHAVIOURAL):
        ax = flat[k]
        zz = z_by[(model, layer)][np.ix_(order, order)]
        np.fill_diagonal(zz, np.nan)
        # Diverging, because z has a meaningful zero -- the model's own mean similarity --
        # with a neutral midpoint and never a rainbow. `nearest` so no cell is interpolated
        # into its neighbour.
        im = ax.imshow(zz, cmap='RdBu_r', vmin=-LIM, vmax=LIM, interpolation='nearest')
        # The dagger is the manuscript's convention, not the release's: ChemDFM is built
        # on Llama-3 rather than 3.1, so a ChemDFM-base gap confounds chemistry training
        # with the base-model change. The paper's tables mark it the same way and the
        # caption explains it; DISPLAY5 stays unmarked for the release's own figures.
        title = DISPLAY5[model] + ('$^{\\dagger}$' if model == CHEMDFM else '')
        ax.set_title(title, pad=3, fontsize=TITLE_FS)
        ax.set_xticks([])
        if k % 3 == 0:
            ax.set_yticks(range(len(groups)))
            ax.set_yticklabels(labels, fontsize=TICK_FS)
        else:
            ax.set_yticks([])
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)

    cell = flat[-1]
    cell.axis('off')
    cax = cell.inset_axes([0.10, 0.08, 0.085, 0.84])
    cb = fig.colorbar(im, cax=cax, extend='both')
    cb.set_label('$z$-cosine (within model)', fontsize=CB_LABEL_FS)
    cb.ax.tick_params(labelsize=CB_TICK_FS)
    cb.outline.set_visible(False)
    return fig


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=DEFAULT_OUT,
                   help='directory to write the PDF into. Point it at the manuscript tree '
                        'to refresh the figure there, e.g. --out-dir ../paper_acs/figures')
    p.add_argument('--layer', type=int, default=FIGURE_LAYER,
                   help=f'layer to draw. Default {FIGURE_LAYER}, the final one.')
    args = p.parse_args()
    style()

    groups, z_by = read_matrices()
    missing = [m for m in BEHAVIOURAL if (m, args.layer) not in z_by]
    if missing:
        raise SystemExit(
            'no rows for ' + ', '.join(missing) + f' at layer {args.layer} in '
            'analysis/data/between_class_matrix.csv -- re-run analysis/analyze_taxonomy.py')

    order = leaf_order(z_by[(BASE, args.layer)])
    fig = figure(groups, z_by, order, args.layer)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f'f2_clustermaps_l{args.layer}.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
