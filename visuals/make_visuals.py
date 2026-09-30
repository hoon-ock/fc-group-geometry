#!/usr/bin/env python3
"""Between-class similarity clustermaps, one file per panel.

    python visuals/make_visuals.py [--only meta-llama-Llama-3.1-8B] [--out-dir DIR]

Emits all FIVE sampled layers and all FIVE 8B checkpoints as **separate** untitled PNGs,
one directory per checkpoint, so each can be placed individually:

  clustermap/<model>/heat_L<layer>.png    25 files: 5 models x 5 layers
  clustermap/<model>/heatrow.png           5 files: all five depths in one row, set in
                                           larger type for reading on its own

`<model>` is the checkpoint's own identifier with `/` replaced by `-`
(`meta-llama-Llama-3.1-8B`), the same string the `results/` and `activations/` trees use and
the same layout `make_visuals_tsne.py` writes its per-model panels into. One model is one
string everywhere a path carries it, so a figure can be traced to the run that produced it
without a lookup table. The model is therefore NOT repeated in the filename.

`heat_figure` and `heatrow_figure` share ONE function for the matrix drawing (`heat_row`)
and differ only in page geometry and type size, so the two forms cannot disagree about a
number.

Nothing is recomputed, rescaled or smoothed here. The within-(model, layer) z-score comes
from `data/between_class_matrix.csv` as `analyze_taxonomy.py` computed it. Never
`results/`, which is gitignored.

Two choices keep 30 separate files comparable to each other rather than 30 unrelated
pictures:

  * ONE leaf order for all 25 clustermaps, taken from the BASE model's final-layer
    dendrogram. Every model is drawn in that layout,
    so a difference between two files is a difference in the numbers and never a
    re-clustering.
  * ONE symmetric colour limit, taken over all five models and all five layers in a first
    pass. Colourbars are therefore interchangeable between files.
"""
import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.cluster.hierarchy import dendrogram, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

from _common import (BASE, CHEM, FIGURE_LAYER, LAYERS, R1,  # noqa: E402
                     REPO, VISUALS, load, num)

# The two chemistry checkpoints added after the paper CSVs were first snapshotted. Slugs are
# the results/ directory names (`/` -> `-`), which is what the CSVs carry in `model`.
CHEM_R = 'weidawang-Chem-R-8B'
CHEMDFM = 'OpenDFM-ChemDFM-v1.5-8B'

# Presentation order: base, the two controlled Llama-3.1 chemistry tunes in training order
# (Chem-R-Faithful is GRPO-trained FROM Chem-R-8B), the off-base chemistry model, then the
# reasoning distill. See src/model_registry.py for the tier argument -- ChemDFM sits on
# Llama-3, not 3.1, so a ChemDFM-vs-base gap is not a controlled contrast.
MODELS5 = [BASE, CHEM_R, CHEM, CHEMDFM, R1]
DEFAULT_VISUALS = os.path.join(VISUALS, 'clustermap')

# Each clustermap is now its own figure rather than one of three panels in a 5.5in row, so
# it carries both axes' group labels: a standalone matrix with only row labels cannot be
# read. Sized so the 19 labels stay legible without the file being larger than a column.
HEAT_SIZE = (4.0, 3.5)
HEAT_TICK = 8.0
HEAT_CB_LABEL = 10.5
HEAT_CB_TICK = 9.5

# The clustermap row read on its own. Type is enlarged over HEAT_* because a row scaled into
# a slide is not a single panel viewed at full size.
#
# The size is measured, not chosen. Two constraints bind it:
#   * the panels must be WIDTH-bound. `imshow` holds them square, so a figure taller than
#     (width per column) opens a band of dead space under the titles that `bbox='tight'`
#     cannot crop, being interior to the figure rather than around it. At 15.0 x 3.2 the
#     five panels come out 155pt square with zero slack in their axes boxes.
#   * 19 group labels must fit down the left edge. 155pt / 19 = 8.2pt of pitch per label,
#     which carries ROW_TICK 8.0 without collision.
# Changing ROW_SIZE without re-checking both of those will quietly break one of them.
ROW_SIZE = (15.0, 3.2)
ROW_TICK = 8.0        # the 19 group labels, column 0
ROW_TITLE = 10.0      # 'Layer N'
ROW_CB_LABEL = 9.5
ROW_CB_TICK = 8.5


def style():
    """Shared matplotlib styling, at a PNG-appropriate save dpi."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
        'axes.linewidth': 0.7, 'lines.linewidth': 1.4,
        'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
        'legend.frameon': False, 'figure.dpi': 200, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.04, 'savefig.dpi': 300,
    })


# --------------------------------------------------------------------------------------
def read_matrices():
    """`(model, layer) -> z-scored 19x19 matrix`, plus the group list.

    The z-score is `f2_block_depth.zmat` verbatim: standardise the off-diagonal cosines by
    their own mean and std within one model at one depth, then park the diagonal at 0. Raw
    cosines differ in level between models for anisotropy reasons that carry no chemistry,
    which is why no cross-model quantity is ever built from the raw values.
    """
    rows = load('between_class_matrix.csv')
    groups = sorted({r['group_a'] for r in rows} | {r['group_b'] for r in rows})
    idx = {g: i for i, g in enumerate(groups)}

    cells = defaultdict(list)
    for r in rows:
        cells[(r['model'], int(r['layer']))].append(r)

    z_by = {}
    for (model, layer), pairs in cells.items():
        sim = np.zeros((len(groups), len(groups)))
        for r in pairs:
            i, j = idx[r['group_a']], idx[r['group_b']]
            sim[i, j] = sim[j, i] = num(r, 'cosine_sim')
        np.fill_diagonal(sim, np.nan)
        off = sim[~np.isnan(sim)]
        z = (sim - off.mean()) / off.std(ddof=1)
        np.fill_diagonal(z, 0.0)
        z_by[(model, layer)] = z
    return groups, z_by


def leaf_order(z):
    """Average-linkage leaf order, as f2_block_depth derives it from the final layer."""
    dist = z.max() - z
    np.fill_diagonal(dist, 0.0)
    return dendrogram(linkage(squareform(dist, checks=False), method='average'),
                      no_plot=True)['leaves']


def heat_figure(model, layer, groups, z_by, order, lim):
    """One clustermap: one model at one depth, standalone and untitled."""
    fig, ax = plt.subplots(figsize=HEAT_SIZE)
    zz = z_by[(model, layer)][np.ix_(order, order)]
    np.fill_diagonal(zz, np.nan)
    # `nearest` so no cell is interpolated into its neighbour, and a symmetric diverging map
    # because z has a meaningful zero -- the model's own mean similarity.
    im = ax.imshow(zz, cmap='RdBu_r', vmin=-lim, vmax=lim, interpolation='nearest')
    ax.set_xticks(range(len(groups)))
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([groups[i] for i in order], fontsize=HEAT_TICK)
    ax.set_xticklabels([groups[i] for i in order], fontsize=HEAT_TICK, rotation=90)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    # Labelled just "z-score". The standardisation is still within (model, layer) -- see
    # read_matrices -- but each file shows one model at one depth, so the qualifier had no
    # contrast to draw inside the frame it sat in.
    cb.set_label('$z$-score', fontsize=HEAT_CB_LABEL)
    cb.ax.tick_params(labelsize=HEAT_CB_TICK)
    cb.outline.set_visible(False)
    return fig


def heat_row(fig, axes, model, groups, z_by, order, lim, tick_fs, cb_label_fs, cb_tick_fs,
             cb_fraction, cb_pad, title_fs=None):
    """Five clustermaps in one row on pre-made axes, sharing one colour limit.

    Used by both `heat_figure` and `heatrow_figure`, so the two can never disagree about
    the arrays, the leaf order or the scale -- they differ only in page geometry and type
    size, which is everything the caller passes in. The axes arrive already allocated
    because the callers lay out differently: one panel per figure against five in a row.

    The colourbar is attached to THESE axes, never to `fig.axes`: in the composite that would
    stretch the bar over the trajectory panel below, which is a different quantity in
    different units.
    """
    im = None
    for col, layer in enumerate(LAYERS):
        ax = axes[col]
        zz = z_by[(model, layer)][np.ix_(order, order)]
        np.fill_diagonal(zz, np.nan)
        im = ax.imshow(zz, cmap='RdBu_r', vmin=-lim, vmax=lim, interpolation='nearest')
        # The one place a title survives: without it the five columns are unidentifiable,
        # which is not true of the individual files, where the filename carries the depth.
        ax.set_title(f'Layer {layer}', pad=3,
                     **({} if title_fs is None else dict(fontsize=title_fs)))
        ax.set_xticks([])
        if col == 0:
            ax.set_yticks(range(len(groups)))
            ax.set_yticklabels([groups[i] for i in order], fontsize=tick_fs)
        else:
            ax.set_yticks([])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=cb_fraction, pad=cb_pad)
    cb.set_label('$z$-score', fontsize=cb_label_fs)
    cb.ax.tick_params(labelsize=cb_tick_fs)
    cb.outline.set_visible(False)


def heatrow_figure(model, groups, z_by, order, lim):
    """All five depths in one row: one colour limit, larger type, read on its own.

    Same `heat_row` that `heat_figure` calls, so this is a re-layout of the same arrays and
    never a second opinion about the numbers.
    """
    fig, axes = plt.subplots(1, len(LAYERS), figsize=ROW_SIZE,
                             gridspec_kw=dict(wspace=0.06))
    heat_row(fig, list(axes), model, groups, z_by, order, lim,
             ROW_TICK, ROW_CB_LABEL, ROW_CB_TICK, 0.018, 0.012, title_fs=ROW_TITLE)
    return fig


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=DEFAULT_VISUALS)
    p.add_argument('--only', choices=MODELS5, metavar='MODEL',
                   help='render a single checkpoint instead of all five, named as it is '
                        'in the CSV and on disk (e.g. meta-llama-Llama-3.1-8B)')
    args = p.parse_args()
    style()

    groups, z_by = read_matrices()

    missing = [m for m in MODELS5 if (m, FIGURE_LAYER) not in z_by]
    if missing:
        raise SystemExit(
            'no rows for ' + ', '.join(missing) + ' in data/between_class_matrix.csv'
            ' -- re-run analysis/analyze_taxonomy.py')

    # First pass, over every model and every layer, so the separate files share one layout
    # and one colour scale. Computing these per file would make two PNGs look different for
    # reasons that are not in the data.
    order = leaf_order(z_by[(BASE, FIGURE_LAYER)])
    lim = max(np.nanmax(np.abs(z_by[(m, L)])) for m in MODELS5 for L in LAYERS)

    wanted = [m for m in MODELS5 if args.only is None or m == args.only]
    for model in wanted:
        sub = os.path.join(args.out_dir, model)
        os.makedirs(sub, exist_ok=True)
        for layer in LAYERS:
            fig = heat_figure(model, layer, groups, z_by, order, lim)
            path = os.path.join(sub, f'heat_L{layer}.png')
            fig.savefig(path)
            plt.close(fig)
            print(f'wrote {os.path.relpath(path, REPO)}')
        fig = heatrow_figure(model, groups, z_by, order, lim)
        path = os.path.join(sub, 'heatrow.png')
        fig.savefig(path)
        plt.close(fig)
        print(f'wrote {os.path.relpath(path, REPO)}')
    print(f'shared z-cosine limit +/-{lim:.3f}, leaf order from base L{FIGURE_LAYER}')


if __name__ == '__main__':
    main()
