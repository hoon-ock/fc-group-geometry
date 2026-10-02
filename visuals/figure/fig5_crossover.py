#!/usr/bin/env python3
"""Manuscript Figure 5: the template-to-chemistry crossover, measured and shown.

    python visuals/figure/fig5_crossover.py [--out-dir DIR]

  f5_crossover.pdf    (a) kNN label agreement vs layer; (b, c) t-SNE at two layers, two colourings

Panel (a) is the measurement: how often a point's 10 nearest neighbours carry its own prompt
template, and how often they carry its own functional group, at every layer. The two curves
cross. Panels (b) and (c) are that crossover shown rather than asserted -- the same embeddings
at layer 8 and at layer 24, each coloured BOTH ways, so the swap is visible rather than
inferred from one half of it.

Models are rows, as in the released grids (`visuals/tsne/tsne_depth_*.png`), and the two layers
sit side by side as column pairs: layer 8 in the two left columns, layer 24 in the two right.
Within each pair the columns are the two colourings. So the crossover is read left-to-right
along a single model's row, and a gutter separates the pairs.

PDF, not PNG, like the other scripts here: the destination is a LaTeX manuscript whose figures
are vector. It is not one of the 70 release figures and `run_local.sh` does not build it.

Composed from the two standalone generators rather than reimplemented. `tsne.panel` already
took an axes; `prompt_structure.agreement_panel` was lifted out of that module's `main()` for
this figure, and both modules' released output is byte-identical across the change.

Reads `analysis/data/tsne_coords.csv` and `analysis/data/label_agreement_by_layer.csv` through
those generators -- never `results/`, which is gitignored.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))

from _common import BEHAVIOURAL, VISUALS  # noqa: E402
import make_visuals_prompt_structure as agr  # noqa: E402
import make_visuals_tsne as tsne  # noqa: E402
from make_visuals_probe import DISPLAY5, style  # noqa: E402

DEFAULT_OUT = os.path.join(VISUALS, 'figure')

WIDTH, HEIGHT = 6.5, 8.45     # included at \textwidth; a dedicated-page figure

# The four scatter columns, left to right: both colourings at layer 8, then both at layer 24.
# `key` is the CSV column `tsne.panel` filters on; the label is the colouring alone, because
# the layer is carried by the spanning header over each pair.
COLS = [
    (8,  'template_index',   'Template'),
    (8,  'functional_group', 'Group'),
    (24, 'template_index',   'Template'),
    (24, 'functional_group', 'Group'),
]

# Width of the empty spacer column between the two layer pairs, relative to one scatter cell.
#
# It is paid for out of cell width that was doing nothing. `tsne.panel` ends in
# `set_aspect('equal', adjustable='datalim')`, so on a cell wider than it is tall matplotlib
# widens the DATA range instead of stretching the points: the cluster is sized by cell HEIGHT
# and the surplus width is padding. Spending some of that padding on a gutter is what makes
# "two left, two right" read as two groups rather than four columns.
GUTTER = 0.5

# Legend geometry. The released grid wraps the 20 functional groups at 7, but that key is the
# widest artist on this page and 7 columns caps the whole figure's type: measured, it is 6.03in
# at 8pt and 6.86in at 9pt against a 6.5in canvas. Six columns is 5.63in at 9pt, which is what
# lets everything else come up too. The 10 templates still fit one row.
FINE_NCOL, TEMPLATE_NCOL = 6, 10

PANEL_LAB = dict(fontsize=10, fontweight='bold', va='top', ha='left')

ROWLAB_FS = 9.0
COLTITLE_FS = 9.0
HEADER_FS = 11.0
LEGEND_FS = 9.0
LEGEND_TITLE_FS = 9.5


def retune():
    """Rebind the generators' sizes for composite scale.

    Module globals read when the drawing functions run, so this changes only this figure --
    the standalone generators keep their own sizes and their released output is unaffected.
    """
    # (a) is drawn at roughly half the standalone's 5.0in width, so its marks come down with
    # it or they swamp 32 layers of line.
    agr.AGR_MARK = 2.6
    agr.AGR_LW = 1.2
    agr.AGR_REF_FS = 8.0
    # Each embedding is ~0.88in tall against the released grid's 1.28in cell.
    tsne.POINT_SIZE = 3.0
    tsne.POINT_ALPHA = 0.75
    # (a)'s y-label is ROTATED, so its LENGTH has to fit the panel's height, and raising the
    # type shortens that panel: the bigger keys below need more bottom margin and the bigger
    # headers a wider inter-panel band, both taken out of (a). 'kNN label agreement (k=10)'
    # measures 1.45in at 9pt and 1.53in at 9.5, against a panel that ends up 1.50in -- so 9.5
    # fitted exactly, with no margin, and 9 is the size that actually has room.
    plt.rcParams.update({'axes.labelsize': 9.0, 'axes.titlesize': 10.0,
                         'xtick.labelsize': 9.0, 'ytick.labelsize': 9.0})


def agreement_block(fig, gs_cell, rows, models, layers):
    """(a) -- the two agreement panels, sharing the model legend below them."""
    sub = gs_cell.subgridspec(1, 2, wspace=0.17)
    axs = [fig.add_subplot(sub[0, i]) for i in range(2)]
    for ax, (label, title) in zip(axs, agr.PANELS):
        agr.agreement_panel(ax, rows, label, title, models, layers)
    # The y-label is the caller's job by the lifted function's contract, and the k comes from
    # the data rather than the prose so the two cannot disagree.
    axs[0].set_ylabel(f'kNN label agreement (k={rows[(models[0], layers[0])]["k"]})')
    return axs


def scatter_block(fig, gs_cell, coords, models, styles):
    """(b) and (c) -- five model rows by four columns, split into two layer pairs.

    Grid column 2 is left empty as the gutter, so the four drawn columns are at grid indices
    0, 1, 3, 4. Returns `axes[row][col]` indexed by the LOGICAL column 0-3, which is what the
    header and panel-label placement measures against.
    """
    sub = gs_cell.subgridspec(len(models), 5, wspace=0.08, hspace=0.06,
                              width_ratios=[1, 1, GUTTER, 1, 1])
    grid_col = (0, 1, 3, 4)
    out = []
    for i, m in enumerate(models):
        row = []
        for j, (layer, key, collab) in enumerate(COLS):
            ax = fig.add_subplot(sub[i, grid_col[j]])
            tsne.panel(ax, coords[(m, layer)], key, styles[key])
            if i == 0:
                ax.set_title(collab, pad=3, fontsize=COLTITLE_FS)
            if j == 0:
                ax.set_ylabel(DISPLAY5[m], labelpad=4, fontsize=ROWLAB_FS)
            row.append(ax)
        out.append(row)
    return out


def key_handles(styles, labels):
    """Legend proxies for one colouring, as `grid_figure` builds them."""
    return [plt.Line2D([], [], linestyle='none', marker=tsne.as_pair(styles[v])[1],
                       markersize=3.0, markeredgewidth=0,
                       color=tsne.as_pair(styles[v])[0], label=tsne.display(v))
            for v in labels]


def _fig_bbox(fig, artist):
    """An artist's drawn extent in figure coordinates."""
    bb = artist.get_window_extent(fig.canvas.get_renderer())
    return bb.transformed(fig.transFigure.inverted())


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    args = p.parse_args()

    style()
    retune()

    rows = agr.read_rows()
    coords = tsne.read_coords()
    models = [m for m in BEHAVIOURAL
              if any((m, L) in rows for L in range(32))
              and all((m, layer) in coords for layer, _, _ in COLS)]
    if not models:
        raise SystemExit('no rows -- run analysis/analyze_prompt_structure.py and '
                         'analysis/compute_tsne_coords.py')
    layers = sorted({L for (_, L) in rows})

    flat = [r for v in coords.values() for r in v]
    fine_styles, fine_labels = tsne.fine_style(flat)
    tmpl_styles, tmpl_labels = tsne.template_style(flat)
    styles = {'functional_group': fine_styles, 'template_index': tmpl_styles}

    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    # Solved from measured furniture, not guessed. Bottom holds the two keys, which at 9pt
    # stack 1.52in; the inter-panel band holds (a)'s x label, the model legend, the two layer
    # headers AND the scatter column titles, about 1.0in; what is left splits 1 : 2.63.
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.63],
                          left=0.085, right=0.985, top=0.966, bottom=0.185,
                          hspace=0.42)
    agr_axes = agreement_block(fig, gs[0, 0], rows, models, layers)
    sc = scatter_block(fig, gs[1, 0], coords, models, styles)

    # Everything below is placed from MEASURED positions rather than figure-fraction constants.
    # The constants this file used to carry were tuned against a 7.3in canvas and silently
    # misplaced themselves the moment HEIGHT changed; deriving them removes that retune.
    fig.canvas.draw()
    pos = [[ax.get_position() for ax in row] for row in sc]
    sc_top = max(b.y1 for b in pos[0])
    # Above the column titles, not just above the axes.
    title_top = max(_fig_bbox(fig, sc[0][j].title).y1 for j in range(4))

    # One spanning header per layer pair, centred over the two columns it covers, with the
    # pair's panel label on the same baseline so '(b)' and 'Layer 8' read as one line.
    head_y = title_top + 0.005
    for j0, layer in ((0, COLS[0][0]), (2, COLS[2][0])):
        mid = (pos[0][j0].x0 + pos[0][j0 + 1].x1) / 2
        fig.text(mid, head_y, f'Layer {layer}', ha='center', va='bottom', fontsize=HEADER_FS)
    lab = dict(PANEL_LAB, va='bottom')
    fig.text(0.004, head_y, '(b)', **lab)
    fig.text(pos[0][2].x0 - 0.034, head_y, '(c)', **lab)
    fig.canvas.draw()
    band_top = max(_fig_bbox(fig, t).y1 for t in fig.texts)

    # Three legends, each under the block it explains. They describe unrelated encodings --
    # five checkpoints, ten templates, twenty groups -- so merging any two would make one
    # oversized box in which none of them reads.
    # Stacked on the header line, not floated at a guessed midpoint: the band between (a) and
    # the scatters holds the headers, the panel labels AND this legend, and measuring only to
    # the scatter axes' top ignores the two that sit above it.
    handles, labels = agr_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, band_top + 0.008),
               ncol=len(models), fontsize=LEGEND_FS, handletextpad=0.4, columnspacing=1.4,
               frameon=False)

    # The 20-group key is the tall one, so it is placed at the canvas floor first and the
    # 10-template key is stacked on whatever it actually occupied.
    fine = fig.legend(handles=key_handles(fine_styles, fine_labels), loc='lower center',
                      bbox_to_anchor=(0.5, 0.0), ncol=FINE_NCOL, fontsize=LEGEND_FS,
                      handletextpad=0.35, columnspacing=1.0, frameon=False,
                      title='Functional group')
    fine.get_title().set_fontsize(LEGEND_TITLE_FS)
    fig.canvas.draw()
    tmpl = fig.legend(handles=key_handles(tmpl_styles, tmpl_labels), loc='lower center',
                      bbox_to_anchor=(0.5, _fig_bbox(fig, fine).y1 + 0.006),
                      ncol=TEMPLATE_NCOL, fontsize=LEGEND_FS,
                      handletextpad=0.35, columnspacing=1.0, frameon=False,
                      title='Prompt template index')
    tmpl.get_title().set_fontsize(LEGEND_TITLE_FS)

    fig.text(0.004, 0.996, '(a)', **PANEL_LAB)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, 'f5_crossover.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
