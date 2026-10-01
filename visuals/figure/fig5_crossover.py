#!/usr/bin/env python3
"""Manuscript Figure 5: the template-to-chemistry crossover, measured and shown.

    python visuals/figure/fig5_crossover.py [--out-dir DIR]

  f5_crossover.pdf    (a) kNN label agreement vs layer; (b-e) t-SNE at two layers, two colourings

Panel (a) is the measurement: how often a point's 10 nearest neighbours carry its own prompt
template, and how often they carry its own functional group, at every layer. The two curves
cross. Panels (b)-(e) are that crossover shown rather than asserted -- the same embeddings at
layer 8 and layer 24, each coloured BOTH ways, so the swap is visible as a 2x2 rather than
inferred from one half of it.

Models are columns here and the four layer/colouring combinations are rows, which transposes
the released grid (`visuals/tsne/tsne_depth_*.png`, models in rows and layers in columns). At
6.5in a five-column row leaves ~1.2in per embedding; ten across would leave 0.65in.

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

WIDTH, HEIGHT = 6.5, 7.3      # included at \textwidth; a near-full-page figure

# The four scatter rows, in reading order: both colourings at each layer, layers grouped.
# `key` is the CSV column `tsne.panel` filters on.
ROWS = [
    (8,  'template_index',   'Layer 8\nTemplate'),
    (8,  'functional_group', 'Layer 8\nGroup'),
    (24, 'template_index',   'Layer 24\nTemplate'),
    (24, 'functional_group', 'Layer 24\nGroup'),
]

# Legend geometry, following grid_figure's own choices so the keys read the same as in the
# released figures: 20 functional groups wrap at 7, 10 templates fit one row.
FINE_NCOL, TEMPLATE_NCOL = 7, 10

PANEL_LAB = dict(fontsize=10, fontweight='bold', va='top', ha='left')

ROWLAB_FS = 7.0
COLTITLE_FS = 8.0
LEGEND_FS = 6.4
LEGEND_TITLE_FS = 6.8


def retune():
    """Rebind the generators' sizes for composite scale.

    Module globals read when the drawing functions run, so this changes only this figure --
    the standalone generators keep their own sizes and their released output is unaffected.
    """
    # (a) is drawn at roughly half the standalone's 5.0in width, so its marks come down with
    # it or they swamp 32 layers of line.
    agr.AGR_MARK = 2.0
    agr.AGR_LW = 1.0
    agr.AGR_REF_FS = 5.4
    # Each embedding is ~1.2in against the released grid's 1.28in, so the points barely move.
    tsne.POINT_SIZE = 3.0
    tsne.POINT_ALPHA = 0.75
    plt.rcParams.update({'axes.labelsize': 7.5, 'axes.titlesize': 8.0,
                         'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5})


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
    """(b)-(e) -- four rows of five embeddings, models in columns."""
    sub = gs_cell.subgridspec(len(ROWS), len(models), wspace=0.06, hspace=0.06)
    for i, (layer, key, rowlab) in enumerate(ROWS):
        for j, m in enumerate(models):
            ax = fig.add_subplot(sub[i, j])
            tsne.panel(ax, coords[(m, layer)], key, styles[key])
            if i == 0:
                ax.set_title(DISPLAY5[m], pad=3, fontsize=COLTITLE_FS)
            if j == 0:
                ax.set_ylabel(rowlab, labelpad=4, fontsize=ROWLAB_FS)


def key_handles(styles, labels):
    """Legend proxies for one colouring, as `grid_figure` builds them."""
    return [plt.Line2D([], [], linestyle='none', marker=tsne.as_pair(styles[v])[1],
                       markersize=3.0, markeredgewidth=0,
                       color=tsne.as_pair(styles[v])[0], label=tsne.display(v))
            for v in labels]


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
              and all((m, layer) in coords for layer, _, _ in ROWS)]
    if not models:
        raise SystemExit('no rows -- run analysis/analyze_prompt_structure.py and '
                         'analysis/compute_tsne_coords.py')
    layers = sorted({L for (_, L) in rows})

    flat = [r for v in coords.values() for r in v]
    fine_styles, fine_labels = tsne.fine_style(flat)
    tmpl_styles, tmpl_labels = tsne.template_style(flat)
    styles = {'functional_group': fine_styles, 'template_index': tmpl_styles}

    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.25],
                          left=0.085, right=0.985, top=0.962, bottom=0.145,
                          hspace=0.26)
    agreement_block(fig, gs[0, 0], rows, models, layers)
    scatter_block(fig, gs[1, 0], coords, models, styles)

    # Three legends, each under the block it explains. They describe unrelated encodings --
    # five checkpoints, ten templates, twenty groups -- so merging any two would make one
    # oversized box in which none of them reads.
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.700),
               ncol=len(models), fontsize=LEGEND_FS, handletextpad=0.4, columnspacing=1.4,
               frameon=False)

    tmpl = fig.legend(handles=key_handles(tmpl_styles, tmpl_labels), loc='lower center',
                      bbox_to_anchor=(0.5, 0.072), ncol=TEMPLATE_NCOL, fontsize=LEGEND_FS,
                      handletextpad=0.35, columnspacing=1.0, frameon=False,
                      title='Prompt template index')
    tmpl.get_title().set_fontsize(LEGEND_TITLE_FS)
    fig.add_artist(tmpl)

    fine = fig.legend(handles=key_handles(fine_styles, fine_labels), loc='lower center',
                      bbox_to_anchor=(0.5, 0.0), ncol=FINE_NCOL, fontsize=LEGEND_FS,
                      handletextpad=0.35, columnspacing=1.0, frameon=False,
                      title='Functional group')
    fine.get_title().set_fontsize(LEGEND_TITLE_FS)

    fig.text(0.004, 0.996, '(a)', **PANEL_LAB)
    fig.text(0.004, 0.690, '(b)', **PANEL_LAB)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, 'f5_crossover.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
