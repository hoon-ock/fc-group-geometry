#!/usr/bin/env python3
"""Manuscript Figure 5: the template-to-chemistry crossover, measured.

    python visuals/figure/fig5_crossover.py [--out-dir DIR]

  f5_crossover.pdf    kNN label agreement vs layer, two labels, five checkpoints

How often a point's 10 nearest neighbours carry its own prompt template, and how often they
carry its own functional group, at every layer. The two curves cross: through the first half of
the stack a point's neighbourhood is organised by which template produced the prompt, and
chemistry overtakes it around layer 19. From there the checkpoints separate -- every chemistry
fine-tune ends BELOW base on template and ABOVE it on fine groups, while the reasoning distill
tracks base.

Each panel carries its own random-neighbour floor rather than one shared line: the two labels
differ in class count and balance (10 balanced templates against 20 unbalanced groups), so a
single reference would make them look comparable when they are not.

The embeddings that SHOW this crossover -- the same points at layers 8 and 24, coloured both
ways -- used to be panels (b) and (c) here. They moved to Figure 4, where they sit under the
probe curve that measures the same transition at the same two depths, so the measurement and
the picture of it are now in one figure rather than two. What is left is the pair of agreement
panels, lettered (a) and (b): they are two measurements against two different label sets with
two different floors, not one plot in two halves, so each is referable on its own.

PDF, not PNG, like the other scripts here: the destination is a LaTeX manuscript whose figures
are vector. It is not one of the 70 release figures and `run_local.sh` does not build it.

Composed from the standalone generator rather than reimplemented.
`prompt_structure.agreement_panel` was lifted out of that module's `main()` for this figure,
and its released output is byte-identical across the change.

Reads `analysis/data/label_agreement_by_layer.csv` through that generator -- never `results/`,
which is gitignored.
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

from _common import BEHAVIOURAL, VISUALS  # noqa: E402
import make_visuals_prompt_structure as agr  # noqa: E402
import fig2_clustermaps as f2  # noqa: E402

DEFAULT_OUT = os.path.join(VISUALS, 'figure')

# A short, wide figure. It was 8.45in tall while it also carried the five-by-four embedding
# grid and that grid's two colour keys; with those in Figure 4 this is two curve panels and a
# model legend, and anything taller would be white space.
WIDTH, HEIGHT = 6.5, 3.05

LEGEND_FS = 9.0

# 12, matching Figure 4. The two figures were one until the embedding grid moved, and they are
# read together, so a reader should not meet two sizes of panel letter across them.
PANEL_LAB = dict(fontsize=12, fontweight='bold', va='bottom', ha='left')

# Margins, measured against the furniture rather than guessed. `bottom` holds the x tick
# labels, the 'Layer' xlabel and the one-row model legend anchored at y=0; `top` holds the two
# panel titles AND the panel letters, which sit on the titles' own top edge and so need their
# own height above it -- at TOP 0.914 the letters ran off the canvas, which a tight bbox does
# not clip but silently pads around. The page grew 2.90 -> 3.05in to pay for that rather than
# taking it out of the axes, which still measure ~1.97in: the rotated y-label needs 1.61in of
# that at 10pt and anything shorter would force the type back down.
LEFT, RIGHT = 0.085, 0.985
TOP, BOTTOM = 0.875, 0.229


def retune():
    """Rebind the generator's sizes for this page.

    Module globals read when the drawing functions run, so this changes only this figure --
    the standalone generator keeps its own sizes and its released output is unaffected.

    These are UP from what this figure used while the panels were squeezed into the top
    quarter of a two-block page. The binding constraint was the rotated y-label: 'kNN label
    agreement (k=10)' measures 1.45in at 9pt against what was then a 1.50in panel, so 9 was
    the size that fit at all. The panel is 1.95in now, so the label fits at 10 with room.
    """
    agr.AGR_MARK = 3.0
    agr.AGR_LW = 1.3
    agr.AGR_REF_FS = 8.5
    plt.rcParams.update({'axes.labelsize': 10.0, 'axes.titlesize': 10.5,
                         'xtick.labelsize': 9.0, 'ytick.labelsize': 9.0})


def _fig_bbox(fig, artist):
    """An artist's drawn extent in figure coordinates."""
    bb = artist.get_window_extent(fig.canvas.get_renderer())
    return bb.transformed(fig.transFigure.inverted())


def agreement_block(fig, gs, rows, models, layers):
    """The two agreement panels, sharing the model legend below them."""
    axs = [fig.add_subplot(gs[0, i]) for i in range(len(agr.PANELS))]
    for ax, (label, title) in zip(axs, agr.PANELS):
        agr.agreement_panel(ax, rows, label, title, models, layers)
    # The y-label is the caller's job by the lifted function's contract, and the k comes from
    # the data rather than the prose so the two cannot disagree.
    axs[0].set_ylabel(f'kNN label agreement (k={rows[(models[0], layers[0])]["k"]})')
    return axs


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    args = p.parse_args()

    # Figure 2's style, not the probe module's. That one is tuned for a PNG save and omits
    # `pdf.fonttype: 42`, which is why this figure used to ship Type-3 bitmapped fonts into a
    # vector manuscript. Figures 2 and 4 already use this one, and Figure 4 now carries the
    # embedding grid that used to live here -- so the two would otherwise disagree on font
    # embedding for panels drawn by the same code.
    f2.style()
    retune()

    rows = agr.read_rows()
    models = [m for m in BEHAVIOURAL if any((m, L) in rows for L in range(32))]
    if not models:
        raise SystemExit('no rows -- run analysis/analyze_prompt_structure.py')
    layers = sorted({L for (_, L) in rows})

    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    gs = fig.add_gridspec(1, len(agr.PANELS), left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM,
                          wspace=0.17)
    axs = agreement_block(fig, gs, rows, models, layers)

    # Panel letters, on the shared top edge of the two titles rather than at a per-axes
    # height, so they sit on one line across the figure. Measured, not a figure-fraction
    # constant: this file's old labels were constants tuned against a 7.3in canvas and
    # misplaced themselves the moment HEIGHT changed, which is the whole reason Figure 4 now
    # derives its own.
    fig.canvas.draw()
    head_y = max(_fig_bbox(fig, a.title).y1 for a in axs) + 0.004
    for a, letter in zip(axs, ('(a)', '(b)')):
        # Left-aligned just outside each panel's own left edge. The first would land on the
        # rotated y-label at its axes edge, so it goes to the canvas margin instead.
        x = 0.004 if a is axs[0] else a.get_position().x0 - 0.052
        fig.text(x, head_y, letter, **PANEL_LAB)

    # One legend for both panels: they show the same five checkpoints against the same depth
    # axis, so a key per panel would be the same key twice.
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=len(models), fontsize=LEGEND_FS, handletextpad=0.4, columnspacing=1.4,
               frameon=False)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, 'f5_crossover.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
