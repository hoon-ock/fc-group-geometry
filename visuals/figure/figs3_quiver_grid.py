#!/usr/bin/env python3
"""Supplementary Figure S3: the three analogy cells of Figure 3b, across all five checkpoints.

    python visuals/figure/figs3_quiver_grid.py [--out-dir DIR]

  fs3_quiver_grid.pdf    5 checkpoints x 3 analogy cells, in the first three PCs

Figure 3b draws three cells for the base model. This is the same three for every checkpoint,
so the geometry that figure argues from can be read as a property of each model rather than of
one. Rows are checkpoints in the usual order; columns are the cells, left to right as in
Figure 3b.

How well a cell draws is a measurement, not a preference, and it varies a lot here. Only the
separation PERPENDICULAR to the arrows renders as a gap -- separation along them just slides
one arrow down the other -- so that over the arrow length decides whether a panel reads. At
layer 31, as a percentage:

                imine C4   imine C6   thioether C5
  Base               51         44            221
  Chem-R             53        107             73
  Chem-F             65         59             21
  Chem-DFM           54         33             36
  Reason             75         92            966

The imine columns sit in a workable 33-107% everywhere. The thioether column does not: at 21%
(Chem-F) the two arrows nearly collapse onto one line, and at 966% (Reason) the pairs are so
far apart that `equalize` zooms out and leaves two short strokes. That spread is the finding,
not a defect in the drawing -- the same quadruple is a tight analogy in some checkpoints and a
loose one in others.

PDF, not PNG, like the other scripts here: the destination is a LaTeX manuscript whose figures
are vector. It is not one of the 70 release figures and `run_local.sh` does not build it.

Reads `analysis/data/analogy_quiver3d_all.csv`, the five-checkpoint snapshot, through
`make_visuals_retrieval_3d.read_cells` -- never `results/`, which is gitignored. The released
figures keep using the single-model `analogy_quiver3d.csv`, so the 70-PNG set is untouched.
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

from _common import BEHAVIOURAL, FIGURE_LAYER, VISUALS  # noqa: E402
import make_visuals_retrieval_3d as q3d  # noqa: E402
from make_visuals_probe import DISPLAY5  # noqa: E402

DEFAULT_OUT = os.path.join(VISUALS, 'figure')
SOURCE = 'analogy_quiver3d_all.csv'

WIDTH, HEIGHT = 6.5, 8.3      # included at \textwidth; a dedicated-page figure

# The three cells of Figure 3b, by (a1, a2, b1, b2, carbon), with the column title each gets.
CELLS = [
    (('imine', 'aldehyde', 'amine', 'alcohol', 4), 'imine / amine, C4'),
    (('imine', 'aldehyde', 'amine', 'alcohol', 6), 'imine / amine, C6'),
    (('thioether', 'thiol', 'ether', 'alcohol', 5), 'thioether / ether, C5'),
]

# Marks are sized for the 8in standalone panel; these cells are ~2in, so they scale with the
# panel or the squares swamp the arrows they sit on.
MARK_SCALE = 0.40

# Vertical only. mplot3d leaves a wide margin around its cube, and letting the axes box overrun
# buys it back -- but any HORIZONTAL overrun throws the vertical axis name into the neighbouring
# column, which is what it did on Figure 3 until the bleed was split per axis.
BLEED_X, BLEED_Y = 0.0, 0.14

COLTITLE_FS = 9.0
ROWLAB_FS = 9.5


def retune():
    """Rebind the generator's type sizes for composite scale.

    Module globals read when the drawing function runs, so this changes only this figure --
    the standalone generator keeps its own sizes and its released output is unaffected.
    """
    q3d.Q3D_LABEL = 6.5       # PC-1 / PC-2 / PC-3
    q3d.Q3D_POINT = 6.5       # the four group names inside the axes
    q3d.Q3D_LEGEND = 9.0
    q3d.Q3D_ALERT = 6.0


def wanted(cells):
    """`(model, key) -> cell`, for the three cells this figure draws."""
    out = {}
    for c in cells:
        m = c['meta']
        key = (m['pair_a1'], m['pair_a2'], m['pair_b1'], m['pair_b2'],
               int(m['carbon_count']))
        out[(m['model'], key)] = c
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    p.add_argument('--layer', type=int, default=FIGURE_LAYER)
    args = p.parse_args()

    q3d.style()
    retune()

    have = wanted(q3d.read_cells(args.layer, source=SOURCE))
    missing = [(m, k) for m in BEHAVIOURAL for k, _ in CELLS if (m, k) not in have]
    if missing:
        raise SystemExit(
            f'no cell for {missing[:3]} at layer {args.layer} in analysis/data/{SOURCE} -- '
            're-run analysis/snapshot_quiver3d.py --all-cells for each model')

    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    gs = fig.add_gridspec(len(BEHAVIOURAL), len(CELLS),
                          left=0.050, right=0.995, top=0.952, bottom=0.070,
                          wspace=0.10, hspace=0.06)
    first = None
    for i, model in enumerate(BEHAVIOURAL):
        for j, (key, title) in enumerate(CELLS):
            ax = fig.add_subplot(gs[i, j], projection='3d')
            cell = have[(model, key)]
            first = first or cell
            q3d.panel(ax, cell, scale=MARK_SCALE)
            if i == 0:
                ax.set_title(title, pad=1, fontsize=COLTITLE_FS)
            bb = ax.get_position()
            ax.set_position([bb.x0 - bb.width * BLEED_X, bb.y0 - bb.height * BLEED_Y,
                             bb.width * (1 + 2 * BLEED_X), bb.height * (1 + 2 * BLEED_Y)])
            if j == 0:
                fig.text(0.012, bb.y0 + bb.height / 2, DISPLAY5[model], rotation=90,
                         ha='left', va='center', fontsize=ROWLAB_FS)

    # One legend for all fifteen panels, which is the point of gridding them.
    fig.legend(handles=q3d.legend_handles(first), loc='lower center',
               bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=q3d.Q3D_LEGEND,
               handlelength=1.9, columnspacing=1.6, frameon=False)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, 'fs3_quiver_grid.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
