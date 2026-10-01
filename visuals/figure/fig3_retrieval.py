#!/usr/bin/env python3
"""Manuscript Figure 3: rank violins beside two analogy quivers.

    python visuals/figure/fig3_retrieval.py [--out-dir DIR]

  f3_retrieval.pdf    (a) rank distributions, three axes; (b, c) two analogy cells in 3-D

Left is the retrieval rank distribution on all three axes, five checkpoints each. Right are
two analogy cells drawn in the first three principal components, sharing one legend.

PDF, not PNG, like the other scripts here: the destination is a LaTeX manuscript whose figures
are vector. It is not one of the 70 release figures and `run_local.sh` does not build it.

Composed from the two standalone generators rather than reimplemented: `violin_panel` and
`panel` both take an axes, so this supplies the layout and they supply the drawing. Their type
sizes are module globals read at call time, which is what makes re-tuning possible -- the
sources are 10x4.8in at 14pt and 8x7.8in at 20pt, so composing is roughly a 0.3x downscale and
a naive shrink would put the violin ticks near 4pt. The constants are rebound below for the
composite's scale; nothing in the shared modules is edited.

Reads `analysis/data/retrieval_by_axis.csv`, `analysis/data/retrieval_rank_hist.csv` and
`analysis/data/analogy_quiver3d.csv` through those generators -- never `results/`, which is gitignored.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))

from _common import FIGURE_LAYER, VISUALS  # noqa: E402
import make_visuals_retrieval_3d as q3d  # noqa: E402
import make_visuals_retrieval_dist as rdist  # noqa: E402
import make_visuals_retrieval_violin as rviolin  # noqa: E402
from make_visuals_probe import COLOR5, DISPLAY5  # noqa: E402
from _common import BEHAVIOURAL  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

DEFAULT_OUT = os.path.join(VISUALS, 'figure')

WIDTH, HEIGHT = 6.5, 5.2      # included at \textwidth, not the 5.5in the other figures use

# The two cells, by the stem `slug()` builds. The second is drawn face-on (see CAMERAS).
CELLS = ('imine-aldehyde_amine-alcohol_C4',
         'ester-carboxylic-acid_ketone-aldehyde_C4')

# `pick_camera` maximises the smallest gap between the five key POINTS, which says nothing
# about the arrows. For the ester cell it returns (32, 105), where the source and target
# offsets -- 97.5% parallel in the data, so genuinely near-collinear -- are also viewed nearly
# end-on and collapse onto each other.
#
# elev 54 is the plane's own tilt: the five points are near-coplanar (third singular value
# 2.5% of the first), so a view along the normal shows the parallelogram at full area. Area is
# not the same as legibility, though -- at azim 135 the two arrows still ran along the lower
# edge of the cube against the axis lines behind them. Ninety-five degrees further about z
# puts them across the open face, where the two strokes read as two.
CAMERAS = {'ester-carboxylic-acid_ketone-aldehyde_C4': (54, 230)}

# Marks are sized for the 8in standalone panel; these are 2.6in, so they scale with the
# panel or the squares swamp the arrows they sit on.
MARK_SCALE = 0.42

# Fraction of each quiver cell the axes box overruns on every side -- see quiver_block.
BLEED = 0.10

PANEL_LAB = dict(fontsize=10, fontweight='bold', va='top', ha='left')


def retune():
    """Rebind the generators' type sizes for composite scale.

    These are module globals used as `fontsize=` arguments, read when the drawing functions
    run, so rebinding here changes this figure only -- the standalone generators keep their
    own sizes and their released output is unaffected.
    """
    rdist.RETR_LABEL = 8.0        # 'Rank (log scale)'
    rdist.RETR_TICK = 6.2         # axis ticks and the per-panel caption
    rdist.DIST_LEGEND = 7.0
    rdist.HIT_FS = 5.0            # five values across a ~1in panel
    q3d.Q3D_LABEL = 7.0           # PC-1 / PC-2 / PC-3
    q3d.Q3D_POINT = 7.0           # the four group names inside the axes
    q3d.Q3D_LEGEND = 7.0
    q3d.Q3D_ALERT = 6.0


def violin_block(fig, gs_cell, layer):
    """(a) -- the three retrieval axes, sharing one legend below them."""
    by, labels = rviolin.read_axis_rows(layer)
    hist = rviolin.rank_hist(layer)
    sub = gs_cell.subgridspec(1, 3, wspace=0.52)
    axs = [fig.add_subplot(sub[0, i]) for i in range(3)]
    for j, (ax, axis) in enumerate(zip(axs, rdist.AXES)):
        rviolin.violin_panel(ax, by, hist, axis, labels[axis])
        # The hit@1 row above each panel, from the same helper the standalone figures use so
        # the two cannot print different values for the same column.
        rdist.annotate_hit1(ax, by, axis, show_key=(j == 0))
        # Axis name only. The standalone label carries '(13 groups, 20 trials)' on a second
        # line, which at this panel width runs into its neighbours; the counts belong in the
        # caption instead, where they are not competing for 1in of column.
        ax.set_xlabel(labels[axis].split('\n')[0], fontsize=rdist.RETR_TICK)
    axs[0].set_ylabel('Rank (log scale)', fontsize=rdist.RETR_LABEL)
    return axs


def quiver_block(fig, cells, gs_cells):
    """(b) and (c) -- two analogy cells, one legend between them."""
    axes = []
    for cell, gc in zip(cells, gs_cells):
        ax = fig.add_subplot(gc, projection='3d')
        q3d.panel(ax, cell, camera=CAMERAS.get(cell['label'].get('stem')),
                  scale=MARK_SCALE)
        # mplot3d fits its cube inside the axes box and leaves a wide margin around it, so a
        # cell-sized axes draws a small cube in a lot of white. The standalone figure buys that
        # back by letting the box run past the figure edges (AXES_BLEED); doing the same per
        # cell is what lets the right column narrow without the cubes shrinking with it.
        bb = ax.get_position()
        ax.set_position([bb.x0 - bb.width * BLEED, bb.y0 - bb.height * BLEED,
                         bb.width * (1 + 2 * BLEED), bb.height * (1 + 2 * BLEED)])
        axes.append(ax)
    return axes


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    p.add_argument('--layer', type=int, default=FIGURE_LAYER)
    args = p.parse_args()

    q3d.style()
    retune()

    cells = {}
    for c in q3d.read_cells(args.layer):
        stem = q3d.slug(c, args.layer)
        for want in CELLS:
            if stem.endswith(want):
                c['label']['stem'] = want
                cells[want] = c
    missing = [w for w in CELLS if w not in cells]
    if missing:
        raise SystemExit(f'no quiver cell for {missing} at layer {args.layer} -- '
                         're-run analysis/snapshot_quiver3d.py')

    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    gs = fig.add_gridspec(2, 2, width_ratios=[4.15, 2.05], height_ratios=[1, 1],
                          left=0.062, right=0.972, top=0.88, bottom=0.115,
                          wspace=0.17, hspace=0.05)
    violin_block(fig, gs[:, 0], args.layer)
    quiver_block(fig, [cells[w] for w in CELLS], [gs[0, 1], gs[1, 1]])

    # The violin's own legend: five checkpoints plus the three summary marks. Kept separate
    # from the quiver legend because the two describe unrelated encodings, and merging them
    # would make one oversized box in which neither reads.
    vhandles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR5[m], edgecolor=COLOR5[m],
                              alpha=rviolin.BODY['alpha']) for m in BEHAVIOURAL]
    vhandles += [Line2D([], [], **rdist.TICK), Line2D([], [], **rdist.MEAN),
                 Line2D([], [], color='0.35', linewidth=1.2, linestyle='--')]
    vnames = [DISPLAY5[m] for m in BEHAVIOURAL] + ['Median', 'Mean', 'Random']
    fig.legend(vhandles, vnames, loc='lower left', bbox_to_anchor=(0.012, 0.005),
               ncol=4, fontsize=rdist.DIST_LEGEND, handlelength=1.5, columnspacing=1.0,
               frameon=False)

    # One legend for both quivers, which is the point of stacking them.
    fig.legend(handles=q3d.legend_handles(cells[CELLS[0]]), loc='upper right',
               bbox_to_anchor=(0.995, 1.0), ncol=2, fontsize=q3d.Q3D_LEGEND,
               handlelength=1.9, columnspacing=1.2, frameon=False)

    fig.text(0.004, 0.985, '(a)', **PANEL_LAB)
    fig.text(0.672, 0.905, '(b)', **PANEL_LAB)
    fig.text(0.672, 0.500, '(c)', **PANEL_LAB)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, 'f3_retrieval.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
