#!/usr/bin/env python3
"""Manuscript Figure 3: rank violins over two analogy quivers.

    python visuals/figure/fig3_retrieval.py [--out-dir DIR]

  f3_retrieval.pdf    (a) rank distributions, three axes; (b, c) two analogy cells in 3-D

Across the top is the retrieval rank distribution on all three axes, five checkpoints each.
Beneath it are two analogy cells drawn in the first three principal components, sharing one
legend. Each block gets the full text width, and each legend sits under its own block.

PDF, not PNG, like the other scripts here: the destination is a LaTeX manuscript whose figures
are vector. It is not one of the 70 release figures and `run_local.sh` does not build it.

Composed from the two standalone generators rather than reimplemented: `violin_panel` and
`panel` both take an axes, so this supplies the layout and they supply the drawing. Their type
sizes are module globals read at call time, which is what makes re-tuning possible -- the
sources are 10x4.8in at 14pt and 8x7.8in at 20pt, so composing is roughly a 0.5x downscale and
a naive shrink would put the violin ticks near 7pt. The constants are rebound below for the
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

WIDTH, HEIGHT = 6.5, 5.8      # included at \textwidth, not the 5.5in the other figures use

# The two cells, by the stem `slug()` builds.
#
# The same analogy at two chain lengths, which is what the two panels together show: the
# offset survives the backbone growing.
#
# Which cells are legible is a measurement, not a preference. Only the separation
# PERPENDICULAR to the arrows can be drawn as a gap -- separation along them just slides one
# arrow down the other -- so the ratio of that to the arrow length decides whether a panel
# reads. At L31, against arrow lengths of 10-24:
#
#   ester-carboxylic-acid / ketone-aldehyde C4     2.8%   arrows collapse into one line
#   imine-aldehyde / amine-alcohol        C4/5/6   44-51% both arrows read, parallel
#   thioether-thiol / ether-alcohol       C5/C6    221%/417%  pairs far apart, arrows tiny
#
# The ester cell was tried first and defeated three cameras before this was measured; the
# thioether pair is the one the abstract names but separates the pairs so widely that
# `equalize` zooms out and leaves two short strokes. 44-51% is the band that works.
CELLS = ('imine-aldehyde_amine-alcohol_C4',
         'imine-aldehyde_amine-alcohol_C6',
         'thioether-thiol_ether-alcohol_C5')

# One camera override, set by measurement rather than by eye.
#
# `pick_camera` maximises the SMALLEST gap between the five labelled points, which is the right
# objective when the risk is two names overprinting. For the thioether cell the binding problem
# is different -- the arrows are 10.03 long where the imine cells' are 23, against a cube
# `equalize` sizes from all 20 points -- so the objective to maximise is projected arrow length,
# subject to the arrows still reading as two (projected separation across them >= 0.6x their
# length) and no labelled pair collapsing (smallest gap within 10% of pick_camera's).
#
# Swept over the generator's own ELEVS x AZIMS through its `screen()`:
#
#   thioether C5   pick_camera (16, 105)  arrow 0.239   ->  (32, 70)  arrow 0.426   1.78x
#   imine C4       pick_camera (20,  30)  arrow 1.050   ->  best is 0.96x, no gain
#   imine C6       pick_camera (32,-145)  arrow 0.836   ->  1.39x, but perp/arrow 0.56 -> 0.42
#
# Only the first is taken. C6's longer arrows come by trading away the separation ratio that
# decides legibility in the first place, and it already renders.
CAMERAS = {'thioether-thiol_ether-alcohol_C5': (32, 70)}

# Marks are sized for the 8in standalone panel; these are far smaller, so they scale with the
# panel or the squares swamp the arrows they sit on.
MARK_SCALE = 0.42

# Fraction of each quiver cell the axes box overruns, per axis -- see quiver_block.
#
# Separate because the two directions are limited by different things. Vertically the only
# constraint is the shared legend below, so the box can overrun and buy back the margin
# mplot3d leaves around its cube. Horizontally it cannot: mplot3d hangs the vertical axis name
# well outside the cube on the left, so any horizontal overrun throws `PC-3` into the
# neighbouring panel -- at three cells across, every panel was printing its PC-3 on top of the
# one to its left, and widening the gutter only moved the collision rather than removing it.
#
# The outermost box must also stay on canvas: `savefig.bbox='tight'` does not clip, it silently
# widens the figure.
BLEED_X, BLEED_Y = 0.0, 0.16

XLAB_FS = 8.5     # 'Inter-group' etc, sized apart from the y ticks

PANEL_LAB = dict(fontsize=10, fontweight='bold', va='top', ha='left')


def retune():
    """Rebind the generators' type sizes for composite scale.

    These are module globals used as `fontsize=` arguments, read when the drawing functions
    run, so rebinding here changes this figure only -- the standalone generators keep their
    own sizes and their released output is unaffected.
    """
    # RETR_TICK drives the y tick labels AND, in violin_panel, the x label. The composite
    # re-sets the x label after calling it, so the two can be sized apart here: ticks small,
    # label large.
    rdist.RETR_TICK = 5.4         # y tick labels only, by the seam above
    rdist.RETR_LABEL = 9.5        # 'Rank (log scale)'
    rdist.DIST_LEGEND = 8.5
    rdist.HIT_FS = 8.0            # five values across a 1.56in panel; 0.24in of its 0.31in slot
    q3d.Q3D_LABEL = 6.2           # PC-1 / PC-2 / PC-3
    q3d.Q3D_POINT = 6.2           # the four group names inside the axes
    q3d.Q3D_LEGEND = 7.5
    q3d.Q3D_ALERT = 6.0


def violin_block(fig, gs_cell, layer):
    """(a) -- the three retrieval axes across the full width, one legend below them."""
    by, labels = rviolin.read_axis_rows(layer)
    hist = rviolin.rank_hist(layer)
    sub = gs_cell.subgridspec(1, 3, wspace=0.36)
    axs = [fig.add_subplot(sub[0, i]) for i in range(3)]
    for j, (ax, axis) in enumerate(zip(axs, rdist.AXES)):
        rviolin.violin_panel(ax, by, hist, axis, labels[axis])
        # The hit@1 row above each panel, from the same helper the standalone figures use so
        # the two cannot print different values for the same column.
        rdist.annotate_hit1(ax, by, axis, show_key=(j == 0))
        # Axis name only. The standalone label carries '(13 groups, 20 trials)' on a second
        # line; the counts belong in the caption, not stacked under every panel.
        ax.set_xlabel(labels[axis].split('\n')[0], fontsize=XLAB_FS)
    axs[0].set_ylabel('Rank (log scale)', fontsize=rdist.RETR_LABEL)
    return axs


def quiver_block(fig, cells, gs_cells):
    """(b) and (c) -- two analogy cells side by side, one legend beneath the pair."""
    axes = []
    for cell, gc in zip(cells, gs_cells):
        ax = fig.add_subplot(gc, projection='3d')
        q3d.panel(ax, cell, camera=CAMERAS.get(cell['label'].get('stem')),
                  scale=MARK_SCALE)
        # mplot3d fits its cube inside the axes box and leaves a wide margin around it, so a
        # cell-sized axes draws a small cube in a lot of white. The standalone figure buys that
        # back by letting the box run past the figure edges (AXES_BLEED); doing the same per
        # cell is what keeps the cubes large in a half-width cell.
        bb = ax.get_position()
        ax.set_position([bb.x0 - bb.width * BLEED_X, bb.y0 - bb.height * BLEED_Y,
                         bb.width * (1 + 2 * BLEED_X), bb.height * (1 + 2 * BLEED_Y)])
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

    # (a) across the top, (b) and (c) side by side beneath it. Stacking the quivers down a
    # narrow right-hand column had squeezed the violins to about 1.2in each; full width gives
    # them 2.0in and the quivers 3.1in, so both blocks roughly double.
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.08],
                          left=0.085, right=0.958, top=0.962, bottom=0.150,
                          hspace=0.34)
    violin_block(fig, gs[0, 0], args.layer)
    qgs = gs[1, 0].subgridspec(1, 3, wspace=0.10)
    quiver_block(fig, [cells[w] for w in CELLS], [qgs[0, i] for i in range(3)])

    # The violin's own legend: five checkpoints plus the three summary marks. Kept separate
    # from the quiver legend because the two describe unrelated encodings, and merging them
    # would make one oversized box in which neither reads.
    vhandles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR5[m], edgecolor=COLOR5[m],
                              alpha=rviolin.BODY['alpha']) for m in BEHAVIOURAL]
    vhandles += [Line2D([], [], **rdist.TICK), Line2D([], [], **rdist.MEAN),
                 Line2D([], [], color='0.35', linewidth=1.2, linestyle='--')]
    vnames = [DISPLAY5[m] for m in BEHAVIOURAL] + ['Median', 'Mean', 'Random']
    fig.legend(vhandles, vnames, loc='upper center', bbox_to_anchor=(0.5, 0.586),
               ncol=8, fontsize=rdist.DIST_LEGEND, handlelength=1.5, columnspacing=1.2,
               frameon=False)

    # One legend for both quivers, under the pair rather than beside them.
    fig.legend(handles=q3d.legend_handles(cells[CELLS[0]]), loc='lower center',
               bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=q3d.Q3D_LEGEND,
               handlelength=1.9, columnspacing=1.6, frameon=False)

    fig.text(0.004, 0.996, '(a)', **PANEL_LAB)
    fig.text(0.004, 0.516, '(b)', **PANEL_LAB)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, 'f3_retrieval.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
