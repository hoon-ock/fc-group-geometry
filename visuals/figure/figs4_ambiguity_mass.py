#!/usr/bin/env python3
"""Supplementary Figure S4: where the probe puts its probability mass on the ambiguous groups.

    python visuals/figure/figs4_ambiguity_mass.py [--out-dir DIR]

  fs4_ambiguity_mass.pdf    4 convention-labelled groups + 2 control blocks, 5 checkpoints each

Accuracy on these four groups is near 1.000 for every checkpoint, but the probability behind it
is split: for an amide the base model holds most of its mass on nitrogen and a sixth of it on
oxygen, which is the honest reading of a group whose label is a naming convention rather than a
fact about the molecule. The control blocks are oxygen-free members of the same families, and
are what make that an argument rather than an observation -- a uniformly diffuse model would
score the same way on both halves.

This was panel (b) of manuscript Figure 4. It moved here because it is a self-contained
argument about one measurement, and because it was the panel paying for that figure's layout:
sharing a page with the probe curve forced the model names and the accuracy column down to 7pt,
measured as 0.77pt of clearance between consecutive accuracy numbers. On its own page the row
pitch is roughly twice that, so `retune` below puts both back up to 9pt.

PDF, not PNG, like the other scripts here: the destination is a LaTeX manuscript whose figures
are vector. It is not one of the 70 release figures and `run_local.sh` does not build it.

Reads `analysis/data/ambiguity_family_probability.csv` through `make_visuals_ambiguity` --
never `results/`, which is gitignored. The released PNG that module writes is unaffected:
`retune` rebinds its module globals in this process only.
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
import make_visuals_ambiguity as amb  # noqa: E402
import fig2_clustermaps as f2  # noqa: E402

DEFAULT_OUT = os.path.join(VISUALS, 'figure')

WIDTH, HEIGHT = 6.5, 7.6      # included at \textwidth; a dedicated SI page

# Panel rectangle, solved against three constraints rather than chosen. `mass_panel` places
# the group names and their formulae OUTSIDE its axes, RIGHT-aligned so they END at axes
# fraction -0.185, so the left margin is a property of the panel rather than of the page.
# Measured on a draw, the widest label ('Nitrogen Controls') is 1.01in at GROUP_FS 9.5:
#
#   left    the label ends at B_LEFT - 0.185*B_WIDTH and runs 1.01in back from there, which
#           must still clear the canvas. At B_LEFT 0.225 / B_WIDTH 0.680 it overran by 0.37in,
#           and because savefig uses a tight bbox that did not clip -- it silently widened the
#           PDF to 6.72in, so LaTeX then scaled the whole figure down to fit \textwidth and
#           every type size landed smaller than the ones measured below.
#   gap     the same labels must not reach the model names inside the axes. They end 0.21in
#           short of them here.
#   right   B_LEFT + B_WIDTH + B_ACC_GAP + B_ACC_W stays inside 0.985.
#
# Re-measure all three together: changing B_WIDTH moves the labels as well as the bars.
B_LEFT, B_WIDTH = 0.290, 0.620
B_BOTTOM, B_HEIGHT = 0.062, 0.884
# The accuracy column, `mass_figure`'s own 0.10 of the bar width, carried over from Figure 4.
B_ACC_W = 0.054
B_ACC_GAP = 0.008


def retune():
    """Rebind the generator's type sizes for this page.

    Module globals read when the drawing function runs, so this changes only this figure --
    `make_visuals_ambiguity`'s own released PNG keeps its 23/21/17 and is unaffected.

    The released figure is 12.6in nominal wide against 6.5in here, so everything comes down --
    but NOT proportionally, because the binding constraint is row pitch rather than width.
    The axes hold 37.7 data units in `B_HEIGHT` of a 7.6in page, which is 6.7in and so about
    12.8pt of pitch per row. That is roughly twice what the same panel had inside Figure 4
    (6.5pt), which is the entire reason this figure exists: the two pitch-bound sizes, one
    model name and one accuracy number per bar, go back to 9pt with ~2pt of clearance instead
    of 7pt with 0.77pt.
    """
    amb.LABEL_FS = 11.0
    amb.DIVIDER_FS = 9.5
    amb.GROUP_FS = 9.5
    amb.LEGEND_FS = 10.0
    amb.TICK_FS = 9.5
    # The two pitch-bound sizes, and also the two that set the left margin -- see B_LEFT.
    # 8.0 against 12.8pt of pitch is generous vertically; what caps it is the horizontal
    # collision with the group names, not the row spacing.
    amb.MODEL_FS = 8.0
    amb.ACC_FS = 8.5
    amb.STRUCT_FS = 6.5


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    p.add_argument('--layer', type=int, default=FIGURE_LAYER,
                   help=f'any layer present in ambiguity_family_probability.csv (0-31). '
                        f'Default {FIGURE_LAYER}, matching the main text.')
    args = p.parse_args()

    # Figure 2's style, not the ambiguity module's: that one is tuned for a PNG save and omits
    # `pdf.fonttype: 42`, which would embed Type-3 bitmapped fonts in a vector manuscript.
    f2.style()
    retune()

    mass = amb.read_mass()
    missing = [m for m in BEHAVIOURAL if (m, args.layer, 'amide') not in mass]
    if missing:
        raise SystemExit(
            f'no layer-{args.layer} rows for {", ".join(missing)} in '
            'analysis/data/ambiguity_family_probability.csv -- re-run src/ambiguity_metric.py')

    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    # Explicit rects rather than a gridspec, and explicit margins rather than `tight_layout`:
    # the group names and the stack legend both sit outside the axes, so neither a gridspec's
    # own margins nor tight_layout can see them.
    axb = fig.add_axes([B_LEFT, B_BOTTOM, B_WIDTH, B_HEIGHT])
    axr = fig.add_axes([B_LEFT + B_WIDTH + B_ACC_GAP, B_BOTTOM, B_ACC_W, B_HEIGHT],
                       sharey=axb)
    amb.mathtext_serif()
    amb.mass_panel(axb, axr, mass, args.layer)

    # `mass_panel` sets the model names with `fontsize=MODEL_FS` and then calls
    # `ax.tick_params(labelsize=TICK_FS)` with no `axis=`, which applies to BOTH axes and
    # silently overrides it. Re-applied here rather than fixed upstream: adding `axis='x'`
    # there would change the released PNG, which is the reproducibility boundary. This
    # travelled with the panel out of fig4_depth_structure.py.
    for t in axb.get_yticklabels():
        t.set_fontsize(amb.MODEL_FS)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, 'fs4_ambiguity_mass.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
