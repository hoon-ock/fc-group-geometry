#!/usr/bin/env python3
"""Supplementary Figure S1: what each checkpoint actually emits, per functional group.

    python visuals/figure/figs1_composition.py [--out-dir DIR]

  fs1_composition.pdf    five checkpoint rows, 20 groups, stacked response composition

Each bar decomposes the 10 free generations for one group into what the model did with them:
answered correctly, named only the superclass, answered with the wrong group, named no group at
all, or emitted something malformed. The black dash is teacher-forced recall for the same
group -- what the model scores when it is not required to produce the answer itself.

The point the SI figure carries is the one the main text argues from: the stacked colours move
a great deal between checkpoints while the dashes barely move, so the behavioural ordering is
set by whether a model commits to an answer rather than by whether it knows the chemistry.

PDF, not PNG, like the other scripts here: the destination is a LaTeX manuscript whose figures
are vector. It is not one of the 70 release figures and `run_local.sh` does not build it.

Composed from the standalone generator rather than reimplemented -- `composition_panel` was
lifted out of `make_visuals_generation_composition.fig_response_composition` for this figure,
and that module's released output is byte-identical across the change.

Reads `analysis/data/group_response_composition.csv` through that generator -- never
`results/`, which is gitignored.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
sys.path.insert(0, HERE)

from _common import BEHAVIOURAL, VISUALS  # noqa: E402
import make_visuals_generation_composition as comp_gen  # noqa: E402
import fig2_clustermaps as f2  # noqa: E402

DEFAULT_OUT = os.path.join(VISUALS, 'figure')

WIDTH, HEIGHT = 6.5, 6.7      # the released 11.1 x 11.4in, scaled to \textwidth

XTICK_ROT = 55


def retune():
    """Rebind the generator's sizes for composite scale.

    Module globals read when the drawing function runs, so this changes only this figure --
    the standalone generator keeps its own sizes and its released output is unaffected.
    """
    scale = WIDTH / comp_gen.COMP_SIZE_W          # 6.5 / 11.1 = 0.586
    comp_gen.COMP_YLABEL = 9.0
    comp_gen.COMP_TICK = 7.0
    comp_gen.COMP_LEGEND = 7.0
    # The teacher-forced dash is a scatter marker, so its width is in POINTS and does not
    # shrink with the axes. `s` is points squared, so matching the narrower bars means scaling
    # by the SQUARE of the width ratio -- at the released 214 every dash would overhang here.
    comp_gen.COMP_TF_MARK = round(comp_gen.COMP_TF_MARK * scale ** 2)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    args = p.parse_args()

    f2.style()
    retune()

    comp = comp_gen.read_composition()
    missing = [m for m in BEHAVIOURAL if not any(k[0] == m for k in comp)]
    if missing:
        raise SystemExit('no rows for ' + ', '.join(missing) + ' in '
                         'analysis/data/group_response_composition.csv -- re-run '
                         'analysis/analyze_generation.py')
    groups = sorted({g for (_, g) in comp})

    # Explicit margins rather than `tight_layout`: the legend sits outside the axes, so
    # tight_layout cannot see it and warns, and the released figure only gets away with it at
    # 11.1in wide.
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    gs = fig.add_gridspec(len(BEHAVIOURAL), 1, left=0.085, right=0.995,
                          top=0.945, bottom=0.115, hspace=0.16)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(len(BEHAVIOURAL))]
    for mi, model in enumerate(BEHAVIOURAL):
        comp_gen.composition_panel(axes[mi], comp, model, groups, label=(mi == 0))
        if mi < len(BEHAVIOURAL) - 1:
            axes[mi].set_xticks([])

    axes[-1].set_xticks(np.arange(len(groups)))
    axes[-1].set_xticklabels(groups, rotation=XTICK_ROT, ha='right',
                             fontsize=comp_gen.COMP_TICK)
    # Six entries in one row, as the released figure has them, but anchored to the FIGURE and
    # centred. The released version hangs the legend off the top axes' left edge, which at
    # 11.1in has room to spare; at 6.5in that pushed it 0.4in past the canvas, and tight bbox
    # widens the page rather than clipping.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=6,
               fontsize=comp_gen.COMP_LEGEND, handletextpad=0.5, columnspacing=1.1,
               frameon=False)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, 'fs1_composition.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
