#!/usr/bin/env python3
"""Per-functional-group free-generation response composition, one panel per checkpoint.

    python visuals/make_visuals_generation_composition.py [--out-dir DIR]

  group_response_composition.png    5 stacked-bar panels, one per checkpoint

Each group is one stacked bar of the five adjudicated response types, with teacher-forced
raw recall overlaid as a tick. The five types PARTITION every prompt, so each bar sums to 1
and the stack is readable as a whole rather than as five independent rates.

**The four halides are the case to read.** Bar after bar of `underspecified`, under a tick
near the top: the model names the family and not the member, so strict scoring files it as a
failure while the forced-choice head ranks the exact class at 0.875-0.950. That is a
resolution effect in the scoring, not a representation deficit -- and it is visible here
because the composition is drawn rather than collapsed to an accuracy number. No correlation
against an embedding metric could have revealed it.

Reads only `analysis/data/group_response_composition.csv`, which
`analysis/analyze_generation.py` snapshots. Never `results/`, which is gitignored.

There is no layer dimension. Generation outcomes come from what the model emitted, not from
a hidden state, so depth does not enter -- unlike every other figure in this directory.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import BEHAVIOURAL, REPO, VISUALS, load, num  # noqa: E402
from make_visuals_probe import DISPLAY5, style  # noqa: E402

DEFAULT_OUT = os.path.join(VISUALS, 'generation')

COMP_SIZE_W = 11.1
COMP_PANEL_H = 2.28       # per model row
COMP_YLABEL = 11.0        # model name down the left
COMP_TICK = 9.6           # x group names and y 0/0.5/1.0
COMP_LEGEND = 9.6
COMP_TF_MARK = 214        # scatter `s` for the teacher-forced tick. Marker size goes as
                          # sqrt(s), so 95 * 1.5**2 keeps the dash spanning the same
                          # fraction of a bar as before.

# Sentence case, capitalised: these are legend labels a reader sees, not field names. The
# first element of each tuple is the CSV column and is what must match the data.
RESPONSE_STACK = [
    ('strict', 'Answered, correct', '#0072B2'),
    ('underspecified', 'Superclass only', '#56B4E9'),
    ('answer_wrong', 'Answered, wrong', '#D55E00'),
    ('non_answer', 'Named no group', '#E69F00'),
    ('malformed', 'Malformed', '#BBBBBB'),
]


def read_composition():
    """`(model, functional_group) -> row` from the committed snapshot."""
    return {(r['model'], r['functional_group']): r
            for r in load('group_response_composition.csv')}


def composition_panel(ax, comp, model, groups, label=False):
    """One model's stacked composition, into a caller-supplied axes.

    Lifted out of `fig_response_composition` so a composite can place these rows in its own
    layout -- the move `make_visuals_probe.probe_panel` and
    `make_visuals_prompt_structure.agreement_panel` both record. `label` replaces the old
    `mi == 0` guard: only the row that passes it contributes legend handles, because every
    row draws the same six artists and six handles are wanted, not thirty. The legend itself
    is the caller's, and so are the x tick labels, which belong to the bottom row of whatever
    grid these land in.
    """
    ax.grid(True, axis='y', color='0.9', linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    for field, lab, color in RESPONSE_STACK:
        vals = np.array([num(comp[(model, g)], field) for g in groups])
        ax.bar(x, vals, 0.76, bottom=bottom, color=color, linewidth=0,
               label=lab if label else None)
        bottom += vals
    tf = [num(comp[(model, g)], 'tf_raw_recall') for g in groups]
    # `s` is in points squared and the dash width is therefore ABSOLUTE -- it does not track
    # the axes width. A composite drawing this narrower must scale COMP_TF_MARK by the square
    # of the width ratio or every dash overhangs its bar.
    ax.scatter(x, tf, marker='_', s=COMP_TF_MARK, color='0.12', linewidth=2.0,
               zorder=4,
               label='Teacher-forced recall' if label else None)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(DISPLAY5[model], fontsize=COMP_YLABEL)
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=COMP_TICK)


def fig_response_composition(comp, out_dir):
    groups = sorted({g for (_, g) in comp})
    fig, axes = plt.subplots(len(BEHAVIOURAL), 1,
                             figsize=(COMP_SIZE_W, COMP_PANEL_H * len(BEHAVIOURAL)),
                             sharex=True)
    for mi, model in enumerate(BEHAVIOURAL):
        composition_panel(axes[mi], comp, model, groups, label=(mi == 0))
    axes[-1].set_xticks(np.arange(len(groups)))
    axes[-1].set_xticklabels(groups, rotation=55, ha='right', fontsize=COMP_TICK)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc='lower left', bbox_to_anchor=(0, 1.02),
                   ncol=6, fontsize=COMP_LEGEND)
    fig.tight_layout()
    path = os.path.join(out_dir, 'group_response_composition.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, REPO)}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    style()

    print("Writing figures:")
    fig_response_composition(read_composition(), args.out_dir)


if __name__ == '__main__':
    main()
