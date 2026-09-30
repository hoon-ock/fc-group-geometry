#!/usr/bin/env python3
"""The probe-depth curve, all five checkpoints on one axes.

    python visuals/make_visuals_probe.py [--out-dir DIR]

  probe_depth.png    balanced accuracy vs layer, five curves

Balanced accuracy under leave-one-fine-group-out against depth, one curve per checkpoint,
with the chance line and the character-n-gram surface baseline drawn in. The gap between a
curve and that baseline is the readable quantity -- the baseline is 0.475 for every model,
because it is fit on the prompt strings and never sees a checkpoint.

No transition marker is drawn. This figure makes no claim about any single layer step; it
shows five curves and lets the reader locate the climb.

This module also owns the five-model palette (`COLOR5`, `DASH5`, `MARKER5`, `DISPLAY5`) that
the other generators import from here rather than each keeping a copy.

Reads `data/layer_curves.csv` and `data/surface_baseline.csv` -- never
`results/`, which is gitignored.
"""
import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import (BASE, BEHAVIOURAL, CHEM, CHEMDFM, CHEM_R,  # noqa: E402
                     LAYERS, R1, REPO, VISUALS, load, num)

DEFAULT_VISUALS = os.path.join(VISUALS, 'probe')
GRID = dict(color='0.9', linewidth=0.6)
REF = dict(color='0.45', linewidth=0.9)

# Read on its own rather than as a column-width float, so the type is set well above the
# paper's 8pt -- the same reasoning as make_visuals.py's COH_* block.
PROBE_SIZE = (7.2, 4.2)
PROBE_LABEL = 13
PROBE_TICK = 12
PROBE_LEGEND = 11
PROBE_REF = 10

# `_common` defines COLOR/MARKER/DASH for the three models the paper draws. Those three keep
# their paper styling exactly; the two additions are given the remaining usable Okabe-Ito
# hues. Extended HERE and not in `_common` so nothing the manuscript draws can change.
#
# Checked rather than eyeballed: CIEDE2000 over all ten pairs puts the closest at 22.2
# (normal) / 12.7 (deuteranopic), above the 11.0 floor `_common`'s palette note cites as the
# level below which colour alone is not enough. That closest pair is chem-r against chem,
# which is the right place for it -- Chem-R-Faithful is GRPO-trained FROM Chem-R-8B, so near
# colours encode a real kinship. Every series also carries its own marker and dash pattern,
# so identity survives greyscale printing and never rests on hue alone.
COLOR5 = {BASE: '#0072B2', CHEM_R: '#E69F00', CHEM: '#D55E00',
          CHEMDFM: '#CC79A7', R1: '#009E73'}
MARKER5 = {BASE: 'o', CHEM_R: 'D', CHEM: 's', CHEMDFM: 'v', R1: '^'}
DASH5 = {BASE: (None, None), CHEM_R: (5, 1.5, 1, 1.5), CHEM: (4, 1.5),
         CHEMDFM: (2, 1.2), R1: (1, 1.5)}
# `_common.SHORT` calls Chem-R-Faithful `chem`, which is unreadable beside `chem-r`.
# Spelled out here.
SHORT5 = {BASE: 'base', CHEM_R: 'chem-r', CHEM: 'chem-faithful',
          CHEMDFM: 'chemdfm', R1: 'reason'}

# `SHORT5` is for TEXT OUTPUT -- the console summaries the retrieval and t-SNE generators
# print, where a full checkpoint name would wrap. No path uses it: files and directories
# carry the model's own identifier instead. `DISPLAY5` is what a reader sees on a figure. Chem-R-Faithful shortens to `Chem-F` rather than spelling `Faithful`
# out -- at this type size the full name is the longest label on every figure that draws it.
DISPLAY5 = {BASE: 'Base', CHEM_R: 'Chem-R', CHEM: 'Chem-F',
            CHEMDFM: 'Chem-DFM', R1: 'Reason'}


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


def recessive(ax):
    """Grid and spines recede; the data is the only thing that should be dark."""
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def read_curves():
    """`model -> layer -> balanced_acc`, and the surface baseline the paper figure draws.

    Both reads are `f1_probe_depth`'s verbatim. The baseline is averaged over every row of
    `surface_baseline.csv` exactly as there: it is a character n-gram classifier that never
    sees a model, and all five rows carry the same 0.475 over 19 folds, so the mean is a
    single number rather than a blend of five different ones.
    """
    curves = defaultdict(dict)
    for r in load('layer_curves.csv'):
        curves[r['model']][int(r['layer'])] = num(r, 'balanced_acc')
    surface = np.mean([num(r, 'mean_balanced_acc')
                       for r in load('surface_baseline.csv')])
    return curves, float(surface)


def probe_figure(curves, surface):
    """One axes: balanced accuracy against depth, five models, untitled."""
    fig, ax = plt.subplots(figsize=PROBE_SIZE)

    for m in BEHAVIOURAL:
        layers = sorted(curves[m])
        dash = DASH5[m]
        ax.plot(layers, [curves[m][L] for L in layers], color=COLOR5[m],
                marker=MARKER5[m], markersize=4.6, linewidth=1.8,
                dashes=dash if dash[0] else (), label=DISPLAY5[m])

    # 0.20 is the balanced accuracy of a guesser drawing uniformly from the FIVE labels the
    # probe can emit -- recall 0.20 on each true family. It is not the only uninformed
    # strategy and not the hardest: a constant 'always oxygen' classifier scores 0.25 over
    # the four evaluated families. Labelled 'Random guess' rather than 'chance' so the
    # number follows from the strategy named.
    ax.axhline(0.20, linestyle=':', **REF)
    # Both reference labels sit at layer 8, where every curve is well clear of them; at the
    # right edge they collided with the curves converging on 1.0.
    ax.text(8, 0.215, 'Random guess', ha='left', va='bottom', fontsize=PROBE_REF,
            color='0.45')
    ax.axhline(surface, linestyle='--', **REF)
    ax.text(8, surface + 0.018, 'Character $n$-gram baseline', ha='left', va='bottom',
            fontsize=PROBE_REF, color='0.45')

    ax.set_xlabel('Layer', fontsize=PROBE_LABEL)
    ax.set_ylabel('Balanced accuracy', fontsize=PROBE_LABEL)
    ax.set_xticks(LAYERS)
    # Unchanged from the paper panel, so the two are read on the same scale. The empty band
    # below the curves is where the chance line and its label live.
    ax.set_ylim(0.15, 1.04)
    ax.tick_params(labelsize=PROBE_TICK)
    # Outside the axes entirely: the plot area has two reference lines across its full width
    # and five curves converging top-right, with no corner left to put a legend in.
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3,
              fontsize=PROBE_LEGEND, handlelength=2.8, columnspacing=1.2)
    recessive(ax)
    return fig


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=DEFAULT_VISUALS)
    args = p.parse_args()
    style()

    curves, surface = read_curves()
    missing = [m for m in BEHAVIOURAL if m not in curves]
    if missing:
        raise SystemExit(
            'no rows for ' + ', '.join(missing) + ' in data/layer_curves.csv -- '
            're-run analysis/analyze_sweep.py')

    os.makedirs(args.out_dir, exist_ok=True)
    fig = probe_figure(curves, surface)
    path = os.path.join(args.out_dir, 'probe_depth.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {os.path.relpath(path, REPO)}')
    print(f'{len(BEHAVIOURAL)} models x {len(curves[BASE])} layers, '
          f'surface baseline {surface:.3f}')


if __name__ == '__main__':
    main()
