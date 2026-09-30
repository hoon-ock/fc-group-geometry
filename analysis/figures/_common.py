#!/usr/bin/env python3
"""Shared loading and naming for the figure generators in this directory.

Every `make_visuals*.py` here reads only `analysis/*/data/*.csv` -- never `results/`,
which is gitignored. That is deliberate and is the reproducibility test: if a generator
ever needs `results/`, a snapshot step was missed upstream, and the figure it draws cannot
be rebuilt from a clean checkout.

Figures are written to `analysis/visuals/<topic>/`. Each generator sets its own output
directory; there is no shared default.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.abspath(os.path.join(ANALYSIS, '..')))
import paths  # noqa: E402

REPO = paths.ROOT

BASE = 'meta-llama-Llama-3.1-8B'
CHEM = 'phenixace-Chem-R-Faithful'
R1 = 'deepseek-ai-DeepSeek-R1-Distill-Llama-8B'
CHEM_R = 'weidawang-Chem-R-8B'
CHEMDFM = 'OpenDFM-ChemDFM-v1.5-8B'

# Presentation order everywhere: base, then the Tier-1 chemistry pair in TRAINING order
# (Chem-R-Faithful is GRPO-trained FROM Chem-R-8B, so they are a pair, not two independent
# samples), then the off-base chemistry model, then the reasoning distill.
#
# Order is append-only. Every generator iterates this list, so appending leaves existing
# panels in place and adds new ones after them; inserting would reshuffle every figure.
BEHAVIOURAL = [BASE, CHEM_R, CHEM, CHEMDFM, R1]

FIGURE_LAYER = 31
# First layer at which ALL FIVE models exceed 0.94 balanced accuracy. Also chem's own best
# probe layer, which is why the N/O error-composition story can be read at exactly the
# depth chem's headline number comes from. Verified against probe/data/layer_curves.csv.

LAYERS = [0, 8, 16, 24, 31]
N_LAYERS = 32


def load(rel):
    """Read one snapshot CSV by path relative to `analysis/`."""
    path = os.path.join(ANALYSIS, rel)
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {os.path.relpath(path, REPO)} -- run the analyze_*.py that writes it")
    with open(path) as fh:
        return list(csv.DictReader(fh))


def num(row, field):
    v = row[field]
    return float(v) if v not in ('', 'None') else float('nan')
