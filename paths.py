#!/usr/bin/env python3
"""Every path in this project, resolved from this file's location.

The layout this code came from required two things that a released repository cannot
assume: that the working directory was the repository root, and that the project
directory was named `fc_group`. Constants like `'fc_group/Results/...'` were relative to
the first and spelled the second. Import this module instead and neither holds -- the
scripts run from any working directory, under any directory name.

The three generated roots -- results/, activations/ and logs/ -- take an environment
override. That is not generality for its own sake: on HCC the home quota cannot hold the
activation tensors, so an extraction run points FCG_ACTIVATIONS at /work and everything
else stays where it is.

    python paths.py     # print every root, marked ok / MISSING
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# Committed: code, inputs, and the analysis snapshots the paper is rebuilt from.
SRC = os.path.join(ROOT, 'src')
DATA = os.path.join(ROOT, 'data')
ANALYSIS = os.path.join(ROOT, 'analysis')
SCRIPTS = os.path.join(ROOT, 'scripts')

# Generated, gitignored, overridable.
RESULTS = os.environ.get('FCG_RESULTS', os.path.join(ROOT, 'results'))
# A sibling of results/ and logs/ rather than a subdirectory of data/: it is ~1.1 GB of
# generated tensors, so it belongs with the other generated roots, not beside the one
# committed input.
ACTIVATIONS = os.environ.get('FCG_ACTIVATIONS', os.path.join(ROOT, 'activations'))
LOGS = os.environ.get('FCG_LOGS', os.path.join(ROOT, 'logs'))

DATASET_CSV = os.path.join(DATA, 'functional_group_dataset.csv')
# At the root, not under data/: it configures the run rather than being an input to it, and
# it is the one file a reader edits to change what the pipeline does.
EXTRACTION_CONFIG = os.path.join(ROOT, 'config_extract_activation.yaml')

# Ordered for the report below: committed roots, generated roots, then the two files.
_REPORT = ['ROOT', 'SRC', 'DATA', 'ANALYSIS', 'SCRIPTS',
           'RESULTS', 'ACTIVATIONS', 'LOGS',
           'DATASET_CSV', 'EXTRACTION_CONFIG']
# Overridable roots, named so the report can show which override is in effect.
_ENV = {'RESULTS': 'FCG_RESULTS', 'ACTIVATIONS': 'FCG_ACTIVATIONS',
        'LOGS': 'FCG_LOGS'}


def main():
    width = max(len(n) for n in _REPORT)
    for name in _REPORT:
        value = globals()[name]
        mark = 'ok' if os.path.exists(value) else 'MISSING'
        env = _ENV.get(name)
        note = f'  [{env}]' if env and env in os.environ else ''
        print(f'{name:<{width}}  {mark:<7}  {value}{note}')


if __name__ == '__main__':
    main()
