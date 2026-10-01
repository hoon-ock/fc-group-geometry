"""Every module must import at its own depth, and probe.py must not be shadowed.

All 30 imports run in ONE subprocess rather than one each: every module pulls in torch or
matplotlib, and paying that 30 times took 30 s. Failures are collected per module so the
report still names exactly which one broke.
"""
import glob
import json
import os
import subprocess
import sys

import pytest

from conftest import REPO

MODULES = sorted(
    os.path.relpath(p, REPO)
    for p in glob.glob(os.path.join(REPO, 'src/*.py'))
    + glob.glob(os.path.join(REPO, 'analysis/*.py'))
    + glob.glob(os.path.join(REPO, 'visuals/*.py'))
    # Manuscript-figure scripts live one level down; the non-recursive glob above would
    # leave them silently untested for importability.
    + glob.glob(os.path.join(REPO, 'visuals/figure/*.py')))

# Mirrors the import stanza each script carries: its own directory, then src/, then ROOT.
_PROBE_ALL = r"""
import importlib.util, json, os, sys, traceback
REPO, mods = sys.argv[1], json.loads(sys.argv[2])
out = {}
for rel in mods:
    p = os.path.join(REPO, rel)
    d = os.path.dirname(p)
    saved = list(sys.path)
    sys.path[:0] = [d, os.path.join(REPO, 'src'), REPO]
    try:
        s = importlib.util.spec_from_file_location('m_' + rel.replace('/', '_')[:-3], p)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        out[rel] = None
    except Exception:
        out[rel] = traceback.format_exc(limit=4)
    finally:
        sys.path[:] = saved
print(json.dumps(out))
"""


@pytest.fixture(scope='module')
def import_results():
    out = subprocess.run([sys.executable, '-c', _PROBE_ALL, REPO, json.dumps(MODULES)],
                         cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_there_are_thirty_modules():
    assert len(MODULES) == 31, MODULES


@pytest.mark.parametrize('rel', MODULES, ids=lambda r: r.replace('/', ':'))
def test_module_imports(rel, import_results):
    err = import_results[rel]
    assert err is None, f'{rel} failed to import:\n{err}'


def test_probe_resolves_to_src_and_is_not_shadowed_by_the_analysis_dir():
    """`probe` is both a src module and (once) an analysis topic. A namespace-package
    shadow would be silent, so this is checked from inside analysis/, where a script's own
    directory goes on sys.path first."""
    code = (f"import sys; sys.path.insert(0, {REPO!r})\n"
            "import paths; sys.path.insert(0, paths.SRC)\n"
            "import probe; print(probe.__file__)\n")
    out = subprocess.run([sys.executable, '-c', code],
                         cwd=os.path.join(REPO, 'analysis'),
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == os.path.join(REPO, 'src', 'probe.py'), out.stdout


def test_common_exposes_what_the_generators_import():
    sys.path.insert(0, os.path.join(REPO, 'visuals'))
    import _common
    for name in ('BEHAVIOURAL', 'FIGURE_LAYER', 'LAYERS', 'N_LAYERS',
                 'REPO', 'VISUALS', 'DATA', 'load', 'num'):
        assert hasattr(_common, name), name
    assert len(_common.BEHAVIOURAL) == 5
    # ANALYSIS went when the data went flat; VISUALS replaced it
    assert not hasattr(_common, 'ANALYSIS')
