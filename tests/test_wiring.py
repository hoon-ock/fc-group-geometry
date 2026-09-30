"""Nothing is orphaned and nothing dangles: every script is reachable, every path resolves.

These are the checks that caught real breakage during the restructures -- a producer that
stopped being invoked, a load() name left with its old topic prefix, a doc command pointing
at a moved file.
"""
import glob
import os
import re

import pytest

from conftest import REPO

RUN_LOCAL = open(os.path.join(REPO, 'scripts/run_local.sh')).read()
PRODUCERS = sorted(os.path.basename(p) for p in glob.glob(os.path.join(REPO, 'analysis/*.py')))
GENERATORS = sorted(os.path.basename(p)
                    for p in glob.glob(os.path.join(REPO, 'visuals/make_visuals*.py')))


def _docs():
    for p in glob.glob(os.path.join(REPO, '**/*.md'), recursive=True):
        if '.git' not in p:
            yield os.path.relpath(p, REPO), open(p).read()


@pytest.mark.parametrize('producer', PRODUCERS)
def test_every_analysis_producer_is_invoked_by_run_local(producer):
    assert f'analysis/{producer}' in RUN_LOCAL, f'{producer} is never run'


def test_run_local_invokes_nothing_that_does_not_exist():
    for m in sorted(set(re.findall(r'python (src/[\w.]+\.py|analysis/[\w.]+\.py)', RUN_LOCAL))):
        assert os.path.exists(os.path.join(REPO, m)), m


def test_the_figures_loop_matches_every_generator():
    assert 'visuals/make_visuals*.py' in RUN_LOCAL
    assert len(GENERATORS) == 9


def test_every_load_name_exists_in_analysis_data():
    """load() takes a bare filename now; a stale '<topic>/data/' prefix would 404 at runtime."""
    names = set()
    for g in glob.glob(os.path.join(REPO, 'visuals/make_visuals*.py')):
        names |= set(re.findall(r"load\('([^']+)'\)", open(g).read()))
    assert names, 'no load() calls found -- did the generators change shape?'
    for n in sorted(names):
        assert '/' not in n, f'{n} still carries a path prefix'
        assert os.path.exists(os.path.join(REPO, 'analysis/data', n)), n


def test_every_python_command_quoted_in_the_docs_resolves():
    bad = []
    for doc, text in _docs():
        for m in re.findall(r'python3? ((?:src|analysis|visuals|scripts)/[\w./]+\.py)', text):
            if not os.path.exists(os.path.join(REPO, m)):
                bad.append(f'{doc}: {m}')
    assert not bad, '\n'.join(bad)


def test_every_script_path_quoted_in_the_docs_exists():
    bad = []
    for doc, text in _docs():
        for m in re.findall(r'\b(scripts/[\w.]+\.(?:sbatch|sh|inc))\b', text):
            if not os.path.exists(os.path.join(REPO, m)):
                bad.append(f'{doc}: {m}')
    assert not bad, '\n'.join(bad)


def test_the_analysis_chain_is_ordered_so_inputs_precede_readers():
    """score_generations writes the adjudicated table analyze_generation reads, and
    analyze_taxonomy writes the matrix the clustermaps need."""
    pos = {p: RUN_LOCAL.index(f'analysis/{p}') for p in PRODUCERS if f'analysis/{p}' in RUN_LOCAL}
    assert pos['score_generations.py'] < pos['analyze_generation.py']
    assert pos['analyze_taxonomy.py'] < pos['analyze_geometry.py']


def test_run_local_passes_no_plots_to_both_analogy_stages():
    """212 + 7 PNGs per model that no released figure reads; 95% of that stage's runtime."""
    compute = RUN_LOCAL[RUN_LOCAL.index('== compute'):RUN_LOCAL.index('== analysis')]
    for script in ('analogy_carbon_matched.py', 'analogy_retrieval.py'):
        block = compute[compute.index(script):]
        assert '--no-plots' in block[:400], f'{script} is drawing unread figures'
