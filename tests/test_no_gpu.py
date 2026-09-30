"""The suite never executes a GPU stage, and that is a property worth pinning.

Three scripts under scripts/ need a GPU: 00_smoke_test, 01_extract and
02_generation_eval. Their outputs -- activations/ and results/generation_eval/ -- are
INPUTS to the tests, never things the tests produce. Everything the suite executes is
CPU-only, so `pytest --runslow` is safe to run anywhere.

Checked by AST rather than by text search, because naming a script is fine and running one
is not: test_structure asserts the sbatch inventory and test_shell parses those files, both
without invoking anything. Only the arguments of an actual call count.
"""
import ast
import glob
import os

import pytest

from conftest import REPO

GPU_SBATCH = ['00_smoke_test.sbatch', '01_extract.sbatch', '02_generation_eval.sbatch']
# src modules that load a checkpoint
GPU_SCRIPTS = ['extract_activations_subset.py', 'generation_eval.py', 'verify_smoke_output.py']
SCHEDULER = ['sbatch', 'srun', 'squeue']

TEST_FILES = sorted(glob.glob(os.path.join(REPO, 'tests/*.py')))


# The only two ways a test can execute anything.
EXECUTORS = {'run', 'check_output', 'check_call', 'Popen'}


def _executed_strings():
    """String literals handed to an executor call, per test file.

    Deliberately narrow. Naming a script is fine -- test_structure asserts the sbatch
    inventory, test_reproduce names 01_extract.sbatch in a skip message telling the reader
    how to produce its input. Only arguments to run()/subprocess.* can actually launch
    something, so only those count.
    """
    for path in TEST_FILES:
        tree = ast.parse(open(path).read())
        strings = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = getattr(fn, 'attr', None) or getattr(fn, 'id', None)
            if name not in EXECUTORS:
                continue
            for a in list(node.args) + [k.value for k in node.keywords]:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    strings.add(a.value)
                elif isinstance(a, (ast.List, ast.Tuple)):
                    for e in a.elts:
                        if isinstance(e, ast.Constant) and isinstance(e.value, str):
                            strings.add(e.value)
        yield os.path.basename(path), strings


def test_the_three_gpu_jobs_are_exactly_what_we_think():
    on_disk = sorted(os.path.basename(p)
                     for p in glob.glob(os.path.join(REPO, 'scripts/*.sbatch')))
    assert on_disk == sorted(GPU_SBATCH)
    for name in GPU_SBATCH:
        assert 'gres=gpu' in open(os.path.join(REPO, 'scripts', name)).read(), name


def test_run_local_requests_no_gpu():
    """The local driver is the CPU pipeline. A GPU stage appearing in it would be news.

    It MENTIONS the sbatch jobs -- "no activations, run scripts/01_extract.sbatch" is how a
    reader learns what is missing -- so this checks command position, not mere presence.
    """
    lines = open(os.path.join(REPO, 'scripts/run_local.sh')).read().split('\n')
    assert not any('gres=gpu' in ln for ln in lines)
    for ln in lines:
        code = ln.split('#')[0].strip()
        for binary in SCHEDULER:
            assert not code.startswith(binary), f'run_local.sh runs {binary}: {ln}'
    for name in GPU_SCRIPTS:
        for ln in lines:
            code = ln.split('#')[0]
            assert f'python src/{name}' not in code, \
                f'run_local.sh invokes the GPU stage {name}'


@pytest.mark.parametrize('name', GPU_SCRIPTS)
def test_no_test_executes_a_checkpoint_loading_script(name):
    hits = [f for f, strings in _executed_strings()
            if any(name in s for s in strings)]
    assert not hits, f'{name} passed to a call in {hits}'


@pytest.mark.parametrize('name', GPU_SBATCH)
def test_no_test_executes_a_gpu_sbatch(name):
    hits = [f for f, strings in _executed_strings()
            if any(name in s for s in strings)]
    assert not hits, f'{name} passed to a call in {hits}'


@pytest.mark.parametrize('binary', SCHEDULER)
def test_no_test_executes_the_scheduler(binary):
    hits = [f for f, strings in _executed_strings()
            if any(s == binary or s.startswith(binary + ' ') for s in strings)]
    assert not hits, f'{binary} invoked from {hits}'


def test_no_test_imports_torch_or_transformers_directly():
    """The pipeline modules may import torch -- test_imports exercises that in a
    subprocess. The tests themselves must not, so collection stays fast and CPU-only."""
    for path in TEST_FILES:
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                root = m.split('.')[0]
                assert root not in ('torch', 'transformers'), \
                    f'{os.path.basename(path)} imports {m}'


def test_the_figures_tier_runs_on_a_bare_clone():
    """It reads only committed CSVs, so it must not gate on activations or results."""
    fig = open(os.path.join(REPO, 'tests/test_figures.py')).read()
    assert 'have_activations' not in fig
    assert 'have_results' not in fig


def test_the_reproduce_tier_skips_rather_than_producing_its_inputs():
    rep = open(os.path.join(REPO, 'tests/test_reproduce.py')).read()
    assert 'pytest.skip' in rep
    assert rep.count('pytest.skip') >= 2, 'both tiers need their own skip'
