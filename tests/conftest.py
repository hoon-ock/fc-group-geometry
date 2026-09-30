"""Shared fixtures and the --runslow gate.

Two rules every test here follows:

1. NEVER WRITE INTO THE REPO. Each generator and producer takes --out-dir, so the slow
   tiers write into pytest's tmp_path and compare against the committed files. Running the
   suite cannot dirty visuals/ or analysis/data/ -- which matters, because those are the
   artifacts under test.

2. SHELL OUT rather than import anything with a main(). A generator that crashes then
   becomes one test failure with its stderr attached, instead of a collection error that
   takes the whole module down with it.
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import paths  # noqa: E402

# The five released checkpoints, in _common.BEHAVIOURAL order.
MODELS = [
    'meta-llama-Llama-3.1-8B',
    'weidawang-Chem-R-8B',
    'phenixace-Chem-R-Faithful',
    'OpenDFM-ChemDFM-v1.5-8B',
    'deepseek-ai-DeepSeek-R1-Distill-Llama-8B',
]

# visuals/<topic> -> PNGs the generators must produce. Sums to 70.
FIGURE_COUNTS = {
    'clustermap': 30,
    'tsne': 28,
    'retrieval': 8,
    'ambiguity': 1,
    'generation': 1,
    'probe': 1,
    'prompt_structure': 1,
}


def pytest_addoption(parser):
    parser.addoption('--runslow', action='store_true', default=False,
                     help='run the end-to-end tiers: rebuild every figure, and reproduce '
                          'the snapshots from activations/ and results/')


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: end-to-end; needs --runslow')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--runslow'):
        return
    skip = pytest.mark.skip(reason='needs --runslow')
    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope='session')
def repo():
    return REPO


@pytest.fixture(scope='session')
def have_activations():
    """True when 01_extract's output is present for all five models."""
    return all(
        os.path.isdir(os.path.join(paths.ACTIVATIONS, m, 'functional_group'))
        for m in MODELS)


# Every results/ tree the analysis chain reads. Only generation_eval needs a GPU; the other
# seven come from `run_local.sh --stages compute`, which is CPU-only.
RESULT_TREES = ['probe', 'anisotropy_diagnostic', 'analogy', 'analogy_retrieval',
                'analogy_retrieval_carbon', 'analogy_retrieval_halide', 'ambiguity_metric',
                'generation_eval']


def missing_result_models():
    """Which (tree, model) pairs are absent, so a skip can say what is needed."""
    return [f'{t}/{m}' for t in RESULT_TREES for m in MODELS
            if not os.path.isdir(
                os.path.join(paths.RESULTS, t, m, 'functional_group', 'data'))]


@pytest.fixture(scope='session')
def have_results():
    """True when every results/ tree holds all five models.

    Checked per model, not just that the tree directory exists. A compute stage part-way
    through its model loop leaves every directory present and most of them incomplete, and
    the analysis scripts carry a fixed five-model list -- so on a partial tree the chain
    fails on a missing model instead of skipping, which is a false red.
    """
    return not missing_result_models()


def run(script, *args, cwd=REPO):
    """Run one pipeline script, returning the CompletedProcess. Never raises."""
    return subprocess.run([sys.executable, script, *args],
                          cwd=cwd, capture_output=True, text=True)


def tracked_text_files():
    """Every tracked file the hygiene sweeps read, as (path, contents).

    tests/ is excluded, and must be: test_hygiene names every forbidden token because
    those lists are what it searches for, and test_structure asserts the file inventory by
    name. Scanning the tests would make each of them flag itself -- which it did, the
    moment tests/ became tracked.
    """
    out = subprocess.run(['git', 'ls-files'], cwd=REPO, capture_output=True, text=True)
    for rel in out.stdout.split():
        if rel.startswith('tests/'):
            continue
        if rel.endswith(('.py', '.md', '.sh', '.sbatch', '.inc', '.yaml', '.txt')):
            full = os.path.join(REPO, rel)
            if os.path.exists(full):
                with open(full, errors='ignore') as fh:
                    yield rel, fh.read()
