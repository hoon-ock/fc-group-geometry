"""[slow] The committed snapshots are reproducible from the inputs they claim.

Two tiers, each skipped when its input is absent so a clone without activations still gets
a green fast+figures run:

  * activations/ present -> the two analyses that read the tensors directly must reproduce
    their three snapshots byte-for-byte.
  * results/ populated   -> the whole nine-script chain must reproduce all 29.

Everything writes into tmp_path and is compared against the committed file. Nothing here
can modify analysis/data/, which matters because that is what is under test.
"""
import filecmp
import os

import pytest

from conftest import REPO, missing_result_models, run

DATA = os.path.join(REPO, 'analysis/data')

pytestmark = pytest.mark.slow

# (script, out-dir flag args, snapshots it must reproduce)
FROM_ACTIVATIONS = [
    ('analysis/compute_tsne_coords.py', [],
     ['tsne_coords.csv', 'tsne_neighbourhood.csv']),
    ('analysis/analyze_prompt_structure.py', ['--out-name', 'label_agreement_by_layer.csv'],
     ['label_agreement_by_layer.csv']),
]

# The chain, in dependency order. score_generations must precede analyze_generation.
CHAIN = [
    'analysis/analyze_sweep.py',
    'analysis/analyze_taxonomy.py',
    'analysis/analyze_geometry.py',
    'analysis/snapshot_quiver3d.py',
    'analysis/analyze_ambiguity.py',
    'analysis/compute_tsne_coords.py',
    'analysis/analyze_prompt_structure.py',
    'analysis/score_generations.py',
    'analysis/analyze_generation.py',
]


def _same(name, produced_dir):
    a = os.path.join(produced_dir, name)
    b = os.path.join(DATA, name)
    assert os.path.exists(a), f'{name} was not produced'
    return filecmp.cmp(a, b, shallow=False)


@pytest.mark.parametrize('script,extra,snapshots', FROM_ACTIVATIONS,
                         ids=[os.path.basename(s) for s, _, _ in FROM_ACTIVATIONS])
def test_activation_reader_reproduces_its_snapshots(script, extra, snapshots,
                                                    have_activations, tmp_path):
    if not have_activations:
        pytest.skip('no activations/ for all five models -- produced by '
                    'scripts/01_extract.sbatch, the one GPU stage this tier needs. The '
                    'tensors are read on CPU (torch.load map_location="cpu").')
    res = run(script, '--out-dir', str(tmp_path), *extra)
    assert res.returncode == 0, f'{script}\n{res.stderr[-2000:]}'
    for name in snapshots:
        assert _same(name, str(tmp_path)), (
            f'{name} differs from the committed snapshot -- the analysis changed, or the '
            'committed file was edited by something other than its producer')


def test_the_whole_chain_reproduces_all_29_snapshots(have_results, tmp_path):
    """The strongest check in the suite. If a rename, a path change or a flag broke
    anything, a snapshot comes back different or a script exits non-zero."""
    if not have_results:
        absent = missing_result_models()
        pytest.skip(
            f'{len(absent)} (tree, model) pair(s) absent from results/, e.g. '
            f'{absent[:3]} -- populate with `bash scripts/run_local.sh --stages compute` '
            '(CPU-only, no GPU); generation_eval is the only tree needing the GPU job')

    out = str(tmp_path)
    for script in CHAIN:
        extra = ['--out-name', 'label_agreement_by_layer.csv'] \
            if script.endswith('analyze_prompt_structure.py') else []
        # score_generations merges a block into results/; --no-results-summary keeps this
        # test read-only with respect to results/.
        if script.endswith('score_generations.py'):
            extra = ['--no-results-summary']
        res = run(script, '--out-dir', out, *extra)
        assert res.returncode == 0, f'{script}\n{res.stdout[-1500:]}\n{res.stderr[-1500:]}'

    committed = sorted(f for f in os.listdir(DATA) if f.endswith('.csv'))
    produced = sorted(f for f in os.listdir(out) if f.endswith('.csv'))
    assert produced == committed, (
        f'produced {len(produced)} snapshots, committed {len(committed)}\n'
        f'missing: {sorted(set(committed) - set(produced))}\n'
        f'extra:   {sorted(set(produced) - set(committed))}')

    differing = [n for n in committed if not _same(n, out)]
    assert not differing, f'{len(differing)} snapshot(s) differ: {differing}'
