"""[slow] The committed snapshots are reproducible from the inputs they claim.

Checked PER SCRIPT, not as one all-or-nothing chain. Each analysis script declares the
results/ trees it reads, and skips only when those are absent -- so a partially populated
results/ still exercises everything it can reach. That matters because the probe is the
expensive stage by a wide margin (~8 min per model, 40 min for the sweep), and needing it
complete should not block the other six analyses from being tested.

No GPU anywhere. The activation tier loads .pt tensors on CPU; the results tier reads CSVs
and npz that the CPU compute stages wrote. The one tree a GPU job produces, generation_eval,
is only ever read.

Everything writes into tmp_path and is compared against the committed file, so running
these cannot modify analysis/data/ -- which is the thing under test.
"""
import filecmp
import os

import pytest

from conftest import MODELS, REPO, run, tree_complete

DATA = os.path.join(REPO, 'analysis/data')

pytestmark = pytest.mark.slow

# script -> (results trees it reads, extra flags, snapshots it must reproduce)
CASES = [
    ('analysis/analyze_sweep.py', ['probe'], [],
     ['layer_curves.csv', 'surface_baseline.csv', 'confusion_errors.csv',
      'depth_by_family.csv', 'per_entity_summary.csv', 'model_diff_bootstrap.csv']),
    ('analysis/analyze_taxonomy.py', ['analogy'], [],
     ['between_class_matrix.csv', 'taxonomy_contrast.csv', 'taxonomy_paired.csv',
      'taxonomy_permutation.csv']),
    ('analysis/analyze_geometry.py',
     ['analogy', 'analogy_retrieval', 'analogy_retrieval_carbon',
      'analogy_retrieval_halide', 'anisotropy_diagnostic'], [],
     ['geometry_by_layer.csv', 'retrieval_by_axis.csv', 'retrieval_rank_hist.csv']),
    ('analysis/snapshot_quiver3d.py', ['analogy', 'analogy_retrieval'], [],
     ['analogy_quiver3d.csv']),
    ('analysis/analyze_ambiguity.py', ['ambiguity_metric', 'analogy', 'probe'], [],
     ['ambiguity_family_probability.csv', 'ambiguity_two_readings.csv']),
]

# analyze_generation is NOT in CASES: it is the one analysis that cannot run alone. It reads
# free_generation_{adjudicated,by_template}.csv out of its own --out-dir, so pointed at an
# empty tmp_path it silently drops the free-generation half of template_sensitivity.csv and
# produces a file that does not match the committed one. score_generations must run into the
# same directory first -- the ordering constraint run_local.sh and the READMEs both state,
# tested below as a chain rather than asserted in prose.
GENERATION = [
    'generation_summary.csv', 'generation_by_class.csv', 'generation_diff_bootstrap.csv',
    'template_sensitivity.csv', 'group_response_composition.csv',
]

FREE_GENERATION = [
    'free_generation_summary.csv', 'free_generation_by_class.csv',
    'free_generation_by_template.csv', 'free_generation_diff_bootstrap.csv',
    'free_generation_adjudicated.csv',
]

# The two that read the tensors directly rather than results/.
FROM_ACTIVATIONS = [
    ('analysis/compute_tsne_coords.py', [],
     ['tsne_coords.csv', 'tsne_neighbourhood.csv']),
    ('analysis/analyze_prompt_structure.py',
     ['--out-name', 'label_agreement_by_layer.csv'],
     ['label_agreement_by_layer.csv']),
]


def _tree_complete(tree):
    """All five models, and each holding what the analysis reads -- see conftest's
    TREE_SENTINELS for why a directory check is not enough."""
    return all(tree_complete(tree, m) for m in MODELS)


def _assert_reproduces(script, produced, snapshots):
    differing = [n for n in snapshots
                 if not filecmp.cmp(os.path.join(produced, n),
                                    os.path.join(DATA, n), shallow=False)]
    missing = [n for n in snapshots if not os.path.exists(os.path.join(produced, n))]
    assert not missing, f'{script} did not produce: {missing}'
    assert not differing, (
        f'{script}: {differing} differ from the committed snapshot -- either the analysis '
        'changed, or the committed file was written by something other than this producer')


@pytest.mark.parametrize('script,extra,snapshots', FROM_ACTIVATIONS,
                         ids=[os.path.basename(s) for s, _, _ in FROM_ACTIVATIONS])
def test_activation_reader_reproduces_its_snapshots(script, extra, snapshots,
                                                    have_activations, tmp_path):
    if not have_activations:
        pytest.skip('no activations/ for all five models -- produced by '
                    'scripts/01_extract.sbatch, the one GPU stage this tier needs. The '
                    'tensors themselves are read on CPU.')
    res = run(script, '--out-dir', str(tmp_path), *extra)
    assert res.returncode == 0, f'{script}\n{res.stderr[-2000:]}'
    _assert_reproduces(script, str(tmp_path), snapshots)


@pytest.mark.parametrize('script,trees,extra,snapshots', CASES,
                         ids=[os.path.basename(s) for s, _, _, _ in CASES])
def test_analysis_reproduces_its_snapshots(script, trees, extra, snapshots, tmp_path):
    """Skips on ITS OWN inputs only. `analyze_sweep` needs the probe sweep, which is the
    40-minute stage; the other five do not, and should not be held hostage to it."""
    absent = [t for t in trees if not _tree_complete(t)]
    if absent:
        pytest.skip(
            f'results/{{{",".join(absent)}}} not complete for all five models. '
            'Fill with `bash scripts/run_local.sh --stages compute` (CPU-only); '
            'generation_eval is the only tree needing a GPU job.')
    res = run(script, '--out-dir', str(tmp_path), *extra)
    assert res.returncode == 0, f'{script}\n{res.stdout[-1500:]}\n{res.stderr[-1500:]}'
    _assert_reproduces(script, str(tmp_path), snapshots)


def test_the_generation_chain_reproduces_its_snapshots(tmp_path):
    """score_generations -> analyze_generation, in that order into one directory.

    Both read results/generation_eval, the only tree a GPU job writes; both run on CPU.
    --no-results-summary keeps score_generations from merging its block back into
    results/summary.json, so the chain stays read-only outside tmp_path.
    """
    if not _tree_complete('generation_eval'):
        pytest.skip('results/generation_eval not complete -- produced by '
                    'scripts/02_generation_eval.sbatch')

    res = run('analysis/score_generations.py', '--out-dir', str(tmp_path),
              '--no-results-summary')
    assert res.returncode == 0, res.stderr[-2000:]
    _assert_reproduces('score_generations.py', str(tmp_path), FREE_GENERATION)

    res = run('analysis/analyze_generation.py', '--out-dir', str(tmp_path))
    assert res.returncode == 0, res.stderr[-2000:]
    _assert_reproduces('analyze_generation.py', str(tmp_path), GENERATION)


def test_analyze_generation_alone_is_not_enough(tmp_path):
    """The ordering constraint, from the other side: run analyze_generation into an empty
    directory and template_sensitivity.csv does NOT match the committed snapshot.

    Without this, the chain above could be passing for the wrong reason -- the two scripts
    might simply be independent, and the ordering the pipeline documents might be decorative.
    """
    if not _tree_complete('generation_eval'):
        pytest.skip('results/generation_eval not complete -- produced by '
                    'scripts/02_generation_eval.sbatch')
    res = run('analysis/analyze_generation.py', '--out-dir', str(tmp_path))
    assert res.returncode == 0, res.stderr[-2000:]
    produced = os.path.join(str(tmp_path), 'template_sensitivity.csv')
    assert os.path.exists(produced), 'expected it to run and write a degraded file, not fail'
    assert not filecmp.cmp(produced, os.path.join(DATA, 'template_sensitivity.csv'),
                           shallow=False), (
        'analyze_generation reproduced template_sensitivity.csv with no free-generation '
        'table present, so score_generations is no longer an input to it and the documented '
        'ordering is stale')


def test_the_cases_cover_every_snapshot_and_every_producer():
    """A snapshot with no case behind it is one nothing verifies."""
    covered = {n for _, _, _, snaps in CASES for n in snaps}
    covered |= {n for _, _, snaps in FROM_ACTIVATIONS for n in snaps}
    covered |= set(FREE_GENERATION) | set(GENERATION)
    committed = {f for f in os.listdir(DATA) if f.endswith('.csv')}
    unverified = committed - covered
    assert not unverified, f'no reproduce case covers: {sorted(unverified)}'
