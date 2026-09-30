"""[slow] Every figure rebuilds from the committed CSVs alone.

This is the property the whole layout is arranged around: no generator may read `results/`,
so a fresh clone with no activations and no pipeline run can still draw all 70. The tests
write into tmp_path, never into visuals/, so running them cannot dirty the artifacts.
"""
import glob
import os

import pytest

from conftest import FIGURE_COUNTS, REPO, run

GENERATORS = sorted(os.path.relpath(p, REPO)
                    for p in glob.glob(os.path.join(REPO, 'visuals/make_visuals*.py')))

# generator -> the visuals/ topic it writes into
TOPIC = {
    'make_visuals.py': 'clustermap',
    'make_visuals_ambiguity.py': 'ambiguity',
    'make_visuals_generation_composition.py': 'generation',
    'make_visuals_probe.py': 'probe',
    'make_visuals_prompt_structure.py': 'prompt_structure',
    'make_visuals_retrieval_3d.py': 'retrieval',
    'make_visuals_retrieval_dist.py': 'retrieval',
    'make_visuals_retrieval_violin.py': 'retrieval',
    'make_visuals_tsne.py': 'tsne',
}

pytestmark = pytest.mark.slow


def test_the_topic_map_covers_every_generator():
    assert sorted(TOPIC) == [os.path.basename(g) for g in GENERATORS]


def test_the_expected_counts_sum_to_seventy():
    assert sum(FIGURE_COUNTS.values()) == 70


@pytest.mark.parametrize('rel', GENERATORS, ids=lambda r: os.path.basename(r))
def test_generator_runs_and_writes_png(rel, tmp_path):
    out = tmp_path / os.path.basename(rel)
    res = run(rel, '--out-dir', str(out))
    assert res.returncode == 0, f'{rel}\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}'
    pngs = list(out.rglob('*.png'))
    assert pngs, f'{rel} exited 0 but drew nothing'


def test_all_seventy_figures_appear_at_the_expected_per_topic_counts(tmp_path):
    """The three retrieval generators share one output directory, so this runs the whole
    set into one tree and counts per topic -- which is what the release actually ships."""
    for rel in GENERATORS:
        res = run(rel, '--out-dir', str(tmp_path / TOPIC[os.path.basename(rel)]))
        assert res.returncode == 0, f'{rel}\n{res.stderr[-2000:]}'

    got = {t: len(list((tmp_path / t).rglob('*.png'))) for t in FIGURE_COUNTS}
    assert got == FIGURE_COUNTS, f'got {got}, want {FIGURE_COUNTS}'
    assert sum(got.values()) == 70


def test_the_3d_generator_also_writes_its_six_text_readouts(tmp_path):
    """The .txt files are tracked where the PNGs are not -- they are the numbers, not the
    rendering."""
    res = run('visuals/make_visuals_retrieval_3d.py', '--out-dir', str(tmp_path))
    assert res.returncode == 0, res.stderr[-2000:]
    assert len(list(tmp_path.glob('analogy3d_*.txt'))) == 6
    assert len(list(tmp_path.glob('analogy3d_*.png'))) == 6


def test_no_generator_touches_results(tmp_path):
    """Enforced by reading, not by trust: a generator that needed results/ would be one
    missed snapshot step, and the whole reproducibility claim would be hollow."""
    for rel in GENERATORS:
        text = open(os.path.join(REPO, rel)).read()
        assert 'paths.RESULTS' not in text, rel
        assert "'results" not in text.replace('`results', ''), rel
