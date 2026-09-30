"""paths.py is the single source of truth -- every other module resolves through it."""
import os
import subprocess
import sys

from conftest import REPO

ROOTS = ['ROOT', 'SRC', 'DATA', 'ANALYSIS', 'VISUALS', 'SCRIPTS',
         'RESULTS', 'ACTIVATIONS', 'LOGS']


def _paths_from(cwd, env=None):
    """Import paths in a fresh interpreter with a given cwd, returning its constants."""
    code = ('import sys; sys.path.insert(0, %r); import paths, json; '
            'print(json.dumps({k: getattr(paths, k) for k in %r}))' % (REPO, ROOTS + [
                'DATASET_CSV', 'EXTRACTION_CONFIG']))
    e = dict(os.environ, **(env or {}))
    out = subprocess.run([sys.executable, '-c', code], cwd=cwd, env=e,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    import json
    return json.loads(out.stdout)


def test_every_root_is_inside_the_repo():
    p = _paths_from(REPO)
    assert p['ROOT'] == REPO
    for k in ROOTS:
        assert p[k].startswith(REPO), (k, p[k])


def test_the_committed_inputs_exist():
    p = _paths_from(REPO)
    assert os.path.exists(p['DATASET_CSV'])
    assert os.path.exists(p['EXTRACTION_CONFIG'])
    # the config is at the root, not under data/ -- it configures a run, it is not an input
    assert os.path.dirname(p['EXTRACTION_CONFIG']) == REPO


def test_resolution_is_independent_of_the_working_directory(tmp_path):
    """The whole point of paths.py: the old layout needed cwd to be the repo root."""
    assert _paths_from(REPO) == _paths_from(str(tmp_path))


def test_the_generated_roots_honour_their_env_overrides(tmp_path):
    p = _paths_from(REPO, env={
        'FCG_RESULTS': str(tmp_path / 'r'),
        'FCG_ACTIVATIONS': str(tmp_path / 'a'),
        'FCG_LOGS': str(tmp_path / 'l'),
    })
    assert p['RESULTS'] == str(tmp_path / 'r')
    assert p['ACTIVATIONS'] == str(tmp_path / 'a')
    assert p['LOGS'] == str(tmp_path / 'l')
    # the committed roots are NOT overridable -- they are repo structure, not scratch
    assert p['ANALYSIS'].startswith(REPO)
    assert p['VISUALS'].startswith(REPO)


def test_activations_is_a_sibling_of_results_not_under_data():
    p = _paths_from(REPO)
    assert p['ACTIVATIONS'] == os.path.join(REPO, 'activations')
    assert os.path.dirname(p['ACTIVATIONS']) == os.path.dirname(p['RESULTS'])


def test_paths_prints_a_report():
    out = subprocess.run([sys.executable, 'paths.py'], cwd=REPO,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    for k in ROOTS:
        assert k in out.stdout
