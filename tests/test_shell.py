"""The shell layer: three GPU jobs, the local driver, and the shared model list."""
import glob
import os
import re
import subprocess

import pytest

from conftest import REPO

SHELL = sorted(os.path.relpath(p, REPO) for p in
               glob.glob(os.path.join(REPO, 'scripts/*.sbatch'))
               + glob.glob(os.path.join(REPO, 'scripts/*.sh')))
SBATCH = [p for p in SHELL if p.endswith('.sbatch')]


@pytest.mark.parametrize('rel', SHELL, ids=lambda r: os.path.basename(r))
def test_shell_script_parses(rel):
    out = subprocess.run(['bash', '-n', rel], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, f'{rel}\n{out.stderr}'


def test_preamble_parses():
    """Sourced, never executed -- but a syntax error in it breaks all three jobs."""
    out = subprocess.run(['bash', '-n', 'scripts/_preamble.inc'],
                         cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


@pytest.mark.parametrize('rel', SBATCH, ids=lambda r: os.path.basename(r))
def test_array_span_matches_the_preflight(rel):
    """`#SBATCH --array` is static and cannot self-update, so each job asserts the product
    of its runtime lists against it. The two must agree or the assert is vacuous."""
    text = open(os.path.join(REPO, rel)).read()
    span = re.search(r'^#SBATCH --array=0-(\d+)', text, re.M)
    declared = re.search(r'check_array_size (\d+)', text)
    assert span, f'{rel} declares no --array'
    assert declared, f'{rel} never calls check_array_size'
    assert int(span.group(1)) + 1 == int(declared.group(1)), rel


@pytest.mark.parametrize('rel', SBATCH, ids=lambda r: os.path.basename(r))
def test_every_job_sources_the_preamble_and_the_model_list(rel):
    text = open(os.path.join(REPO, rel)).read()
    assert '_preamble.inc' in text, rel
    assert 'models.sh' in text, rel


def test_models_sh_yields_the_five_swept_models_and_one_entity_type():
    script = ('ROOT="$PWD"; source scripts/models.sh >/dev/null 2>&1; '
              'printf "%s\\n" "${MODELS[@]}"; echo "--"; read_entity_types')
    out = subprocess.run(['bash', '-c', script], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    models, entities = out.stdout.split('--\n')
    assert len(models.split()) == 5, models
    assert entities.split() == ['functional_group'], entities


def test_check_array_size_accepts_five_and_rejects_anything_else():
    """The guard has to be able to fail, or it is decoration."""
    def run(n):
        return subprocess.run(
            ['bash', '-c', f'ROOT="$PWD"; source scripts/models.sh >/dev/null 2>&1; '
                           f'check_array_size {n}'],
            cwd=REPO, capture_output=True, text=True)
    assert run(5).returncode == 0
    bad = run(115)
    assert bad.returncode != 0
    assert '115' in bad.stdout + bad.stderr


def test_models_sh_names_the_registered_but_unswept_models():
    """A model may be registered without being swept -- the 70B pair is. The subset check
    says so out loud rather than staying silent about it."""
    out = subprocess.run(['bash', '-c', 'ROOT="$PWD"; source scripts/models.sh'],
                         cwd=REPO, capture_output=True, text=True)
    note = out.stdout + out.stderr
    assert 'registered, not in this sweep' in note
    assert 'meta-llama/Llama-3.1-70B' in note


def test_run_local_rejects_an_unknown_stage():
    out = subprocess.run(['bash', 'scripts/run_local.sh', '--stages', 'nonsense'],
                         cwd=REPO, capture_output=True, text=True)
    assert out.returncode != 0
    assert 'compute' in out.stdout + out.stderr
