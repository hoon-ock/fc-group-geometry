"""The 30 committed CSVs are the reproducibility boundary: past them nothing reads results/."""
import csv
import glob
import os

import pytest

from conftest import MODELS, REPO

DATA = os.path.join(REPO, 'analysis/data')
SNAPSHOTS = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA, '*.csv')))

# The few that are a single paired contrast or a single model by construction, so a
# five-model assertion would be wrong rather than merely strict.
NOT_FIVE_MODELS = {
    'analogy_quiver3d.csv',              # base only -- all the released 3D figures
                                         # draw; analogy_quiver3d_all.csv is the
                                         # five-model companion figure S3 reads
    'model_diff_bootstrap.csv',          # one paired chem-vs-base contrast
    'generation_diff_bootstrap.csv',
    'free_generation_diff_bootstrap.csv',
    'taxonomy_paired.csv',
}


def test_there_are_thirty_snapshots():
    assert len(SNAPSHOTS) == 30, SNAPSHOTS


def test_the_snapshot_names_are_distinct_without_a_topic_prefix():
    """Which is what let analysis/data/ go flat."""
    assert len(set(SNAPSHOTS)) == len(SNAPSHOTS)


@pytest.mark.parametrize('name', SNAPSHOTS)
def test_snapshot_parses_and_is_not_empty(name):
    rows = list(csv.DictReader(open(os.path.join(DATA, name))))
    assert rows, f'{name} has a header but no rows'
    assert all(len(r) == len(rows[0]) for r in rows), f'{name} has ragged rows'


@pytest.mark.parametrize('name', [s for s in SNAPSHOTS if s not in NOT_FIVE_MODELS])
def test_snapshot_covers_the_five_released_models(name):
    rows = list(csv.DictReader(open(os.path.join(DATA, name))))
    if 'model' not in rows[0]:
        pytest.skip('no model column')
    assert {r['model'] for r in rows} == set(MODELS), name


@pytest.mark.parametrize('name', SNAPSHOTS)
def test_snapshot_record_terminators_are_uniform(name):
    """Every RECORD terminator must be the same. That is how you tell a file was written by
    its own producer rather than rewritten by something else -- which is how a CRLF got into
    two of these mid-session.

    Counted csv-aware, not by raw bytes. `free_generation_adjudicated.csv` legitimately
    holds bare LFs *inside* quoted fields: 222 of its rows carry a model generation
    containing a newline, the same ones that make `wc -l` overcount generations.csv. So this
    compares the CRLF count against the record count instead of looking for any LF at all.
    """
    path = os.path.join(DATA, name)
    with open(path, newline='') as fh:
        n_records = sum(1 for _ in csv.reader(fh))      # header included
    crlf = open(path, 'rb').read().count(b'\r\n')
    assert crlf in (0, n_records), (
        f'{name}: {crlf} CRLF against {n_records} records -- neither all-LF nor all-CRLF, '
        'so some records carry a different terminator than the rest')


def test_only_the_functional_group_entity_type_appears():
    for name in SNAPSHOTS:
        rows = list(csv.DictReader(open(os.path.join(DATA, name))))
        if 'entity_type' in rows[0]:
            assert {r['entity_type'] for r in rows} == {'functional_group'}, name


def test_the_probe_snapshot_carries_all_32_layers():
    rows = list(csv.DictReader(open(os.path.join(DATA, 'layer_curves.csv'))))
    for m in MODELS:
        layers = {int(r['layer']) for r in rows if r['model'] == m}
        assert layers == set(range(32)), m


def test_the_five_figure_layers_are_what_the_geometry_snapshot_holds():
    rows = list(csv.DictReader(open(os.path.join(DATA, 'geometry_by_layer.csv'))))
    assert {int(r['layer']) for r in rows} == {0, 8, 16, 24, 31}
