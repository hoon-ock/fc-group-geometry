"""The two committed inputs: the prompt config and the molecule dataset."""
import csv

import yaml

import paths

KEPT_COLUMNS = ['iupac_name', 'common_name', 'formula', 'smiles',
                'carbon_count', 'functional_group', 'functional_group_structure']
DROPPED_PROPERTIES = ['mw', 'pka', 'pkah', 'tpsa', 'avg_carbon_oxidation_state',
                      'hbd', 'hba', 'boiling_point_c', 'water_solubility']


def _config():
    with open(paths.EXTRACTION_CONFIG) as fh:
        return yaml.safe_load(fh)['extraction']


def test_exactly_one_prompt_type_ships():
    """The source tree swept 23. Every released number comes from functional_group alone."""
    entities = _config()['entities']
    assert len(entities) == 1
    assert entities[0]['entity_type'] == 'functional_group'


def test_ten_declarative_templates_interpolating_only_identifiers():
    e = _config()['entities'][0]
    assert len(e['templates']) == 10
    for t in e['templates']:
        # no measured property may enter a prompt -- that is why the dataset carries none
        placeholders = set(__import__('re').findall(r'\{(\w+)\}', t))
        assert placeholders <= {'iupac_name', 'smiles'}, t


def test_the_config_embeds_no_paths():
    """save_dir and data_file were repo-root-relative strings; paths.py owns them now."""
    ex = _config()
    assert 'save_dir' not in ex
    assert 'data_file' not in ex['entities'][0]


def test_dataset_is_92_molecules_with_the_seven_identifier_columns():
    rows = list(csv.DictReader(open(paths.DATASET_CSV)))
    assert len(rows) == 92
    assert list(rows[0]) == KEPT_COLUMNS


def test_the_measured_property_columns_are_gone():
    rows = list(csv.DictReader(open(paths.DATASET_CSV)))
    for col in DROPPED_PROPERTIES:
        assert col not in rows[0], col


def test_the_dataset_covers_twenty_functional_groups():
    rows = list(csv.DictReader(open(paths.DATASET_CSV)))
    groups = {r['functional_group'] for r in rows}
    assert len(groups) == 20
    assert 'none (alkane)' in groups     # the backbone every diff vector subtracts


def test_every_group_has_a_structure_string():
    """analyze_ambiguity reads functional_group_structure as the canonical formula."""
    rows = list(csv.DictReader(open(paths.DATASET_CSV)))
    for r in rows:
        assert r['functional_group_structure'].strip(), r['iupac_name']
