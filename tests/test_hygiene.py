"""No stale name survives, and no name that must survive was swept away by accident.

Every rename in this repo was a bulk text substitution, and the two hazards are symmetric:
a token left behind, or a token clobbered because it shared a prefix with the target. These
tests pin both directions.

With MIGRATION.md gone there is no file that legitimately quotes the old names, so none of
these needs an exception -- they are stricter than the equivalent grep sweeps were.
"""
import pytest

from conftest import tracked_text_files

# Renamed away. results/functional_group_probe/<model>/functional_group/ said it twice.
GONE = [
    'functional_group_probe',
    'functional_group_analogy',
    'functional_group_separability',
    'functional_group_dataset.csv.bak',
]

# Path prefixes from before analysis/ was flattened and visuals/ split out.
STALE_PATHS = [
    'analysis/figures',
    'analysis/visuals',
    'analysis/narrative',
    'analysis/probe/',
    'analysis/tsne/',
    'analysis/taxonomy/',
    'analysis/geometry/',
    'analysis/analogy/',
    'analysis/ambiguity/',
    'analysis/generation/',
    'analysis/prompt_structure/',
    'analysis/*/data',
    'data/activation',          # activations moved back to a top-level activations/
]

# Dropped scripts and figures. Checked in CODE only: a docstring or a comment naming one
# is a dangling pointer, but PROSE is allowed to say "X was dropped, here is what replaced
# it" -- analysis/README.md and visuals/retrieval/README.md both do that deliberately, and
# those notes are the reason a reader does not go hunting for the file.
DROPPED = [
    'make_figures.py', 'make_tables.py', 'make_setup_tables.py',
    'check_claims.py', 'check_numbers.py',
    'analyze_analogy.py', 'analyze_block_cohesion.py', 'analyze_error_composition.py',
    'analyze_generation_vs_representation.py', 'analyze_tsne_context.py',
    'tsne_functional_groups.py', 'extract_properties.py',
    'hit1.png', 'hit2.png', 'hit3.png', 'meanrank.png', 'rankmixed.png',
    'label_agreement_depth_raw.png', 'cohesion_', 'combined_',
]

# Must NOT have been swept: the entity type, the label column, the structure column.
PROTECTED = ['functional_group_dataset', 'functional_group_structure']

FILES = list(tracked_text_files())


def _hits(needle):
    return [rel for rel, text in FILES if needle in text]


@pytest.mark.parametrize('token', GONE)
def test_renamed_token_does_not_survive(token):
    assert not _hits(token), f'{token} still in: {_hits(token)}'


@pytest.mark.parametrize('prefix', STALE_PATHS)
def test_stale_path_prefix_does_not_survive(prefix):
    assert not _hits(prefix), f'{prefix} still in: {_hits(prefix)}'


CODE_SUFFIXES = ('.py', '.sh', '.sbatch', '.inc', '.yaml')


@pytest.mark.parametrize('name', DROPPED)
def test_dropped_artifact_is_not_referenced_from_code(name):
    """Caught a real one: make_visuals.py's docstrings still described `combined_figure`
    and `combined_<model>.png` after the cohesion cut removed both."""
    hits = [rel for rel in _hits(name) if rel.endswith(CODE_SUFFIXES)]
    assert not hits, f'{name} still referenced from code: {hits}'


@pytest.mark.parametrize('token', PROTECTED)
def test_protected_token_was_not_clobbered(token):
    """These share the `functional_group` prefix with the renamed ones, so a careless
    substitution would have taken them out."""
    assert _hits(token), f'{token} has vanished -- a rename over-reached'


def test_the_entity_type_still_appears_widely():
    """`functional_group` is the entity type, the label column and the results leaf. A
    blanket removal of the prefix would have destroyed all three."""
    assert len(_hits('functional_group')) > 20


def test_no_absolute_developer_path_leaked_in():
    for needle in ('/Users/hoon', '/work/ock'):
        hits = [rel for rel in _hits(needle) if not rel.startswith('scripts/')]
        assert not hits, f'{needle} in {hits}'
