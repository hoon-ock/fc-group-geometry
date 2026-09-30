"""The documentation points at things that exist.

Written after a fresh clone of the published repo failed `test_structure`: `visuals/clustermap/`
had no tracked file, so git could not carry the directory. Checking for more of the same
turned up that **every markdown link in the repository was broken** -- all four of them -- plus
fourteen backticked paths naming files that no longer exist. All of it rot from flattening
`analysis/`, which dissolved the per-topic subdirectories (`analysis/taxonomy/`,
`analysis/probe/`, `analysis/figures/`) into one flat `analysis/data/`.

None of it was catchable by the existing hygiene sweeps, which look for `analysis/figures` and
`analysis/<topic>/` prefixes. These references use the bare (`taxonomy/`), parent-relative
(`../probe/`) and capitalised (`Analysis/`) forms instead.

The hard part is not finding broken paths, it is not drowning in false positives: plenty of
backticked strings contain a slash and are not paths at all. Rather than guess what looks like
a path, `_anchored_paths` only considers strings starting with a real top-level entry of this
repository. That admits `analysis/data/x.csv` and rejects the model identifiers
(`meta-llama/Llama-3.1-8B`), the `results/` leaf shorthand used inside tables already headed
`results/` (`probe/`, `anisotropy_diagnostic/`), and the maths (`k/n`,
`p(O)/[p(O)+p(home)]`, `sulfone@4/5`).
"""
import os
import re

import pytest

from conftest import REPO, tracked_text_files

# Real top-level entries. A backticked string starting with one of these is meant to be a
# path; anything else is not assumed to be one. `Analysis/` is the SOURCE repository's
# spelling and no longer exists here -- included precisely so citing it fails.
ANCHORS = ('analysis/', 'visuals/', 'src/', 'scripts/', 'tests/', 'data/', 'Analysis/')

MD = [(rel, txt) for rel, txt in tracked_text_files() if rel.endswith('.md')]
MD_IDS = [rel for rel, _ in MD]

LINK = re.compile(r'\[[^\]]*\]\(([^)\s]+)\)')
HEADING = re.compile(r'^#{1,6}\s+(.*)$', re.M)
CODE = re.compile(r'`([^`\s]+)`')
LINENO = re.compile(r':\d+(?:-\d+)?$')


def _slug(heading):
    """GitHub's anchor rule: lower-case, drop punctuation, spaces to hyphens."""
    s = re.sub(r'[^\w\s-]', '', heading.strip().lower())
    return re.sub(r'\s+', '-', s)


def _headings(path):
    with open(path, errors='ignore') as fh:
        return {_slug(m.group(1)) for m in HEADING.finditer(fh.read())}


def _anchored_paths(text):
    for m in CODE.finditer(text):
        raw = m.group(1)
        if not raw.startswith(ANCHORS) and not raw.startswith('../'):
            continue
        # A glob or a <placeholder> is not a literal path and cannot be resolved.
        if any(c in raw for c in '*{}<>'):
            continue
        yield raw, LINENO.sub('', raw)


@pytest.mark.parametrize('rel,txt', MD, ids=MD_IDS)
def test_every_markdown_link_resolves(rel, txt):
    base = os.path.join(REPO, os.path.dirname(rel))
    broken = []
    for m in LINK.finditer(txt):
        target = m.group(1).split('#')[0]
        if not target or target.startswith(('http://', 'https://', 'mailto:')):
            continue
        if not os.path.exists(os.path.normpath(os.path.join(base, target))):
            broken.append(target)
    assert not broken, f'{rel} links to nothing: {broken}'


@pytest.mark.parametrize('rel,txt', MD, ids=MD_IDS)
def test_every_link_anchor_resolves_to_a_real_heading(rel, txt):
    """Covers analysis/README.md's own table of contents, which is where a renamed section
    would otherwise break silently -- a wrong anchor still renders as a working link."""
    base = os.path.join(REPO, os.path.dirname(rel))
    broken = []
    for m in LINK.finditer(txt):
        path, _, anchor = m.group(1).partition('#')
        if not anchor or path.startswith(('http://', 'https://')):
            continue
        target = os.path.normpath(os.path.join(base, path)) if path else os.path.join(REPO, rel)
        if not os.path.exists(target):
            continue                      # the link test above owns this failure
        if anchor not in _headings(target):
            broken.append(m.group(1))
    assert not broken, f'{rel} anchors at no such heading: {broken}'


@pytest.mark.parametrize('rel,txt', [(r, t) for r, t in tracked_text_files()],
                         ids=[r for r, _ in tracked_text_files()])
def test_every_anchored_path_reference_resolves(rel, txt):
    """Applies to code and shell as well as markdown -- a docstring citing a moved file is
    the same defect. Paths resolve against the repository root, or against the citing file's
    own directory for the `../` forms the topic READMEs use."""
    base = os.path.join(REPO, os.path.dirname(rel))
    broken = []
    for raw, path in _anchored_paths(txt):
        if os.path.exists(os.path.join(REPO, path)):
            continue
        if os.path.exists(os.path.normpath(os.path.join(base, path))):
            continue
        broken.append(raw)
    assert not broken, f'{rel} cites paths that do not exist: {sorted(set(broken))}'


def test_no_reference_to_the_source_repositorys_capitalised_analysis_dir():
    """`Analysis/` was fc_group's spelling. Here the directory is `analysis/`, so any
    surviving capitalised reference is inherited text that was never updated."""
    hits = [rel for rel, txt in tracked_text_files() if 'Analysis/' in txt]
    assert not hits, f'stale `Analysis/` (the source repo spelling) in {hits}'


def test_no_pre_flattening_topic_prefix_in_the_generator_table():
    """The 29 snapshots live flat in `analysis/data/`. Before `analysis/` was flattened they
    sat under per-topic subdirectories, and visuals/README.md's "Reads" column still labelled
    each CSV with its old topic directory -- which reads as a real path and is not one."""
    with open(os.path.join(REPO, 'visuals/README.md'), errors='ignore') as fh:
        txt = fh.read()
    # Only the "Reads" column names snapshots, and every snapshot is a .csv. The "Output"
    # column's `generation/group_response_composition.png` is legitimate visuals/<topic>/
    # shorthand, so matching on the topic prefix alone flags the wrong column.
    stale = re.findall(
        r'`((?:taxonomy|geometry|analogy|probe|tsne|ambiguity|generation|prompt_structure)'
        r'/[^`]*\.csv)`', txt)
    assert not stale, (
        f'visuals/README.md still names snapshots by their pre-flattening topic: {stale} '
        '-- all 29 now live flat in analysis/data/')


# --- Badges -----------------------------------------------------------------------------
# A badge is a reference like any other, and a wrong one fails loudly in public: a workflow
# that does not exist renders "no status", and a slug belonging to another repository renders
# THAT project's build result on this README. The second is the likelier mistake -- badge
# markup is usually copied from an existing project and the slug edited by hand.

BADGE_WORKFLOW = re.compile(
    r'https://github\.com/([^/\s]+/[^/\s]+)/actions/workflows/([^/\s]+)/badge\.svg')


def _readme():
    with open(os.path.join(REPO, 'README.md'), errors='ignore') as fh:
        return fh.read()


def test_every_actions_badge_names_a_workflow_that_exists():
    missing = [wf for _, wf in BADGE_WORKFLOW.findall(_readme())
               if not os.path.exists(os.path.join(REPO, '.github/workflows', wf))]
    assert not missing, (
        f'README badges reference workflows that do not exist: {missing} -- the badge will '
        'render "no status" forever')


def test_every_actions_badge_points_at_THIS_repository():
    """The slug is the part that gets copied from another project and forgotten, which would
    quietly publish someone else's build status on this README."""
    import subprocess
    url = subprocess.run(['git', 'remote', 'get-url', 'origin'], cwd=REPO,
                         capture_output=True, text=True).stdout.strip()
    if not url:
        pytest.skip('no origin remote configured')
    slug = re.sub(r'\.git$', '', url.split('github.com')[-1].lstrip(':/'))
    wrong = [s for s, _ in BADGE_WORKFLOW.findall(_readme()) if s != slug]
    assert not wrong, f'badges point at {wrong}, but origin is {slug}'


def test_the_license_badge_agrees_with_the_license_file():
    badges = re.findall(r'img\.shields\.io/badge/License-([A-Za-z0-9._%-]+)-', _readme())
    if not badges:
        pytest.skip('no license badge')
    with open(os.path.join(REPO, 'LICENSE'), errors='ignore') as fh:
        text = fh.read()
    wrong = [b for b in badges if b.replace('%20', ' ').split('-')[0] not in text]
    assert not wrong, f'license badge says {wrong}, which LICENSE does not mention'
