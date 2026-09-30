"""The layout is a contract: the docs, run_local.sh and paths.py all assume this shape."""
import glob
import os
import subprocess

from conftest import FIGURE_COUNTS, REPO


def _names(pattern):
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(REPO, pattern)))


def test_src_holds_the_eleven_pipeline_modules():
    assert _names('src/*.py') == [
        'ambiguity_metric.py', 'analogy_carbon_matched.py', 'analogy_retrieval.py',
        'anisotropy_diagnostic.py', 'check_model_registry.py',
        'extract_activations_subset.py', 'free_generation_scoring.py',
        'generation_eval.py', 'model_registry.py', 'probe.py', 'verify_smoke_output.py',
    ]


def test_analysis_holds_nine_producers():
    assert _names('analysis/*.py') == [
        'analyze_ambiguity.py', 'analyze_generation.py', 'analyze_geometry.py',
        'analyze_prompt_structure.py', 'analyze_sweep.py', 'analyze_taxonomy.py',
        'compute_tsne_coords.py', 'score_generations.py', 'snapshot_quiver3d.py',
    ]


def test_analysis_is_flat():
    """data/ is the only subdirectory -- the topic directories were flattened away."""
    subdirs = sorted(d for d in os.listdir(os.path.join(REPO, 'analysis'))
                     if os.path.isdir(os.path.join(REPO, 'analysis', d))
                     and d != '__pycache__')
    assert subdirs == ['data']


def test_visuals_holds_nine_generators():
    assert len(_names('visuals/make_visuals*.py')) == 9
    assert os.path.exists(os.path.join(REPO, 'visuals/_common.py'))


def test_every_figure_topic_has_a_TRACKED_readme():
    """Tracked, not merely present on disk -- which is the whole point.

    Every PNG under visuals/ is gitignored and git does not track empty directories, so a
    topic directory survives a clone only if some tracked file keeps it alive. For six of the
    seven that is the README. clustermap had none, so it existed locally as generated output
    and vanished on clone -- and the predecessor of this test, which read os.listdir, passed
    in the working tree and failed on a fresh clone of the published repo. Checking the index
    instead of the filesystem is what makes the test see what a reader gets.

    Topics come from conftest.FIGURE_COUNTS so this list cannot drift from the one
    test_figures proves the generators actually produce.
    """
    tracked = set(subprocess.run(['git', 'ls-files'], cwd=REPO,
                                 capture_output=True, text=True).stdout.split())
    missing = [t for t in FIGURE_COUNTS
               if f'visuals/{t}/README.md' not in tracked]
    assert not missing, (
        f'no tracked visuals/<topic>/README.md for {missing} -- git cannot carry the '
        'directory, so it will be absent from a fresh clone')


def test_no_unexpected_topic_directory_on_disk():
    """The filesystem may hold FEWER topics than FIGURE_COUNTS (nothing has been rendered
    yet, as on a fresh clone) but never a topic nothing knows about."""
    present = {d for d in os.listdir(os.path.join(REPO, 'visuals'))
               if os.path.isdir(os.path.join(REPO, 'visuals', d)) and d != '__pycache__'}
    assert not present - set(FIGURE_COUNTS), (
        f'unknown directory under visuals/: {sorted(present - set(FIGURE_COUNTS))}')


def test_scripts_holds_three_gpu_jobs_and_the_local_driver():
    assert _names('scripts/*.sbatch') == [
        '00_smoke_test.sbatch', '01_extract.sbatch', '02_generation_eval.sbatch']
    assert _names('scripts/*.sh') == ['models.sh', 'pull_results.sh', 'run_local.sh']
    assert os.path.exists(os.path.join(REPO, 'scripts/_preamble.inc'))


def test_the_dropped_things_are_gone():
    for gone in ('MIGRATION.md', 'analysis/narrative', 'analysis/figures',
                 'analysis/visuals', 'src/functional_group_probe.py',
                 'src/functional_group_analogy_carbon_matched.py'):
        assert not os.path.exists(os.path.join(REPO, gone)), gone


def test_config_and_dataset_sit_where_paths_says():
    assert os.path.exists(os.path.join(REPO, 'config_extract_activation.yaml'))
    assert os.path.exists(os.path.join(REPO, 'data/functional_group_dataset.csv'))
