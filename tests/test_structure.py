"""The layout is a contract: the docs, run_local.sh and paths.py all assume this shape."""
import glob
import os

from conftest import REPO


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


def test_visuals_holds_nine_generators_and_seven_output_dirs():
    assert len(_names('visuals/make_visuals*.py')) == 9
    assert os.path.exists(os.path.join(REPO, 'visuals/_common.py'))
    subdirs = sorted(d for d in os.listdir(os.path.join(REPO, 'visuals'))
                     if os.path.isdir(os.path.join(REPO, 'visuals', d))
                     and d != '__pycache__')
    assert subdirs == ['ambiguity', 'clustermap', 'generation', 'probe',
                       'prompt_structure', 'retrieval', 'tsne']


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
