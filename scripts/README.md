# Cluster jobs

Nine SLURM array jobs that fill `results/` and `activations/`. Everything downstream — the
snapshot CSVs under `analysis/*/data/` and the figures under `analysis/visuals/` — runs on a
laptop from what these produce.

Submit from the repository root:

```bash
sbatch scripts/01_extract.sbatch
```

Each job resolves the root from `$SLURM_SUBMIT_DIR` and fails immediately if `paths.py` is
not there. For a manual run from elsewhere, set `FCG_ROOT=/path/to/repo`.

## Order

`01` must finish before anything else. After that, `02`/`03`/`04`/`07`/`08` are independent;
`05` needs `04`, and `06` needs `02`.

| Job | Stage | Resources | Reads | Writes under `results/` |
|---|---|---|---|---|
| `00_smoke_test` | one template, three layers, then verify shapes against the registry | 1 GPU | — | — |
| `01_extract` | forward pass, per-layer hidden states via hooks | 1 GPU, 8 cores | `data/` | → `activations/` |
| `02_probe` | layer-wise linear probe, all 32 layers | 1 core, 4 GB | `activations/` | `functional_group_probe/` |
| `03_anisotropy` | generic-anisotropy baseline for the diff-vector geometry | 1 core | `activations/` | `anisotropy_diagnostic/` |
| `04_analogy` | diff-vector geometry, carbon-matched | 1 core | `activations/` | `functional_group_analogy/` |
| `05_retrieval` | does `b2 + (a1 - a2)` retrieve `b1`? three axes | 1 core | `04`'s npz files | `functional_group_analogy_retrieval{,_carbon,_halide}/` |
| `06_ambiguity` | how much mass lands on the defensible rival family | 1 core | `02`'s `oof_proba_*.npz` | `ambiguity_metric/` |
| `07_separability` | per-group rival margin, all 32 layers | 1 core | `activations/` | `functional_group_separability/` |
| `08_generation_eval` | the behavioural stage: forced choice + free generation | 1 GPU, 8 cores | `data/` | `generation_eval/` |

Only `01`, `00` and `08` need a GPU. `02` deliberately asks for one core and 4 GB with no
GPU: it is entirely scikit-learn (`torch` appears only as `torch.load(map_location="cpu")`),
extra cores do not help because the matrices are too small for BLAS parallelism and lbfgs is
serial, and an earlier 32 GB request pinned the array to a few concurrent tasks with the rest
in `PD (Resources)`. Throughput comes from array width, not from per-task size.

## Every array is `0-4`

Five models × one entity type. Both lists are derived at runtime — the models from
`models.sh`, the entity types from `data/config_extract_activation.yaml` — so they cannot
drift from each other or from what was actually extracted. The one thing that cannot
self-update is the static `#SBATCH --array` span, so every job calls
`check_array_size 5` and refuses to start if the product disagrees:

```
ERROR: 5 models x 1 entity types = 5 runs,
       but '#SBATCH --array' declares 115.
       Update the --array line and ARRAY_SIZE together.
```

Add a prompt type to the YAML and that message tells you exactly which lines to widen.

## `models.sh`

The sweep is declared there; `src/model_registry.py` is the catalogue it draws from. The
check is a **subset** check, not equality: a model must be registered for
`get_model_config` to resolve its depth, but it may be registered without being swept. The
70B pair is in that position — registered, unswept, and used by nothing released here — so
sourcing `models.sh` prints:

```
note: registered, not in this sweep: deepseek-ai/DeepSeek-R1-Distill-Llama-70B meta-llama/Llama-3.1-70B
```

A swept name that is *not* registered is a hard error, which is the failure the file exists
to catch: a typo would otherwise narrow the sweep silently rather than stopping it.

## Scratch space

`01_extract` writes ~1.1 GB (5 models × 32 layers × 920 prompts) and `results/` adds a few
hundred MB more. On a cluster whose home quota cannot hold that, redirect both **before**
extracting — `paths.py` reads these, and every later stage must see the same values:

```bash
export FCG_ACTIVATIONS=/work/$USER/fc_group/activations
export FCG_RESULTS=/work/$USER/fc_group/results
```

`01_extract` echoes the resolved activation path on startup so a misconfigured run is
visible in the first lines of its log rather than after the first model loads.

Gated checkpoints need Hugging Face auth: `export HF_TOKEN=...` before submitting. The jobs
also set `HF_HOME` to a shared cache, overridable in the environment.

## Triage

Every job ends with one `SUMMARY:` line per task, so a whole sweep reduces to one grep:

```bash
grep -h SUMMARY logs/slurm-probe-*.out | sort | uniq -c
```

The outcomes are distinguished on purpose rather than collapsed into pass/fail:
`no activation data` and `no retrieval input` mean a stage ran before its input existed — an
ordering problem. `bad-output` means `analogy_retrieval.py`'s own self-check failed, so the
analysis ran but its outputs do not agree with each other, which is a bug in the code rather
than the environment. `crashed` is an ordinary traceback.

## Bringing results back

```bash
bash scripts/pull_results.sh --dry-run    # then without the flag
```

Override `FCG_REMOTE`, `FCG_REMOTE_ROOT` and `FCG_RESULTS` rather than editing the script.
It copies only the `functional_group` leaf of each method, and it does **not** use
`--delete`, which matters for one file: `generation_eval/<model>/functional_group/data/
summary.json` carries a `free_generation` block merged in locally, and the remote copy
overwrites it. Re-derive it after every pull — it is CPU-only and takes seconds:

```bash
python analysis/generation/score_generations.py
```

## Files

| | |
|---|---|
| `_preamble.inc` | sourced first by every job: resolves `ROOT`, activates the env, pins BLAS thread counts (the compute nodes cap `RLIMIT_NPROC` and OpenBLAS blows past it) |
| `models.sh` | the five swept models, the registry subset check, `read_entity_types`, `check_array_size` |
| `pull_results.sh` | selective rsync of `results/` back from the cluster |
