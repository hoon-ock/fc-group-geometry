# Running the pipeline

**Only the GPU work needs a cluster.** Three SLURM jobs here do that. Everything else —
probe, anisotropy, analogy, retrieval, ambiguity, every `analyze_*.py`, and all 70 figures —
runs locally from the activations in one command:

```bash
bash scripts/run_local.sh
```

That was not always true. The five CPU stages used to be SLURM array jobs
(`02_probe` … `06_ambiguity`) because they were written beside the extraction that does need
a GPU. Each asks for one core and 4–8 GB and never touches a GPU, so on a machine that
already has the activations, a cluster round-trip plus `pull_results.sh` bought nothing.
They now live inside `run_local.sh`.

## What still needs a GPU

| Job | Stage | Resources | Writes |
|---|---|---|---|
| `00_smoke_test` | one template, three layers, then verify shapes against the registry | 1 GPU | — |
| `01_extract` | forward pass, per-layer hidden states via forward hooks | 1 GPU, 8 cores | `activations/` |
| `02_generation_eval` | the behavioural stage: forced choice + free generation | 1 GPU, 8 cores | `results/generation_eval/` |

Submit from the repository root. Each job resolves the root from `$SLURM_SUBMIT_DIR` and
fails immediately if `paths.py` is not there; for a manual run elsewhere, set `FCG_ROOT`.

```bash
sbatch scripts/01_extract.sbatch
```

Both arrays are `0-4` — five models × one entity type. Both lists are derived at runtime
(models from `models.sh`, entity types from `config_extract_activation.yaml`), so they cannot
drift from each other or from what was extracted. The static `#SBATCH --array` span is the
one thing that cannot self-update, so each job calls `check_array_size 5` and refuses to
start on a mismatch:

```
ERROR: 5 models x 1 entity types = 5 runs,
       but '#SBATCH --array' declares 115.
       Update the --array line and ARRAY_SIZE together.
```

## `run_local.sh`

Three stages, and any one can run alone:

```bash
bash scripts/run_local.sh                     # compute -> analysis -> figures
bash scripts/run_local.sh --stages compute    # the five CPU stages, per model
bash scripts/run_local.sh --stages analysis   # results/ -> analysis/*/data/
bash scripts/run_local.sh --stages figures    # analysis/*/data/ -> analysis/visuals/
bash scripts/run_local.sh --models meta-llama/Llama-3.1-8B
```

Measured per model: anisotropy ~6 s, analogy ~1–3 min (the permutation null dominates),
probe ~8 min for all 32 layers, retrieval and ambiguity seconds each — those two read the
npz the earlier stages wrote rather than the tensors. The probe is the long pole; the whole
five-model compute stage is well under an hour.

**`--stages figures` needs nothing but the repository.** Every `make_visuals*.py` reads only
the committed `analysis/*/data/*.csv`, so a fresh clone with no activations draws all 70
PNGs in about 15 seconds. The compute and analysis stages exist to *re-derive* those
snapshots; the figures consume whatever is on disk.

**`--stages analysis` skips the GPU-only half gracefully.** `score_generations.py` and
`analyze_generation.py` need `results/generation_eval/`, which only `02_generation_eval`
produces. Without it those two are skipped with a note, the committed generation snapshot
stays as it is, and the figure still draws from it.

## Which stages a figure needs

Two analysis scripts read the `.pt` tensors directly and no `results/` tree at all, so
**29 of the 70 figures need only the activations** — not even the local compute stage:

| Figures | Count | Producer | Needs |
|---|---|---|---|
| `tsne/` | 28 | `analysis/tsne/compute_tsne_coords.py` | activations |
| `prompt_structure/` | 1 | `analysis/prompt_structure/analyze_prompt_structure.py` | activations |
| `clustermap/` | 30 | `analysis/taxonomy/analyze_taxonomy.py` | local: analogy |
| `retrieval/rankdist`, `rankviolin` | 2 | `analysis/geometry/analyze_geometry.py` | local: anisotropy + analogy + retrieval |
| `retrieval/analogy3d_*` | 6 | `analysis/analogy/snapshot_quiver3d.py` | local: analogy + retrieval |
| `probe/` | 1 | `analysis/probe/analyze_sweep.py` | local: probe |
| `ambiguity/` | 1 | `analysis/ambiguity/analyze_ambiguity.py` | local: probe + analogy + ambiguity |
| `generation/` | 1 | `score_generations.py` → `analyze_generation.py` | **`02_generation_eval` (GPU)** |

Worth being explicit about the division, because the directory names invite the opposite
reading: **`src/` computes and `analysis/` summarises.** `analysis/probe/analyze_sweep.py`
does not fit a probe — it contains no sklearn calls at all and reads the `summary_*.json`
that `src/functional_group_probe.py` wrote. The same holds for ambiguity, taxonomy, geometry
and generation. The two activations-only scripts above are the exceptions.

Each `analysis/<topic>/README.md` carries the command that regenerates its own snapshot.

## `models.sh`

The sweep is declared there; `src/model_registry.py` is the catalogue it draws from. The
check is a **subset** check, not equality: a model must be registered for
`get_model_config` to resolve its depth, but it may be registered without being swept. The
70B pair is in that position, so sourcing `models.sh` prints:

```
note: registered, not in this sweep: deepseek-ai/DeepSeek-R1-Distill-Llama-70B meta-llama/Llama-3.1-70B
```

A swept name that is *not* registered is a hard error — the failure this file exists to
catch, since a typo would otherwise narrow the sweep silently instead of stopping it.

## Scratch space

`01_extract` writes ~1.3 GB (5 models × 32 layers × 920 prompts, fp16). On a cluster whose home
quota cannot hold that, redirect it **before** extracting — `paths.py` reads these, and
every later stage must see the same values:

```bash
export FCG_ACTIVATIONS=/work/$USER/fc_group/activations
export FCG_RESULTS=/work/$USER/fc_group/results
```

`FCG_LOGS` works the same way. `01_extract` echoes the resolved activation path on startup,
so a misconfigured run is visible in the first lines of its log rather than after the first
model loads.

Gated checkpoints need Hugging Face auth: `export HF_TOKEN=...` before submitting. The jobs
also set `HF_HOME` to a shared cache, overridable in the environment.

## Triage

Every sbatch job ends with one `SUMMARY:` line per task, so a sweep reduces to one grep:

```bash
grep -h SUMMARY logs/slurm-extract-*.out | sort | uniq -c
```

`run_local.sh` reports `ok` with a duration per step, or `FAILED` with the last 12 lines of
that step's output, and stops there rather than carrying a broken input forward.

## Bringing work back from the cluster

The activations, once:

```bash
rsync -a --partial swan:/work/$USER/fc_group/activations/ activations/
```

The generation eval, the only `results/` tree the cluster still writes:

```bash
bash scripts/pull_results.sh --dry-run    # then without the flag
```

Override `FCG_REMOTE`, `FCG_REMOTE_ROOT` and `FCG_RESULTS` rather than editing the script.
It does **not** use `--delete`, which matters for one file:
`generation_eval/<model>/functional_group/data/summary.json` carries a `free_generation`
block merged in locally, and the remote copy overwrites it. `run_local.sh --stages analysis`
re-derives it.

## Files

| | |
|---|---|
| `run_local.sh` | the whole CPU pipeline: five compute stages, nine analysis scripts, nine figure generators |
| `_preamble.inc` | sourced first by every sbatch: resolves `ROOT`, activates the env, pins BLAS thread counts (the compute nodes cap `RLIMIT_NPROC` and OpenBLAS blows past it) |
| `models.sh` | the five swept models, the registry subset check, `read_entity_types`, `check_array_size` |
| `pull_results.sh` | selective rsync of `results/generation_eval/` back from the cluster |
