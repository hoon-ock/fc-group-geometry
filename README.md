# Functional Group Geometry in LLM Hidden States

Do LLM hidden states encode organic functional groups as a consistent geometric structure,
analogous to word2vec-style analogies (`king - man + woman ~= queen`)?

For each non-alkane molecule the pipeline defines

```
diff_vec(group, C) = act(molecule with `group` at chain length C) - act(alkane backbone at the same chain length C)
```

e.g. `diff_vec(alcohol, 3) = act(propan-1-ol) - act(propane)`, then measures the geometry of
those difference vectors: within-class consistency across chain lengths, between-class
distinctness, analogy retrieval, an anisotropy baseline, a layer-wise linear probe, and a
behavioural free-generation counterpart to check whether any of it shows up in what the
model actually says.

Extraction never generates text — one forward pass per prompt, per-layer hidden states
pulled via forward hooks, last-token pooled.

**The figures under `visuals/` are the output.** Everything else exists to produce
them.

## What is measured, on what

92 molecules × 20 functional groups × 10 declarative prompt templates = 920 prompts, run
through five 8B checkpoints at all 32 layers.

| | |
|---|---|
| `meta-llama/Llama-3.1-8B` | base |
| `weidawang/Chem-R-8B` | chemistry-tuned, same base |
| `phenixace/Chem-R-Faithful` | chemistry-tuned, GRPO-trained *from* Chem-R-8B |
| `OpenDFM/ChemDFM-v1.5-8B` | chemistry-tuned, but on Llama-3 rather than 3.1 |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | reasoning-distilled |

The set is deliberately all-Llama at one size, so every contrast varies one thing against a
shared architecture and tokenizer. ChemDFM is the exception and is marked as such everywhere
it appears: it sits on Llama-3, so a ChemDFM-vs-base gap confounds chemistry training with
the 3 → 3.1 difference and is never a controlled contrast. Chem-R-8B is what carries that.

## Layout

| Path | Contents |
|---|---|
| `paths.py` | every path in the project, resolved from its own location. Import this; never hardcode |
| `config_extract_activation.yaml` | the model, pooling and prompt templates for a run — the one file to edit to change what the pipeline does |
| `data/` | the 92-molecule dataset |
| `src/` | the pipeline: extraction, probe, analogy geometry, retrieval, anisotropy, generation eval |
| `analysis/` | one script per analysis, flat, plus the 29 snapshot CSVs in `data/` and one README covering every topic |
| `visuals/` | the 9 figure generators *and* their output — **70 PNGs**, gitignored, rebuilt from the snapshots. Read only `analysis/data/*.csv`, never `results/` |
| `scripts/` | `run_local.sh` (the whole CPU pipeline, no cluster) plus three SLURM jobs for the GPU stages |
| `results/` | generated, gitignored. Raw per-model pipeline output |
| `activations/` | the saved hidden states, one directory per checkpoint. Gitignored, ~1.3 GB |

## Pipeline

Four stages. Only the first two need a GPU; the rest is one local command.

```
[1] 01_extract.sbatch          GPU  ->  activations/                 160 .pt, ~1.3 GB
[2] 02_generation_eval.sbatch  GPU  ->  results/generation_eval/     3 files per model
[3] run_local.sh --stages compute   ->  results/  (7 trees, named below)
[4] run_local.sh --stages analysis  ->  analysis/data/*.csv        29 snapshots, committed
    run_local.sh --stages figures   ->  visuals/            70 PNGs
```

### [1] Activations

```bash
sbatch scripts/01_extract.sbatch
```

One forward pass per prompt, per-layer hidden states pulled via forward hooks, last-token
pooled. 5 models × 32 layers × 920 prompts (92 molecules × 10 templates), fp16:

```
activations/<model-slug>/functional_group/functional_group.last.10_templates.layer_<N>.pt
```

160 files, 258 MB per model. `<model-slug>` is the checkpoint id with `/` replaced by `-`
(`meta-llama-Llama-3.1-8B`) — the same spelling `results/` and the figure directories use,
so one model is one string everywhere a path carries it.

### [2] Generation eval

```bash
sbatch scripts/02_generation_eval.sbatch
```

The only stage that asks the model to *answer* rather than probing frozen activations, which
is why it needs a GPU and stays on the cluster. Three files per model:

```
results/generation_eval/<model-slug>/functional_group/data/
  scores.csv        920 rows -- the forced choice
  generations.csv   920 rows -- the free continuation
  summary.json      aggregates, plus a free_generation block merged in at [4]
```

| File | Contents |
|---|---|
| `scores.csv` | The **forced choice**: no text is emitted. Each class's surface forms are scored by log-sum-exp and the argmax is the prediction. Per prompt: `pred_raw`, `pred_norm`, `correct_raw`, `correct_norm`, `best_surface_form`, `p_true`, `margin`, plus `logp_<class>` and `logpnorm_<class>` for all 20 classes. |
| `generations.csv` | The **free continuation**, 10 new tokens, greedy: `iupac_name`, `template_index`, `true_label`, `prompt`, `generation`. Logged verbatim and *not scored by this job* — `analysis/score_generations.py` adjudicates it at [4]. |
| `summary.json` | `accuracy`, `balanced_accuracy` and `log_loss` under both `raw` and `length_normalized`, plus `chance` and `mean_top1_margin`. |

Generations contain embedded newlines, so `wc -l` reports more lines than records — read
`generations.csv` as CSV, not by line.

### [3] The CPU compute

```bash
bash scripts/run_local.sh --stages compute
```

Five stages per model, single-core, no GPU, ~8 min per model with the probe dominating:

| Stage | Reads | Writes under `results/` |
|---|---|---|
| probe, all 32 layers | activations | `probe/` |
| anisotropy, L 0/8/16/24/31 | activations | `anisotropy_diagnostic/` |
| analogy, carbon-matched, same layers | activations | `analogy/` |
| retrieval, 3 axes | the analogy stage's `.npz` | `analogy_retrieval{,_carbon,_halide}/` |
| ambiguity | the probe's `oof_proba_*.npz` | `ambiguity_metric/` |

The last two never touch the tensors — they read what the stages above wrote, which is why
they take seconds.

### [4] Snapshots, then figures

```bash
bash scripts/run_local.sh          # or --stages analysis / --stages figures
```

**Analysis** turns `results/` into the 29 committed CSVs under `analysis/data/`. That is
the reproducibility boundary: past this point nothing reads `results/`. Two ordering
constraints — `analyze_taxonomy` writes the matrix the clustermaps need, and
`score_generations` must precede `analyze_generation`, which reads its adjudicated table:

```
analyze_sweep -> analyze_taxonomy -> analyze_geometry -> snapshot_quiver3d ->
analyze_ambiguity -> compute_tsne_coords -> analyze_prompt_structure ->
score_generations -> analyze_generation
```

`compute_tsne_coords` and `analyze_prompt_structure` are the two analyses that read
activations directly, so **their 29 figures need stage [1] only** — not [2] or [3].

**Figures** draw all 70 PNGs from those CSVs in about 15 seconds and need nothing else.

## Where to start

| You have | Run | Time |
|---|---|---|
| just this repository | `bash scripts/run_local.sh --stages figures` | ~15 s |
| the activations | `bash scripts/run_local.sh` | under an hour |
| nothing | the four stages above | a cluster job plus the hour |

From nothing, end to end:

```bash
pip install -r requirements.txt
export HF_TOKEN=...                                   # the Llama checkpoints are gated

sbatch scripts/01_extract.sbatch                      # [1]
sbatch scripts/02_generation_eval.sbatch              # [2]

rsync -a --partial swan:/work/$USER/fc_group/activations/ activations/
bash scripts/pull_results.sh                          # results/generation_eval/ only

bash scripts/run_local.sh                             # [3] + [4]
```

Without `results/generation_eval/`, `run_local.sh` skips the two generation steps with a
note and the committed generation snapshot still draws its figure. That every
`make_visuals*.py` reads only committed CSVs is the property the whole layout is arranged
around: if one ever needs `results/`, a snapshot step was missed upstream.

`scripts/README.md` carries the operational detail — resources, array sizing, triage, and a
figure-by-figure table of which stage each one needs.

## Paths

`paths.py` is the single source of truth, resolved from `__file__`, so every script runs
from any working directory:

```bash
python paths.py          # print every root, marked ok / MISSING
```

The three generated roots take an environment override, which is what a cluster run needs
when the home quota cannot hold the activation tensors:

```bash
export FCG_ACTIVATIONS=/work/$USER/fc_group/activations
export FCG_RESULTS=/work/$USER/fc_group/results
```

`FCG_LOGS` works the same way. Output directories are created by the scripts that write
them, so a fresh clone needs no `mkdir`.

## Adding a model

`model_name` is a config/CLI parameter, not hardcoded. Since the pipeline only does forward
passes — no chat template, no output parsing — a new checkpoint needs no prompt-format work.
It needs an entry in `src/model_registry.py` and must satisfy two constraints: it exposes
`model.model.layers` (true for Llama/Mistral/Qwen-family causal LMs in `transformers`, not
guaranteed generally), and the *installed* `transformers` must recognise its `model_type` —
a newer architecture than the environment knows fails inside `AutoConfig.from_pretrained`
only after the weights have downloaded.

```bash
python src/check_model_registry.py
```

checks both, plus `hidden_dim`/`num_layers` against the model's real `config.json`, in
seconds and without a GPU. Run it before adding an entry or submitting a sweep.

Registered is not the same as swept: `scripts/models.sh` does a subset check and names any
registered-but-unswept model on startup.

## Scope

One prompt type ships. The working tree this was released from swept 23 — the same 92
molecules asked about pKa, TPSA, boiling point and so on — as an attribute-specificity
matrix asking whether functional-group identity is present in every last-token state or only
when the prompt asks for it. Every released result comes from the `functional_group`
declarative prompts alone, so that is all the config declares, and the dataset here carries
identifiers only (no measured property columns — nothing interpolates them into a prompt).

The `entity_type` and `family` columns survive in the CSVs because the analysis scripts are
unchanged and would emit more if more prompt types were extracted.

## Tests

```bash
pip install -r requirements-dev.txt
pytest                    # 228 checks, ~5 s
pytest --runslow          # 247 -- + rebuild all 70 figures, + reproduce the snapshots
```

**Nothing in the suite needs a GPU**, and `tests/test_no_gpu.py` pins that: it asserts by
AST that no test hands a checkpoint-loading script or the scheduler to a subprocess, and
that no test imports torch. The GPU stages' outputs are inputs to the tests, never things
the tests produce.

The fast tier covers structure, `paths.py` resolution (including cwd-independence and the
`FCG_*` overrides), that all 30 modules import, that every producer is wired into
`run_local.sh` and every `load()` name exists, the 29 snapshots' schema and record
terminators, shell syntax and the `--array`/`check_array_size` agreement, and that no
renamed or dropped name survives in code.

`--runslow` adds the two end-to-end tiers:

| tier | needs | what it proves |
|---|---|---|
| figures | nothing but this repo | all 9 generators run and produce exactly 70 PNGs at the right per-topic counts |
| reproduce · activations | `activations/` | `compute_tsne_coords` and `analyze_prompt_structure` reproduce their 3 snapshots byte-for-byte |
| reproduce · results | whichever `results/` trees exist | each analysis reproduces its own snapshots byte-for-byte |

The reproduce tiers **skip rather than fail** when their inputs are absent, and each script
is checked **on its own inputs** rather than as one chain — so a partially populated
`results/` still exercises everything it can reach. That division is there for one stage in
particular: the probe sweep is ~8 min per model, 40 min for all five, and `analyze_sweep` is
the only analysis that needs it. The other analyses are not held behind it. Each skip message
names the missing trees and the CPU-only command that fills them.

The one exception is the generation pair, tested as a chain because it is one:
`analyze_generation` reads `free_generation_by_template.csv` from its own `--out-dir`, so
`score_generations` must have run there first. A second test asserts that running
`analyze_generation` alone produces a `template_sensitivity.csv` that *does not* match the
committed snapshot — otherwise the chain could be passing for the wrong reason, with the
documented ordering being decorative.

Everything writes to a pytest `tmp_path` and is compared against the committed file, so
running the suite cannot modify `analysis/data/` or `visuals/` — the artifacts under test.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Everything under `analysis/` is CPU-only. `bitsandbytes` 4-bit quantization is Linux +
NVIDIA only; on macOS/CPU extraction falls back to fp16/bf16. Gated checkpoints need
`export HF_TOKEN=...` (checked first, and the one to use on a cluster) or a one-time
`huggingface-cli login`. Do not put a token in the tracked config.

## License

See `LICENSE`.
