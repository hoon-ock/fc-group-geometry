# Functional Group Geometry in LLM Hidden States

[![tests](https://github.com/hoon-ock/fc-group-geometry/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/hoon-ock/fc-group-geometry/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)

This project tackles two questions:

 - Do LLM hidden states encode organic functional groups as a consistent geometric structure,
e.g. word2vec-style analogies (`king - man + woman ~= queen`)?
 - How does the training procedure impact the embedding geometry and generation performance?

## Models

92 molecules × 20 functional groups × 10 declarative prompt templates = 920 prompts, run
through five 8B checkpoints at all 32 layers.

| Checkpoint | Role | Access | License |
|---|---|---|---|
| [`meta-llama/Llama-3.1-8B`](https://huggingface.co/meta-llama/Llama-3.1-8B) | base | **request, reviewed by hand** — not instant | `llama3.1` |
| [`weidawang/Chem-R-8B`](https://huggingface.co/weidawang/Chem-R-8B) | chemistry-tuned, same base | accept terms, automatic | `apache-2.0` |
| [`phenixace/Chem-R-Faithful`](https://huggingface.co/phenixace/Chem-R-Faithful) | chemistry-tuned, GRPO-trained *from* Chem-R-8B | open | `cc-by-4.0` |
| [`OpenDFM/ChemDFM-v1.5-8B`](https://huggingface.co/OpenDFM/ChemDFM-v1.5-8B) | chemistry-tuned, but on Llama-3 rather than 3.1 | open | `agpl-3.0` |
| [`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) | reasoning-distilled | open | `mit` |

The set is deliberately all-Llama at one size, so every contrast varies one thing against a
shared architecture and tokenizer. ChemDFM is the exception and is marked as such everywhere
it appears: it sits on Llama-3, so a ChemDFM-vs-base gap confounds chemistry training with
the 3 → 3.1 difference and is never a controlled contrast. Chem-R-8B is what carries that.

### Getting access to the checkpoints

Two of the five are gated, and they gate differently — the base model's request is reviewed
by a human, so start it before you plan anything else. Request access while logged in, from
each model's page linked above. Then create a **read** token at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and put it in the
environment — never in the tracked config:

```bash
export HF_TOKEN=hf_...
```

`HF_TOKEN` is checked first and is the one to use on a cluster; a one-time
`huggingface-cli login` works locally instead. Only the three GPU stages download weights —
everything downstream reads the saved activations, so a reader reproducing the figures needs
no token at all. See [Where to start](#where-to-start) for which of those applies to you.

## Data

`data/functional_group_dataset.csv` — 92 molecules, identifiers only:

| iupac_name | common_name | formula | smiles | carbon_count | functional_group | functional_group_structure |
|---|---|---|---|---|---|---|
| Propane | propane | CH3CH2CH3 | `CCC` | 3 | none (alkane) | C-H / C-C |
| Propan-1-ol | n-propyl alcohol | CH3CH2CH2OH | `CCCO` | 3 | alcohol | -OH |
| Propan-2-ol | isopropanol | CC(O)C | `CC(O)C` | 3 | alcohol | -OH |
| Propanal | propionaldehyde | CH3CH2CHO | `CCC=O` | 3 | aldehyde | -CHO |
| … 88 more | | | | | | |

20 functional groups at chain lengths C3–C6. Seventeen carry one molecule per length, 4 each;
`alcohol`, `amine` and `thiol` carry two isomers per length, 8 each — `Propan-1-ol` beside
`Propan-2-ol` above. 17 × 4 + 3 × 8 = **92**. `none (alkane)` is a group here because the
alkanes are the backbone every diff-vector subtracts.

Each molecule is asked 10 ways, so 92 × 10 = **920 prompts** per model:

```
Propan-1-ol (CCCO) contains the functional group
The primary functional group in Propan-1-ol (CCCO) is
For Propan-1-ol (CCCO), the functional-group class is
...                                                     (10 templates in total)
```

Declarative and unpunctuated, so the next token is the answer. They live in
`config_extract_activation.yaml` and interpolate only `{iupac_name}` and `{smiles}`.

Each prompt is one forward pass, and the last token's hidden state is kept at every layer:

```
activations/<model-slug>/functional_group/functional_group.last.10_templates.layer_<N>.pt

  torch.Size([920, 4096])   float16   7.5 MB per layer
```

Row order is molecule-major, template-minor — rows 0–9 are `Propane` under all ten templates,
rows 10–19 are `Butane`, and so on. `template_index` is not a stored column; the probe and the
t-SNE labelling both reconstruct it from that layout, so it is the one thing here that cannot
change silently.

## Layout

**The figures under `visuals/` are the output** — the 70 release PNGs and the 8 manuscript
PDFs beside them. Everything else exists to produce them.

| Path | Contents |
|---|---|
| `paths.py` | every path in the project, resolved from its own location. Import this; never hardcode |
| `config_extract_activation.yaml` | the model, pooling and prompt templates for a run — the one file to edit to change what the pipeline does |
| `data/` | the 92-molecule dataset |
| `src/` | the pipeline: extraction, probe, analogy geometry, retrieval, anisotropy, generation eval |
| `analysis/` | one script per analysis, flat, plus the 30 snapshot CSVs in `data/` and one README covering every topic |
| `visuals/` | the 9 release figure generators *and* their output — **70 PNGs**, gitignored, rebuilt from the snapshots. Read only `analysis/data/*.csv`, never `results/` |
| `visuals/figure/` | the 8 **manuscript** figure generators — 4 main, 4 supplementary — writing PDFs for the paper. Outside the 70-PNG set and not built by `run_local.sh`; same snapshots-only rule. See `visuals/figure/README.md` |
| `scripts/` | `run_local.sh` (the whole CPU pipeline, no cluster) plus three SLURM jobs for the GPU stages |
| `results/` | generated, gitignored. Raw per-model pipeline output |
| `activations/` | the saved hidden states, one directory per checkpoint. Gitignored, ~1.2 GB |

## Pipeline

Four stages. Only the first two need a GPU; the rest is one local command.

```
[1] 01_extract.sbatch          GPU  ->  activations/                 160 .pt, ~1.2 GB
[2] 02_generation_eval.sbatch  GPU  ->  results/generation_eval/     3 files per model
[3] run_local.sh --stages compute   ->  results/  (7 trees, named below)
[4] run_local.sh --stages analysis  ->  analysis/data/*.csv        30 snapshots, committed
    run_local.sh --stages figures   ->  visuals/            70 PNGs

    visuals/figure/fig*.py          ->  visuals/figure/     8 PDFs   (not run_local.sh)
```

### [1] Activations

```bash
sbatch scripts/01_extract.sbatch
```

**Extraction never generates text.** One forward pass per prompt, per-layer hidden states
pulled via forward hooks, last-token pooled. 5 models × 32 layers × 920 prompts
(92 molecules × 10 templates), fp16:

```
activations/<model-slug>/functional_group/functional_group.last.10_templates.layer_<N>.pt
```

160 files, 241 MB per model. `<model-slug>` is the checkpoint id with `/` replaced by `-`
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

Five independent measurements over the same frozen activations, each asking a different
question. None loads a model; all are single-core and CPU-only:

- **probe** — the supervised question: *is a molecule's functional group linearly decodable
  from the last-token residual stream, and at what depth does decodability peak?* All 32
  layers, leave-one-fine-group-out, so withholding `alkyl iodide` entirely tests whether its
  molecules still land with the halides. The long pole by a wide margin.
- **anisotropy** — the control the rest depends on. Transformer hidden states share a dominant
  direction present almost regardless of content, so this measures how much of the geometry
  below is that concentration rather than chemistry.
- **analogy** — the unsupervised counterpart, carbon-matched. Builds
  `diff_vec(group, C) = act(molecule with group at chain length C) − act(alkane backbone at
  the same C)` — e.g. `diff_vec(alcohol, 3) = act(propan-1-ol) − act(propane)` — then scores
  within-class consistency across chain lengths against between-class distinctness.
- **retrieval** — the same analogy question asked the way word2vec actually asked it. The
  cosine test above tops out near +0.66, which reads as *"the analogy isn't here"*; retrieval
  instead asks whether `thioether − thiol + alcohol` lands **nearest** to `ether`. Three axes.
- **ambiguity** — where the probe's mass goes for the four groups whose family label is a
  naming convention rather than a chemical fact: `amide` and `nitro` both contain oxygen but
  are labelled nitrogen, `sulfone` and `sulfoxide` sulfur.

Mechanically, ~8 min per model with the probe dominating:

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

**Analysis** turns `results/` into the 30 committed CSVs under `analysis/data/`. That is
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

#### Manuscript figures

The paper's own figures — four main and four supplementary — are built by the eight scripts
under `visuals/figure/`, as PDFs rather than PNGs because their destination is a LaTeX
document. They read the same committed snapshots and need nothing else either, but they are
**not** part of the 70 and `run_local.sh` does not build them:

```bash
for f in visuals/figure/fig*.py; do python "$f"; done   # 8 PDFs into visuals/figure/
```

Each one composes the release generators rather than reimplementing them, so a manuscript
panel and the released PNG of the same measurement cannot disagree.
`visuals/figure/README.md` has the figure-by-figure table and the conventions they share.

## Where to start

| You have | Run | Time |
|---|---|---|
| just this repository | `bash scripts/run_local.sh --stages figures` | ~15 s |
| the activations | `bash scripts/run_local.sh` | under an hour |
| nothing | the four stages above | a cluster job plus the hour |

From nothing, end to end:

```bash
pip install -r requirements.txt
export HF_TOKEN=...                                   # two of the five are gated

sbatch scripts/01_extract.sbatch                      # [1]
sbatch scripts/02_generation_eval.sbatch              # [2]

rsync -a --partial <cluster>:/work/$USER/fc_group/activations/ activations/
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
pytest                    # 335 checks, ~5 s
pytest --runslow          # + the 25 skipped above: rebuild all 70 figures, reproduce the snapshots
```

**Nothing in the suite needs a GPU**, and `tests/test_no_gpu.py` pins that: it asserts by
AST that no test hands a checkpoint-loading script or the scheduler to a subprocess, and
that no test imports torch. The GPU stages' outputs are inputs to the tests, never things
the tests produce.

The fast tier covers structure, `paths.py` resolution (including cwd-independence and the
`FCG_*` overrides), that all 38 modules import — the eight under `visuals/figure/` included,
which a non-recursive glob would have left untested — that every producer is wired into
`run_local.sh` and every `load()` name exists, the 30 snapshots' schema and record
terminators, shell syntax and the `--array`/`check_array_size` agreement, and that no
renamed or dropped name survives in code.

It also checks that **the documentation points at things that exist** — every markdown link
and `#anchor`, and every backticked path anchored on a real directory, in prose *and* in
docstrings. This exists because a fresh clone once failed: `visuals/clustermap/` had no
tracked file, so git could not carry the directory, and checking for more of the same found
that all four markdown links in the repo were broken along with 32 stale path references —
all of it rot from flattening `analysis/`. The structural test that missed it read
`os.listdir`; it now reads `git ls-files`, so it sees what a reader gets rather than what the
working tree happens to hold. The same rule covers the README badges: an Actions badge must
name a workflow that exists, in *this* repository.

`--runslow` adds the two end-to-end tiers:

| tier | needs | what it proves |
|---|---|---|
| figures | nothing but this repo | all 9 release generators run and produce exactly 70 PNGs at the right per-topic counts. The 8 manuscript scripts are checked for importability only, not rendered |
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
NVIDIA only; on macOS/CPU extraction falls back to fp16/bf16. For the checkpoints and the
token the GPU stages need, see
[Getting access to the checkpoints](#getting-access-to-the-checkpoints).

## License

See `LICENSE`.
