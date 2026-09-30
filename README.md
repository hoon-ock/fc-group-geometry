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

**The figures under `analysis/visuals/` are the output.** Everything else exists to produce
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
| `data/` | the 92-molecule dataset, and the extracted activations under `activation/` |
| `src/` | the pipeline: extraction, probe, analogy geometry, retrieval, anisotropy, generation eval |
| `analysis/<topic>/` | one directory per analysis, each with its `analyze_*.py`, the `data/*.csv` snapshot it writes, and a README arguing what the numbers show |
| `analysis/figures/` | the 9 figure generators. Read only `analysis/*/data/*.csv` — never `results/` |
| `analysis/visuals/` | **the output.** 70 PNGs, gitignored and rebuilt from the snapshots |
| `analysis/narrative/` | interpretation notes, each number traced to a committed CSV |
| `scripts/` | eight SLURM array jobs that fill `results/` and `data/activation/` |
| `results/` | generated, gitignored. Raw per-model pipeline output |
| `data/activation/` | the saved hidden states, one directory per checkpoint. Gitignored, ~1.1 GB |

## Reproducing the figures

Two entry points, depending on how far back you want to start.

**From the committed snapshots** — no GPU, no cluster, no `results/`, about a minute:

```bash
pip install -r requirements.txt
cd analysis/figures
for f in make_visuals*.py; do python "$f"; done
```

That rebuilds all 70 PNGs from the `analysis/*/data/*.csv` files in this repository. That it
works is the reproducibility test the whole layout is arranged around: no generator may read
`results/`, so if one ever needs it, a snapshot step was missed upstream.

**From scratch**, which needs a GPU and re-derives the snapshots too:

```bash
export HF_TOKEN=...                        # the Llama checkpoints are gated
sbatch scripts/01_extract.sbatch           # then 02..07 -- see scripts/README.md
python analysis/probe/analyze_sweep.py     # and the other analyze_*.py
```

`scripts/README.md` has the job order and dependencies; each `analysis/<topic>/README.md`
has the command that regenerates its own snapshot. The order among the analyses is:
`analyze_sweep` → `analyze_taxonomy` → `analyze_geometry` → `snapshot_quiver3d` →
`analyze_ambiguity` → `compute_tsne_coords` →
`analyze_prompt_structure` → `score_generations` → `analyze_generation`.

## Paths

`paths.py` is the single source of truth, resolved from `__file__`, so every script runs
from any working directory:

```bash
python paths.py          # print every root, marked ok / MISSING
```

The three generated roots take an environment override, which is what a cluster run needs
when the home quota cannot hold the activation tensors:

```bash
export FCG_ACTIVATIONS=/work/$USER/fc_group/activation
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
