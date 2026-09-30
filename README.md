# Functional Group Geometry in LLM Hidden States

Do LLM hidden states encode organic functional groups as a consistent geometric
structure, analogous to word2vec-style analogies (`king - man + woman ~= queen`)?

For each non-alkane molecule the pipeline defines

```
diff_vec(group, C) = act(molecule with `group` at chain length C) - act(alkane backbone at the same chain length C)
```

e.g. `diff_vec(alcohol, 3) = act(propan-1-ol) - act(propane)`, then analyzes the geometry
of those difference vectors: within-class consistency across chain lengths, between-class
distinctness, retrieval, an anisotropy baseline, a layer-wise linear probe, and a
behavioural free-generation counterpart. Extraction never generates text -- one forward
pass per prompt, per-layer hidden states pulled via forward hooks.

> **Migration in progress.** This repository is being assembled from the `fc_group/`
> subdirectory of an earlier working tree. `MIGRATION.md` tracks what has landed and the
> path rewrites each file needs. Until it is complete, this README is a skeleton and the
> sections below are the layout contract, not a usage guide.

## Layout

| Path | Contents |
|---|---|
| `paths.py` | Every path in the project, resolved from its own location. Import this; never hardcode. |
| `data/` | Committed inputs: the curated molecule dataset and the extraction/prompt config |
| `src/` | The pipeline: extraction, probe, analogy geometry, retrieval, anisotropy, t-SNE, generation eval |
| `analysis/` | One directory per analysis topic, each with the `analyze_*.py` and the `data/*.csv` snapshot it writes |
| `analysis/paper/` | Table, figure and verification generators. Read only `analysis/*/data/*.csv` -- never `results/` |
| `scripts/` | SLURM jobs for the sweep, plus `scripts/chem/` for the chemistry-model track |
| `results/` | Generated, gitignored. Raw per-model pipeline output |
| `activations/` | Generated, gitignored. Saved hidden-state tensors, one subdirectory per model |
| `paper_out/` | Generated, gitignored. Figure and table output |

## Paths

`paths.py` is the single source of truth, resolved from `__file__`. Scripts therefore run
from any working directory:

```bash
python paths.py          # print every root, marked ok / MISSING
```

The four generated roots take an environment override, which is what a cluster run needs
when the home quota cannot hold the activation tensors:

```bash
export FCG_ACTIVATIONS=/work/$USER/fc_group/activations
export FCG_RESULTS=/work/$USER/fc_group/results
```

`FCG_LOGS` and `FCG_PAPER_OUT` work the same way. Output directories are created by the
scripts that write them, so a fresh clone needs no `mkdir`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`bitsandbytes` 4-bit quantization is Linux + NVIDIA only; on macOS/CPU the extraction
falls back to plain fp16/bf16. Gated checkpoints need Hugging Face auth -- `export
HF_TOKEN=...` (checked first, and the one to use on a cluster) or `huggingface-cli login`
once. Do not put a token in the tracked config.

## Reproducibility

The paper is rebuilt from committed artifacts alone -- no GPU, no `results/`:

```bash
python analysis/paper/check_claims.py     # assert the numbers the argument rests on
python analysis/paper/make_figures.py     # -> paper_out/
python analysis/paper/make_tables.py      # -> paper_out/
```

That these read only `analysis/*/data/*.csv` is the reproducibility test: if one of them
ever needs `results/`, a snapshot step was missed upstream.
