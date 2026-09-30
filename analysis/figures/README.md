# The figure generators

Everything that turns `analysis/*/data/*.csv` into the PNGs under `analysis/visuals/`. These
figures are the repository's end product; every stage upstream exists to feed them.

## The rule this directory enforces

**No generator here may read `results/`.** Each reads only the committed snapshot CSVs under
`analysis/*/data/`, which is what makes every figure rebuildable from a clean checkout with
no GPU, no activations and no pipeline run. That is a test, not a convention:

```bash
mkdir -p /tmp/repro && cp -R ../../analysis ../../paths.py /tmp/repro/
cd /tmp/repro/analysis/figures && python3 make_visuals_probe.py && python3 make_visuals_tsne.py
```

If a generator needs `results/`, a snapshot step is missing upstream. Three such gaps were
found and closed this way: `generation/` (the behavioural half of the argument),
`taxonomy/between_class_matrix.csv` (the clustermaps), and `probe/confusion_errors.csv` (the
nitrogen→oxygen counts, which had been read from an uncommitted tree).

`_common.load()` is what enforces it — it resolves paths relative to `analysis/` only, and
raises naming the `analyze_*.py` that should have written a missing file.

## Generators

Each writes into `analysis/visuals/<topic>/` and takes `--out-dir` to write elsewhere.

| Generator | Output | Reads |
|---|---|---|
| `make_visuals.py` | `clustermap/<model>/`: `heat_L<layer>.png` (25), `heatrow.png` (5) | `taxonomy/between_class_matrix.csv` |
| `make_visuals_tsne.py` | `tsne/tsne_depth_{coarse,fine,template}.png`, plus per-model `fine_L<layer>.png` | `tsne/tsne_coords.csv`, `probe/layer_curves.csv` |
| `make_visuals_probe.py` | `probe/probe_depth.png` | `probe/layer_curves.csv`, `probe/surface_baseline.csv` |
| `make_visuals_ambiguity.py` | `ambiguity/mass_split_L<layer>.png` | `ambiguity/ambiguity_family_probability.csv` |
| `make_visuals_prompt_structure.py` | `prompt_structure/label_agreement_depth.png` | `prompt_structure/label_agreement_by_layer.csv` |
| `make_visuals_generation_composition.py` | `generation/group_response_composition.png` | `generation/group_response_composition.csv` |
| `make_visuals_retrieval.py` | `retrieval/hit1.png`, `hit2.png`, `hit3.png` | `geometry/retrieval_by_axis.csv` |
| `make_visuals_retrieval_rank.py` | `retrieval/meanrank.png` | `geometry/retrieval_by_axis.csv` |
| `make_visuals_retrieval_dist.py` | `retrieval/rankdist.png` | `geometry/retrieval_rank_hist.csv` |
| `make_visuals_retrieval_violin.py` | `retrieval/rankviolin.png` | `geometry/retrieval_rank_hist.csv` |
| `make_visuals_retrieval_mixed.py` | `retrieval/rankmixed.png` | `geometry/retrieval_rank_hist.csv` |
| `make_visuals_retrieval_3d.py` | `retrieval/analogy3d_*.png` + a `.txt` readout each (6 pairs) | `analogy/analogy_quiver3d.csv` |

Two generators split their output per checkpoint — `make_visuals.py` and
`make_visuals_tsne.py` — and both name the subdirectory with the model's own identifier
(`meta-llama-Llama-3.1-8B`), matching the `results/` and `activations/` trees and the
`analogy3d_*` filenames. One model is one string everywhere a path carries it, so a figure
traces back to the run that produced it without a lookup table. The lower-case slugs
(`base`, `chem-faithful`, …) appear only in console output and figure legends, never in a
path.

The four `retrieval_*` rank variants draw the same measurement four ways — dots, violins, a
mix, and a reduction to mean and median. `rankdist.png` is the one the
`visuals/retrieval/README.md` argues from; the others exist because a rank distribution over
19 candidates is easy to misread from summary statistics alone.

`_common.py` holds what they share: the model order, the figure layer, and `load`/`num`.

## Model order and identity

`BEHAVIOURAL` in `_common.py` fixes the order in every figure: base, then the Tier-1
chemistry pair in **training** order (Chem-R-Faithful is GRPO-trained *from* Chem-R-8B),
then the off-base chemistry model, then the reasoning distill. The list is **append-only** —
every generator iterates it, so appending adds panels after the existing ones while
inserting would reshuffle every figure.

ChemDFM sits on Llama-3 rather than 3.1. A ChemDFM-vs-base gap therefore confounds chemistry
training with the 3 → 3.1 difference and is never a controlled contrast; Chem-R-8B is what
carries that. Figures mark it accordingly.

## Colour

`make_visuals_probe.py` defines the five-model palette (`COLOR5`, `DASH5`, `MARKER5`,
`DISPLAY5`) that the other generators import from it, rather than each keeping its own copy.
It is Okabe-Ito-derived and was chosen by running a colour-vision validator, not by eye: the
seaborn default it replaced put red beside green at deuteranopic ΔE 7.3, inside the band that
is only legal with secondary encoding. Every series additionally carries its own marker and
dash pattern, so identity survives greyscale printing and does not rest on hue alone.

If you change a colour, re-validate rather than eyeballing it.

## Output is not committed

`analysis/visuals/**/*.png` is gitignored. The PNGs are rebuilt from the committed CSVs by
the commands above, so tracking them would add a binary blob on every regeneration for no
information the repository does not already hold. What **is** tracked beside them is each
topic's `README.md` — the argument each figure supports — and the `analogy3d_*.txt` readouts,
which are numbers rather than renderings.
