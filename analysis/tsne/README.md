# t-SNE of functional-group activations

`compute_tsne_coords.py` embeds the last-token activations and releases the **coordinates**,
so `analysis/figures/make_visuals_tsne.py` can draw the figures without touching
activations. It is the only analysis here besides `prompt_structure/` that reads
`activations/` directly.

```bash
python analysis/tsne/compute_tsne_coords.py     # -> data/tsne_coords.csv, data/tsne_neighbourhood.csv
```

**23,000 rows**: 5 models × 5 layers (0, 8, 16, 24, 31) × 920 prompts (92 molecules × 10
templates). Row order is molecule-major, template-minor — the layout `generate_prompts()` in
`src/extract_activations_subset.py` writes and that `src/probe.py` already
relies on. PCA to 50 components, then t-SNE at perplexity 30, both at `random_state=42`.

## Read the neighbourhood table, not the picture

A t-SNE plot invites cluster-quality judgements it cannot support, so the honest quantity is
released alongside the coordinates. `data/tsne_neighbourhood.csv` asks, for each point, what
share of its **k = 10 nearest neighbours in the embedding** carries the same label — for
three competing labels at once:

| model | L0 tmpl / coarse / fine | L16 tmpl / coarse / fine | L31 tmpl / coarse / fine |
|---|---|---|---|
| base | 0.66 / 0.58 / 0.44 | **0.91** / 0.71 / 0.40 | 0.74 / **0.94** / 0.59 |
| chem | 0.65 / 0.56 / 0.42 | **0.88** / 0.69 / 0.42 | 0.46 / **0.93** / **0.83** |
| chem-r | 0.65 / 0.56 / 0.42 | **0.89** / 0.66 / 0.41 | 0.59 / **0.90** / 0.72 |
| chemdfm | 0.67 / 0.51 / 0.36 | **0.91** / 0.70 / 0.41 | 0.52 / **0.95** / 0.81 |
| reason | 0.66 / 0.60 / 0.46 | **0.93** / 0.72 / 0.42 | 0.79 / 0.88 / 0.58 |

`tmpl` = same prompt template (10 classes, chance 0.10), `coarse` = same heteroatom family
(5 classes: halide / hydrocarbon / nitrogen / oxygen / sulfur), `fine` = same functional
group (20 classes, chance 0.05).

**The middle layers encode the prompt, not the chemistry.** At L16 every model puts 0.88–0.93
of a point's neighbourhood on its own template — against 0.66–0.72 for the chemical family.
Whatever a middle-layer t-SNE plot looks like, the dominant axis of the geometry there is
which of the ten sentences was used.

**That inverts by the final layer.** Template agreement falls to 0.46–0.79 while coarse-family
agreement rises to 0.88–0.95. The representation reorganizes from prompt-shaped to
chemistry-shaped over the second half of the network — the same depth range where the probe
in `analysis/probe/` becomes accurate, and consistent with it.

**The models differ in how completely they finish that inversion.** `chem` ends at 0.46
template / 0.83 fine, `chemdfm` at 0.52 / 0.81 — both largely done. `reason` ends at 0.79
template / 0.58 fine: it still carries the prompt at the last layer and resolves the fine
group least well. `base` sits between (0.74 / 0.59) despite being the most accurate probe
target, which is worth noting — linear decodability and neighbourhood purity are not the same
property, and the probe can read a direction the k-NN geometry does not make dominant.

Chance for `fine` is 0.05, so even 0.58 is far above it. The comparison that matters is
between models and between depths, not against chance.

## What this does not measure

- **Silhouette scores are deliberately absent.** The upstream plotting script annotated every
  figure with a silhouette computed on the **2-D t-SNE coordinates** rather than on the
  activations, and in practice it did not rank plots by how well they clustered — sometimes
  it ranked them backwards. A number that cannot be quoted does not belong on a figure, so it
  is neither computed nor stored here. The k-NN purity above replaces it, and is computed on
  the embedding with the labels stated.
- **t-SNE distances are not metric.** Between-cluster distances and cluster sizes in the
  rendered figures carry no interpretation. Only neighbourhood composition does, which is why
  that is the released number.
- **Five layers, not 32.** The embedding is expensive and the figures sample depth. The probe
  runs every layer for a fraction of the cost; use `analysis/probe/data/layer_curves.csv` for
  anything that needs depth resolution.
- **One prompt type.** These are the `functional_group` declarative prompts. The
  `entity_type` column exists because the script is unchanged and would emit more if more
  prompt types were extracted.

## Files

| file | contents |
|---|---|
| `data/tsne_coords.csv` | model × layer × 920 prompts: `x`, `y`, plus `molecule`, `functional_group`, `coarse_family`, `carbon_count`, `template_index` |
| `data/tsne_neighbourhood.csv` | model × layer × k: k-NN label agreement for template, coarse family and fine group |
| `compute_tsne_coords.py` | the embedding; reads `activations/`, writes both CSVs |

The figures are drawn by `analysis/figures/make_visuals_tsne.py` into
`analysis/visuals/tsne/` — three overview PNGs (`tsne_depth_{coarse,fine,template}.png`) plus
a per-model layer series. See `analysis/visuals/tsne/README.md` for what each shows.
