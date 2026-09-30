# Between-class similarity clustermaps — which taxonomy the geometry follows

```bash
python analysis/analyze_taxonomy.py      # -> between_class_matrix.csv
python visuals/make_visuals.py           # all 30 files
```

Thirty files, the largest topic here, written by
[`visuals/make_visuals.py`](../make_visuals.py) from
`analysis/data/between_class_matrix.csv` and never from `results/` — the rule every
generator in that directory follows. The matrix is snapshotted by
[`analysis/analyze_taxonomy.py`](../../analysis/analyze_taxonomy.py), and the full argument
with its confidence intervals and controls is
[`analysis/README.md` § Which taxonomy the geometry follows](../../analysis/README.md#which-taxonomy-the-geometry-follows).
This file covers what the pictures show.

```
clustermap/<model>/heat_L<layer>.png     25 files: 5 models x layers 0/8/16/24/31
clustermap/<model>/heatrow.png            5 files: all five depths in one row
```

`<model>` is the checkpoint's own identifier with `/` replaced by `-`
(`meta-llama-Llama-3.1-8B`) — the same string `results/`, `activations/` and
`make_visuals_tsne.py` use, so a figure traces back to the run that produced it without a
lookup table. The model is therefore **not** repeated in the filename.

- **`heat_L<layer>.png`** — one model at one depth, standalone and untitled, so each can be
  placed individually in a paper. Rows and columns are the 19 non-alkane functional groups;
  the cell is the cosine between their diff-vectors, z-scored (below). The diagonal is blank
  rather than 1.0: a group's self-similarity is not a measurement and would otherwise own one
  end of the colour scale.
- **`heatrow.png`** — the same five depths for one model in a single row, in larger type for
  reading on its own. Layers are titled here, which the individual files do not need because
  the filename carries the depth.

## What the figure shows

`COARSE_MAP` sorts a group into a family by the heteroatom that **names** it. Reactivity class
is a different defensible partition, and the two disagree in four places — `amide` (an acyl
derivative, so carbonyl by reactivity but nitrogen by name), `nitro` (nitrogen-named, no
carbonyl carbon), `sulfone`/`sulfoxide` (sulfur-named, but oxidised rather than divalent like
thiol/thioether), and the alcohol/ether *vs* thiol/thioether isostere pairs.

The block structure in these panels is which partition the geometry is following, and it
**moves with depth, differently per checkpoint**. Written as
`delta = contrast(reactivity) - contrast(naming)`, so positive means the geometry sorts by
reactivity and negative by nomenclature (`analysis/data/taxonomy_contrast.csv`, rows in
`_common.BEHAVIOURAL` order):

| delta | L0 | L8 | L16 | L24 | L31 |
|---|---|---|---|---|---|
| base | +0.534 | +0.506 | +0.260 | +0.003 | **−0.205** |
| chem-r | +0.256 | +0.388 | +0.375 | +0.289 | **+0.162** |
| chem-faithful | +0.245 | +0.400 | +0.423 | +0.367 | **+0.268** |
| chemdfm | +0.175 | +0.430 | +0.443 | +0.200 | **+0.055** |
| reason | +0.365 | +0.366 | +0.341 | +0.153 | **−0.002** |

**All five start reactivity-leaning; only the base model ends up on the other side.** Its
geometry migrates monotonically from reactivity toward nomenclature across the stack. All
three chemistry fine-tunes stay positive to layer 31, and the reasoning distill lands on zero
without crossing. Read along a `heatrow.png` and that is the visible story: the base model's
blocks reorganise between layers 16 and 31, while Chem-R's soften without changing side
(+0.375 to +0.162).

That is the mechanism the clustermaps exist to show — **Llama ends organised by the naming
convention its probe labels encode; the chemistry models end organised by chemistry** — and
it explains without a second hypothesis why Llama scores *higher* on the probe than the
chemistry model it is paired against below (`coarse_group` best balanced accuracy 1.000
against Chem-R-Faithful's 0.983): the probe's labels are the naming partition, so a model
organised that way is being scored on its own axis.

## Two decisions that make 30 files comparable

Thirty separate PNGs are only worth more than one composite if they can be read against each
other, which takes two choices made once in a first pass over all of them:

- **One leaf order for all 25 panels**, from the base model's final-layer average-linkage
  dendrogram. Every model and depth is drawn in that same layout, so a difference between two
  files is a difference in the numbers and never a re-clustering. This is why the figures are
  heatmaps in a fixed order rather than per-panel clustermaps with their own dendrograms — a
  dendrogram per panel would make every panel incomparable to every other.
- **One symmetric colour limit**, the maximum absolute z over all five models and all five
  layers. Colourbars are therefore interchangeable between files, and the diverging map is
  centred on a meaningful zero.

## Three things the figure does not say

**The z-score is within one (model, layer), so no cross-model claim rests on cell colour.**
Off-diagonal cosines are standardised by their own mean and standard deviation inside a single
model at a single depth. Raw cosine *levels* differ between these checkpoints for anisotropy
reasons that carry no chemistry — base is the most anisotropic of the five — so a raw
comparison would mostly measure that. The colourbar is labelled just `z-score` because each
file shows one model at one depth, and the qualifier had no contrast to draw inside its own
frame. The consequence: two panels being uniformly redder or bluer is not a finding. Only the
*block structure* within a panel is.

**Only one model pair has a significance test.** The paired bootstrap
(`taxonomy_paired.csv`, resampling unit = group, n = 19) covers **Chem-R-Faithful against
Llama-3.1-8B only** — at layer 31, paired delta **+0.474**, 95% CI [+0.108, +0.798],
P(>0) = 0.992, with layer 0 reversed and significant the other way as the control that makes
the rest worth believing. The per-model deltas in the table above do **not** individually
clear zero at n = 19. The other three models have deltas but no paired interval, so treat
their ordering as descriptive.

**Five layers, not 32.** 0/8/16/24/31 is what the analogy tree holds, so nothing here locates
a crossover the way the 32-layer probe and prompt-structure curves can. And the reactivity
partition is hand-specified: defensible group by group, and it beats family-size-matched
random partitions from layer 16 on, but it is not the only such partition and no search over
partitions was run.

## Sources

Committed CSVs only, never `results/`:

- the matrix — `analysis/data/between_class_matrix.csv` (4275 rows = 5 models × 5 layers ×
  171 group pairs; `cosine_sim` plus the `same_naming` and `same_reactivity` flags the
  contrast is built from)
- the contrast and its tests — `analysis/data/taxonomy_contrast.csv`,
  `taxonomy_paired.csv`, `taxonomy_permutation.csv`
- generator — `visuals/make_visuals.py`. `--only <model>` renders a single checkpoint;
  `--out-dir` writes elsewhere. It refuses to start if any of the five is missing from the
  CSV at layer 31, naming the models and pointing at `analyze_taxonomy.py`, rather than
  quietly emitting a short sweep.

**All five checkpoints**, in `_common.BEHAVIOURAL` order — base, chem-r, chem-faithful,
chemdfm, reason — the order the probe and geometry tables use, so a reader moving between
them never re-sorts.

**ChemDFM sits on Llama-3, not 3.1.** A chemdfm–base gap confounds chemistry training with
the base-model change and is never a controlled contrast. Chem-R-8B carries that, and it moves
the same way (+0.162 at layer 31 against base's −0.205), so the result above does not rest on
ChemDFM.

A note on the unit, the same one the probe and t-SNE carry: a diff-vector is averaged over the
molecules carrying that group, and each molecule is measured through 10 near-duplicate
templates. n = 19 groups is the honest sample size for anything computed on this matrix, which
is why the paired test resamples groups and not cells.
