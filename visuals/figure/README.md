# The manuscript figures

Eight scripts that build the paper's figures — four main, four supplementary — as **PDFs**,
because their destination is a LaTeX manuscript whose figures are vector.

They are **deliberately outside the 70-PNG release set**. `scripts/run_local.sh --stages
figures` globs `visuals/make_visuals*.py` and never descends here, so nothing in the normal
pipeline builds them; `tests/test_figures.py` asserts exactly 70 PNGs and would fail if they
were folded in. `tests/test_structure.py` carries the other half of that arrangement: `figure/`
is on a code-directory allow-list rather than being a figure topic, and a second test requires
it to actually hold tracked `.py` so the exclusion cannot quietly become a hole.

Everything else about them matches the release generators. Each reads only the committed
snapshots under `analysis/data/`, never `results/`, so all eight build from a clean checkout
with no GPU, no activations and no pipeline run. Each takes `--out-dir` and writes beside
itself by default.

## Building them

There is no driver script. Run them directly:

```bash
for f in visuals/figure/fig*.py; do python "$f"; done   # 8 PDFs into visuals/figure/
```

The PDFs are gitignored, like the PNGs and for the same reason — they are rebuilt from the
committed CSVs, so tracking them would add a binary blob on every regeneration.

## The figures

| Script | Output | What it shows | Reads |
|---|---|---|---|
| `fig2_clustermaps.py` | `f2_clustermaps_l31.pdf` | Five checkpoints' between-class clustermaps at layer 31, side by side | `analysis/data/between_class_matrix.csv` |
| `fig3_retrieval.py` | `f3_retrieval.pdf` | (a) retrieval rank distributions on three axes, five checkpoints; (b, c) two analogy cells in the first three PCs | `analysis/data/retrieval_by_axis.csv`, `analysis/data/retrieval_rank_hist.csv`, `analysis/data/analogy_quiver3d.csv` |
| `fig4_depth_structure.py` | `f4_depth_structure.pdf` | (a) probe accuracy against depth, with the base model's clustermap at layers 8 and 24 as insets on leaders; (b) t-SNE at those same two depths, each coloured by template and by group | `analysis/data/layer_curves.csv`, `analysis/data/surface_baseline.csv`, `analysis/data/between_class_matrix.csv`, `analysis/data/tsne_coords.csv` |
| `fig5_crossover.py` | `f5_crossover.pdf` | kNN label agreement against depth for two label sets; template organisation gives way to chemistry around layer 19. Each panel carries its own random-neighbour floor | `analysis/data/label_agreement_by_layer.csv` |
| `figs1_composition.py` | `fs1_composition.pdf` | **S1** — what each checkpoint emits per functional group: five rows, 20 groups, stacked response composition, with teacher-forced recall as a black dash | `analysis/data/group_response_composition.csv` |
| `figs2_clustermap_grid.py` | `fs2_clustermap_grid.pdf` | **S2** — the clustermaps at every checkpoint and every depth, 5 × 5 | `analysis/data/between_class_matrix.csv` |
| `figs3_quiver_grid.py` | `fs3_quiver_grid.pdf` | **S3** — Figure 3b's three analogy cells drawn for all five checkpoints, 5 × 3 | `analysis/data/analogy_quiver3d_all.csv` |
| `figs4_ambiguity_mass.py` | `fs4_ambiguity_mass.pdf` | **S4** — where the probe's probability mass goes on the four convention-labelled groups, against two oxygen-free control blocks | `analysis/data/ambiguity_family_probability.csv` |

`analogy_quiver3d_all.csv` is the one snapshot that exists only for this directory. The
released 3D figures draw one PNG per cell in the single-model `analogy_quiver3d.csv`, so
extending that file in place would have grown the release set from 70 to 94; the
five-checkpoint data went to a companion snapshot instead and the released set is untouched.

## They compose the release generators, they do not reimplement them

Every script here supplies a layout and borrows the drawing. That is what keeps a manuscript
panel from disagreeing with the released PNG of the same measurement — there is one
implementation, called twice.

| Script | Borrows |
|---|---|
| `fig2_clustermaps.py` | `leaf_order`, `read_matrices` from `visuals/make_visuals.py`; `DISPLAY5` from `visuals/make_visuals_probe.py` |
| `fig3_retrieval.py` | `visuals/make_visuals_retrieval_3d.py`'s `panel`, plus `make_visuals_retrieval_dist` and `make_visuals_retrieval_violin` |
| `fig4_depth_structure.py` | `make_visuals_probe`, `make_visuals_tsne`, `make_visuals.read_matrices` |
| `fig5_crossover.py` | `visuals/make_visuals_prompt_structure.py` |
| `figs1_composition.py` | `composition_panel` from `visuals/make_visuals_generation_composition.py` |
| `figs2_clustermap_grid.py` | `leaf_order`, `read_matrices` from `visuals/make_visuals.py` |
| `figs3_quiver_grid.py` | `visuals/make_visuals_retrieval_3d.py` |
| `figs4_ambiguity_mass.py` | `visuals/make_visuals_ambiguity.py` |

`fig2_clustermaps.py` is the base of the set: `style()` and `clustermap_panel()` live there and
the other seven import them, so one change to the type or the colour map moves every figure.

The borrowed modules' type sizes are **module globals read at call time**, which is what makes
composing possible at all. A manuscript panel is roughly a 0.5× downscale of the standalone
figure, so the scripts rebind those globals in their own process before drawing. Nothing in
`visuals/make_visuals*.py` is edited, and the released PNGs stay byte-identical across every
change made here.

## The one cross-figure invariant

Figures 2, 4 (the insets) and S2 all draw the same between-class matrix, and all three use
**one leaf order — the base model's dendrogram at layer 31**. A given cell is therefore the
same group pair in every panel of every one of those figures, and a difference between two
panels is a difference in the numbers rather than a re-clustering.

Ordering each panel by its own dendrogram would be wrong here rather than merely different.
Clustering always returns some order, so layer 8 would be drawn with blocks it does not have.

The insets and the S2 grid carry no group labels: at those panel sizes the 19 names would land
on 3–4 pt of pitch. Figure 2 is the labelled reference for all of them.
