# Analogy retrieval and the 3D projection

Scope is the **declarative `functional_group` prompt only**; `snapshot_quiver3d.py` takes
`--entity-type` if others are extracted.

This directory holds one producer, `snapshot_quiver3d.py`, which writes
`data/analogy_quiver3d.csv` — the 3D coordinates behind
`visuals/retrieval/analogy3d_*.png`. The retrieval *numbers* live in `../geometry/`; §1 and
§2 below document the test those numbers come from, because it is
`src/analogy_retrieval.py`'s method rather than `analyze_geometry.py`'s.

> **`analogy_quiver3d.csv` is base-only**, which is all the 3D figure draws. Per-model
> retrieval across all five checkpoints is in `../geometry/data/retrieval_by_axis.csv` and
> `retrieval_rank_hist.csv`.

> **The activation space is strongly anisotropic** — raw pairwise cosine runs 0.78–1.00 — so
> any cosine quoted here without mean-centring is largely reporting that. The centred numbers
> from `results/anisotropy_diagnostic/` are the real ones, and §3 is why this projection
> centres too.

> **The second-order analogy rests on four hand-chosen quadruples**, and they disagree in
> sign, so their mean is not a summary of anything. §1 is scored per quadruple for that
> reason.

An earlier revision of this directory also carried `analyze_analogy.py` and three CSVs
(`layer_trend`, `analogy_quadruples`, `within_by_group`) scoring the analogies by **offset
cosine** rather than by retrieval. That analysis fed no figure and was dropped; its prose,
written against an 8B/70B pair whose data does not ship here, went with it.

## 1. The retrieval test

The dropped cosine analysis scored each quadruple by the **cosine between two offset
vectors**. That is not what the word2vec analogy result measures. That result is *retrieval* — `king − man + woman` lands
nearest to `queen` — and offset cosines in those spaces are well short of 1. Scoring by
cosine alone measures the strictly harder quantity, and it changes the verdict on half the
quadruples.

`src/analogy_retrieval.py` runs the retrieval version over the diff vectors
`src/analogy_carbon_matched.py` saved to
`results/analogy/*/data/diff_vectors_layer_{L}.npz`, so it needs no rerun
and no GPU. For
each quadruple it tests all four corner predictions (`b1 = b2 + (a1 − a2)`, and the three
rotations) at each chain length: 4 quadruples × 4 corners × 4 chain lengths = 64 trials per
layer, against 19 candidate groups.

**Retrieval sees 5 analogies where the offset cosine sees 7.** Retrieval is blind to how four
groups are paired — every corner target is two of them minus the third, and the exclusion
set is always the other three, so re-pairing `{w,x,y,z}` poses the same four questions with
the sources relabelled. `cos(a1−a2, b1−b2)` *does* depend on the pairing, which is why
`ANALOGY_QUADRUPLES` carries both pairings of two group sets as notation controls, and why
`analogy_retrieval.py` collapses them (`distinct_by_group_set`). Counting both would double
those group sets' weight in every pooled figure.

`functional_group`, carbon-matched, 80 trials per layer, random-baseline hit@1 = 6.2%:

| depth | 0.00 | 0.25 | 0.51 | 0.76 | 1.00 |
|---|---|---|---|---|---|
| 8B hit@1 | 42.5% | 45.0% | 71.2% | **75.0%** | 73.8% |
| 70B hit@1 | 46.2% | 48.8% | **82.5%** | 81.2% | 81.2% |

> **Three of those 8B cells have been re-measured on the SMILES prompts** — the first
> numbers in this file to be replaced rather than merely annotated. Depth 0.00 goes
> 42.5% → **53.8%** (mean rank 3.79 → 2.41), depth 0.51 goes 71.2% → **68.8%**, and depth
> 1.00 goes 73.8% → **80.0%**. The shape is unchanged; what moves is the floor. Layer 0 is
> where notation lives, and a SMILES string spells the heteroatom out where a condensed
> formula hid it, so the shallow layers now solve analogies that character identity alone
> is enough to solve. Read the depth-0.00 column as an orthographic baseline, not as
> evidence that the untrained embedding knows chemistry. The other cells, and every 70B
> cell, are still old-prompt figures.

**The peak-then-fall shape reported for the four-quadruple set was the halogen quadruple.**
With the halogens gone, both models rise and then plateau across the whole second half
rather than falling back at the end. The earlier "falls back at
the final layer" reading was one notation-confounded quadruple collapsing and dragging the
pooled mean with it.

O↔S retrieves 16/16 from the midpoint on. **The sulfur ladder, which is negative by offset
cosine, retrieves at rank 1–2 on every trial at 8B layer 31** — the clearest case of cosine
and retrieval disagreeing.

Two caveats that have to travel with these numbers:

- **Excluding the source terms is doing real work.** Standard word2vec practice excludes the
  three input terms from the candidate pool, and that convention is also why those results
  flatter themselves. Here the unexcluded nearest neighbour is simply the source group in
  **29–58%** of trials — the offset is real, but small next to the distance between groups.
  `rank_incl` and `degenerate_top1_rate` are emitted in every output for that reason.
- **The halogen failure is asymmetric by corner.** The two corners predicting `alkyl bromide`
  sit at rank 11–17 while predicting chloride and iodide is rank 1 at every chain length —
  which is what the SMILES-vs-condensed-formula confound in the dataset predicts (the
  halides were mixed notation when these activations were extracted), and is invisible in
  the cosine view.

```bash
python src/analogy_retrieval.py                    # functional_group
python src/analogy_retrieval.py --entity-type pka  # any other prompt
python src/analogy_retrieval.py --no-plots         # CSVs only
```

Outputs land under `results/`, in the same shape every other experiment here uses —
`results/analogy_retrieval/{model}/{entity_type}/`, figures at the
leaf and CSVs under `data/`. Model and entity are in the path, so they are not repeated in
the filenames, and each CSV holds exactly one model.

| file in `data/` | contents |
|---|---|
| `retrieval_trials.csv` | one row per trial: ranks with and without the source terms excluded, hit@1/@2/@3, cosine to the target, the unexcluded top-1 |
| `retrieval_layer_trend.csv` | per layer × mode: hit@1/@2/@3, mean and median rank, MRR, degenerate rate, and the matching `random_*` baselines |
| `retrieval_by_quadruple.csv` | the same aggregates split by quadruple |
| `retrieval_by_group.csv` | the same aggregates split by the functional group being predicted, with `n_trials` — groups are **unevenly sampled**, since one that answers two different corners gets twice the trials |

Seven figures per model:

| figure | shows |
|---|---|
| `retrieval_accuracy_pooled.png` | hit@1/@2/@3 pooled over the five distinct quadruples, on one axis so the three can be compared |
| `retrieval_accuracy_hit{1,2,3}_by_quadruple.png` | one file per k: four quadruple lines against that k's own baseline |
| `retrieval_rank_by_quadruple.png` | mean rank per quadruple. **No pooled mean line** — the four quadruples span six rank positions at some depths, so their average describes none of them |
| `retrieval_rank_by_group.png` | mean rank per functional group, plus the overall mean |
| `retrieval_rank_heatmap.png` | rank of the correct group for every corner prediction × chain length × layer |

**"Random baseline", not "chance".** Ordering the candidates at random puts the answer in
the top *k* with probability `k/n`, so each hit@k has its own baseline — 6.1% / 12.1% /
18.2%, and 8.75 for mean rank. They are not interchangeable: reading hit@3 against the
hit@1 line would overstate it threefold. The baselines are averaged over the *actual*
candidate pools rather than assumed constant, because pool size is 16 or 17 depending on
whether the quadruple reuses a group.

A quadruple keeps one colour across all four by-quadruple figures, so they can be read
side by side.

Every series in these plots is a solid line distinguished by colour and marker. Dash
patterns are deliberately not used to encode identity — at these font sizes `--` and `-.`
are indistinguishable from `-` in a legend swatch, so a dash-encoded series ends up
mislabelled in practice even when the code is right. The self-checks in §2 have no opinion
on this — it is a figure property, not a data one — so it is asserted separately by
building each figure and comparing every legend handle's linestyle, marker and colour
against the line it labels, with solid the only non-baseline style permitted.

## 2. Checking it

`analogy_retrieval.py` used to exit 0 whenever it did not crash, and both bugs found while
writing it were silent-wrong-answer bugs: a heatmap row key that collided for the two
quadruples reusing a group (14 rows drawn for 16 trials), and an exclusion set that deleted
the answer for a quadruple. Neither showed up in the printed table.

It now **self-checks before it writes**, so a run that fails leaves no plausible-looking but
wrong results tree behind — there is no separate verifier to remember to run:

```bash
python src/analogy_retrieval.py                    # checks, then writes
python src/analogy_retrieval.py --skip-checks      # escape hatch, not a normal flag
```

It asserts trial counts; that no two quadruples share a group set (which is what the first
seven-quadruple run needed and did not have); that no quadruple was silently skipped; that every trial is
uniquely addressable by (quadruple, source, predicted, chain length); that candidate-pool
sizes match the exclusion rule (`n_groups − 2` for the quadruple built from three distinct
groups, `n_groups − 3` otherwise); that ranks and hit flags agree row by row; that the
per-group rows partition the trials; that the random baselines equal `k/n` over the actual
pools; and that every reported aggregate matches the trials it summarizes.

That last one recomputes the aggregates with `recompute()` rather than `summarize()`. The
near-duplication is deliberate — folding them together would make the check vacuous — but
it is worth being clear about what it buys: it catches the wrong rows reaching `summarize`,
rows mutated between summarizing and writing, and a typo in one copy. It does not catch a
concept that is wrong in both.

It also holds a fixed regression on 8B/`functional_group` — hit@1 34/57/59 of 80 at layers
0/16/31, mean rank 3.79/2.17/1.35, and O↔S at 16/16. Those track the contents of
`results/` **and** `ANALOGY_QUADRUPLES`, so after a legitimate re-extraction or a change
to the quadruple set the constants at the top of the file are what should change.

## 3. The 3D projection snapshot

`snapshot_quiver3d.py` writes `data/analogy_quiver3d.csv`, the coordinates
`analysis/figures/make_visuals_retrieval_3d.py` draws under `visuals/retrieval/` as one PNG
plus one `.txt` per selected cell, stems `analogy3d_{model}_L{layer}_{quadruple}_C{N}`. The
numbers live in the `.txt`; the image carries only what identifies it. It reads two results
trees — `analogy/` for the diff vectors and
`analogy_retrieval/` for the scored trials — plus the dataset.

```bash
python analysis/analogy/snapshot_quiver3d.py \
    [--layer 31] [--mode carbon_matched|lumped] [--top-n N] [--all-cells]
```

### Every point is a compound

The default pool is **carbon-matched**: one chain length at a time, so `ether` at C5 is methyl
butyl ether rather than an average over C3–C6, and the `molecules` column names it. Seven of
the ten groups these quadruples use are a single compound at a fixed chain length. **Alcohol,
thiol and amine are not** — each carries an `n-`/`sec-` isomer pair at every chain length in
`functional_group_dataset.csv`, so those three vectors are a 2-molecule mean. `n_molecules`
records which, and any figure showing one says so on its face. `--mode lumped` restores the
chain-length average, in which no point is a molecule at all.

### Which cells it picks

`rank_cells` orders the layer's trials per **(quadruple, chain length)** cell on three
criteria, in order: every scoreable corner at hit@1; then *no* corner flagged `degenerate`;
then highest mean cosine as a tiebreak. Six of the twenty cells pass at base/L31:

| quadruple | C | hit@1 | degen | mean cos | recon | drawn |
|---|---|---|---|---|---|---|
| thioether : thiol :: ether : alcohol | 5 | 4/4 | 0 | 0.836 | 0.47 | yes |
| thioether : thiol :: ether : alcohol | 6 | 4/4 | 0 | 0.805 | 0.42 | yes |
| ester : carboxylic acid :: ketone : aldehyde | 4 | 4/4 | 0 | 0.796 | 0.64 | yes |
| imine : aldehyde :: amine : alcohol | 4 | 4/4 | 0 | 0.744 | 0.53 | yes |
| imine : aldehyde :: amine : alcohol | 5 | 4/4 | 0 | 0.739 | 0.56 | yes |
| imine : aldehyde :: amine : alcohol | 6 | 4/4 | 0 | 0.715 | 0.55 | yes |
| thioether : thiol :: ether : alcohol | 4, 3 | 4/4 | 1 | 0.811, 0.792 | — | no |
| ester : carboxylic acid :: ketone : aldehyde | 3, 6, 5 | 4/4 | 1, 1, 2 | 0.734, 0.733, 0.702 | — | no |
| amide : carboxylic acid :: amine : alcohol | 4 | 4/4 | 2 | 0.742 | — | no |
| amide : carboxylic acid :: amine : alcohol | 5, 6, 3 | 3/4 | 0, 1, 2 | 0.740, 0.679, 0.698 | — | no |
| imine : aldehyde :: amine : alcohol | 3 | 2/4 | 0 | 0.696 | — | no |
| sulfoxide : thioether :: sulfone : sulfoxide | 3 | 2/4 | 3 | 0.731 | — | no |
| sulfoxide : thioether :: sulfone : sulfoxide | 5, 4, 6 | 1/4 | 3 | 0.739, 0.739, 0.726 | — | no |

The second criterion is the one that matters. A cell can be 4/4 on hit@1 and still be mostly
degenerate — the constructed point's unexcluded nearest neighbour is the source group itself,
so the offset never left the neighbourhood it started in and the exclusion did the work.
Ranking on hit@1 alone would put six such cells on the figure.

Keying per **cell** rather than per quadruple matters too: `ester` is 0 degenerate at C4 and 2
at C5, and averaging those together is exactly what the lumped pool used to do. Under lumped,
that quadruple had 2 degenerate corners and the rule dropped it. At C4 it is clean and it is
drawn — with its panel marked `DELIBERATE NEGATIVE CONTROL`
(`analogy_carbon_matched.py:306-311`): formally matched legs (H → CH3) that
are chemically mismatched, and the one that should fail if leg symmetry is what makes an
analogy work. It does not fail. That is a result, not a reason to hide it, but it must not
read as a plain success. `NEGATIVE_CONTROLS` is named in the script and asserted against
`ANALOGY_QUADRUPLES`, so editing the quadruple list breaks loudly rather than silently
dropping the annotation.

### The projection is mean-centered, and the cosine tables are not

`analogy_carbon_matched.fit_uncentered_3d_basis` deliberately does not
center: the origin there reads as "no change from the alkane baseline", a real zero. This
snapshot deliberately does, and the two should not be reconciled — they answer different
questions.

An uncentered basis is dominated by the anisotropic mean direction (PC-1 coordinates run −32
to −9 against ±16 on the other two), so every group's arrow points the same way and the
parallelogram the analogy *is* collapses. Offsets are translation-invariant, so centering
costs the parallelogram nothing; what it gives up is the origin's reading, which this figure
does not use. Measured over the three quadruples that retrieve perfectly at base/L31:

| basis | variance kept | 3D rank of the true answer |
|---|---|---|
| uncentered | 0.73 | 1, **3**, 1 |
| **centered (this file)** | 0.57–0.61 | **1, 1, 1** |
| quadruple-local (span of both offsets) | 0.17–0.27 | 1, **3**, 1 |

The uncentered basis does not merely blur the imine/amine quadruple — it inverts it, putting
`alkyl fluoride` nearest the constructed point and dropping the true `amine` to rank 3 in a
case that is rank 1 in full 4096-D. This is the same position §1 takes about raw cosine: an
uncentered number here is largely quoting the anisotropy.

### Columns

One row per plotted point. Cell-level fields repeat on every row of that cell, as in
`layer_trend.csv`.

| column | |
|---|---|
| `cell_rank` | 1-based position in the ranking above |
| `carbon_count` | the chain length this cell is drawn at; empty under `--mode lumped` |
| `role` | `a1`/`a2`/`b1`/`b2`, `predicted` for the constructed point (`group` empty), `other` for the rest of the pool |
| `group`, `molecules`, `n_molecules` | the functional group, the compound(s) behind it, and how many |
| `is_negative_control` | 1 for the ester quadruple |
| `svd1..3` | the centered PCA coordinates, the only thing the figure positions anything by |
| `explained_var` | variance the three components hold, over all 19 groups at that chain length |
| `recon_frac` | how much of the constructed point those three dimensions hold — 0.42 to 0.64 across the six. Under half of some panels is on the page |
| `rank_excl`, `cos_to_target` | **copied** from `retrieval_trials.csv` for the `b1 = b2 + (a1 − a2)` corner, never recomputed, so a panel cannot disagree with `f4_retrieval` about an outcome |
| `rank3d_excl` | that rank recomputed from the three plotted coordinates, by Euclidean distance — what a reader's eye does with a scatter. When it disagrees with `rank_excl` the projection is misleading, and the script warns and the panel says so on its face |
| `dist3d_true`, `dist3d_rival`, `rival_group` | the same comparison as a margin, so the figure can show that the answer *won* rather than only that the residual was short |
| `n_corners`, `n_hit1`, `n_degenerate`, `mean_cos` | the selection criteria, carried so the ranking can be audited from the CSV |

### Limits

Base model and one layer at a time. `--model`, `--layer`, `--mode`, `--top-n` and
`--all-cells` widen it; nothing downstream assumes a particular number of cells, since the
figure writes one file per cell.
