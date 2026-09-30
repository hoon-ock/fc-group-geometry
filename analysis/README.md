# The analyses

One script per analysis, flat in this directory, each reading `results/` (or the activations
directly) and writing its snapshot into `data/`. Those 29 CSVs are the reproducibility
boundary: past them nothing reads `results/`, and every figure in `visuals/` is drawn from
them alone.

```bash
bash scripts/run_local.sh --stages analysis     # all of them, in dependency order
```

| Analysis | Script | Section |
|---|---|---|
| probe | `analyze_sweep.py` | [The linear probe](#the-linear-probe) |
| taxonomy | `analyze_taxonomy.py` | [Which taxonomy the geometry follows](#which-taxonomy-the-geometry-follows) |
| geometry | `analyze_geometry.py` | [What fine-tuning did to the representation](#what-fine-tuning-did-to-the-representation) |
| analogy | `snapshot_quiver3d.py` | [Analogy retrieval and the 3D projection](#analogy-retrieval-and-the-3d-projection) |
| ambiguity | `analyze_ambiguity.py` | [The convention-labelled groups, read two ways](#the-convention-labelled-groups-read-two-ways) |
| tsne | `compute_tsne_coords.py` | [t-SNE of the activations](#t-sne-of-the-activations) |
| prompt_structure | `analyze_prompt_structure.py` | [Prompt wording or chemistry? A depth curve](#prompt-wording-or-chemistry-a-depth-curve) |
| generation | `analyze_generation.py`, `score_generations.py` | — (never had a write-up) |

Two of these read the activations rather than `results/` — `compute_tsne_coords.py` and
`analyze_prompt_structure.py` — so their figures need only the extraction stage. The figure
generators and their output live in `visuals/`.

---

## The linear probe

*Producer: `analysis/analyze_sweep.py`*


`analyze_sweep.py` reads `results/probe/` across all five models and writes
the tables below into `data/`. Nothing here loads a model or touches activations — it reads
the probe's own outputs and runs CPU-only in a few seconds.

The probe asks whether functional-group identity is *linearly decodable* from the last-token
residual stream, and at what depth decodability appears. The headline metric is balanced
accuracy under **leave-one-fine-group-out** (`--split group`, the default): withhold
`alkyl iodide` entirely and ask whether its molecules still land with the halides, having
seen only F/Cl/Br. That is the only split with dynamic range — `molecule` and `carbon` are
saturated by the molecule's own spelling, which the surface baseline below quantifies.

> **One prompt type.** The working tree this was released from swept 23 entity types — the
> same 92 molecules asked about pKa, TPSA, boiling point and so on — as an
> attribute-specificity matrix. Every released result comes from the `functional_group`
> declarative prompts alone, so `family` is `declarative` in every row and the
> cross-family comparisons that sweep supported are not reproducible here. The
> `family` and `entity_type` columns are retained because `analyze_sweep.py` is
> unchanged and would emit them again if more prompt types were extracted.

### The five models

`base` = `meta-llama/Llama-3.1-8B`, `chem` = `phenixace/Chem-R-Faithful`,
`chem-r` = `weidawang/Chem-R-8B`, `chemdfm` = `OpenDFM/ChemDFM-v1.5-8B`,
`reason` = `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`. All 8B, all 32 layers.

ChemDFM sits on Llama-3 rather than 3.1, so a ChemDFM-vs-base gap confounds chemistry
training with the 3 → 3.1 difference and is never a controlled contrast. Chem-R-8B is what
carries that. Chem-R-Faithful is GRPO-trained *from* Chem-R-8B, so those two are a
training-order pair, not independent samples.

### 1. Every model solves the task, and at the same depth

From `per_entity_summary.csv` (target `coarse`, split `group`, chance 0.20):

| model | best | best layer | final layer | max-jump layer | 90%-of-best layer |
|---|---|---|---|---|---|
| base | **1.000** | 29 | 0.998 | 16 | 18 |
| chemdfm | 0.993 | 21 | 0.955 | 16 | 19 |
| chem | 0.983 | 19 | 0.964 | 18 | 18 |
| chem-r | 0.980 | 31 | 0.980 | 16 | 18 |
| reason | 0.963 | 29 | 0.958 | 16 | 19 |

The interesting quantity is not the ceiling — all five clear 0.96 — but **where the climb
happens**. Every model's largest single-layer jump is at L16 or L18, and every model reaches
90% of its own best by L18–L19: `depth_by_family.csv` puts the median saturation depth at
d = 0.58–0.61 for all five. Functional-group identity becomes linearly available around
**half depth** regardless of fine-tuning, and nothing after that adds much.

Four of five models peak before the final layer and slide slightly into it. The slide is
small (0.002–0.038) but it is consistent in direction, and it is why `FIGURE_LAYER = 31` in
`visuals/_common.py` is chosen as the first layer where all models are jointly above
0.94 rather than as anyone's individual best.

### 2. The surface baseline is what makes §1 mean anything

`surface_baseline.csv`: **0.475 for all five models**, over the same 19 leave-one-group-out
folds.

That number is a character-n-gram classifier fit on the raw prompt strings — no model
involved. Every template embeds `{iupac_name} ({smiles})`, so "Propan-1-ol (CCCO)" hands a
probe the substrings `-ol` and `O` directly. It is identical across models because the
prompts are identical across models; the baseline is a property of the text, not the network.

So the quantity to read in §1 is **probe minus surface**: 0.49 to 0.53. Half of the coarse
decision is available from spelling alone, and the probe recovers essentially all of the
rest. On the `molecule` and `carbon` splits the surface baseline runs 0.98–1.00 — which is
exactly why those splits are not the default and their near-perfect probe scores say
nothing.

### 3. Where the errors go — the one place the models differ

Reduced from `confusion_errors.csv` at L31 (sum the off-diagonal cells per model, then
the `nitrogen`->`oxygen` cell against that total):

| model | total errors | nitrogen → oxygen | share |
|---|---|---|---|
| base | 2 | 0 | 0.00 |
| chem-r | 18 | 7 | 0.39 |
| chem | 35 | 20 | 0.57 |
| chemdfm | 44 | 31 | 0.70 |
| reason | 41 | 37 | **0.90** |

Accuracy in §1 spans 0.037 across these five models. Error *composition* spans 0.90. The
models that make more mistakes make them almost entirely on one boundary — and it is a
defensible boundary: `amide` and `nitro` carry oxygen but are labelled `nitrogen` by naming
convention, so nitrogen→oxygen is the chemically motivated error, not a random one.
`analysis/` is the analysis that scores *how much* mass lands on that rival rather
than just counting errors.

Read the share, not the count. Below the transition every model misclassifies heavily in
every direction, so a large raw count at low depth reflects a probe that does not work yet.
`n_total_errors` is carried in the file so the denominator is visible — base's falls to
single digits in the deep layers, which makes its share correspondingly noisy.

### 4. base vs chem, with an interval

`model_diff_bootstrap.csv` is a **paired** bootstrap on the contrast the study is built
around: **chem − base = −0.017, 95% CI [−0.032, −0.006]**, excluding zero.

The resampling unit is the **molecule** (88 of them — 92 minus alkane's 4, since alkane is
the only hydrocarbon and is skipped as a leave-one-group-out fold), not the row. Rows are
molecule-major / template-minor, so a molecule's 10 template rows are near-duplicates;
resampling rows would narrow the interval by treating them as independent evidence.

The sign is worth stating plainly: the chemistry-tuned model is *slightly worse* at this
task than the base model, and the interval excludes zero. Chemistry fine-tuning did not make
functional-group identity more linearly decodable. §3 is where the two actually differ.

### 5. Readings the data does not support

- **"Chemistry training improves the representation."** It does not, on this metric — §4 has
  the interval, and the sign is negative.
- **"base is the best model."** It has the cleanest error profile here, on one task, with a
  0.475 surface floor underneath it. That is not a general claim about the checkpoints.
- **"The N→O share measures chemical understanding."** It measures where errors fall once a
  model has essentially solved the coarse task. Below L19 — the depth at which every model
  is above 0.94 — the share is not interpretable, because a model that misclassifies in
  every direction will put some of that everywhere.
- **Anything about prompt families or attribute specificity.** One prompt type ships; see the
  note at the top.

### 6. Known gaps in the released outputs

**Reference nulls are absent.** The sweep runs with `--num-null-samples 0`, so `null_mean`
and `p_value_best_layer` are `null` in every `summary_*.json`. That is deliberate: the null
is ~91% of the job's runtime and does not depend on the activations — shuffled labels leave
no signal regardless of what `X` holds, so it is a property of the fold structure, class
distribution, classifier and metric, all identical across every task. Measured once it is
flat and sits at chance (0.184–0.199 against a 0.20 chance line). It still matters once, as
a **leak detector**: if the molecule-level shuffle ever broke, it would rise above chance.
Run it per model and quote it as a reference band:

```bash
python src/probe.py --model-name meta-llama/Llama-3.1-8B \
  --entity-type functional_group --num-null-samples 50 \
  --output-dir results/null_ref_base
```

### 7. Regenerating

```bash
python analysis/analyze_sweep.py                        # -> data/*.csv
```

`--tag` selects which probe target to summarise and defaults to `coarse_group`, which is
what `scripts/run_local.sh` runs and what the committed `data/` carries. Another tag needs the
job re-run with `PROBE_TARGET` set, and would overwrite these same filenames — the
snapshot holds one target at a time.

Needs a populated `results/probe/`. CPU-only, no model loading, a few
seconds. `--n-boot` and `--seed` control the bootstrap; CI bounds move by < 0.002 between
seeds at the default 20,000 draws.

| file in `data/` | contents |
|---|---|
| `layer_curves.csv` | model × entity × layer × balanced_acc — the long-format source for everything else |
| `per_entity_summary.csv` | best/last, peak-to-last slide, final-layer step, whether the peak is final, jump layer, 90%-of-best layer, each with normalized depth |
| `depth_by_family.csv` | per model: modal jump layer, 90%-of-best range, median depth, floor-to-peak span |
| `model_diff_bootstrap.csv` | the paired chem-vs-base contrast: bAcc per model, difference, 95% CI, significance flag |
| `surface_baseline.csv` | the character-n-gram baseline, per model |
| `confusion_errors.csv` | the confusion matrix as countable rows at every layer, not just the best one — the source for §3 |

---

## Which taxonomy the geometry follows

*Producer: `analysis/analyze_taxonomy.py`*


The paper's aim-2 headline, and the one result that turns "Chem-R's clustermap looks
different" into a number with a confidence interval.

Source: `results/analogy/{model}/functional_group/data/
between_class_similarity_layer_*.csv` — the diff-vector cosine between every pair of the
19 non-alkane groups, which the analogy script already writes. That tree is **gitignored**,
so `data/` here is the durable record.

```bash
python analysis/analyze_taxonomy.py
```

### The question

`COARSE_MAP` sorts a group into a family by the heteroatom that **names** it. That is one
defensible partition of the 19 groups. Reactivity class is another, and it disagrees in
exactly four places:

| group | naming | reactivity | why |
|---|---|---|---|
| `amide` | nitrogen | carbonyl | an acyl derivative; reacts as one |
| `nitro` | nitrogen | C–N | no carbonyl carbon, so not an acyl derivative |
| `sulfone`, `sulfoxide` | sulfur | S=O | oxidised sulfur, not divalent like thiol/thioether |
| `alcohol`/`ether` vs `thiol`/`thioether` | oxygen vs sulfur | one family | the two isostere pairs |

For each model and layer, with cosines z-scored inside that model's own between-class
distribution so a globally flatter model is not penalised:

```
contrast(partition) = mean z-cosine(pairs inside a family) - mean z-cosine(pairs across)
delta               = contrast(reactivity) - contrast(naming)
```

`delta > 0` means the geometry sorts by reactivity; `delta < 0`, by nomenclature.

### What it found

`data/taxonomy_contrast.csv`. **All three models start reactivity-leaning and only the base
model crosses over**:

| delta | L0 | L8 | L16 | L24 | L31 |
|---|---|---|---|---|---|
| Llama-3.1-8B | +0.534 | +0.506 | +0.260 | +0.003 | **−0.205** |
| Chem-R-Faithful | +0.245 | +0.400 | +0.423 | +0.367 | **+0.268** |
| DeepSeek-R1-Distill | +0.365 | +0.366 | +0.341 | +0.153 | −0.002 |

The base model's geometry migrates, monotonically, from reactivity toward nomenclature
across the stack. Chem-R's does not. That is the mechanism: **Llama ends up organised by the
naming convention its probe labels encode; Chem-R ends up organised by chemistry.**

It also explains, without needing a second hypothesis, why Llama scores higher on the probe
(`coarse_group` 1.000 vs 0.983, molecule-level CI [−0.032, −0.006] — see `../probe/`): the
probe's labels *are* the naming partition, so a model organised that way is being scored on
its own axis.

### Why the paired bootstrap is the only test reported

`data/taxonomy_paired.csv`. Resampling unit is the **group**, n = 19.

| layer | Chem-R delta | Llama delta | paired | 95% CI | P(>0) |
|---|---|---|---|---|---|
| 0 | +0.245 | +0.534 | **−0.289** | [−0.535, +0.005] | 0.029 |
| 16 | +0.423 | +0.260 | +0.163 | [−0.138, +0.468] | 0.863 |
| 24 | +0.367 | +0.003 | +0.363 | [+0.004, +0.698] | 0.976 |
| 31 | +0.268 | −0.205 | **+0.474** | [+0.108, +0.798] | 0.992 |

Per-model deltas do **not** individually clear zero at n = 19 — quote the paired difference
only. Both models are evaluated on the same drawn groups, so their sampling noise is largely
shared and the paired interval is far tighter than either marginal.

Layer 0 is the control that makes the rest worth believing: at the input embedding the sign
is *reversed* and significant. The divergence is produced by the stack, not inherited from a
tokenizer the three models share.

### The permutation control

`data/taxonomy_permutation.csv`. A partition into more, smaller families scores a higher
within-minus-between contrast for purely combinatorial reasons, so the control shuffles group
labels while **matching the family-size profile** — reactivity's [4,5,4,4,2] against naming's
[4,5,6,4].

Both nulls centre on 0.00 ± 0.19, so the shape-matching worked: a 5-family partition gets no
free advantage over a 4-family one. Against those nulls:

| min z over the six (model × partition) cells | L0 | L8 | L16 | L24 | L31 |
|---|---|---|---|---|---|
| | 1.94 | 2.78 | **5.08** | **6.97** | **7.58** |

**From layer 16 on, both named partitions clear the matched null in every model** (z = 5.1–9.9,
p ≤ 0.001). At layers 0 and 8 they do not reliably: the *naming* partition is weakest of all
(z = 1.9–4.1, p up to 0.045), while reactivity already clears at L0 in every model (z = 3.9–4.5).

That is a limit and a result at once. It limits the L0 paired control in the previous section —
a difference between two weakly-structured contrasts — so read that row as "the sign is not yet
what it becomes", not as a significant reversal. And it is consistent with the trajectory: at
the input embedding there is barely any nomenclature structure to find, and what taxonomy is
present is already the reactivity one.

### Limits

- Five layers only (0/8/16/24/31) — that is what the analogy tree holds.
- n = 19 groups, 171 pairs. Fine for the paired test, not for per-model claims.
- The reactivity partition is hand-specified. It is defensible group-by-group (table above)
  and it beats shape-matched random partitions, but it is not the only such partition, and no
  search over partitions was run — that would need a correction this design does not have.

---

## What fine-tuning did to the representation

*Producer: `analysis/analyze_geometry.py`*


The aim-2 supporting table. Joins three result trees that each hold one piece of the same
story and are never otherwise read together.

```bash
python analysis/analyze_geometry.py
```

Sources (all **gitignored**, so `data/geometry_by_layer.csv` is the durable record):
`results/anisotropy_diagnostic/`, `results/analogy/`,
`results/analogy_retrieval/`.

### 1. The honest-signal check comes first

Raw between-class cosines run 0.40–0.45 at layer 31. They are not evidence of anything on
their own: the residual stream is anisotropic — mean pairwise cosine **0.944** in Llama at
layer 31 — so two unrelated vectors are similar by default.

After mean-centering, the between-class mean lands **exactly on the empirical null** at every
layer in every model (−0.044 vs −0.042 for Llama at L31; the pattern is identical for the
other two). Within-class does not: 0.754 / 0.616 / 0.736 against a null of ±0.17–0.25.

So the durable geometric claim is **within-class**: the functional-group diff vector is
consistent across chain lengths. Any claim resting on between-class similarity — including
every dendrogram in the paper — has to be framed as *relative* structure, with the centered
null stated. `../taxonomy/` is built that way on purpose: it z-scores within each model's own
between-class distribution and never quotes a level.

### 2. Fine-tuning de-anisotropises

`raw_pairwise_cosine`:

| layer | Llama | Chem-R | DeepSeek |
|---|---|---|---|
| 0 | 0.989 | 0.986 | 0.988 |
| 16 | 0.950 | 0.896 | 0.924 |
| 24 | 0.859 | **0.589** | 0.649 |
| 31 | **0.944** | **0.676** | 0.806 |

and `diffvec_svd_top1` at L31 — the share of diff-vector variance on a single direction —
0.469 / **0.353** / 0.421.

Both fine-tunes end far less anisotropic than the base, and Chem-R's diff vectors spread over
more directions. This is worth stating explicitly because it *partially produces* the visual
impression that Chem-R's groups are "more standalone" in a clustermap: its raw between-class
max is 0.742 against Llama's 0.956. The impression is real; the cause is not extra chemical
separation alone.

### 3. Linear compositionality degrades

`degenerate_rate` — the share of retrieval trials where `b2 + (a1 − a2)` is nearest to `b2`
itself once the exclusion is lifted, i.e. the offset never left the source group's own
neighbourhood. Scored on the same trials as hit@1, so a model can retrieve well and still
have a weak offset.

| L31 | Llama | Chem-R | DeepSeek |
|---|---|---|---|
| pooled hit@1 | 0.80 | 0.80 | 0.60 |
| hit@1 excluding sulfoxide | 0.889 | 0.889 | 0.667 |
| **degenerate rate** | **0.25** | **0.65** | **0.70** |

Read as a **cost**: the fine-tuned models sharpen categorical identity and lose the linear
offset structure that makes the analogy an analogy. Chem-R ties Llama on hit@1 while failing
the degeneracy check nearly three times as often.

`hit1_ex_sulfoxide` is emitted beside `hit1` rather than in place of it. Sulfoxide is
hit@1 = 0.00 in **every** model at **every** layer, so dropping it shifts all three equally —
it is a property of the group, not a limitation of any one model. Chem-R's defensible
retrieval claim is that it **peaks earlier and higher**: 1.000 ex-sulfoxide at L24 against
Llama's 0.833, matching its probe peak at L19 against Llama's L29.

### Limits

- Five layers (0/8/16/24/31) — the anisotropy and analogy trees sample only these.
- Pooled retrieval is **20 trials over 13 groups**. A 0.80-vs-0.775 gap is one trial. Never
  quote a between-model retrieval difference without a bootstrap; there is not enough
  resolution here for one.

---

## Analogy retrieval and the 3D projection

*Producer: `analysis/snapshot_quiver3d.py`*


Scope is the **declarative `functional_group` prompt only**; `snapshot_quiver3d.py` takes
`--entity-type` if others are extracted.

This directory holds one producer, `snapshot_quiver3d.py`, which writes
`data/analogy_quiver3d.csv` — the 3D coordinates behind
`visuals/retrieval/analogy3d_*.png`. The retrieval *numbers* live in `../geometry/`; §1 and
§2 below document the test those numbers come from, because it is
`src/analogy_retrieval.py`'s method rather than `analyze_geometry.py`'s.

> **`analogy_quiver3d.csv` is base-only**, which is all the 3D figure draws. Per-model
> retrieval across all five checkpoints is in `../data/retrieval_by_axis.csv` and
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

### 1. The retrieval test

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

### 2. Checking it

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

### 3. The 3D projection snapshot

`snapshot_quiver3d.py` writes `data/analogy_quiver3d.csv`, the coordinates
`visuals/make_visuals_retrieval_3d.py` draws under `visuals/retrieval/` as one PNG
plus one `.txt` per selected cell, stems `analogy3d_{model}_L{layer}_{quadruple}_C{N}`. The
numbers live in the `.txt`; the image carries only what identifies it. It reads two results
trees — `analogy/` for the diff vectors and
`analogy_retrieval/` for the scored trials — plus the dataset.

```bash
python analysis/snapshot_quiver3d.py \
    [--layer 31] [--mode carbon_matched|lumped] [--top-n N] [--all-cells]
```

#### Every point is a compound

The default pool is **carbon-matched**: one chain length at a time, so `ether` at C5 is methyl
butyl ether rather than an average over C3–C6, and the `molecules` column names it. Seven of
the ten groups these quadruples use are a single compound at a fixed chain length. **Alcohol,
thiol and amine are not** — each carries an `n-`/`sec-` isomer pair at every chain length in
`functional_group_dataset.csv`, so those three vectors are a 2-molecule mean. `n_molecules`
records which, and any figure showing one says so on its face. `--mode lumped` restores the
chain-length average, in which no point is a molecule at all.

#### Which cells it picks

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

#### The projection is mean-centered, and the cosine tables are not

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

#### Columns

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

#### Limits

Base model and one layer at a time. `--model`, `--layer`, `--mode`, `--top-n` and
`--all-cells` widen it; nothing downstream assumes a particular number of cells, since the
figure writes one file per cell.

---

## The convention-labelled groups, read two ways

*Producer: `analysis/analyze_ambiguity.py`*


```bash
python analysis/analyze_ambiguity.py
```

Four groups carry oxygen but are named for another heteroatom — `amide` and `nitro` are
labelled `nitrogen`, `sulfoxide` and `sulfone` `sulfur`. Under leave-one-group-out they are
genuinely underdetermined: nothing in the chemistry picks nitrogen over oxygen for an amide,
only IUPAC does. A model that puts real mass on oxygen there is wrong in the chemically right
direction, and accuracy cannot see it.

This script emits the same question answered from two independent places, both contrasted
against the group's own **oxygen-free** family controls so a merely-diffuse model scores zero:

- **posterior** — the probe's out-of-fold `p(family)`, as `r = p(O) / [p(O) + p(home)]`,
  from `results/ambiguity_metric/`.
- **geometric** — mean z-cosine from the group to the six oxygen-named groups, from
  `results/analogy/`. No probe, no softmax, no regularization constant.

### What holds: the within-model result, in all three models

Layer 31, `data/ambiguity_two_readings.csv`:

| group | post. Δ Llama | post. Δ Chem-R | post. Δ DSR1 | geom. Δ Llama | geom. Δ Chem-R | geom. Δ DSR1 |
|---|---|---|---|---|---|---|
| **amide** | +0.147 | +0.269 | +0.398 | +1.14 | +1.17 | +1.66 |
| **nitro** | +0.082 | +0.126 | +0.317 | +0.80 | −0.15 | +1.21 |
| sulfone | −0.003 | −0.095 | +0.033 | +0.69 | +0.08 | +0.73 |
| sulfoxide | −0.007 | +0.007 | −0.013 | +0.63 | +0.55 | +0.80 |

**Amide leans toward oxygen, well above its own nitrogen controls, on both readings in all
three models.** That is the durable claim, and it is an aim-1 claim: graded, chemically
correct ambiguity is present in the representation regardless of fine-tuning. It also shows
up with no metric at all — in Llama's layer-31 clustermap amide sits inside the nitrogen
block while carrying a visible warm band against ester/ketone/aldehyde/acid.

The sulfur half is the weak half by design, and comes out near zero or negative: `thiol`
(−SH) and `thioether` (−S−) are the direct analogues of `alcohol` (−OH) and `ether` (−O−),
so the sulfur *controls* sit close to the oxygen family themselves. That is a statement about
the controls. Quote amide and nitro; report the sulfur rows, do not lean on them.

### What does not hold: ranking models on this

Two independent reasons, both visible in the table.

**The two readings disagree.** The posterior puts Chem-R above Llama on both amide and nitro.
The geometry — which has no `C` in it — does not: amide is a tie (+1.17 vs +1.14) and nitro
is *reversed* (−0.15 vs +0.80). The two measure different objects (the probe reads whole
activations, the geometry reads chain-length-matched diff vectors), so this is not a
contradiction — but it does mean neither reading settles the ranking alone.

**`select_C` is not held fixed across models.** `best_C_mode` and `best_C_frac` are on every
row for this reason. At layer 31, the layer the probabilities are read at:

| | modal `best_C` | fraction of the 19 folds |
|---|---|---|
| Llama-3.1-8B | 1e-3 | 0.947 |
| Chem-R-Faithful | 1e-3 | 0.842 |
| DeepSeek-R1-Distill | **1e-4** | 1.000 |

Llama and Chem-R are **matched**, so that one posterior comparison is like-for-like. DeepSeek
sits two orders looser, emits flatter posteriors on everything, and tops the posterior
column — exactly the artifact §10 was written about.

> An earlier methods note tabulated `best_C` at each model's *own best layer* and reported
> Chem-R at 1e-4 ×15. That is a different quantity from the one above and, on the current
> sweep, a different value; `select_C` varies with depth (Chem-R: 1e-3 ×18 at L19, 1e-2 ×15 at L24,
> 1e-3 ×16 at L31). Take it at the layer the probabilities are read at, which is what this
> script does.

Note this cuts against the tempting story either way: DeepSeek has the *worst* generation
accuracy of the three (0.804 vs Chem-R's 0.978), so even taken at face value the ambiguity
ranking does not explain the behavioural ranking.

**What would settle it:** `ambiguity_metric.py --full --mode projection`, which reads the
same question off the probe's own geometry with no softmax, plus `--c-grid 1e-3` to pin the
temperature identically across models. Neither has been run —
`results/ambiguity_metric/*/functional_group/data/ambiguity_contrast.csv` does not exist for
any model. Until it does, report this section as within-model and say so.

---

## t-SNE of the activations

*Producer: `analysis/compute_tsne_coords.py`*


`compute_tsne_coords.py` embeds the last-token activations and releases the **coordinates**,
so `visuals/make_visuals_tsne.py` can draw the figures without touching
activations. It is the only analysis here besides `prompt_structure/` that reads
`activations/` directly.

```bash
python analysis/compute_tsne_coords.py     # -> data/tsne_coords.csv, data/tsne_neighbourhood.csv
```

**23,000 rows**: 5 models × 5 layers (0, 8, 16, 24, 31) × 920 prompts (92 molecules × 10
templates). Row order is molecule-major, template-minor — the layout `generate_prompts()` in
`src/extract_activations_subset.py` writes and that `src/probe.py` already
relies on. PCA to 50 components, then t-SNE at perplexity 30, both at `random_state=42`.

### Read the neighbourhood table, not the picture

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
in `analysis/` becomes accurate, and consistent with it.

**The models differ in how completely they finish that inversion.** `chem` ends at 0.46
template / 0.83 fine, `chemdfm` at 0.52 / 0.81 — both largely done. `reason` ends at 0.79
template / 0.58 fine: it still carries the prompt at the last layer and resolves the fine
group least well. `base` sits between (0.74 / 0.59) despite being the most accurate probe
target, which is worth noting — linear decodability and neighbourhood purity are not the same
property, and the probe can read a direction the k-NN geometry does not make dominant.

Chance for `fine` is 0.05, so even 0.58 is far above it. The comparison that matters is
between models and between depths, not against chance.

### What this does not measure

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
  runs every layer for a fraction of the cost; use `analysis/data/layer_curves.csv` for
  anything that needs depth resolution.
- **One prompt type.** These are the `functional_group` declarative prompts. The
  `entity_type` column exists because the script is unchanged and would emit more if more
  prompt types were extracted.

### Files

| file | contents |
|---|---|
| `data/tsne_coords.csv` | model × layer × 920 prompts: `x`, `y`, plus `molecule`, `functional_group`, `coarse_family`, `carbon_count`, `template_index` |
| `data/tsne_neighbourhood.csv` | model × layer × k: k-NN label agreement for template, coarse family and fine group |
| `compute_tsne_coords.py` | the embedding; reads `activations/`, writes both CSVs |

The figures are drawn by `visuals/make_visuals_tsne.py` into
`visuals/tsne/` — three overview PNGs (`tsne_depth_{coarse,fine,template}.png`) plus
a per-model layer series. See `visuals/tsne/README.md` for what each shows.

---

## Prompt wording or chemistry? A depth curve

*Producer: `analysis/analyze_prompt_structure.py`*


```bash
python analysis/analyze_prompt_structure.py
python visuals/make_visuals_prompt_structure.py
```

For each of the 920 prompts, the share of its 10 nearest neighbours **in activation space**
carrying the same label — the prompt template that produced it, its heteroatom family, or its
functional group. One row per (model, layer), all 32 layers, all five checkpoints.

This promotes a result that `visuals/tsne/README.md` found at five layers on 2-D t-SNE
coordinates. That measure describes a panel's layout; this one is computed on the activations,
so it is a claim about the model. The two are companions, not substitutes.

### What it shows

**The first half of every stack is about the prompt, not the molecule.** Template agreement
climbs to ~0.99 by layer 3 and holds a plateau through roughly layer 15, against a chance floor
of 0.099. Whatever those layers encode, the nearest-neighbour structure is dominated by
wording.

**Family agreement overtakes template agreement at layer 19** — and within one layer of
that in every checkpoint (base 19, chem-r 19, chem-faithful 19, chemdfm 19, reason 20). This is *later* than the probe's accuracy step — the largest
single-layer jump is 15→16 in four of the five checkpoints and 17→18 in chem-faithful, and the
probe first clears 0.90 balanced accuracy at layer 16–18 (`data/layer_curves.csv`). The
gap is the interesting part: a linear reader can recover the family two to three layers before
the representation's own neighbourhood structure reorganises around it. Decodability and
dominant structure are not the same event, and the probe sees the earlier one.

**The checkpoints separate only after that, and only on two of the three labels.** Layer 31:

| model | template | family | fine group |
|---|---|---|---|
| base | 0.759 | 0.934 | 0.593 |
| chem-r | 0.587 | 0.913 | 0.753 |
| chem-faithful | 0.489 | 0.942 | 0.843 |
| chemdfm | 0.568 | 0.934 | 0.789 |
| reason | 0.778 | 0.907 | 0.609 |

Family agreement is saturated for everyone (0.90–0.94) — chemistry training does not improve
the coarse partition, which base already has. The separation is entirely in *suppressing
wording* and *resolving within family*.

**The family column is released but not plotted.** `COARSE_MAP` files four groups by the atom
that *names* them rather than the one they contain — `amide` (−CONH₂) and `nitro` (−NO₂) under
nitrogen, `sulfoxide` and `sulfone` under sulfur. A point whose neighbours are chemically
sensible, an amide sitting among esters and ketones, therefore scores as *disagreement*, and
the label's ceiling is not 1.0 for reasons that have nothing to do with the model. That does
not bias the comparison across checkpoints — every model is scored against the same labels —
but it makes the panel easy to misread, so the figure draws only `template` and
`functional_group`, which carry no such convention. The columns stay in the CSV because the
crossover layer is measured against them and because a number worth computing is worth
releasing. The same caveat is the subject of `Analysis/ambiguity/`.

### What is robust, and what is not

**Robust.** At every k tested (5, 10, 20), all three chemistry fine-tunes sit **below** base on
template and **above** base on fine group. Three independent chemistry runs, one direction. The
crossover layer moves only from 18–19 at k=5 to 20–21 at k=20.

**Not robust: where `reason` sits relative to base.** Template agreement at layer 31:

| | k=5 | k=10 | k=20 |
|---|---|---|---|
| base | 0.824 | 0.759 | 0.709 |
| chem-r | 0.709 | 0.587 | 0.532 |
| chem-faithful | 0.641 | 0.489 | 0.404 |
| chemdfm | 0.715 | 0.568 | 0.481 |
| reason | 0.862 | 0.778 | 0.683 |

reason is above base at k=5 and k=10 and **below** it at k=20, and the gap is small at every k.
So the safe statement is that reason *tracks base closely* while the chemistry models pull away
from both — not that reason is above base. What the reason curve establishes is that the effect
is specific to **chemistry** training rather than to fine-tuning in general; its exact rank
against base is not evidence of anything.

The fine ordering *within* the three chemistry models is also not k-stable (chem-r and chemdfm
swap between k=5 and k=10). Quote the group, not the ranking.

### The centring choice, and why it turned out not to matter

Neighbours are ranked on **mean-centred** cosine. The residual stream is strongly anisotropic —
mean pairwise cosine 0.944 in base at layer 31 (`data/geometry_by_layer.csv`) — so on
raw vectors the shared mean direction contributes to every neighbour ranking, and a model could
be scored partly on how collapsed its space is. `analyze_geometry.py` applies the same
correction before it will quote a between-class cosine at all. Centring is therefore the
defensible default and costs nothing.

**It also changes almost nothing here, which is why the uncentred reading is no longer
shipped.** An uncentred column and a second figure were released alongside this one while the
question was open. Measured over all 160 rows, centred minus uncentred spans **−0.027 to
+0.032, mean 0.000** — per label: template −0.011/+0.032, family −0.009/+0.009, group
−0.027/+0.013. At layer 31 on the template panel, the widest case:

| model | centred | uncentred |
|---|---|---|
| base | 0.759 | 0.751 |
| chem-r | 0.587 | 0.563 |
| chem-faithful | 0.489 | 0.460 |
| chemdfm | 0.568 | 0.559 |
| reason | 0.778 | 0.773 |

The model ordering is identical under both readings, and no claim in this document changes.
The comparison is recorded here rather than carried as a duplicate column and a duplicate
figure indefinitely. Re-deriving it needs the activations and one line in
`analyze_prompt_structure.py`: pass the uncentred `acts` to `knn_agreement` beside the
centred call.

### Against the t-SNE measure

Layer 31 template agreement, activation space against the 2-D coordinates:

| model | activation | t-SNE 2-D |
|---|---|---|
| base | 0.759 | 0.744 |
| chem-r | 0.587 | 0.594 |
| chem-faithful | 0.489 | 0.455 |
| chemdfm | 0.568 | 0.520 |
| reason | 0.778 | 0.786 |

Close in value and identical in ordering, which is the check that the t-SNE panels were a
faithful picture rather than a projection artifact. They are still different quantities and
should not be quoted interchangeably.

### Caveats

**ChemDFM is built on Llama-3, not 3.1** — the same dagger Table 1 carries. A chemdfm–base gap
confounds chemistry training with the base-model change. Chem-R-8B is the controlled contrast
and moves the same way, so the result does not rest on ChemDFM.

**n = 3 chemistry checkpoints.** A direction, not an effect size. Nothing here carries an
interval.

**920 points are 92 molecules measured 10 ways.** A molecule's ten templates are near-duplicates
of each other, which is exactly why template agreement can reach 0.99 — and why the independent
sample size behind every number on this page is 92, not 920.

**Chance floors differ per label** and are on every row, because the labels are not otherwise
comparable to each other: template 0.099 / family 0.239 / group 0.054. Computed analytically as `sum_c n_c (n_c - 1) / n (n - 1)`,
and checked against a 200-draw label shuffle.

### Sources

- `analysis/data/label_agreement_by_layer.csv` — 160 rows
  (5 models × 32 layers), `knn_*_centered` and `chance_*` for all three labels.
- generators — `analyze_prompt_structure.py` (reads activations),
  `visuals/make_visuals_prompt_structure.py` (figure, reads only the CSV).
- `template_index` is not a dataset column: it is reconstructed from the molecule-major /
  template-minor row layout, the same assumption `Analysis/tsne/compute_tsne_coords.py` makes.

---
