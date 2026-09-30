# Functional-group probe — what the released sweep shows

`analyze_sweep.py` reads `results/functional_group_probe/` across all five models and writes
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

## The five models

`base` = `meta-llama/Llama-3.1-8B`, `chem` = `phenixace/Chem-R-Faithful`,
`chem-r` = `weidawang/Chem-R-8B`, `chemdfm` = `OpenDFM/ChemDFM-v1.5-8B`,
`reason` = `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`. All 8B, all 32 layers.

ChemDFM sits on Llama-3 rather than 3.1, so a ChemDFM-vs-base gap confounds chemistry
training with the 3 → 3.1 difference and is never a controlled contrast. Chem-R-8B is what
carries that. Chem-R-Faithful is GRPO-trained *from* Chem-R-8B, so those two are a
training-order pair, not independent samples.

## 1. Every model solves the task, and at the same depth

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
`analysis/figures/_common.py` is chosen as the first layer where all models are jointly above
0.94 rather than as anyone's individual best.

## 2. The surface baseline is what makes §1 mean anything

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

## 3. Where the errors go — the one place the models differ

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
`analysis/ambiguity/` is the analysis that scores *how much* mass lands on that rival rather
than just counting errors.

Read the share, not the count. Below the transition every model misclassifies heavily in
every direction, so a large raw count at low depth reflects a probe that does not work yet.
`n_total_errors` is carried in the file so the denominator is visible — base's falls to
single digits in the deep layers, which makes its share correspondingly noisy.

## 4. base vs chem, with an interval

`model_diff_bootstrap.csv` is a **paired** bootstrap on the contrast the study is built
around: **chem − base = −0.017, 95% CI [−0.032, −0.006]**, excluding zero.

The resampling unit is the **molecule** (88 of them — 92 minus alkane's 4, since alkane is
the only hydrocarbon and is skipped as a leave-one-group-out fold), not the row. Rows are
molecule-major / template-minor, so a molecule's 10 template rows are near-duplicates;
resampling rows would narrow the interval by treating them as independent evidence.

The sign is worth stating plainly: the chemistry-tuned model is *slightly worse* at this
task than the base model, and the interval excludes zero. Chemistry fine-tuning did not make
functional-group identity more linearly decodable. §3 is where the two actually differ.

## 5. Readings the data does not support

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

## 6. Known gaps in the released outputs

**Reference nulls are absent.** The sweep runs with `--num-null-samples 0`, so `null_mean`
and `p_value_best_layer` are `null` in every `summary_*.json`. That is deliberate: the null
is ~91% of the job's runtime and does not depend on the activations — shuffled labels leave
no signal regardless of what `X` holds, so it is a property of the fold structure, class
distribution, classifier and metric, all identical across every task. Measured once it is
flat and sits at chance (0.184–0.199 against a 0.20 chance line). It still matters once, as
a **leak detector**: if the molecule-level shuffle ever broke, it would rise above chance.
Run it per model and quote it as a reference band:

```bash
python src/functional_group_probe.py --model-name meta-llama/Llama-3.1-8B \
  --entity-type functional_group --num-null-samples 50 \
  --output-dir results/null_ref_base
```

## 7. Regenerating

```bash
python analysis/probe/analyze_sweep.py                        # -> data/*.csv
```

`--tag` selects which probe target to summarise and defaults to `coarse_group`, which is
what `02_probe.sbatch` runs and what the committed `data/` carries. Another tag needs the
job re-run with `PROBE_TARGET` set, and would overwrite these same filenames — the
snapshot holds one target at a time.

Needs a populated `results/functional_group_probe/`. CPU-only, no model loading, a few
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
