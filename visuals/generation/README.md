# `visuals/generation/` — what per-group generation failure is made of

One figure. Written by
[`visuals/make_visuals_generation_composition.py`](../make_visuals_generation_composition.py),
which reads only `analysis/data/group_response_composition.csv` and never `results/` — the
same rule every generator in that directory follows. That CSV is snapshotted by
[`analysis/analyze_generation.py`](../../analysis/analyze_generation.py).

Regenerate:

```bash
python analysis/score_generations.py          # -> free_generation_adjudicated.csv
python analysis/analyze_generation.py         # -> group_response_composition.csv
python visuals/make_visuals_generation_composition.py
```

The order matters: the composition is rebuilt from the adjudicated per-prompt table, so
`score_generations.py` has to run first.

---

## `group_response_composition.png`

Each functional group is one stacked bar of the five adjudicated response types from
`src/free_generation_scoring.py`, one row per checkpoint, with teacher-forced raw recall
overlaid as a dark tick. The five types partition every prompt, so each bar sums to 1.

**The four halides are the case to read.** Base at layer 31:

| group | strict | + superclass credit | underspecified | non-answer | teacher-forced |
|---|---|---|---|---|---|
| alkyl bromide | 0.325 | 0.775 | **0.450** | 0.050 | 0.875 |
| alkyl chloride | 0.150 | 0.625 | **0.475** | 0.100 | 0.950 |
| alkyl fluoride | 0.250 | 0.350 | 0.100 | **0.300** | 0.425 |
| alkyl iodide | 0.050 | 0.375 | 0.325 | 0.175 | 0.475 |

`alkyl chloride` scores **0.150** by strict free generation and **0.950** by teacher-forced
recall. The model is not failing to identify chlorine — it answers "alkyl halide", and
`free_generation_scoring.py` files that as `underspecified`. Credit the superclass and the
group moves to 0.625. Bromide behaves the same way. In the figure this reads as a tall light
blue `Superclass only` block sitting under a tick near the top.

**Three failure modes, pooled into one number.** Strict per-group recall cannot tell them
apart, and only the third is the kind of thing a representation metric is even about:

| mode | signature in the figure | example |
|---|---|---|
| Answers the family, not the member | tall `Superclass only`, tick near the top | chloride 0.150 → 0.950 |
| Names no group at all | tall `Named no group`, tick still high | `imine` non-answer 0.400, tf 0.800; `thioether` 0.450, tf 0.800 |
| Genuinely weak on the class | the tick itself is low | `alkyl fluoride` tf 0.425, `alkyl iodide` tf 0.475 |

Fluoride and iodide are the only two groups where the forced-choice head is also weak, so
those two are the only real candidates for a representation deficit. `imine` and `thioether`
fail by declining to commit while the forced-choice head has the answer at 0.800.

That is the per-group form of the commitment story: **conditional on committing, the
chemistry is there.** A model that declines to answer, or answers the family instead of the
member, is not a model that lacks the class -- the forced-choice head resolves it at
0.800-0.950 in exactly those cases. Strict free-generation recall pools the two together
and reports the sum as failure.

## No layer axis

Every other figure in this repository is drawn against depth. This one is not, and cannot be:
generation outcomes come from what the model emitted, not from a hidden state at some layer.
The CSV therefore carries no `layer` column.

## What this figure replaced

Two correlation figures — per-group generation accuracy scattered against retrieval hit@1
and mean rank, and rank correlation by layer for six representation predictors — were dropped
along with the analysis behind them. Their result was a null: the point cloud had no
orientation to fit, `sulfone` sitting at hit@1 0.000 with strict recall 0.875 while `imine`
sat at hit@1 0.667 with 0.275. The composition above is what answers the question those
figures could not, and it needs no embedding metric to do it.
