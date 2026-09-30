#!/bin/bash
#SBATCH --job-name=fcg_retrieval
#SBATCH --partition=ock
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=8G
# 5 models x 1 entity type; each task runs all three quadruple sets.
#SBATCH --array=0-4
#SBATCH --output=logs/slurm-retrieval-%A_%a.out

source "${FCG_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/_preamble.inc" || exit 1
source "$ROOT/scripts/models.sh" || exit 1
check_array_size 5 || exit 1

python -c "import yaml, numpy, scipy, matplotlib" || {
  echo "ERROR: conda env not active or deps missing (python=$(command -v python))"; exit 1; }

# This job reads what 04_analogy wrote, not the activations. No GPU, no model load. If that
# tree is absent the whole array is pointless, so say so once, here.
ANALOGY_DIR="$(python -c 'import os, paths; print(os.path.join(paths.RESULTS, "functional_group_analogy"))')"
[ -d "$ANALOGY_DIR" ] || {
  echo "ERROR: $ANALOGY_DIR/ missing -- run 04_analogy first; this job reads its"
  echo "       diff_vectors_layer_*.npz files."
  exit 1; }

MODEL="${MODELS[$TASK_ID]}"
ENTITY="${ENTITIES[0]}"
echo "=== task=$TASK_ID model=$MODEL entity=$ENTITY ==="

# All three quadruple sets in one task: cross-family (inter), within-family down the halogen
# column (halide), and within-group up the chain length (carbon). Each writes its own tree,
# and running them together is what guarantees all three come from the same diff vectors.
# analysis/geometry/ reads all three as the "three independent retrieval axes".
STATUS=0
for QSET in inter halide carbon; do
  RUN_LOG=$(mktemp)
  if python src/analogy_retrieval.py \
       --analogy-dir "$ANALOGY_DIR" \
       --model-name "$MODEL" \
       --entity-type "$ENTITY" \
       --quadruple-set "$QSET" 2>&1 | tee "$RUN_LOG"; then
    # analogy_retrieval.py self-checks before it writes, so reaching here means the trials,
    # the summaries and each other all agreed. No postflight step needed.
    echo "SUMMARY: ok  $MODEL | $ENTITY | $QSET"
  else
    # Three different non-zero exits, and telling them apart matters. Missing input is an
    # ordering problem; a failed self-check is a bug in the analysis, not the environment.
    if grep -qE "no diff_vectors_layer_|nothing to report" "$RUN_LOG"; then
      echo "SUMMARY: no retrieval input  $MODEL | $ENTITY | $QSET"
    elif grep -q "FAILED SELF-CHECK" "$RUN_LOG"; then
      echo "SUMMARY: bad-output  $MODEL | $ENTITY | $QSET"
    else
      echo "SUMMARY: crashed  $MODEL | $ENTITY | $QSET"
    fi
    STATUS=1
  fi
  rm -f "$RUN_LOG"
done
exit $STATUS
