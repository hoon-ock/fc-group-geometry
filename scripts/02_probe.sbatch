#!/bin/bash
#SBATCH --job-name=fcg_probe
#SBATCH --partition=ock
# One core, 4 GB, no GPU, and all three are deliberate. Measured peak RSS is well under
# 1 GB, so 4G is ~4x headroom; an earlier 32 GB request pinned the array to a handful of
# concurrent tasks on one node with the rest in PD (Resources). The probe is entirely
# scikit-learn -- torch appears only as torch.load(..., map_location="cpu") -- so a GPU
# would idle while competing with extraction. Extra cores do not help either: the matrices
# are too small for BLAS parallelism and lbfgs is serial. Throughput comes from array width.
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=4G
# 5 models x 1 entity type. Append '%N' (--array=0-4%3) to cap concurrency.
#SBATCH --array=0-4
#SBATCH --output=logs/slurm-probe-%A_%a.out

source "${FCG_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/_preamble.inc" || exit 1
source "$ROOT/scripts/models.sh" || exit 1
check_array_size 5 || exit 1

python -c "import yaml, torch, scipy, seaborn, sklearn, matplotlib, pandas" || {
  echo "ERROR: conda env not active or deps missing (python=$(command -v python))"; exit 1; }

# Overridable at submit time:
#   sbatch --export=ALL,PROBE_TARGET=fine,PROBE_SPLIT=molecule scripts/02_probe.sbatch
# An empty PROBE_SPLIT leaves the probe on its own default (leave-one-group-out), which is
# the only split with dynamic range -- molecule and carbon are saturated by the molecule's
# own spelling, where character n-grams alone score 0.975-1.00.
PROBE_TARGET="${PROBE_TARGET:-all}"
PROBE_SPLIT="${PROBE_SPLIT:-}"

MODEL="${MODELS[$TASK_ID]}"
ENTITY="${ENTITIES[0]}"
echo "=== task=$TASK_ID model=$MODEL entity=$ENTITY target=$PROBE_TARGET split=${PROBE_SPLIT:-<default>} ==="

# --num-null-samples 0 is deliberate. The permutation null is ~91% of this job's runtime and
# does not depend on the activations: shuffled labels leave no signal regardless of what X
# holds, so it is a property of the fold structure, class distribution, classifier and
# metric -- identical across every task here, and measured flat at chance. Run it separately
# once per model as a leak detector; see "Reference nulls" in analysis/probe/README.md.
#
# --surface-baseline, by contrast, is ON. It costs ~0.2 s and it is the control that
# separates recalled chemistry from reading "-ol" out of the prompt.
SPLIT_ARGS=()
[ -n "$PROBE_SPLIT" ] && SPLIT_ARGS+=(--split "$PROBE_SPLIT")

RUN_LOG=$(mktemp)
if python src/functional_group_probe.py \
     --model-name "$MODEL" \
     --entity-type "$ENTITY" \
     --target "$PROBE_TARGET" \
     "${SPLIT_ARGS[@]}" \
     --surface-baseline \
     --num-null-samples 0 2>&1 | tee "$RUN_LOG"; then
  if grep -q "No layer files found" "$RUN_LOG"; then
    echo "SUMMARY: no activation data  $MODEL | $ENTITY"; rm -f "$RUN_LOG"; exit 1
  fi
  echo "SUMMARY: ok  $MODEL | $ENTITY"; rm -f "$RUN_LOG"
else
  echo "SUMMARY: crashed  $MODEL | $ENTITY"; rm -f "$RUN_LOG"; exit 1
fi
