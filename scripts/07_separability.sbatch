#!/bin/bash
#SBATCH --job-name=fcg_separability
#SBATCH --partition=ock
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=8G
# 5 models x 1 entity type.
#SBATCH --array=0-4
#SBATCH --output=logs/slurm-separability-%A_%a.out

source "${FCG_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/_preamble.inc" || exit 1
source "$ROOT/scripts/models.sh" || exit 1
check_array_size 5 || exit 1

python -c "import numpy, pandas, sklearn" || {
  echo "ERROR: conda env not active or deps missing (python=$(command -v python))"; exit 1; }

MODEL="${MODELS[$TASK_ID]}"
ENTITY="${ENTITIES[0]}"
echo "=== task=$TASK_ID model=$MODEL entity=$ENTITY (all layers) ==="

# The one per-group representation number that covers all 20 groups. Every other readout is
# either pooled over groups (retrieval_by_axis collapses the group axis) or reaches only 17
# of them at 1-16 trials each -- neither can go on a scatter against per-group generation
# accuracy, which is what analysis/generation/analyze_generation_vs_representation.py needs.
#
# --all-layers because the released separability_by_group.csv carries all 32.
if python src/functional_group_separability.py \
     --model-name "$MODEL" \
     --entity-type "$ENTITY" \
     --all-layers; then
  echo "SUMMARY: ok  $MODEL | $ENTITY"
else
  echo "SUMMARY: failed  $MODEL | $ENTITY"; exit 1
fi
