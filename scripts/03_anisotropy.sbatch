#!/bin/bash
#SBATCH --job-name=fcg_aniso
#SBATCH --partition=ock
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=8G
# 5 models x 1 entity type.
#SBATCH --array=0-4
#SBATCH --output=logs/slurm-aniso-%A_%a.out

source "${FCG_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/_preamble.inc" || exit 1
source "$ROOT/scripts/models.sh" || exit 1
check_array_size 5 || exit 1

python -c "import yaml, torch, scipy, sklearn, matplotlib, pandas" || {
  echo "ERROR: conda env not active or deps missing (python=$(command -v python))"; exit 1; }

MODEL="${MODELS[$TASK_ID]}"
ENTITY="${ENTITIES[0]}"
echo "=== task=$TASK_ID model=$MODEL entity=$ENTITY ==="

# How much of the diff-vector geometry is generic transformer anisotropy rather than
# chemistry. analysis/analogy/ and analysis/geometry/ both read this to state their cosines
# against a baseline instead of in absolute terms. Five layers, matching the analogy job.
if python src/anisotropy_diagnostic.py \
     --model-name "$MODEL" \
     --entity-type "$ENTITY" \
     --layers 0,8,16,24,31; then
  echo "SUMMARY: ok  $MODEL | $ENTITY"
else
  echo "SUMMARY: failed  $MODEL | $ENTITY"; exit 1
fi
