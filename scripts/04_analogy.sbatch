#!/bin/bash
#SBATCH --job-name=fcg_analogy
#SBATCH --partition=ock
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem=8G
# 5 models x 1 entity type.
#SBATCH --array=0-4
#SBATCH --output=logs/slurm-analogy-%A_%a.out

source "${FCG_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/_preamble.inc" || exit 1
source "$ROOT/scripts/models.sh" || exit 1
check_array_size 5 || exit 1

python -c "import yaml, torch, scipy, seaborn, sklearn, matplotlib, pandas" || {
  echo "ERROR: conda env not active or deps missing (python=$(command -v python))"; exit 1; }

MODEL="${MODELS[$TASK_ID]}"
ENTITY="${ENTITIES[0]}"
echo "=== task=$TASK_ID model=$MODEL entity=$ENTITY ==="

# The carbon-matched version, not the plain one: it writes a strict superset, including the
# second_order_analogy_lumped_layer_*.csv and the diff_vectors_layer_*.npz that 05_retrieval
# and analysis/taxonomy/ both read. Five layers -- the depths every downstream figure
# reports at (_common.LAYERS).
if python src/functional_group_analogy_carbon_matched.py \
     --model-name "$MODEL" \
     --entity-type "$ENTITY" \
     --layers 0,8,16,24,31; then
  echo "SUMMARY: ok  $MODEL | $ENTITY"
else
  echo "SUMMARY: failed  $MODEL | $ENTITY"; exit 1
fi
