#!/bin/bash
#SBATCH --job-name=fcg_ambiguity
#SBATCH --partition=ock
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=8G
# 5 models x 1 entity type.
#SBATCH --array=0-4
#SBATCH --output=logs/slurm-ambiguity-%A_%a.out

source "${FCG_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/_preamble.inc" || exit 1
source "$ROOT/scripts/models.sh" || exit 1
check_array_size 5 || exit 1

python -c "import numpy, pandas, scipy, matplotlib" || {
  echo "ERROR: conda env not active or deps missing (python=$(command -v python))"; exit 1; }

# Reads the probe's out-of-fold probability tables (data/oof_proba_*.npz), not activations.
# 02_probe writes those by default; --no-save-proba would have suppressed them.
PROBE_DIR="$(python -c 'import os, paths; print(os.path.join(paths.RESULTS, "functional_group_probe"))')"
[ -d "$PROBE_DIR" ] || {
  echo "ERROR: $PROBE_DIR/ missing -- run 02_probe first; this job reads its"
  echo "       oof_proba_*.npz files."
  exit 1; }

MODEL="${MODELS[$TASK_ID]}"
ENTITY="${ENTITIES[0]}"
echo "=== task=$TASK_ID model=$MODEL entity=$ENTITY ==="

# Four groups are labelled by naming convention rather than composition -- amide and nitro
# carry oxygen but are `nitrogen`, sulfoxide and sulfone carry oxygen but are `sulfur`. Under
# leave-one-group-out they are genuinely underdetermined, and accuracy cannot see that. This
# scores how much mass lands on the defensible rival, contrasted against the group's own
# oxygen-free family controls so a merely-diffuse probe scores zero.
#
# --mode proba keeps this off the activations entirely. The projection mode additionally
# needs activations/ present; --full adds the calibrated contrast with bootstrap CIs.
if python src/ambiguity_metric.py \
     --model-name "$MODEL" \
     --entity-type "$ENTITY" \
     --mode proba; then
  echo "SUMMARY: ok  $MODEL | $ENTITY"
else
  echo "SUMMARY: failed  $MODEL | $ENTITY"; exit 1
fi
