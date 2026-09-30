#!/bin/bash
#SBATCH --job-name=fcg_smoke
#SBATCH --partition=ock
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=32G
# 5 models x 1 entity type.
#SBATCH --array=0-4
#SBATCH --output=logs/slurm-smoke-%A_%a.out

source "${FCG_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/_preamble.inc" || exit 1
source "$ROOT/scripts/models.sh" || exit 1
check_array_size 5 || exit 1

python -c "import yaml, torch, transformers" || {
  echo "ERROR: conda env not active or deps missing (python=$(command -v python))"; exit 1; }

# Not defaulted to a path: the five 8B checkpoints are ~16 GB each at fp16, ~80 GB for the
# sweep, which exceeds a typical cluster home quota. Left unset, transformers caches under
# $HOME/.cache/huggingface; redirect it before submitting rather than after the first
# download fills the quota.
#   export HF_HOME=/work/$USER/hf_cache
[ -n "${HF_HOME:-}" ] || echo "note: HF_HOME unset -- weights cache under $HOME/.cache/huggingface"

MODEL="${MODELS[$TASK_ID]}"
echo "=== task=$TASK_ID model=$MODEL (smoke: one template, three layers) ==="

# One template and three layers: enough to prove the checkpoint loads, the hooks fire and
# the tensor shapes match the registry, in minutes rather than hours. verify_smoke_output.py
# is what checks the last part -- a silently wrong hidden_dim would otherwise surface only
# when the probe fails to load a fold.
if python src/extract_activations_subset.py \
     --config "$ROOT/config_extract_activation.yaml" \
     --model-name "$MODEL" \
     --layers bottom middle top \
     --max-templates 1; then
  python src/verify_smoke_output.py --model-name "$MODEL" \
    && echo "SUMMARY: ok  $MODEL" \
    || { echo "SUMMARY: verify failed  $MODEL"; exit 1; }
else
  echo "SUMMARY: extraction crashed  $MODEL"; exit 1
fi
