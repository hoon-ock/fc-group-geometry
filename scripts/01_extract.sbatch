#!/bin/bash
#SBATCH --job-name=fcg_extract
#SBATCH --partition=ock
#SBATCH --gres=gpu:1
# Not quantized, so the CPU cost is safetensors deserialization before .to(device).
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=32G
# 5 models x 1 entity type.
#SBATCH --array=0-4
#SBATCH --output=logs/slurm-extract-%A_%a.out

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

# Every downstream stage reads these tensors, and 5 models x 32 layers x 920 prompts is
# ~1.3 GB. On a cluster whose home quota cannot hold that, point FCG_ACTIVATIONS at scratch
# BEFORE this runs -- paths.py reads it, and every analysis stage must see the same value.
echo "activations -> $(python -c 'import paths; print(paths.ACTIVATIONS)')"

MODEL="${MODELS[$TASK_ID]}"
echo "=== task=$TASK_ID model=$MODEL (all layers) ==="

# All 32 layers, not the registry's default three: the probe reports a curve over depth and
# the released layer_curves.csv has every layer in it.
if python src/extract_activations_subset.py \
     --config "$ROOT/config_extract_activation.yaml" \
     --model-name "$MODEL"; then
  echo "SUMMARY: ok  $MODEL"
else
  echo "SUMMARY: failed  $MODEL"; exit 1
fi
