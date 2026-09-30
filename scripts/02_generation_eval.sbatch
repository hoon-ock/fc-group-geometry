#!/bin/bash
#SBATCH --job-name=fcg_generation
#SBATCH --partition=ock
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=32G
# 5 models x 1 entity type.
#SBATCH --array=0-4
#SBATCH --output=logs/slurm-generation-%A_%a.out

source "${FCG_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/_preamble.inc" || exit 1
source "$ROOT/scripts/models.sh" || exit 1
check_array_size 5 || exit 1

python -c "import yaml, torch, transformers, pandas" || {
  echo "ERROR: conda env not active or deps missing (python=$(command -v python))"; exit 1; }

export HF_HOME="${HF_HOME:-/work/ock/hoon/hf_cache}"

MODEL="${MODELS[$TASK_ID]}"
echo "=== task=$TASK_ID model=$MODEL ==="

# The one behavioural stage: every other job probes frozen activations, this one asks the
# model to answer. It records two things per prompt -- a forced choice scored by log-sum-exp
# over the surface forms of each class (scores.csv), and the free continuation it actually
# generates (generations.csv). analysis/ reads both: analyze_generation.py the
# forced choice, score_generations.py the free text through the same adjudicator this job
# imports, so the HPC run and the local scoring cannot disagree.
#
# Needs a GPU -- it is the only stage that runs a forward pass with generation.
if python src/generation_eval.py \
     --config "$ROOT/config_extract_activation.yaml" \
     --model-name "$MODEL" \
     --entity-types functional_group \
     --max-new-tokens 10; then
  echo "SUMMARY: ok  $MODEL"
else
  echo "SUMMARY: failed  $MODEL"; exit 1
fi
