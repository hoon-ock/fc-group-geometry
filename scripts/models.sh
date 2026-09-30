#!/bin/bash
# Shared configuration for every job in this directory. Source it, do not run it.
#
# Expects ROOT to be set by the caller (each sbatch resolves it before sourcing).
#
# The sweep is declared here; src/model_registry.py is the catalogue it draws from.
# That direction matters: a model can be registered so `get_model_config` resolves it --
# the probe, anisotropy, analogy and ambiguity scripts all raise otherwise -- while being
# outside this sweep. So the check below is a subset check, not an equality one. What it
# still catches is the failure it was written for: a typo'd or unregistered name failing
# loudly instead of silently narrowing the sweep.

# The five released checkpoints. All 8B, all 32 layers, so all fit on one GPU at full
# precision and there is no size split to make -- earlier revisions of this sweep carried a
# SMALL/LARGE divide because the 70B pair needed 4-bit sharded across four GPUs. Those two
# stay registered and unswept; nothing released here uses them.
#
# Order matches _common.BEHAVIOURAL: base, the Tier-1 chemistry pair in training order,
# the off-base chemistry model, then the reasoning distill.
MODELS=(
  "meta-llama/Llama-3.1-8B"
  "weidawang/Chem-R-8B"
  "phenixace/Chem-R-Faithful"
  "OpenDFM/ChemDFM-v1.5-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)

_REGISTRY_MODELS=()
while IFS= read -r line; do
  [ -n "$line" ] && _REGISTRY_MODELS+=("$line")
done < <(python -c "
import sys
sys.path.insert(0, '$ROOT/src')
from model_registry import MODEL_CONFIGS
print('\n'.join(MODEL_CONFIGS))
") || { echo "ERROR: could not read models from $ROOT/src/model_registry.py"; exit 1; }

_registry_sorted=$(printf '%s\n' "${_REGISTRY_MODELS[@]}" | sort)
_sweep_sorted=$(printf '%s\n' "${MODELS[@]}" | sort)

# Every swept model must be registered, or its depth lookup fails mid-job.
_unregistered=$(comm -13 <(echo "$_registry_sorted") <(echo "$_sweep_sorted"))
if [ -n "$_unregistered" ]; then
  echo "ERROR: swept but not in src/model_registry.py: $(echo "$_unregistered" | tr '\n' ' ')"
  echo "  Add it to MODEL_CONFIGS, or fix the spelling in MODELS above."
  exit 1
fi

# Registered but not swept is legal, so say so rather than staying silent: a model dropping
# out of a sweep unnoticed is the failure the old equality check existed to prevent.
_unswept=$(comm -23 <(echo "$_registry_sorted") <(echo "$_sweep_sorted"))
if [ -n "$_unswept" ]; then
  echo "note: registered, not in this sweep: $(echo "$_unswept" | tr '\n' ' ')"
fi
unset _registry_sorted _sweep_sorted _unregistered _unswept _REGISTRY_MODELS

# Entity types come from the extraction config for the same reason -- one source of truth,
# so a job cannot sweep something that was never extracted. The released config declares
# exactly one (`functional_group`), which is why every --array below is 0-4 rather than
# models x entities. Add a prompt type to the YAML and the preflights will tell you which
# --array lines to widen.
read_entity_types() {
  python -c "
import yaml
with open('$ROOT/data/config_extract_activation.yaml') as f:
    cfg = yaml.safe_load(f)
for e in cfg['extraction']['entities']:
    print(e['entity_type'])
"
}

# Every job repeats this: resolve the entity list, then assert models x entities matches
# the static --array span, which is the one thing that cannot self-update.
check_array_size() {
  local declared="$1"
  ENTITIES=()
  while IFS= read -r entity; do
    [ -n "$entity" ] && ENTITIES+=("$entity")
  done < <(read_entity_types)
  [ ${#ENTITIES[@]} -gt 0 ] || { echo "ERROR: no entity types parsed from config"; return 1; }

  local expected=$(( ${#MODELS[@]} * ${#ENTITIES[@]} ))
  if [ "$expected" -ne "$declared" ]; then
    echo "ERROR: ${#MODELS[@]} models x ${#ENTITIES[@]} entity types = $expected runs,"
    echo "       but '#SBATCH --array' declares $declared."
    echo "       Update the --array line and ARRAY_SIZE together."
    return 1
  fi
}
