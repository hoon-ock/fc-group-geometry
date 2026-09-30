#!/bin/bash
# Activations in, 70 figures out -- no cluster, no SLURM.
#
#   bash scripts/run_local.sh                     # everything
#   bash scripts/run_local.sh --stages analysis   # skip the compute, re-derive snapshots
#   bash scripts/run_local.sh --stages figures    # just redraw from the committed CSVs
#   bash scripts/run_local.sh --models meta-llama/Llama-3.1-8B
#
# WHY THIS EXISTS. Five of the pipeline's stages -- probe, anisotropy, analogy, retrieval
# and ambiguity -- ask for one core and 4-8 GB and never touch a GPU. They were SLURM array
# jobs because they were written beside the extraction that does need one, but on a machine
# that already has the activations they run in well under an hour, and a cluster round-trip
# plus pull_results.sh buys nothing. What still needs a GPU is in scripts/*.sbatch: the
# extraction itself and the behavioural generation eval.
#
# Measured on an M-series laptop, per model: anisotropy ~6 s, analogy ~1-3 min (the null
# dominates), probe ~8 min for all 32 layers, retrieval and ambiguity seconds each -- they
# read the npz the earlier stages wrote rather than the tensors. The probe is the long pole.
#
# THE FIGURES DO NOT DEPEND ON ANY OF THIS. Every make_visuals*.py reads only the committed
# analysis/*/data/*.csv, so `--stages figures` works on a fresh clone with no activations at
# all. The compute stages exist to RE-DERIVE those snapshots; the figures consume whatever
# is on disk.

set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$ROOT/paths.py" ] || { echo "ERROR: cannot locate repo root from $ROOT" >&2; exit 1; }
cd "$ROOT" || exit 1

# numpy's OpenBLAS will happily oversubscribe every core for matrices this small and end up
# slower. One thread per process, same as the sbatch jobs.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

STAGES=all
MODELS_OVERRIDE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --stages) STAGES="${2:-}"; shift 2 ;;
    --models) MODELS_OVERRIDE="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$STAGES" in all|compute|analysis|figures) ;;
  *) echo "ERROR: --stages must be all, compute, analysis or figures" >&2; exit 2 ;; esac

source "$ROOT/scripts/models.sh" || exit 1
[ -n "$MODELS_OVERRIDE" ] && MODELS=($MODELS_OVERRIDE)
ENTITY="$(read_entity_types | head -1)"

ACT="$(python -c 'import paths; print(paths.ACTIVATIONS)')"
RESULTS="$(python -c 'import paths; print(paths.RESULTS)')"

echo "root        $ROOT"
echo "activations $ACT"
echo "results     $RESULTS"
echo "models      ${#MODELS[@]}"
echo "entity      $ENTITY"
echo "stages      $STAGES"
echo

# `python -` rather than a heredoc into a file: keeps the driver a single file.
step() {
  local label="$1"; shift
  local t0=$SECONDS
  printf '  %-46s ' "$label"
  if "$@" >/tmp/run_local.$$ 2>&1; then
    printf 'ok   %ds\n' "$(( SECONDS - t0 ))"
    rm -f /tmp/run_local.$$
  else
    printf 'FAILED\n'
    sed 's/^/      /' /tmp/run_local.$$ | tail -12
    rm -f /tmp/run_local.$$
    return 1
  fi
}

# ---- 1. compute: the five CPU stages, per model -----------------------------------------
if [ "$STAGES" = all ] || [ "$STAGES" = compute ]; then
  [ -d "$ACT" ] || { echo "ERROR: no activations at $ACT -- run scripts/01_extract.sbatch"; exit 1; }
  echo "== compute (5 stages x ${#MODELS[@]} models) =="
  for MODEL in "${MODELS[@]}"; do
    echo "-- $MODEL"
    # --target coarse: nothing downstream reads the other targets. --num-null-samples 0:
    # the permutation null is ~91% of the runtime and is a property of the fold structure,
    # not the activations. --surface-baseline IS on -- it is the control that separates
    # recalled chemistry from reading "-ol" out of the prompt.
    step "probe (32 layers)" \
      python src/probe.py --model-name "$MODEL" --entity-type "$ENTITY" \
        --target coarse --surface-baseline --num-null-samples 0 || exit 1
    # Five layers, the depths every downstream figure reports at.
    step "anisotropy" \
      python src/anisotropy_diagnostic.py --model-name "$MODEL" --entity-type "$ENTITY" \
        --layers 0,8,16,24,31 || exit 1
    # Carbon-matched: writes the diff-vector npz that retrieval and taxonomy both read.
    step "analogy (carbon-matched)" \
      python src/analogy_carbon_matched.py --model-name "$MODEL" \
        --entity-type "$ENTITY" --layers 0,8,16,24,31 || exit 1
    # Three axes in one pass over the npz the analogy stage just wrote, so all three come
    # from the same diff vectors. No activations, no model.
    for QSET in inter halide carbon; do
      step "retrieval ($QSET)" \
        python src/analogy_retrieval.py \
          --analogy-dir "$RESULTS/analogy" \
          --model-name "$MODEL" --entity-type "$ENTITY" --quadruple-set "$QSET" || exit 1
    done
    # Reads the probe's out-of-fold probabilities; --mode proba keeps it off the tensors.
    step "ambiguity" \
      python src/ambiguity_metric.py --model-name "$MODEL" --entity-type "$ENTITY" \
        --mode proba || exit 1
  done
  echo
fi

# ---- 2. analysis: results/ -> the committed snapshot CSVs --------------------------------
if [ "$STAGES" = all ] || [ "$STAGES" = analysis ]; then
  echo "== analysis (results/ -> analysis/*/data/) =="
  # Order matters twice: analyze_taxonomy writes the matrix analyze_geometry does not need
  # but the clustermaps do, and score_generations must precede analyze_generation, which
  # reads its adjudicated table to build the response composition.
  step "probe/analyze_sweep"                python analysis/probe/analyze_sweep.py || exit 1
  step "taxonomy/analyze_taxonomy"          python analysis/taxonomy/analyze_taxonomy.py || exit 1
  step "geometry/analyze_geometry"          python analysis/geometry/analyze_geometry.py || exit 1
  step "analogy/snapshot_quiver3d"          python analysis/analogy/snapshot_quiver3d.py || exit 1
  step "ambiguity/analyze_ambiguity"        python analysis/ambiguity/analyze_ambiguity.py || exit 1
  step "tsne/compute_tsne_coords"           python analysis/tsne/compute_tsne_coords.py || exit 1
  step "prompt_structure/analyze_prompt_structure" \
                                            python analysis/prompt_structure/analyze_prompt_structure.py || exit 1
  # The behavioural half needs results/generation_eval/, which only the GPU job produces.
  # Absent, the committed snapshots stay as they are and the figure still draws from them.
  if [ -d "$RESULTS/generation_eval" ]; then
    step "generation/score_generations"     python analysis/generation/score_generations.py || exit 1
    step "generation/analyze_generation"    python analysis/generation/analyze_generation.py || exit 1
  else
    printf '  %-46s skipped (no results/generation_eval -- needs a GPU,\n' "generation"
    printf '  %-46s  scripts/02_generation_eval.sbatch)\n' ""
  fi
  echo
fi

# ---- 3. figures --------------------------------------------------------------------------
if [ "$STAGES" = all ] || [ "$STAGES" = figures ]; then
  echo "== figures (analysis/*/data/ -> analysis/visuals/) =="
  for f in analysis/figures/make_visuals*.py; do
    step "$(basename "$f")" python "$f" || exit 1
  done
  echo
  echo "wrote $(find analysis/visuals -name '*.png' | wc -l | tr -d ' ') PNGs under analysis/visuals/"
fi
