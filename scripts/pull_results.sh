#!/bin/bash
# Pull results from the cluster down to this machine. Run it locally, not on the login node.
#
#   bash scripts/pull_results.sh [--dry-run]
#
# Override the endpoints with environment variables rather than editing this file:
#
#   FCG_REMOTE       ssh host or alias                (default: swan)
#   FCG_REMOTE_ROOT  the results tree on that host    (default: /work/$USER/fc_group/results)
#   FCG_RESULTS      local destination                (default: <repo>/results, via paths.py)
#
# The remote tree is results/<method>/<model>/<leaf>, and only a slice of it is ever under
# analysis. Copying the whole thing means gigabytes dominated by plot leaves nobody is
# looking at, so the selection lives here as explicit lists rather than as a half-remembered
# rsync filter typed by hand.
#
# One rsync per source directory, with an explicit destination, instead of a single
# --files-from run: macOS ships openrsync (/usr/bin/rsync, "rsync 2.6.9 compatible"), which
# takes --files-from but not the --relative that real rsync 3.x implies under it to rebuild
# the tree. Forty small invocations over one multiplexed SSH connection cost nothing and put
# the destination layout beyond doubt.
#
# ONE FILE HERE IS NOT A PURE MIRROR. generation_eval/<model>/functional_group/data/
# summary.json carries a "free_generation" block merged in locally by
# analysis/generation/score_generations.py -- the behavioural readout that generation_eval.py
# writes generations.csv for but never scores. summary.json exists on both sides and there is
# no --delete here, so the remote copy overwrites it and that block goes away. Nothing warns
# you: results/ is gitignored, so there is no diff to notice. Restore it after a pull:
#
#   python analysis/generation/score_generations.py
#
# CPU-only, seconds. Excluding the file instead would make the sync rules conditional on one
# leaf, which is worse than re-deriving something this cheap.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$ROOT/paths.py" ] || { echo "ERROR: cannot locate repo root from $ROOT" >&2; exit 1; }

# The ~/.ssh/config alias, which should set ControlMaster auto + ControlPersist: the first
# connection authenticates and the rest reuse the socket, so 2FA is not prompted per
# directory.
REMOTE="${FCG_REMOTE:-swan}"
REMOTE_ROOT="${FCG_REMOTE_ROOT:-/work/$USER/fc_group/results}"
LOCAL_ROOT="$(cd "$ROOT" && python -c 'import paths; print(paths.RESULTS)')"

# Every method whose output an analysis stage reads. Sibling prompt types can include names
# with spaces ("functional_group question"), so the leaf is named exactly and never globbed.
METHODS=(
  ambiguity_metric
  anisotropy_diagnostic
  functional_group_analogy
  functional_group_analogy_retrieval
  functional_group_analogy_retrieval_carbon
  functional_group_analogy_retrieval_halide
  functional_group_probe
  generation_eval
)
LEAF=functional_group

# Directory spellings, not registry names: model_registry.py says "meta-llama/Llama-3.1-8B",
# the results tree writes "meta-llama-Llama-3.1-8B". Kept in sync with MODELS in models.sh
# by hand, because that list carries the slash form and this one cannot.
MODELS=(
  meta-llama-Llama-3.1-8B
  weidawang-Chem-R-8B
  phenixace-Chem-R-Faithful
  OpenDFM-ChemDFM-v1.5-8B
  deepseek-ai-DeepSeek-R1-Distill-Llama-8B
)

DRY=""
if [ "${1:-}" = "--dry-run" ]; then
  DRY="-n"
  echo "DRY RUN -- nothing will be written."
elif [ -n "${1:-}" ]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi

echo "from: $REMOTE:$REMOTE_ROOT"
echo "to:   $LOCAL_ROOT"
echo

missing=()
for method in "${METHODS[@]}"; do
  for model in "${MODELS[@]}"; do
    src="$REMOTE_ROOT/$method/$model/$LEAF/"
    dst="$LOCAL_ROOT/$method/$model/$LEAF/"
    printf '  %-42s %-42s ... ' "$method" "$model"
    [ -n "$DRY" ] || mkdir -p "$dst"
    # No -z (the payload is mostly PNGs, already compressed) and no -v (thousands of plot
    # filenames drown the one line that matters). --partial so an interrupted transfer
    # resumes. Trailing slashes on both sides: copy the contents in, so a re-run tops up
    # rather than nesting another $LEAF/.
    if rsync -a $DRY --partial "$REMOTE:$src" "$dst" 2>"/tmp/pull_results_err.$$"; then
      echo "ok"
    else
      echo "FAILED"
      sed 's/^/      /' "/tmp/pull_results_err.$$"
      missing+=("$method/$model/$LEAF")
    fi
    rm -f "/tmp/pull_results_err.$$"
  done
done

echo
if [ ${#missing[@]} -gt 0 ]; then
  # Not fatal: a partially-complete sweep should still yield what does exist.
  echo "${#missing[@]} directory(ies) did not transfer:"
  printf '  %s\n' "${missing[@]}"
  echo
fi

if [ -z "$DRY" ]; then
  echo "local sizes:"
  du -sh "$LOCAL_ROOT"/* 2>/dev/null
  echo
  echo "next: re-merge the free-generation block, then rebuild the snapshots and figures"
  echo "  python analysis/generation/score_generations.py"
fi
