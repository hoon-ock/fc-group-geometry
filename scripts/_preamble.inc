# Sourced by every job here, immediately after the #SBATCH block. Not executable.
#
# Resolves ROOT and fails fast if it is wrong. A SLURM job's $0 is a spooled copy, so
# BASH_SOURCE cannot locate the repo -- SLURM_SUBMIT_DIR is the reliable handle, and
# FCG_ROOT overrides it for a manual run from anywhere.

set -o pipefail

module load anaconda
conda activate llm-perception

ROOT="${FCG_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -f "$ROOT/paths.py" ] || {
  echo "ERROR: $ROOT is not the repository root (no paths.py)."
  echo "       Submit from the root, or set FCG_ROOT=/path/to/repo."
  exit 1; }
cd "$ROOT" || exit 1

# The compute node caps per-user threads (RLIMIT_NPROC) and numpy's OpenBLAS backend blows
# past it. Load-bearing, not decoration.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
