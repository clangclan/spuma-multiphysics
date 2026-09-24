# Source this file from bash before using the local solver.
# The upstream shell setup contains expected nonzero feature probes; sourcing
# it as a tested command prevents a caller's `set -e` from aborting mid-setup.
source "${REACTIVE_SPUMA_ENV:-/home/jsw/cae-gpu/spuma-env.sh}" || return $?
export REACTIVE_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$REACTIVE_PROJECT_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$REACTIVE_PROJECT_ROOT/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
