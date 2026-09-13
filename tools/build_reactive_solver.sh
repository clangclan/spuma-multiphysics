#!/usr/bin/env bash
set -eo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source ./env.sh
export PINTLE_REACTIVE_PREFIX="${PINTLE_REACTIVE_PREFIX:-$project_root/research/reactive-env}"
# Run this whole build under /home/jsw/cae-benchmark/run.lock.
bash tools/build_reactive_backend.sh
bash tools/build_reactive_transport.sh
wmake -j1 src/reactiveFoam
