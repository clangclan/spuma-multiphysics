#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "$0")/.."
source ./env.sh
mkdir -p lib logs
wmake -j1 libso src/asyncSmoother 2>&1 | tee logs/build-async-smoother.log
