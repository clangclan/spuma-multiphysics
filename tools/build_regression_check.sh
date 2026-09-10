#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "$0")/.."
source ./env.sh
mkdir -p bin logs
wmake -j1 src/regressionCheck 2>&1 | tee logs/build-regression-check.log
