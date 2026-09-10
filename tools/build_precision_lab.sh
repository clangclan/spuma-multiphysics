#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "$0")/.."
source ./env.sh
mkdir -p bin lib logs
nvcc -std=c++17 -O3 -arch=sm_120 --ftz=false --prec-div=true \
    --prec-sqrt=true --fmad=true -lineinfo -Xcompiler=-ffp-contract=off \
    -Xptxas=-v src/precisionLab/precisionLab.cu -o bin/pintlePrecisionLab \
    2>&1 | tee logs/build-precision-lab.log
wmake -j1 libso src/precisionProbe 2>&1 | tee logs/build-precision-probe.log
