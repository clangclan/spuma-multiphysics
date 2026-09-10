#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "$0")/.."
source ./env.sh
mkdir -p bin lib logs
nvcc -std=c++17 -O3 -arch=sm_120 --ftz=false --prec-div=true \
    --prec-sqrt=true --fmad=true --cudart shared -lineinfo -Xcompiler=-fPIC \
    -Xptxas=-v -shared src/precisionSolver/pintleIrKernels.cu \
    -o lib/libpintleIrKernels.so 2>&1 | tee logs/build-ir-kernels.log
wmake -j1 libso src/precisionSolver 2>&1 | tee logs/build-ir-wrapper.log
