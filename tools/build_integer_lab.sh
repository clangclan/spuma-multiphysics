#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "$0")/.."
source ./env.sh
mkdir -p bin logs
nvcc -std=c++17 -O3 -arch=sm_120 --ftz=false --prec-div=true --prec-sqrt=true \
    --fmad=true -lineinfo -Xcompiler=-ffp-contract=off -Xptxas=-v \
    src/integerLab/integerLab.cu -o bin/pintleIntegerLab 2>&1 | tee logs/build-integer-lab.log
