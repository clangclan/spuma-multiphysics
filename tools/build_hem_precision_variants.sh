#!/usr/bin/env bash
# HEM-only experiment libraries; the production transport library is untouched.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:?Usage: build_hem_precision_variants.sh OUTPUT_DIRECTORY}"
mkdir -p -- "$output"
arch="${REACTIVE_CUDA_ARCH:-120}"
[[ "$arch" =~ ^[0-9]+$ ]] || exit 2
shift
modes=("$@")
if ((${#modes[@]}==0)); then modes=(fp64 fp32-seeds fp32-linear int32-linear fp32-combined); fi
for mode in "${modes[@]}"; do
    flags=()
    case "$mode" in
        fp32-seeds) flags+=(-DREACTIVE_HEM_FP32_SEEDS=1);;
        fp32-linear) flags+=(-DREACTIVE_HEM_FP32_LINEAR=1);;
        int32-linear) flags+=(-DREACTIVE_HEM_INT_LINEAR=1);;
        fp32-combined) flags+=(-DREACTIVE_HEM_FP32_SEEDS=1 -DREACTIVE_HEM_FP32_LINEAR=1);;
        fp32-nasa) flags+=(-DREACTIVE_HEM_NASA_PRECISION=1);;
        ds-nasa) flags+=(-DREACTIVE_HEM_NASA_PRECISION=2);;
        fp64);;
        *) echo "Unknown mode: $mode" >&2; exit 2;;
    esac
    "${NVCC:-nvcc}" -std=c++17 -O2 -Xcompiler=-fPIC --fmad=false \
        --ftz=false --prec-div=true --prec-sqrt=true -lineinfo -Xptxas=-v \
        "-gencode=arch=compute_${arch},code=[sm_${arch},compute_${arch}]" \
        "${flags[@]}" -shared "$root/src/reactiveTransport/reactiveGpuHemFma.cu" \
        -o "$output/libHem-$mode.so" > "$output/build-$mode.log" 2>&1
    echo "Built $mode"
done
