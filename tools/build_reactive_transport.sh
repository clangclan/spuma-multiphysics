#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$project_root/lib"
mode="${REACTIVE_TRANSPORT_BUILD:-cpu}"
case "$mode" in
  cpu)
    "${CXX:-g++}" -std=c++17 -O2 -fPIC -shared -Wall -Wextra -fno-fast-math \
      "$project_root/src/reactiveTransport/reactiveTransport.cpp" \
      -o "$project_root/lib/libreactiveTransport.so" ;;
  cuda)
    arch="${REACTIVE_CUDA_ARCH:-80}"
    [[ "$arch" =~ ^[0-9]+$ ]] || { echo 'REACTIVE_CUDA_ARCH must be a compute capability number' >&2; exit 1; }
    arithmetic="${REACTIVE_GPU_HEM_ARITHMETIC:-separateRn}"
    common=(-std=c++17 -O2 -Xcompiler=-fPIC,-Wall,-Wextra --ftz=false --prec-div=true --prec-sqrt=true
      "-gencode=arch=compute_${arch},code=[sm_${arch},compute_${arch}]")
    case "$arithmetic" in
      separateRn)
        "${NVCC:-nvcc}" "${common[@]}" -shared --fmad=false \
          "$project_root/src/reactiveTransport/reactiveTransport.cu" \
          -o "$project_root/lib/libreactiveTransport.so" ;;
      fp64FmaRnGuard)
        # The new rounding policy applies only to HEM. Keep transport and
        # scalar caloric kernels at their existing separate-rounding policy.
        hem_build_dir="$(mktemp -d)"
        trap 'rm -rf -- "$hem_build_dir"' EXIT
        "${NVCC:-nvcc}" "${common[@]}" -c --fmad=false -DREACTIVE_EXTERNAL_GPU_HEM=1 \
          "$project_root/src/reactiveTransport/reactiveTransport.cu" \
          -o "$hem_build_dir/transport.o"
        "${NVCC:-nvcc}" "${common[@]}" -c --fmad=true -DREACTIVE_HEM_FMA_POLICY=1 \
          "$project_root/src/reactiveTransport/reactiveGpuHemFma.cu" \
          -o "$hem_build_dir/hem.o"
        "${NVCC:-nvcc}" -shared "$hem_build_dir/transport.o" "$hem_build_dir/hem.o" \
          -o "$project_root/lib/libreactiveTransport.so" ;;
      *) echo 'REACTIVE_GPU_HEM_ARITHMETIC must be separateRn or fp64FmaRnGuard' >&2; exit 1 ;;
    esac ;;
  *) echo 'REACTIVE_TRANSPORT_BUILD must be cpu or cuda' >&2; exit 1 ;;
esac
