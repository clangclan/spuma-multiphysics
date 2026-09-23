#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$project_root/lib"
mode="${PINTLE_REACTIVE_TRANSPORT_BUILD:-cpu}"
case "$mode" in
  cpu)
    "${CXX:-g++}" -std=c++17 -O2 -fPIC -shared -Wall -Wextra -fno-fast-math \
      "$project_root/src/reactiveTransport/pintleReactiveTransport.cpp" \
      -o "$project_root/lib/libpintleReactiveTransport.so" ;;
  cuda)
    arch="${PINTLE_CUDA_ARCH:-80}"
    [[ "$arch" =~ ^[0-9]+$ ]] || { echo 'PINTLE_CUDA_ARCH must be a compute capability number' >&2; exit 1; }
    arithmetic="${PINTLE_GPU_HEM_ARITHMETIC:-separateRn}"
    common=(-std=c++17 -O2 -Xcompiler=-fPIC,-Wall,-Wextra --ftz=false --prec-div=true --prec-sqrt=true
      "-gencode=arch=compute_${arch},code=[sm_${arch},compute_${arch}]")
    case "$arithmetic" in
      separateRn)
        "${NVCC:-nvcc}" "${common[@]}" -shared --fmad=false \
          "$project_root/src/reactiveTransport/pintleReactiveTransport.cu" \
          -o "$project_root/lib/libpintleReactiveTransport.so" ;;
      fp64FmaRnGuard)
        # The new rounding policy applies only to HEM. Keep transport and
        # scalar caloric kernels at their existing separate-rounding policy.
        hem_build_dir="$(mktemp -d)"
        trap 'rm -rf -- "$hem_build_dir"' EXIT
        "${NVCC:-nvcc}" "${common[@]}" -c --fmad=false -DPINTLE_EXTERNAL_GPU_HEM=1 \
          "$project_root/src/reactiveTransport/pintleReactiveTransport.cu" \
          -o "$hem_build_dir/transport.o"
        "${NVCC:-nvcc}" "${common[@]}" -c --fmad=true -DPINTLE_HEM_FMA_POLICY=1 \
          "$project_root/src/reactiveTransport/pintleGpuHemFma.cu" \
          -o "$hem_build_dir/hem.o"
        "${NVCC:-nvcc}" -shared "$hem_build_dir/transport.o" "$hem_build_dir/hem.o" \
          -o "$project_root/lib/libpintleReactiveTransport.so" ;;
      *) echo 'PINTLE_GPU_HEM_ARITHMETIC must be separateRn or fp64FmaRnGuard' >&2; exit 1 ;;
    esac ;;
  *) echo 'PINTLE_REACTIVE_TRANSPORT_BUILD must be cpu or cuda' >&2; exit 1 ;;
esac
