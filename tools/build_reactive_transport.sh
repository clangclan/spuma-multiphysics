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
    "${NVCC:-nvcc}" -std=c++17 -O2 -shared -Xcompiler=-fPIC,-Wall,-Wextra \
      --fmad=false --prec-div=true --prec-sqrt=true \
      "-gencode=arch=compute_${arch},code=[sm_${arch},compute_${arch}]" \
      "$project_root/src/reactiveTransport/pintleReactiveTransport.cu" \
      -o "$project_root/lib/libpintleReactiveTransport.so" ;;
  *) echo 'PINTLE_REACTIVE_TRANSPORT_BUILD must be cpu or cuda' >&2; exit 1 ;;
esac
