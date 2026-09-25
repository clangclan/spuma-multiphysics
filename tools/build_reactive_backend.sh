#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
reactive_prefix="${REACTIVE_PREFIX:-$project_root/research/reactive-env}"
mkdir -p "$project_root/lib" "$project_root/logs"
test -f "$reactive_prefix/include/cantera/thermo/PengRobinson.h"
# The caller holds the shared run lock, including when this is part of a larger
# validation campaign. Host-only Cantera calls are isolated behind a C ABI.
# Header-only NVTX ranges (reactiveNvtx.h) when the CUDA headers are present.
nvtx_include=()
if [[ -n "${CUDA_HOME:-}" && -f "$CUDA_HOME/include/nvtx3/nvToolsExt.h" ]]; then nvtx_include=(-isystem "$CUDA_HOME/include"); fi
build_backend() {
local link_flags=(-fPIC -shared)
if [[ "${3:-shared}" == executable ]]; then link_flags=(); fi
g++ -std=c++17 -O2 "${link_flags[@]}" -Wall -Wextra -fno-fast-math -pthread \
    -isystem "$reactive_prefix/include" -isystem "$reactive_prefix/include/eigen3" "${nvtx_include[@]}" \
    "$1" \
    -L"$reactive_prefix/lib" -Wl,-rpath,"$reactive_prefix/lib" \
    -lcantera -lfmt -lpthread -lcrypto -lsundials_cvode -lsundials_nvecserial \
    -lsundials_sunmatrixdense -lsundials_sunlinsoldense -lsundials_sunlinsolspgmr -lsundials_core \
    -o "$2"
}
build_backend "$project_root/src/reactiveThermo/reactiveThermo.cpp" "$project_root/lib/libreactiveBackend.so"
if [[ "${REACTIVE_CACHE_TEST:-0}" == 1 ]]; then
    build_backend "$project_root/tools/test_reactive_sparse_cache.cpp" "$project_root/lib/libreactiveCacheTest.so"
fi

if [[ "${REACTIVE_RF21_CONTRACT_TEST:-0}" == 1 ]]; then
    mkdir -p "$project_root/bin"
    build_backend "$project_root/tools/check_real_fluid_v21_contract.cpp" "$project_root/bin/check-real-fluid-v21-contract" executable
fi
