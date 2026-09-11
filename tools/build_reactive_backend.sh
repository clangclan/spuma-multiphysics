#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
reactive_prefix="${PINTLE_REACTIVE_PREFIX:-$project_root/research/reactive-env}"
mkdir -p "$project_root/lib" "$project_root/logs"
test -f "$reactive_prefix/include/cantera/thermo/PengRobinson.h"
# The caller holds the shared run lock, including when this is part of a larger
# validation campaign. Host-only Cantera calls are isolated behind a C ABI.
g++ -std=c++17 -O2 -fPIC -shared -Wall -Wextra -fno-fast-math \
    -isystem "$reactive_prefix/include" -isystem "$reactive_prefix/include/eigen3" \
    "$project_root/src/reactiveThermo/pintleReactiveThermo.cpp" \
    -L"$reactive_prefix/lib" -Wl,-rpath,"$reactive_prefix/lib" \
    -lcantera -lfmt -lpthread -lcrypto -lsundials_cvode -lsundials_nvecserial \
    -lsundials_sunmatrixdense -lsundials_sunlinsoldense -lsundials_core \
    -o "$project_root/lib/libpintleReactiveBackend.so"
