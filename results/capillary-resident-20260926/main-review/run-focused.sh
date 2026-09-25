#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
out="$repo/results/capillary-resident-20260926/main-review"
review_python="${REACTIVE_PREFIX:-/home/jsw/문서/analysis/spuma-multiphysics/research/reactive-env}/bin/python"
cuda_root=/home/jsw/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2
cd "$repo/examples/impinging-n2o-supercooled"
"$review_python" "$repo/tools/validate_capillary_flash.py" \
  --configuration "$PWD/cold-pr-148K-config.yaml" --library "$repo/lib/libreactiveBackend.so" \
  --cuda-library "$repo/lib/libreactiveTransport.so" > "$out/curved-flash.json"
cd "$repo"
for api in v1 v2; do
  "$review_python" tools/validate_wale_pr_epochs.py \
    --configuration "$repo/examples/impinging-n2o-supercooled/cold-pr-148K-config.yaml" \
    --backend "$repo/lib/libreactiveBackend.so" --cuda-library "$repo/lib/libreactiveTransport.so" \
    --output "$out/wale-$api.json" --api "$api" > "$out/wale-$api.log"
done
"$review_python" tools/validate_reactive_transport.py --library "$repo/lib/libreactiveTransport.so" \
  --backend cuda --output "$out/cuda-transport.json" > "$out/cuda-transport.log"
"$cuda_root/compute-sanitizer/compute-sanitizer" --tool memcheck --error-exitcode 86 \
  "$review_python" tools/validate_capillary_resident.py --library "$repo/lib/libreactiveTransport.so" \
  --cudart "$cuda_root/targets/x86_64-linux/lib/libcudart.so.13.2.75" \
  --output "$out/resident-memcheck.json" > "$out/resident-memcheck.log" 2>&1
