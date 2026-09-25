#!/usr/bin/env bash
# usage: bench500.sh NAME BUILD_DIR STEPS [reactiveProperties overrides...]
set -euo pipefail
name=$1; build=$2; steps=$3; shift 3
P=/home/jsw/문서/analysis/runs/n160_40bar_200to500us_gpu_20260925/production/40bar/to-500us
root=/home/jsw/문서/analysis/runs/gpu_opt_validation_20260925
case=$root/$name
[[ -e $case ]] && { echo "exists: $case" >&2; exit 1; }
mkdir -p "$case/constant"
cp -a "$P/0.0005" "$case/"; chmod -R u+w "$case/0.0005"
cp -a "$P/system" "$case/"; chmod -R u+w "$case/system"
cp -P -r "$P/constant/polyMesh" "$case/constant/"
sed "s#^closureLibrary .*#closureLibrary \"$build/lib/libreactiveTransport.so\";#" "$P/constant/reactiveProperties" > "$case/constant/reactiveProperties"
for override in "$@"; do echo "$override" >> "$case/constant/reactiveProperties"; done
sed -i -e 's/startTime [0-9.e-]*;/startTime 0.0005;/' -e 's/endTime [0-9.e-]*;/endTime 0.0006;/' \
    -e "s/maxAcceptedSteps [0-9]*;/maxAcceptedSteps $steps;/" "$case/system/controlDict"
cd /home/jsw/문서/analysis/spuma-multiphysics-gpu-opt
export REACTIVE_SPUMA_ENV=/home/jsw/cae-gpu-pr1-compatible/env.sh
set +eu; source ./env.sh >/dev/null 2>&1; set -eu
export LD_LIBRARY_PATH="$build/lib:$LD_LIBRARY_PATH"
exec 9>/home/jsw/cae-benchmark/run.lock; flock 9
nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,clocks.sm --format=csv,noheader,nounits -lms 100 > "$case/nvml.csv" &
smi=$!
trap 'kill $smi 2>/dev/null || true' EXIT
stdbuf -oL "$build/bin/ReactiveFoam" -case "$case" 2>&1 | python3 -u -c 'import sys,time
for l in sys.stdin: sys.stdout.write(f"{time.time_ns()} {l}")' > "$case/solver.stamped.log"
cut -d' ' -f2- "$case/solver.stamped.log" > "$case/solver.log"
echo "case=$case"
