# 현재 솔버 정밀도 실험 자료

보고서: [현재 ReactiveFoam FP32·정수 실험](../../../reports/current-solver-precision-20260924.ko.md).

`summary.json`은 **21개 배치 replay, 6개 전체 계산, 12개 NCU counter capture, 4개 전체 체크포인트 비교**를 집계한다. strict FP32 NASA의 실패 replay도 실패로 보존하며, 중단된 1기압 실행은 `aborted.json`에 기록했다. 실패 상태를 물리 오차로 평가하지 않는다. 예전 CSR 실험과 중간 v1 파일은 집계에 포함하지 않았다.

- `replay-v2-*`: 현재 솔버에서 추출한 같은 입력, FP64 대조군과 네 변환. 워밍업 1회+측정 7회.
- `replay-*-nasa`: FP32 또는 2×FP32 NASA 다항식, 기존 허용오차.
- `relaxed-*`: 명시적으로 허용오차만 바꾼 입력 복사본. 워밍업 1회+측정 3회. 같은 tolerance 쌍의 입력 SHA256 일치를 집계 시 확인한다.
- `full-*`: 160³, 같은 초기 primitive 필드, 같은 물리 옵션, 0→1 μs. Nsight 없이 NVML만 사용했다. 각각 1회 측정이다.
- `ncu-*`: 별도 kernel replay 계측. 대표 HEM 호출의 파이프·명령·점유율·캐시·메모리·스케줄러·SM 분포 지표. 이 실행의 프로세스 시간을 전체 속도 비교에 쓰지 않는다.
- `comparison-*`: 전체 q/state 비교 및 N₂O가 있는 셀만의 오차, 상 선택, 적응 Δt·재시도 이력, 총 보존량. 비교 도구는 물리·초기 입력·시간 설정 및 실행 바이너리가 같고 HEM 라이브러리만 다른지 확인한다.
- `variants-*/build-*.log`: NVCC/ptxas 레지스터·스택·spill 기록. 빌드 옵션은 `tools/build_hem_precision_variants.sh`에 있다.

자료 원본은 `/home/jsw/문서/analysis/runs/precision_revisit_20260924`다. 큰 메시, 체크포인트, `.npy`, `.ncu-rep`, `.so`, 실제 입력 `hem-*.bin`은 원본 폴더에 남겼다. 이 폴더에는 소형 검증·로그·CSV/JSON과 도표를 보관했다. 배치 입력과 라이브러리 SHA256은 `summary.json`, 전체 실행 바이너리는 각 `profiler-command.json` 및 runtime manifest에 있다.

`benchmark-definition.json`의 모델/수치 해시는 원본 케이스의 정의를 담고 있다. **실제로 로드된 수치 정책은 `solver.log`의 `REACTIVE_RUNTIME manifest`, 집계의 `full[].runtime`을 기준으로 확인한다.** 다른 정밀도 정책의 체크포인트 해시를 바꿔 읽지 않았다. 전체 비교는 항상 새 primitive 초기화에서 시작한다.

기본 산술은 FP64다. 실험용 경로는 HEM만 교체한다. 엄격한 FP32 NASA는 회귀 실패가 있어 production 설정에 사용하지 않는다. 같은 허용오차의 FP64보다 느린 완화 FP32 NASA도 채택하지 않았다.

## 재현 예시

저장소에서 아래를 실행한다. 새 출력 디렉터리가 필요하며 기존 측정값을 덮어쓰지 않는다. CUDA 실행은 한 번에 하나씩 수행한다.

```bash
cd /home/jsw/문서/analysis/spuma-multiphysics-spray-physics
export PINTLE_SPUMA_ENV=/home/jsw/cae-gpu-pr1-compatible/env.sh
export PINTLE_REACTIVE_PREFIX=/home/jsw/문서/analysis/spuma-multiphysics/research/reactive-env
precision_python="$PINTLE_REACTIVE_PREFIX/bin/python"
precision_campaign=/home/jsw/문서/analysis/runs/precision_revisit_20260924

# NVCC 경로가 설정된 환경에서 새 sidecar 라이브러리 빌드
tools/build_hem_precision_variants.sh /tmp/reactive-precision-new fp64 fp32-linear int32-linear

# 실제 입력 배치에서 기준을 만든 뒤 동일 입력 후보 비교
flock /home/jsw/cae-benchmark/run.lock "$precision_python" tools/replay_hem_precision.py \
  "$precision_campaign/hem-1atm.bin" --library /tmp/reactive-precision-new/libHem-fp64.so \
  --output /tmp/reactive-replay-fp64 --repeats 7
flock /home/jsw/cae-benchmark/run.lock "$precision_python" tools/replay_hem_precision.py \
  "$precision_campaign/hem-1atm.bin" --library /tmp/reactive-precision-new/libHem-fp32-linear.so \
  --output /tmp/reactive-replay-fp32 --reference /tmp/reactive-replay-fp64 --repeats 7

# 전체 계산: 도구 내부에서 GPU lock을 잡는다.
"$precision_python" tools/benchmark_hem_optimization.py \
  /home/jsw/문서/analysis/runs/impinging_n2o_cube80_z40_counters_1us_20260923/n160/full-physics/ambient_1atm \
  --output /tmp/reactive-full-fp32 --batch 1048576 --reuse true --fresh \
  --closure-library /tmp/reactive-precision-new/libHem-fp32-linear.so

# 별도 상세 계측: 동일 입력의 두 번째 HEM launch를 NCU kernel replay한다.
"$precision_python" tools/profile_hem_precision.py "$precision_campaign/hem-1atm.bin" \
  --library /tmp/reactive-precision-new/libHem-fp32-linear.so \
  --ncu /home/jsw/nvidia/hpc_sdk/Linux_x86_64/26.5/profilers/13.2/Nsight_Compute/ncu \
  --output /tmp/reactive-ncu-fp32
```

배치 입력 파일의 ABI는 같은 소스의 `Model`, `PintleThermoState`, 계면 구조체에 종속된다. 다른 ABI 버전에서는 현재 솔버에서 다시 캡처한다. `hem_capture.cpp`는 실제 HEM 호출을 저장하고 원래 라이브러리로 그대로 전달하는 도구이며, 모델·q·에너지·seed·계면 입력을 기록한다.

공학적 비교 screen은 p/ρ/U scaled 차이 ≤10⁻³, |ΔT|≤0.1 K, |Δα|≤10⁻⁴와 기존 보존·상 상태·GPU 검증이다. 이 기준은 이번 수치 실험의 선별 기준이며 장시간 충돌·분열의 물리 검증을 대체하지 않는다.
