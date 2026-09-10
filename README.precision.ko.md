# 컨슈머 GPU의 FP64 대체 실험

RTX 5080에서 FP32, 2개의 FP32 확장수(DS), 3개의 FP32 확장수(TS)를 직접 구현하고 실제 Pintle 행렬과 온도 방정식에서 비교한다. [측정·정확도 결과](reports/precision-exploration.md)를 함께 읽어야 한다. 기존 FP64 솔버의 기본 설정은 그대로이며 이 문서의 모듈은 별도로 선택하는 실험 기능이다.

## 구현 범위

- `src/precisionLab/floatExpansion.cuh`: `x=hi+lo`와 `x=hi+mid+lo`를 나타내는 확장수의 덧셈·뺄셈·곱셈. TwoSum, FMA 기반 TwoProd, 명시적 RN 연산을 사용한다. TS는 Fabiano–Muller–Picot의 Algorithms 4/5/8/9를 구현했다.
- `src/precisionLab/precisionLab.cu`: 실제 CSR의 `Ax`와 `b−Ax`, 계산 위주 recurrence, 원래 입력과 변환된 입력을 구분한 113비트 기준 검산. 전체 행렬의 보조 기준은 host `long double`이며, 113비트 행렬 검산은 결정론적으로 선택한 32,768행이다.
- `src/precisionProbe`: 기존 풀이를 호출한 뒤 첫 번째 선형계를 내보내는 진단 모듈. 본 성능 측정에는 사용하지 않는다.
- `src/precisionSolver`: 온도 증분 선형계의 내부 Jacobi를 FP64/FP32/DS/TS 중 하나로 수행한다. 원래 RHS, 누적 보정, 실제 잔차와 수렴 판정은 FP64다. 수렴 실패나 유효하지 않은 변환이 있으면 원래 FP64 `smoothSolver`로 다시 풀고 실제 잔차를 재검사한다.

DS/TS는 IEEE FP64의 완전한 구현이 아니다. FP32의 지수 범위를 확장하지 않고, IEEE 예외·부호 있는 0·올바른 반올림의 모든 경우를 보장하지 않는다. 나눗셈·제곱근·초월함수는 구현하지 않았다. 온도 대각 역수는 매 solve의 준비 단계에서 FP64로 구한 뒤 선택 정밀도로 변환한다. 현재 범위는 FP64/Int32 ABI, 하나의 프로세스, 비결합 경계를 쓰는 고정 Pintle mesh다.

## 재현

기존 프로젝트를 먼저 빌드하고 `cases/pintle45us`가 준비되어 있어야 한다. 각 명령의 출력 디렉터리는 새 경로여야 한다. 벤치마크 도구는 원본을 실행하지 않고 복사하며 공용 실행 lock을 사용한다. 이 구현에 맞는 `sm_120` 빌드다.

```bash
cd /home/jsw/문서/analysis/spuma-multiphysics
bash tools/build_precision_lab.sh
bash tools/build_precision_solver.sh
source ./env.sh

# 원래 압력 풀이 이후 실제 행렬을 추출한다.
python3 tools/snapshot_precision.py --field pressure \
  --output benchmarks/precision-pressure-new
bin/pintlePrecisionLab \
  benchmarks/precision-pressure-new/pressure.csr \
  benchmarks/precision-pressure-new/lab.json

# 별도 4스텝 실행에서 원래 온도 선형계의 잔차를 점검한다.
python3 tools/benchmark_precision.py \
  --precision baseline fp64 fp32 ds ts --steps 4 --repeats 1 \
  --diagnostics --output benchmarks/precision-pilot-new

# 동일 허용오차, 45 μs 재시작, dt=30 ns. 첫 2스텝과 출력 스텝을 제외한다.
python3 tools/benchmark_precision.py \
  --precision baseline fp64 fp32 ds ts --steps 12 --repeats 3 \
  --output benchmarks/precision-full-new

python3 tools/summarize_precision.py benchmarks/precision-full-new/benchmark.json \
  --lab benchmarks/precision-pressure-new/lab.json \
  --pilot benchmarks/precision-pilot-new/benchmark.json \
  --output reports/precision-new.json

# 실제 mesh에서 강제 fallback과 FP32 내부 커널의 선택적 메모리 검사를 한다.
python3 tools/check_precision_fallback.py \
  --output benchmarks/precision-fallback-new
```

`baseline`은 현재의 FP64 multicolor 온도 증분 솔버다. `fp64`는 새 Jacobi·잔차보정 알고리즘을 FP64로 실행하는 대조군이다. 둘을 구분해야 알고리즘 변경의 효과를 FP32의 효과로 잘못 해석하지 않는다. `--sweeps`의 기본값은 6이며 각 보정 뒤 실제 FP64 잔차를 평가한다. `PINTLE_IR` 로그에 보정 횟수, 실제 내부 sweep 수, 최종 잔차와 fallback 여부가 남는다.

## 개별 케이스에서 선택

복사한 케이스의 `controlDict.libs`에 `"libpintleMixedTemperature.so"`를 추가하고, `fvSolution.solvers`의 `T`와 `TFinal` 항목에서 다음 키를 설정한다. 기존 smoother와 허용오차 설정은 유지한다.

```text
solver          pintleMixedTemperature;
innerPrecision  fp32;  // fp64, ds, ts도 가능
innerSweeps     6;
maxRefinements  8;
diagnostics     no;
```

이 실험은 온도 내부 반복만 선택 정밀도로 바꾼다. 압력·보존량·체적분율·EOS 전체를 낮은 정밀도로 바꾸는 기능은 아니다. 짧은 체크포인트 실험에서의 수렴·성능 결과를 긴 물리 시간, 다른 초기 조건, 다중 GPU/MPI 또는 원본 Foundation 14와의 동등성으로 확대하지 않는다.
