# 정수 정밀도와 GPU 실행 간격 실험

이 모듈은 FP64 기본 솔버와 별도로 선택하는 실험이다. MAIN이 구현·빌드·실행 검증했고, 서브에이전트는 자료 조사와 읽기 전용 검토를 맡았다. 결과와 적용 범위는 [실험 보고서](reports/integer-gpu-exploration.md)에 기록한다.

## 정수 CSR 실험

```bash
source ./env.sh
bash tools/build_integer_lab.sh
python3 tools/analyze_integer_exponents.py
flock -n /home/jsw/cae-benchmark/run.lock \
  bin/pintleIntegerLab benchmarks/precision-pressure-snapshot1/pressure.csr \
  reports/integer-pressure-new.json
```

실제 압력·온도 행렬 스냅샷은 별도의 대용량 산출물이다. 저장소에 포함하지 않는다. 생성 방법은 [기존 정밀도 실험 설명](README.precision.ko.md)의 `tools/snapshot_precision.py`를 참고한다.

정수 표현은 `a_ij ≈ qA_ij × 2^(eA_i−F)`, `x_j ≈ qX_j × 2^(eX−F)`다. `eA`는 행마다 최대 계수로 결정하고, `eX`는 전체 x의 최대값으로 결정한다. 기본 실험은 다음 두 방식이다.

- **INT32 입력, F=30:** INT64에 곱과 합을 정확히 누적한다.
- **INT64 입력, F=62:** 입력 한 개가 두 INT32 레지스터에 해당하며, 곱·합은 128비트 누산기를 사용한다. 누산기에는 네 INT32가 필요하다.

원래 FP64 문제와의 오차는 입력 양자화와 별개로 평가한다. RHS b도 곱과 같은 스케일에 반올림해 잔차를 정수로 계산한다. `|b_q| + Σ|qA qX|`를 CPU의 넓은 정수로 검사하고, signed 누산기 용량을 넘으면 GPU 누적 전에 실패한다. 조용한 overflow나 saturation은 허용하지 않는다. GPU limb 누적을 모든 행에서 독립 CPU `__int128` 계산과 대조하고, 최종 FP64 변환도 검사한다. 원래 FP64 행렬의 오차 기준은 32,768행의 113비트 계산이다.

이 구현은 정상 범위의 유한 FP64 출력만 허용한다. 부호 있는 0, NaN, 무한대, 전체 FP64 지수 범위를 재현하는 소프트웨어 IEEE FP64 구현이 아니다. 임의 크기 행렬용 정수 솔버도 아니다. 행당 최대 32개 원소를 지원하며, 현재 실제 행렬의 최대는 19개다.

CUDA event는 5개 round를 사용한다. 각 round의 packing은 3회, SpMV/잔차는 20회다. 행별 지수 결정은 packing 시간에 포함되지만, CPU의 전역 x 지수 검색·전송·할당·검증 시간은 제외된다. 정수 경로 출력은 FP64이고, FP 경로 출력은 해당 native 형식이다. `reverse`를 마지막 인자로 주면 부동소수점 비교를 정수 경로 뒤에서 수행한다.

INT32 저장값을 FP32로 변환해 FMA하는 hybrid SpMV도 측정한다. 이는 FP32 산술이며, 별도의 공짜 정수 연산 자원을 추가하는 방식은 아니다. 실제 CFD에 정수 내부 반복을 연결하지는 않았다.

## 선택형 비동기 Gauss–Seidel smoother

```bash
source ./env.sh
wmake -j1 libso src/asyncSmoother
python3 tools/benchmark_async.py --output benchmarks/async-new --steps 12 --repeats 4
```

벤치마크는 원본 case를 매번 새로 복사한다. 원본 case는 실행하거나 수정하지 않는다. 실행 순서는 B→A, A→B를 반복하고, 처음 2스텝과 마지막 저장 스텝은 시간 집계에서 제외한다. 12스텝이면 run당 9개 스텝을 측정한다.

직접 선택할 때는 `controlDict`의 기존 `libs`에 `libpintleAsyncSmoother.so`를 추가하고, 대상 solver의 `smoother`를 `pintleAsyncGaussSeidel`로 설정한다. 그래프 색상 설정은 기존 `multicolorGaussSeidel` 하위 dictionary를 계속 읽는다.

색상 의존성은 같은 default CUDA stream의 실행 순서로 유지한다. 기본 구현은 다음 경계를 사용한다.

- 단일 프로세스, 활성 matrix interface 없음, pooled 우변, 해 벡터와 다른 우변: 우변 복사를 생략하고 모든 sweep을 enqueue한 뒤 한 번 완료를 기다린다.
- 그 밖의 경우: 기존 `bPrime` 복사와 interface 갱신을 보존하고, 각 sweep 끝에서 완료를 기다린다.

`pintleDirectSource false;`로 우변 복사 생략을 끌 수 있다. 공용 SPUMA executor와 설치된 원본 라이브러리는 수정하지 않는다. reduction, 경계조건, MPI의 동기화 계약도 유지한다. 현재 실측은 단일 GPU·고정 mesh·연결 경계 없는 Pintle case에 한정한다.

```bash
python3 tools/profile_async.py --mode baseline --output benchmarks/profile-async-B-new
python3 tools/profile_async.py --mode async --output benchmarks/profile-async-A-new
python3 tools/summarize_async_trace.py \
  benchmarks/profile-async-B-new/trace.sqlite \
  benchmarks/profile-async-A-new/trace.sqlite \
  --output reports/async-trace-new.json
python3 tools/profile_async.py --memcheck --output benchmarks/async-memcheck-new
```

프로파일은 시작 20초 후 10초를 추적하며, 프로파일 실행 시간은 성능 결과에 포함하지 않는다. 메모리 검사는 `pintleAsync` 커널과 명시적 CUDA API 오류를 대상으로 한다. 전체 SPUMA race 검사를 대신하지 않는다. 파일별 실제 실행 기록·바이너리 해시·성능 제한은 보고서를 참고한다.
