# HEM 활용도 최적화 — 결과 자료

[해석 보고서](../../../reports/hem-utilization-optimization-20260924.ko.md) ·
[수치 및 검증 집계](summary.json) · [비교 그림](optimization.png)

서로 다른 계측 범위의 시간을 합치거나 속도 비율로 비교하지 않는다.

| 폴더 | 측정 범위 |
|---|---|
| `batch*-40bar` | 같은 1 μs 체크포인트에서 한 스텝, 배치 크기만 변경, Nsight Systems+GPU 지표 |
| `reuse*-40bar` | 같은 한 스텝, 완전한 capillary 입력 키로 중복 계산 제거, 같은 계측 |
| `ncu-*` | 대표 HEM 호출의 Nsight Compute replay; 전체 실행 시간 비교에 사용하지 않음 |
| `full-baseline-*`, `full-optimized-*` | 같은 primitive 입력부터 1 μs까지 새로 실행한 맞춤 비교; NVML 모니터만 사용 |

`summary.json` 생성 시 아래를 대조한다.

- 모든 실행의 물리 옵션·CUDA 경로·CPU 폴백 0·보존량 검증 통과
- 중간 여섯 설정의 최종 checkpoint SHA 동일
- 배치 크기만 바꾼 네 설정의 HEM 작업 카운터 동일
- 환경별 전체 기준/최적화 실행의 바이너리·스텝 수·종료 시간·checkpoint SHA 동일
- 각 쌍의 초기 입력 및 적응 Δt·시각·재시도 이력 동일, 물리 설정 차이 없음
- 두 추가 NCU 실행과 정확 키 GPU 회귀 통과

정확 키 회귀는 [capillary-reuse-validation.json](capillary-reuse-validation.json),
기존 곡면 flashing 검증은 [capillary-flash-validation.log](capillary-flash-validation.log)에 있다.
`optimization-validation.json`에는 전체 로그 해시와 물리 잔차,
`benchmark-definition.json`에는 입력 파일 해시,
`profiler-command.json`에는 실제 계측 명령·바이너리 해시가 있다.

원시 메시·체크포인트·Nsight 파일은 다음 로컬 경로에 보존한다.

```text
/home/jsw/문서/analysis/runs/hem_utilization_optimization_20260924
```

저장소 루트에서 실행 도구를 사용한다. 환경:

```bash
export PINTLE_SPUMA_ENV=/home/jsw/cae-gpu-pr1-compatible/env.sh
HEM_PY=/home/jsw/문서/analysis/spuma-multiphysics/research/reactive-env/bin/python
HEM_SOURCE=/home/jsw/문서/analysis/runs/impinging_n2o_cube80_z40_counters_1us_20260923/n160/full-physics/ambient_40bar_abs
```

새 경로에서 한 스텝을 재현하는 예시:

```bash
"$HEM_PY" tools/benchmark_hem_optimization.py "$HEM_SOURCE" \
  --output /tmp/hem-optimization-new-run --batch 1048576 --reuse true
```

전체 0→1 μs는 `--fresh`, Nsight Systems 계측은 `--nsys /절대경로/nsys`를 추가한다.
기준은 `--batch 4096 --reuse false`다. 각 도구는 GPU 공용 실행 잠금을 사용하며
기존 실행 결과 폴더를 덮어쓰지 않는다.

NCU는 `tools/profile_reactive_kernels.py`의 `--batch 1048576 --reuse true --hem-only`
옵션을 사용한다. `--ncu`와 새 `--output` 경로가 필요하다. 실제 사용한 모든 옵션은
각 실행의 명령 JSON을 참조한다. NCU kernel replay 메모리의 호스트 백업은
프로파일러 동작이며 솔버 물리의 CPU 폴백을 뜻하지 않는다.

[구현 이력](implementation-provenance.json)에 기준 커밋, 수정된 소스/도구 및 실제 바이너리의
SHA256을 기록했다. [솔버 변경분](solver-changes.patch)과 빌드 로그도 보존했다.
이번 변경은 로컬 작업 트리에 반영된 상태다.

GPU를 다시 실행하지 않고 자료를 재집계하는 명령:

```bash
"$HEM_PY" tools/summarize_hem_optimization.py \
  /home/jsw/문서/analysis/runs/hem_utilization_optimization_20260924 \
  --output /tmp/hem-optimization-review
```

새 충돌 벤치마크는 `tools/prepare_impinging_resolution_series.py`가 큰 배치와 정확 재사용을
기본으로 설정한다. 기준 생성은 `--thermo-batch-cells 4096 --disable-thermo-exact-reuse`를 쓴다.
개별 CUDA ALU 코어 사용률이나 캐시 상주 바이트를 측정한 자료는 아니다.
