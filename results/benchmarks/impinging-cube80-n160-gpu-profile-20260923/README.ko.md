# 160³ N₂O 충돌 GPU 계측 자료

대기압 및 40 bar(abs), 각 1 μs 완료. 측정 2026-09-23~24.
**프로파일러가 켜진 진단 실행이며 이전 비계측 벤치마크와 시간 배율을 비교하지 않는다.**
물리·정밀도·솔버 바이너리는 변경하지 않았다.

- [해석 보고서](../../../reports/impinging-cube80-n160-gpu-profile-20260924.ko.md)
- [벤치마크 요약](transient/summary.json), [검증 기록](transient/validation.json)
- [최종 자료 대조 기록](artifact-validation.json)
- [진단 그림](diagnosis.png)

## 파일 구성

| 경로 | 내용 |
|---|---|
| `transient/` | 물리·시간 간격·단계별 시간, 솔버 로그, 메시 검사, 입력 정의, 바이너리 해시 |
| `ambient_1atm/`, `ambient_40bar_abs/` | Nsight Systems 커널/API/복사/할당, 1 kHz 하드웨어 지표, HEM 배치 분포, NVML·CPU 샘플, 타임라인 |
| `ncu-ambient_1atm/`, `ncu-ambient_40bar_abs/` | 환경별 대표 21개 커널 호출의 Nsight Compute 카운터·명령·검증 |
| `ncu-wale_1atm/`, `ncu-wale_40bar/` | 준비 호출 이후 실제 WALE PR 물성 호출의 추가 계측, 각 1개 |
| `device.json`, `resources.txt`, `static-kernels.json` | 장치 속성, 커널 레지스터·스택 및 정적 명령 자료 |
| `permission-probe.log` | GPU 성능 카운터 접근 성공 확인 |

`all-counters.json`은 Nsight Compute의 원시 값과 단위를 보존한다.
`kernel-summary.csv/json`은 선택 지표와 FLOP·sector 처리량의 파생값이다.
느린 HEM L2 적중률의 100% 초과 원시값은 replay 간 변동을 포함하므로
유효 확률로 해석하지 않는다. 원시값을 임의로 잘라내지 않았다.

NCU는 최종 1 μs 체크포인트에서 한 스텝을 이어서 계산한 별도 실행이다.
원래 마지막 스텝의 동일 입력을 replay한 수치나 전체 실행의 가중 평균이 아니다.
44개 계측 호출의 시간은 주 벤치마크 시간에 합산하지 않는다.
SM active·활성 SM 기준 occupancy·파이프 활성률은 각각 분모가 다르다.
캐시 sector 처리량은 캐시 용량 점유율과 다르다.

## 원본과 재현

전체 메시·체크포인트·`.nsys-rep`·SQLite·`.ncu-rep`는 다음 로컬 경로에 보존했다.

```text
/home/jsw/문서/analysis/runs/impinging_n2o_cube80_z40_counters_1us_20260923
  n160/full-physics/ambient_1atm
  n160/full-physics/ambient_40bar_abs
  ncu/ambient_1atm
  ncu/ambient_40bar_abs
  ncu/wale_1atm
  ncu/wale_40bar
```

환경은 `/home/jsw/cae-gpu-pr1-compatible/env.sh`, Python 및 물성 의존성은
`/home/jsw/문서/analysis/spuma-multiphysics/research/reactive-env`다.
실제 프로파일러 옵션은 각 `profiler-command.json`에 보존했다.
NCU 입력 파일 해시는 `counter-profile-definition.json`에 있다.
권한 변경 전 사용자가 중단한 `impinging_n2o_cube80_z40_profiled_1us_20260923`
실행은 이번 완료 결과에 포함하지 않는다.

아래는 저장된 자료만 재집계하는 예시다. 솔버를 다시 실행하지 않는다.
저장소 루트에서 실행하며, 출력은 별도 `/tmp` 폴더를 쓴다.

```bash
PROFILE_RUN=/home/jsw/문서/analysis/runs/impinging_n2o_cube80_z40_counters_1us_20260923
PROFILE_PY=/home/jsw/문서/analysis/spuma-multiphysics/research/reactive-env/bin/python

"$PROFILE_PY" tools/summarize_impinging_transient.py \
  "$PROFILE_RUN/profiled-validation.json" --output /tmp/reactive-profile-review/transient

"$PROFILE_PY" tools/analyze_gpu_trace.py \
  "$PROFILE_RUN/n160/full-physics/ambient_1atm" \
  "$PROFILE_RUN/n160/full-physics/ambient_40bar_abs" \
  --output /tmp/reactive-profile-review

"$PROFILE_PY" tools/summarize_ncu_counters.py \
  "$PROFILE_RUN/ncu/ambient_1atm/counters.csv" \
  --output /tmp/reactive-profile-review/ncu-ambient_1atm
```

새 계측 실행용 도구는 `tools/profile_impinging_gpu.py`와
`tools/profile_reactive_kernels.py`다. 기존 결과 경로를 재사용하지 않고
새 케이스/출력 경로를 지정한다. GPU 계측 도구끼리는 동시에 실행하지 않는다.

솔버 SHA256: `4f59195d821988335697c8a515a55ffc960da7442f77f91dadd8d1b803b44858`.
전체 바이너리 및 분석 도구 해시는 최종 자료 대조 기록에 있다.
