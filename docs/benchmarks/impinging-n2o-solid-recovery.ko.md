# 저온 복원과 고체 N₂O 생성·승화 — 2026-09-22

**과거 프로필:** 사용자가 고체·승화를 제외하고 과냉각 액체로 처리하도록 변경했다. 현재 입력과 검증은 [과냉각 액체 전환 보고서](impinging-n2o-supercooled.ko.md)를 따른다. 아래 고체 모델과 미달 검증 결과는 비교 근거로 보존하며 현재 벤치마크에는 사용하지 않는다.

기존 182.34 K 하한에서 멈추던 문제에 대해 탐색 초기화 수정과 저온 기체·고체 N₂O 상평형을 구현했다. 두 256k 케이스가 실제 GPU에서 각각 10스텝을 재시도·CPU fallback 없이 완료했다. **수치 실행 검증은 통과했지만, 문헌 승화압과 최대 9.39% 차이가 있어 8% 물성 정확도 기준은 미달이다.** 고체 모델은 연구용 잠정 구현으로 유지한다.

## 원인과 수정

초기 대기압 실패 대표 상태 5개 중 3개는 기존 기체 상태의 온도에서 액체 후보 탐색을 시작해 올바른 상평형 해를 놓쳤다. 동일 보존 질량·에너지에서 유효한 액체 온도 영역을 시작점으로 삼으면 해가 존재했다. CPU와 CUDA에 같은 유한한 온도 재시작 목록을 추가했다. 기존 후보가 모두 실패한 경우에만 적용하며, 전역 해 탐색을 보장하는 알고리즘은 아니다. 원래 성공하던 경로는 그대로 유지한다.

그 수정만 적용한 유체 전용 연속 계산은 대기압에서 7스텝 이후 182.34 K 물성 경계에 도달해 실패했다. 이는 초기화만의 문제가 아니므로 N₂/O₂/N₂O 기체 열량 물성을 148 K까지 확장하고, 액체·고체·기체의 화학퍼텐셜을 비교하는 상평형에 고체 N₂O를 연결했다. **온도 clipping이나 질량·에너지 수정으로 하한을 우회하지 않는다.** 원래 유체 프로필은 종전 물성 하한과 물리 해시를 유지한다.

고체 생성과 승화는 평형 상분배로 계산한다. 냉각하면 기체에서 고체로 질량이 이동하고 가열하면 고체가 승화한다. 잠열은 각 상의 h/e/s에서 보존 에너지에 결합한다. 유한속도 핵생성·입자 추적·고체 입자 운동·슬립·벽 부착은 포함하지 않는다.

## 물성 및 적용 범위

고체 Cp는 Atake·Chihara의 실험 표를 구간별 선형 보간하고 h와 s를 해석적으로 적분한다. 기준 온도 182.408 K에서 PR 액체 h/s와 융해잠열 6516 J/mol로 고체 기준을 맞췄다. 승화압은 독립 문헌식과 비교하며 이 검증점에 맞춰 기준 상수를 보정하지 않았다. 출처는 [Atake·Chihara 논문](https://doi.org/10.1246/bcsj.47.2126)과 [Atake 학위논문 표·식](https://ir.library.osaka-u.ac.jp/repo/ouka/all/2122/02913_Dissertation.pdf), 저장소의 [물성·출처 JSON](../model-data/n2o-solid-atake1974.json)이다.

| 항목 | 구현 범위·제한 |
|---|---|
| 저온 기체 | N₂/O₂/N₂O에 148–182.34 K NASA9 구간 추가. CoolProp 6.7.0의 이상기체 Cp₀ 사용 |
| 기존 구간 | 모든 원래 계수와 IPA 계수 보존. 182.34 K에서 h/s 연속; Cp 접합 차이는 최대 0.694% |
| 고체 | 148–183 K, **총 절대압 ≤ 2 bar**, 기체와 공존하는 비압축성 고체 |
| 고체 Cp 상단 | 182–183 K는 181–182 K 기울기를 사용한 짧은 외삽 |
| 고체 부피 | 잠정 40 cm³/mol 고정. backend 분자량 44.013 kg/kmol 사용 시 밀도 1100.325 kg/m³ |
| 지원하지 않는 상태 | 전량 고체, 기체가 없는 액체·고체 혼합, 고압 고체 EOS, 반응성 고체, 148 K 미만 |
| 응축상 구성 | 슬롯 0=액체 N₂O, 슬롯 1=고체 N₂O. 기존 IPA 액상 슬롯을 대체 |
| IPA | 기체 종은 남지만 기존 열량 범위 아래에서 양의 IPA 질량은 거부 |
| 상평형 방법 | 일반 HEM 복원. 중복 종에 맞지 않는 기존 log-vapor `boundaryFallback` 설정은 거부 |

기체 Cp₀는 [CoolProp N₂O 물성 구현](https://coolprop.org/fluid_properties/fluids/NitrousOxide.html)을 사용했다. 희박 밀도 입력으로 이상기체 Cp₀만 평가했으며, 삼중점 아래 실제 유체 EOS나 포화선을 외삽한 것이 아니다. 전체 계수·접합·검증 결과는 [준비 manifest](../../results/n2o-solid-20260922/thermo-preparation.json)에 있다.

40 bar 환경 계산의 성공은 고압 고체의 정확도를 검증하지 않는다. 해당 계산의 최저 온도는 222.34 K였으며 고체가 필요한 영역으로 들어가지 않았다. 고체가 필요한 저온에서 2 bar를 넘으면 복원을 명시적으로 거부한다.

## 상태 저장과 GPU

`PintleThermoState`의 176바이트 레이아웃과 schema-3 체크포인트는 유지한다. 기존 `liquidMass[2]`, `alphaLiquid[2]`는 ABI 이름을 보존하고, 메타데이터로 각 슬롯의 액체/고체 종류와 종을 구분한다. 같은 N₂O 보존 질량에서 액체와 고체를 모두 차감하며 합계가 보존 질량을 넘는 입력은 거부한다. 기상 N₂O 분석은 `q_N2O - massLiquid.N2O - massSolid.N2O`를 사용한다.

출력은 `alphaLiquid.N2O`, `rhoLiquid.N2O`, `massLiquid.N2O`와 `alphaSolid.N2O`, `rhoSolid.N2O`, `massSolid.N2O`를 제공한다. `mass*` 필드는 셀 체적당 상 질량(kg/m³)이며, 적분 질량은 셀 체적을 곱한다. 기존 `alphaLiquid0/1`은 호환을 위해 유지하므로 고체 프로필의 `alphaLiquid1`은 실제로 고체 체적분율이다.

고체 물성 계산·안정성 검사·상평형 탐색은 CUDA에서도 수행한다. 기존 분석 Jacobian이 서로 다른 응축 종을 가정하므로 고체 프로필은 **GPU 내 유한차분 Jacobian**을 사용한다. CPU fallback과는 다르다. 기존 일반 유체 프로필의 분석 Jacobian은 유지한다. 순수 N₂O의 기체·액체·고체 공존점에서는 평형 음속 0을 허용하며, 수송/CFL에는 양의 동결 음속을 사용한다.

새 고체 물성과 저온 기구는 별도 물리 해시를 가진다. 기존 물성의 체크포인트를 새 프로필의 상태로 재해석하지 않는다. 예제 생성기가 새 초기 상태와 identity를 만든다.

## 실제 256k GPU 검증

각 케이스는 80×80×40 정육면체 메시, 0.25 mm 셀, 55 barg·293.15 K 공급을 유지했다. CUDA 수송과 CUDA HEM, WALE stress-v1을 사용하며 CPU fallback은 껐다. 각 환경을 한 번씩 10스텝으로 제한했고 장시간 충돌류는 계산하지 않았다.

| 환경 | 수용 스텝 | dt 범위 | 전체 스텝 최저 T | 재시도 / CPU fallback / GPU 실패 | 프로세스 실행 시간 |
|---|---:|---:|---:|---:|---:|
| 1 atm | 10 | 25.884–26.136 ns | **170.187 K** | 0 / 0 / 0 | **151.162 s** |
| 40 bar(abs) | 10 | 25.636–25.855 ns | 222.335 K | 0 / 0 / 0 | **65.786 s** |

최종 물리 시각은 각각 0.2610 μs, 0.2583 μs다. 질량·에너지·종·원소 보존 수지 기준 1e-8과 체크포인트·상 질량 검사를 통과했다. 전체 대기압 스텝 중 최대 에너지 수지 잔차는 5.80e-12다. 두 최종 체크포인트의 고체 질량은 0이며, 중간 스텝의 고체 총질량은 저장하지 않았다. 연속 계산의 저온 통과와 별도 0D 생성·승화 시험을 구분해 해석해야 한다.

dt는 30 ns 상한보다 작지만 실패 재시도로 줄어든 결과가 아니다. 0.25 mm 셀의 현재 음향·수송 CFL 제한이다. 기존 메시 품질 검사도 통과했다. 30–50 ns를 강제하거나 메시를 바꾸지 않았으며, CFL 완화·셀 크기 변경의 안정성 검증은 이번 작업에 포함하지 않았다.

실행 자료: [검증 JSON](../../results/n2o-solid-20260922/gpu-ten-steps.json), [대기압 스텝 로그](../../results/n2o-solid-20260922/ambient_1atm-step-evidence.log), [40 bar 스텝 로그](../../results/n2o-solid-20260922/ambient_40bar_abs-step-evidence.log). 전체 로그·체크포인트는 `logs/n2o-temperature-recovery-v1/solid-ten-steps`에 있다.

## 검증 결과와 남은 정확도 문제

- 기존 유체 복원 회귀검사 **388/388 통과**. [결과](../../results/n2o-solid-20260922/legacy-recovery.json)
- 기존 실패 대표 상태·CPU/CUDA 재생 및 해시 검사 **18/18 통과**. 유체 모델 범위를 벗어난 두 상태는 계속 거부하는 것이 기대 결과다. [결과](../../results/n2o-solid-20260922/fluid-recovery-analytic.json)
- 고체 검사 **수치 17/17 통과, 물성 참조 1개 미달**. h−e=p/ρ, Cp 미분, 융해 h/s, 삼상 공존, 냉각에 의한 고체 증가, 가열에 의한 승화, 범위·초과 질량 거부와 CPU/CUDA 일치를 검사했다. CPU/GPU 상태 차이는 최대 7.23e-15(스케일 정규화)다. [결과](../../results/n2o-solid-20260922/solid-unit.json)
- 고체 몰 부피 ±10% 민감도에서 동일 질량·에너지의 T 변화는 최대 0.00272 K, p 변화 1.35 Pa, 고체 질량 상대 변화 6.48e-6이었다. 고체 체적분율은 거의 ±10% 변하므로 부피 데이터 불확실성은 여전히 중요하다. Cp 표 36개 절점의 적분/미분도 검사했다. [결과](../../results/n2o-solid-20260922/density-sensitivity.json)

독립 승화압 문헌식에 대한 순수 N₂O 평형 압력 오차는 다음과 같다.

| 온도 | 모델 p | 문헌 p | 상대 오차 |
|---|---:|---:|---:|
| 150 K | 3228.37 Pa | 2951.32 Pa | **9.39%** |
| 160 K | 10502.79 Pa | 9770.97 Pa | 7.49% |
| 170 K | 29549.14 Pa | 28027.51 Pa | 5.43% |
| 180 K | 73845.69 Pa | 71340.42 Pa | 3.51% |
| 182 K | 87619.17 Pa | 84920.24 Pa | 3.18% |

검증기 기준 8%를 완화하지 않았다. `numericalPassed=true`, `sublimationReferencePassed=false`, 전체 `passed=false`를 남기고 종료 코드 1을 반환한다. PR 액체와 연동한 기준 h/s 및 저온 기체/고체 열량의 일관성을 추가 개선해야 물리 정확도 완료로 볼 수 있다. 데이터나 허용오차를 조정해 검증을 통과한 것으로 표시하지 않았다.

## 계산 비용

고체 후보와 삼상 탐색으로 계산량이 크게 증가했다. 대기압 한 스텝은 1.74–72.03 s, 40 bar는 1.04–9.00 s였다. 대기압 HEM GPU kernel 합계는 149.184 s로 프로세스 시간의 약 98.7%다. 기존 유체 전용 40 bar 10스텝의 12.589 s와 비교하면 이번 단일 실행은 약 5.23배 느리다. 반복 표본을 이용한 성능 통계는 아니다.

고체 프로필의 유한차분 Jacobian, 추가 상 후보, 경계 부근 반복 탐색이 현재의 비용 요인이다. 일반 분석 최적화 작업은 중단 상태를 유지했다. 사용자의 계산량 제한에 따라 검증 후 전체 유동을 다시 실행하거나 종료시간 1 ms까지 확장하지 않았다.

## 입력과 재현

준비한 두 케이스는 `/home/jsw/문서/analysis/runs/impinging_n2o_solid_20260922/benchmark`에 있다. 저장소에 포함한 [입력 예제](../../examples/impinging-n2o-solid/README.ko.md)는 동일 기구를 사용한다. 원래 유체 전용 입력과 초기 실패 기록은 보존했다.

```bash
export PINTLE_SPUMA_ENV=/home/jsw/cae-gpu-pr1-compatible/env.sh
export PINTLE_REACTIVE_PREFIX=/home/jsw/문서/analysis/spuma-multiphysics/research/reactive-env
export PINTLE_REACTIVE_TRANSPORT_BUILD=cuda
export PINTLE_CUDA_ARCH=120
flock /home/jsw/cae-benchmark/run.lock ./Allwmake

# 새 경로를 사용한다. 기존 케이스를 덮어쓰지 않는다.
"$PINTLE_REACTIVE_PREFIX/bin/python" tools/prepare_impinging_n2o.py /path/to/new-solid-benchmark \
  --configuration examples/impinging-n2o-solid/cold-pr-148K-config.yaml --cell-mm 0.25

# 아래 연속 검증 명령은 재현용이다. 비용 때문에 자동 반복하지 않는다.
"$PINTLE_REACTIVE_PREFIX/bin/python" tools/validate_impinging_n2o.py \
  --benchmark /path/to/new-solid-benchmark --output /path/to/new-validation --steps 10

# 빠른 0D 수치·물성 검사. 현재 참조 정확도 미달로 종료 코드 1이 예상된다.
flock /home/jsw/cae-benchmark/run.lock "$PINTLE_REACTIVE_PREFIX/bin/python" tools/validate_n2o_solid.py \
  --configuration /path/to/new-solid-benchmark/thermo/cold-pr-config.yaml \
  --library lib/libpintleReactiveBackend.so --cuda-library lib/libpintleReactiveTransport.so \
  --output /path/to/solid-unit.json
```

저온 기구를 원본에서 다시 만들려면 `extend_n2o_gas_thermo.py --configuration <원본 cold-pr-config.yaml> --solid-data docs/model-data/n2o-solid-atake1974.json --output <새 경로> --library lib/libpintleReactiveBackend.so`를 사용한다. 최종 생성기 검사는 원래 준비된 기구 바이트와 물리 해시가 동일함을 확인했다.

수치 라이브러리는 위 검증과 동일하다. 마지막 솔버 정리에서는 로그에 실제 유한차분 선택을 표시하고 출력 필드의 종 이름을 메타데이터에서 읽도록 수정했다. 이 출력·로그 변경은 빌드 검증했으며 256k 계산을 반복하지 않았다. 실행 당시와 최종 바이너리 해시는 [전달 manifest](../../results/n2o-solid-20260922/delivery.json)에서 구분한다.

발달한 충돌류, 액막 분열, 액적 생성, VOF·표면장력, 유한속도 flashing 또는 고압 고체 물성의 검증 완료를 의미하지 않는다.
