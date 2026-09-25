# 40 bar 125 μs 중단: PR 근 판정 수정 (2026-09-25)

대상 실행은 `runs/n160_40bar_100to200us_gpu_20260925/production/40bar/to-125us`다. 실행은 125 μs까지 정상 도달했지만, 검증기가 `GPU closure failure/fallback`으로 판정해 150·200 μs 구간이 차단되었다. 외부 분석이 제시한 원인 두 가지를 로그와 GPU 재현으로 확인하고 둘 다 수정했다.

## 로그로 확인한 사실

| 항목 | 로그 근거 |
|---|---|
| 실패 시점 | 116.7354 μs, `stage=rk2`, `attemptId=328`, `batchOffset=1048576` (`reactiveFailures.jsonl`) |
| 실패 셀 | 1,945,613과 1,947,680. `localCell` 897,037/899,104, 오류는 `CUDA HEM candidate search failed; CPU fallback disabled` |
| 입력 | 초기 추정 T 274.477 K, p 40.650 bar, 색 0.34949, 모세관 압력차 0.3185 Pa. 두 셀은 x·y 운동량만 서로 바뀐 거의 같은 상태 |
| 처리 | `REACTIVE_RETRY dt=5.1328e-08`, 이후 `dt=2.5664e-08 retries=1`로 수락 (`solver.log` 1681–1683행) |
| 이후 | dt가 51.33 ns로 돌아옴. 488스텝이 수락되었고 125 μs 체크포인트 저장 후 `End` |
| 카운터 | 재시도한 스텝의 `REACTIVE_GPU_HEM_STEP deviceFailures=2`, 누적 `REACTIVE_GPU_HEM deviceFailures=2 cpuFallbacks=0` |
| 차단 | `validate_impinging_initial_steps.py`의 `deviceFailures!=0` 판정 |

외부 분석의 사건 순서와 수치는 로그와 모두 일치했다.

## 원인 1: 근 판정이 쓸 수 있는 근까지 거부

`src/reactiveThermo/reactiveDevicePR.h`에서 `rootsTP`와 `rootsFromMixingCached`(HEM이 쓰는 고정 용량 경로)는 같은 판정을 쓴다. Cantera처럼 `|h|≈|q|`이면 disc를 0으로 강제한 뒤 `|disc| < 1e-14`이면 `NoPhysicalRoot`를 반환했다. 이 판별식은 스케일되지 않은 절대값이다. 그래서 두 근이 스피노달 근처에서 거의 겹치면, 멀리 떨어진 세 번째 근도 함께 거부되었다. 이번 상태의 세 번째 근은 쓸 수 있는 기체 근이었다.

**수정.** 이 구간에서도 모든 근을 거부하지 않는다. 새 함수 `detail::isolatedRoot`가 떨어져 있는 근 하나만 계산하고, 다음을 모두 만족할 때만 받아들인다.

- 삼각함수 공식에서 떨어진 근의 위치(`center ± 2δ`)를 시작점으로 FP64 다항식 Newton을 수렴시킨다.
- 공유체적 b보다 크다.
- 다항식 잔차가 `64·ε·scale` 이하다.
- 겹치는 두 근의 위치(`center ∓ δ`)에서 상대적으로 1e-3 이상 떨어져 있다.
- 기계적 안정성 `∂p/∂V < 0`을 만족한다.

겹치는 두 근은 반올림에 따라 존재 여부가 바뀌므로 계속 사용할 수 없다. 떨어진 근이 기체 쪽(q < 0)이면 기체만, 액체 쪽이면 액체만 가능으로 표시한다. 조건을 하나라도 못 맞추면 예전처럼 `NoPhysicalRoot`를 반환한다. 삼중근 근처(δ→0)에서는 분리 조건을 만족하지 못하므로 기존 동작이 그대로 유지된다. `|h|≈|q|`이면서 `disc > 1e-10`일 때의 `NonFinite` 판정과 HEM 수렴 허용오차는 바꾸지 않았다.

수치 정책 식별자에 `:pr-isolated-root-v1`을 추가했다(`reactiveThermo.cpp`). 이전 정책으로 저장한 체크포인트에서 재시작하면 `REACTIVE_RESTART_POLICY ... decision=supported_current_settings`가 기록된다.

`reactiveDevicePRSelectedRootPrototype.h`에도 같은 판정이 있다. 이 파일은 prototype이고 솔버 빌드에서 쓰지 않아 고치지 않았다.

## 원인 2: 검증기가 복구된 실패도 최종 실패로 판정

`tools/validate_impinging_initial_steps.py`는 누적 카운터 `deviceFailures`가 0이 아니면 실패로 판정했다. 이 카운터에는 되돌린 시도의 실패도 남는다.

**수정.** `recovered_device_failures()`를 추가했다. 규칙은 다음과 같다.

- `cpuFallbacks`는 지금처럼 0이어야 한다.
- 모든 장치 실패는 수락된 스텝의 `REACTIVE_GPU_HEM_STEP` 기록에 들어 있어야 한다. 스텝별 합계가 누적 합계와 같아야 한다.
- 장치 실패가 있는 스텝은 `retries > 0`인 수락 스텝이어야 한다.
- `REACTIVE_RETRY` 줄의 수는 수락 스텝의 `retries` 합과 같아야 한다.

이 조건을 통과하면 실패 대신 `row['recovery']`에 복구 건수, 해당 스텝 수, 재시도 수, 시각을 기록한다. 실제 로그에서는 `recoveredDeviceFailures=2, stepsWithRecoveredFailures=1, retries=1, recoveredAtUs=[116.761]`이 나왔다. 두 반례에서는 거부됨을 확인했다. 하나는 스텝 기록에서 실패를 빼 누적 합계와 맞지 않게 한 경우이고, 다른 하나는 해당 스텝의 `retries`를 0으로 바꾼 경우다.

## 검증

실패 입력 재현에는 새 도구 `tools/hem_failure_batch.py`를 썼다. `reactiveFailures.jsonl`의 q, bulk 에너지, seed 상태, 색, 압력차를 그대로 넣고, 두 셀을 각각 500번 복제한 1,000셀 배치다. 모델 이미지는 같은 열역학 설정의 캡처 `runs/precision_revisit_20260924/hem-40bar.bin`에 해석적 Jacobian 플래그를 켜서 썼다.

| 실험 | 결과 |
|---|---|
| 실패 입력, 실행에 쓴 라이브러리(`runtime/lib`) | 1,000셀 중 0개 성공, 즉 재현됨 |
| 실패 입력, 수정 라이브러리 | 1,000셀 모두 성공. 복제본끼리 비트 동일 |
| └ 셀 1,945,613 결과 | T 274.484435 K, p 40.649552 bar, 체적 잔차 5.41e-13, 에너지 잔차 3.20e-14, 화학퍼텐셜 잔차 1.90e-13, 반복 5회 |
| 120 μs 체크포인트의 N₂O 포함 셀 전체(724,022셀) 재생, 기존 대비 수정 | 둘 다 전부 성공. **모든 셀이 바이트 단위로 동일**. HEM 커널 1.655 → 1.658 s |
| 검증 사례 3스텝(4,096,000셀) | `reactiveState.bin`이 수정 전 빌드와 비트 단위로 동일. 스텝 5.49/5.24/5.20 s |

수정 결과는 외부 분석의 별도 GPU 진단 결과(T 274.484435 K, p 40.649552 bar, 잔차 5.41e-13/3.20e-14/1.90e-13)와 같다. 이 구간에 들어가지 않는 셀은 결과가 바뀌지 않는다.

**실제 실패 구간 재실행.** 수정 빌드를 원래 실행의 110.011 μs 체크포인트에서 재시작해 116.8 μs까지 133스텝을 돌렸다(`runs/gpu_opt_validation_20260925/eosfix-110to117us`).

- 앞 131스텝은 시각과 dt가 원래 로그와 모든 자릿수까지 같았다.
- 원래 실행이 실패한 스텝은 116.7354 μs에서 dt 5.1327887666892626e-08로 시작한다. 수정 빌드는 **같은 dt를 재시도 없이 수락**해 116.7867 μs에 도달했다.
- `REACTIVE_RETRY` 0회, 누적 `deviceFailures=0`, `cpuFallbacks=0`, 정상 종료했다.

## 남은 일

- 캠페인 재개는 하지 않았다. 125 μs 체크포인트에서 150·200 μs 구간을 다시 돌리려면 `tools/run_long_gpu_campaign.py`를 수정된 `bin`/`lib`와 검증기로 다시 실행해야 한다. 캠페인 `runtime/` 사본은 실행 당시 상태를 보존하기 위해 건드리지 않았다.
- 떨어진 근 판정의 분리 기준(1e-3)과 잔차 기준(64ε)은 이번 상태와 120 μs 전체 재생에서 확인했다. 다만 다른 중근, 상 경계, 초임계 근처 조성에서 체계적으로 탐색한 것은 아니다. 다른 조건(1 atm, 과냉각, 고체 사례)으로 넓힐 때는 `hem_failure_batch.py` 재생과 전체 셀 재생 비교를 같은 방식으로 반복해야 한다.
