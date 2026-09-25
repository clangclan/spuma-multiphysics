# GPU HEM 최적화 변경 기록 (2026-09-25)

`origin/main`(`57a8c31`)에서 분기한 `claude/gpu-hem-optimization` 브랜치의 변경 기록이다. 변경은 아직 커밋하지 않은 작업 트리 상태이며, 작업 위치는 `spuma-multiphysics-gpu-opt` 워크트리다.

**요약.** 160³·40 bar·100 μs 캠페인에서 전체 시간의 65%를 차지한 GPU HEM 비용을 두 방향에서 줄였다.

1. 모세관(곡면) 셀까지 해석적 Jacobian을 확장하고, N₂O 모세관 케이스의 기본 Jacobian을 `analytic`으로 바꿨다.
2. 모세관 외부 반복에서 입력이 바뀌지 않은 셀은 다시 flash하지 않도록 했다.
3. 수송 staging 슬롯의 기본값을 1 MiB에서 64 MiB로 늘렸다.

100 μs 체크포인트의 실제 N₂O 셀 549,292개로 HEM 배치를 재생한 결과, HEM 커널 시간은 **3.496 s에서 1.450 s(2.41배)**로 줄었고 상(phase) 평가는 **83% 감소**했다.

**솔버 전체 실행 결과:** 100 μs 체크포인트에서 2스텝을 계산했을 때 스텝 시간이 **20.0 s에서 7.9–8.1 s(약 2.5배)**로 줄었다. 2스텝 뒤 q는 셀의 99.92%에서 기준과 비트 단위로 같았고, 상태의 최대 상대차는 2.4×10⁻⁹였다. 실행에는 AVX-512가 없는 호환 SPUMA 빌드를 사용했다([AVX-512 우회 문서](../docs/avx512-runtime-workaround.ko.md)). ③ staging 변경의 시간 효과는 작았다(수송 시간 약 0.84 s → 0.81 s).

## 배경

리뷰에서 확인한 병목은 다음과 같다(자료: `reports/n160-40bar-100us-gpu-analysis-20260925.ko.md`, 원시 로그 `runs/n160_40bar_100us_gpu_20260924`).

- 마지막 스텝(20.07 s) 가운데 HEM 커널이 15.46 s였다. 이 스텝에서 FD Jacobian은 1.46억 회, 상 평가는 19.35억 회였다. FD Jacobian 하나가 잔차 6회, 곧 상 평가 12회를 요구하므로 **상 평가의 약 90%가 FD Jacobian에서 나온다**.
- 캠페인은 `closureJacobian finiteDifference`로 실행됐다. 게다가 해석적 Jacobian에는 `pressureJump==0` 조건이 걸려 있어, `analytic`을 켜더라도 곡면 셀은 FD로 처리됐다.
- 모세관 복원 1회는 언제나 외부 반복 2회로 끝났다(3,896/1,948). 반복 1은 사실상 검증 패스인데도 전체 메시를 다시 flash했다. 한 스텝에 복원이 2회이므로 **전체 메시 제출이 4회**였다.
- 수송은 1 MiB pinned 슬롯 하나로 전체 q를 타일 단위로 옮기면서 타일마다 event 동기화를 했다(스텝당 `Layout` 커널 1,144회, 누적 event 대기 218만 회).

100 μs 체크포인트를 보면 N₂O가 있는 셀은 549,292개, 액체가 있는 셀은 772개, 곡면(J≠0) 셀은 1,612개다. HEM 비용의 대부분은 **액체가 없는 N₂O 증기 셀**이 두 상 후보 5개를 Newton으로 시도하는 데서 나온다. 이 셀들의 최종 해는 단상 기체다.

## 변경 1: 곡면 셀의 해석적 Jacobian

**파일:** `src/reactiveThermo/reactiveDeviceFlashJacobian.h`, `src/reactiveThermo/reactiveDeviceFlash.h`, `src/reactiveThermo/reactiveThermo.cpp`

곡면 복원에서는 J(=σκ)와 color c를 국소 풀이 동안 고정한 채, 기체를 p_g = p̄ − cJ에서, 액체를 p_l = p̄ + (1−c)J에서 평가한다. 그러면 dp_g/dp̄ = dp_l/dp̄ = 1이다. 따라서 ln p̄ 열과 ln T 열의 해석식은 형태가 그대로이고, 상 상태를 각 상의 압력에서 평가하도록만 바꾸면 된다.

- `build()`와 `buildCached()`에 기본값이 있는 `gasPressure`, `liquidPressure` 인자를 추가했다. 값이 0이면 평면(p_g = p_l = p̄)으로 처리하므로 기존 호출과 호환된다.
  - `build()`: 기체 상태를 p_g에서, 액체 상태를 p_l에서 평가한다.
  - `buildCached()`: 기체 부분 몰 성질(`activeGasDerivatives`)의 EOS 압력 검사와 부피 분모에 p_g를 쓰고, 액체 내부에너지 ū_L = h̄_L − p_l·v̄_L에 p_l을 쓴다.
  - 화학 퍼텐셜 행의 ln p̄ 열(p̄·(v̄_L − v̄_G)/RT)과 부피·에너지 행(Vp, VT, Ep, ET)은 원래대로 p̄를 쓴다. Vp와 Ep는 `evaluate()`가 이미 각 상의 압력으로 누적한 값이다.
- `Flash::jacobian()`에서 `pressureJump==0` 조건을 제거하고 `gasPressure(p̄)`, `liquidPressure(p̄)`를 전달한다. 고체가 있는 모델은 이전처럼 FD만 쓴다.
- 수치 정책 식별값: `closureJacobian analytic`인 CUDA HEM 정책에 `:curved-analytic-v1`을 덧붙였다. 이전 analytic 정책과 해시가 달라지며, FD 정책의 해시는 바뀌지 않는다.
- `tools/prepare_capillary_impingement.py`는 원래 `analytic`을 `finiteDifference`로 강제로 바꿨다. 이제는 `--closure-jacobian`(기본값 `analytic`)을 따른다.
- README의 예시 설정과 옵션 설명을 새 동작에 맞게 고쳤다.

## 변경 2: 모세관 외부 반복의 입력 재사용

**파일:** `src/reactiveFoam/reactiveCapillary.H`, `src/reactiveFoam/ReactiveFoam.C`

**이전 동작.** 반복 k ≥ 1에서도 모든 셀을 다시 flash했다. 이때 seed는 반복 k−1의 출력이었다.

**변경 후 동작.**

- **seed를 고정한다.** 외부 반복 전체에서 같은 seed를 쓴다. seed는 복원에 들어올 때의 상태이고, 액체 재고(liquidMass)만 수송된 값으로 바꾼다. 이는 이전의 반복 0 seed와 같다.
- **입력이 그대로인 셀은 재사용한다.** 반복 동안 q는 고정돼 있으므로 셀의 flash 결과는 (color, J, 표면에너지)의 결정적 함수다. 세 값이 마지막 flash 때와 **비트 단위로 같은** 셀은 다시 계산해도 같은 결과가 나오므로, 기존 상태를 그대로 쓴다. 다시 flash하는 셀만 모아 배치로 보낸다.
- **수렴 판정은 바꾸지 않았다.** 잔차 계산, 완화, 40회 반복 상한은 이전과 같다.
- **메모리 추정을 갱신했다.** seed(22 double), 마지막 flash 입력(3), dirty 인덱스(1)를 모세관 메모리 추정에 더했다(`12` → `38` double/셀). 160³ 메시에서 약 0.85 GB가 늘어난다.
- **수치 정책 식별값:** 모세관 케이스의 수치 문맥에 `capillaryClosure=fixed-seed-input-reuse-v1`을 추가했다.
- **로그:** `REACTIVE_CAPILLARY_STEP`에 `flashedCells`와 `reusedCells`(누적)를, 종료 시 `REACTIVE_CAPILLARY_PROFILE`에 `closureFlashedCells`와 `closureReusedCells`를 추가했다.

**정확성 근거.** 이 변경은 "같은 입력과 같은 seed를 주면 셀 결과가 비트 단위로 같고, 배치 구성과도 무관하다"는 가정에 기댄다. 이 가정은 GPU에서 확인했다(아래 검증 3). 이전 동작과 비교하면 반복 k ≥ 1의 seed가 달라지므로 결과는 반올림 수준에서 달라질 수 있다. 물리 모델이나 수렴 기준은 바뀌지 않았다.

**기대 효과(추정).** 반복 1에서 다시 flash할 셀은 계면 띠에 한정된다. 100 μs 체크포인트 기준으로 곡면 셀은 1,612개이고, 색·곡률·표면에너지·액체 중 하나라도 0이 아닌 셀을 3층 확장해도 5,266개다. 이전에는 반복 1에서도 N₂O 셀 549,292개와 공기 셀을 모두 다시 계산했다.

## 변경 3: 수송 staging 기본값

**파일:** `src/reactiveFoam/ReactiveFoam.C`, `README.md`

`transportStagingBytes`와 `maxPinnedTransportBytes`의 기본값을 1,048,576에서 67,108,864로 바꿨다. 4.1M 셀 × 9 변수의 q 한 번 전송이 약 280 타일(타일마다 event 대기)에서 5 타일로 줄어든다. 작은 메시에서는 기존 코드가 슬롯을 메시 크기로 제한한다. 값을 명시적으로 지정한 케이스(예: `prepare_reactive_case.py`)는 영향을 받지 않는다. 수치 결과에는 영향이 없다.

## 도구 변경

| 파일 | 변경 |
|---|---|
| `tools/hem_checkpoint_batch.py` (신규) | 체크포인트(`reactiveState.bin` + 계면 필드)에서 모세관 HEM 배치를 만든다. OpenFOAM이 필요 없다. 출력은 `hem_capture.cpp` 형식이다. `--analytic-jacobian 0|1`로 모델 이미지의 Jacobian 플래그만 바꿀 수 있다. |
| `tools/replay_hem_precision.py` | 기본 separate-rounding 빌드는 `reactive_gpu_hem_numerical_policy_v1`을 export하지 않는다. 이 경우 `legacy-fp64-separate-rn`으로 기록하도록 했다. |
| `tools/hem_capture.cpp` | `REACTIVE_CAPTURE_MIN_CURVED`: 곡면·액체 seed 셀이 지정한 수 이상 들어 있는 배치만 캡처한다. |
| `tools/prepare_capillary_impingement.py` | `--closure-jacobian` 옵션을 추가했다(기본값 `analytic`). |

## 검증

**환경:** RTX 5080, CUDA 13.2, sm_120, 기본 FP64 separate-rounding HEM 빌드. 기준 라이브러리는 같은 워크트리에서 `57a8c31`을 빌드한 것이다.

**입력 배치:** 100 μs 체크포인트(`runs/n160_40bar_100us_gpu_20260924/production/40bar/to-100us/0.0001`)에서 만든 배치다. 입력은 수렴한 복원의 검증 패스에 해당한다(bulk 에너지 = 내부에너지 − 표면에너지, J = σκ, seed = 체크포인트 상태). 과도(transient) 스텝의 반복 0과 seed가 다르므로, 결과는 커널의 정확성과 처리량을 보는 근거일 뿐 스텝 시간의 근거는 아니다. 모델 이미지는 같은 thermo 설정의 기존 캡처(`runs/precision_revisit_20260924/hem-40bar.bin`)에서 복사했다.

### 검증 1: N₂O 셀 전체 549,292개 (반복 3회, 커널 중앙값)

| 변형 | HEM 커널 [s] | 상 평가 | 잔차 평가 | 해석적 Jacobian | FD Jacobian | 기준 대비 최대 상대차 |
|---|---:|---:|---:|---:|---:|---|
| 기존 코드 + FD (캠페인 설정) | **3.496** | 476,598,677 | 801,215,699 | 0 | 36,544,597 | — |
| 수정 코드 + FD | 3.487 | 476,598,677 | 801,215,699 | 0 | 36,544,597 | 0 (비트 동일) |
| 기존 코드 + analytic | 1.544 | 82,416,755 | 587,178,820 | 36,428,895 | 225,839 | 0 (비트 동일) |
| 수정 코드 + analytic | **1.450** | 81,470,615 | 586,677,202 | 36,513,275 | 141,638 | 음속 1.2×10⁻⁹, 나머지 ≤ 2.2×10⁻¹³ |

- 모든 변형이 549,292셀 전부 성공했고, 장치 실패와 CPU fallback은 0이었다. 반복 실행 간 결과도 결정적이었다.
- 수정 코드의 FD 경로는 기존 코드와 상태·카운터가 모두 비트 단위로 같다.
- 증기 셀의 최종 해는 FD든 analytic이든 비트 단위로 같다. 이 셀들의 최종 해는 Newton이 아니라 단상 기체 경로에서 나오기 때문이다.
- 수정 코드 + analytic에서 결과가 달라진 셀은 곡면 셀뿐이다(검증 2).
- 표의 `iterations` 필드 차이 1은 Newton 반복 수가 달라졌다는 뜻이며, 상태 값의 차이가 아니다.

### 검증 2: 곡면 셀 1,612개

| 변형 | 커널 [s] | 상 평가 | FD Jacobian | 해석적 Jacobian |
|---|---:|---:|---:|---:|
| 기존 코드 + FD | 0.2287 | 1,345,621 | 84,201 | 0 |
| 기존 코드 + analytic | 0.2290 | 1,345,621 | 84,201 | 0 (J≠0 조건 때문에 전부 FD) |
| 수정 코드 + analytic | 0.2008 | 399,481 | 0 | 84,380 |

FD 기준 대비 최대 상대차는 평형 음속 1.22×10⁻⁹, 기체 질량 2.1×10⁻¹³, 기체 밀도 1.7×10⁻¹³이다. 1,612셀 모두 성공했고, 활성 액체 mask 변화는 없었다. 음속 차이는 기존 문서가 밝힌 FD 음속의 잡음 수준(약 5×10⁻⁷)과 음향 게이트 기준(10⁻⁸)보다 작다. 배치가 작아서 긴 꼬리 셀이 시간을 지배하므로, 커널 시간 감소(12%)는 상 평가 감소(70%)보다 작다.

### 검증 3: 배치 구성과 무관한 셀 결과 (변경 2의 전제)

수정 코드 + analytic으로 N₂O 셀을 7개마다 하나씩 골라 만든 배치(78,471셀)의 결과는, 전체 배치에서 같은 셀의 결과와 **비트 단위로 같았다**. 최종 빌드 라이브러리(`libreactiveTransport.so` sha256 `33a1b052…`)로 전체 배치를 다시 돌려도 결과가 비트 단위로 같았고, 커널 시간은 1.452 s였다.

원자료: `results/gpu-hem-optimization-20260925/summary.json`, 변형별 `replay/*.json`. 배치 파일(최대 약 132 MB)과 `states.npy`는 크기 때문에 저장소에 넣지 않았고, 해시만 요약에 기록했다.

재현 명령:

```bash
PY=/home/jsw/문서/analysis/spuma-multiphysics/research/reactive-env/bin/python
CK=/home/jsw/문서/analysis/runs/n160_40bar_100us_gpu_20260924/production/40bar/to-100us/0.0001
MODEL=/home/jsw/문서/analysis/runs/precision_revisit_20260924/hem-40bar.bin
cd tools
$PY hem_checkpoint_batch.py $CK --model-from $MODEL --sigma 0.0020577273399028486 \
  --select condensable --analytic-jacobian 1 --output /tmp/ck-condensable-an.bin
flock /home/jsw/cae-benchmark/run.lock $PY replay_hem_precision.py /tmp/ck-condensable-an.bin \
  --library ../lib/libreactiveTransport.so --output /tmp/replay-an --repeats 3
```

### 검증 4: 솔버 전체 실행 (100 μs 체크포인트에서 2스텝)

**조건:** `REACTIVE_SPUMA_ENV=/home/jsw/cae-gpu-pr1-compatible/env.sh`(x86-64-v3 SPUMA 빌드), `runs/gpu_opt_validation_20260925`, `maxAcceptedSteps 2`. 기준은 `57a8c31` 빌드이고, 수정본은 이 워크트리의 빌드다. 설정은 캠페인과 같고, 세 번째 실행에만 `closureJacobian analytic`을 추가했다.

| 실행 | 스텝 시간 [s] | 복원 [s] | HEM 커널 [s] | HEM 제출 | 상 평가 | FD Jacobian | 수송 [s] | CFL [s] |
|---|---|---|---|---:|---:|---:|---|---|
| 기준(FD) | 20.03 / 20.02 | 18.23 / 18.24 | 15.19 / 15.23 | 6,974,554 | 1.937×10⁹ | 1.463×10⁸ | 0.84 / 0.83 | 0.60 / 0.60 |
| 수정본, FD | 12.20 / 12.01 | 10.43 / 10.24 | 8.02 / 8.11 | 3,498,029 | 0.976×10⁹ | 0.733×10⁸ | 0.82 / 0.81 | 0.60 / 0.61 |
| 수정본, analytic | **8.07 / 7.86** | 6.29 / 6.09 | **3.87 / 3.95** | 3,498,029 | 0.184×10⁹ | 2.8×10⁵ | 0.81 / 0.81 | 0.61 / 0.61 |

HEM 제출·상 평가·FD Jacobian은 첫 스텝의 값이다.

- 세 실행 모두 재시도 0, GPU 실패 0, CPU fallback 0으로 `End`까지 끝났다. 외부 반복은 모두 복원당 2회였다.
- 수정본에서는 2스텝(복원 4회) 동안 16,389,824개 셀을 flash하고 16,378,176개 셀을 재사용했다. 반복 1에서 다시 flash한 셀은 복원당 평균 약 1,456개다.
- 시간 간격(dt)은 세 실행 모두 출력 자릿수까지 같았다. 보존 잔차(질량 약 2.14×10⁻¹¹, 에너지 약 1.80×10⁻¹¹)도 같은 수준이다.
- 2스텝 뒤 `reactiveState.bin`을 비교한 결과, 두 수정 실행 모두 q가 4,092,895/4,096,000셀에서 비트 단위로 같았다. 상태의 최대 상대차는 FD 수정본 2.9×10⁻¹⁰, analytic 수정본 2.4×10⁻⁹(액체 분율)였다. 활성 액체 mask 변화는 0, 액체 셀 수는 모두 772개였다.
- staging 변경으로 event 대기는 2,272회에서 42회로 줄었다(초기화 포함 전체 실행 기준). 그러나 수송 시간은 스텝당 약 0.02–0.03 s 줄었을 뿐이다. 수송 시간의 대부분은 타일 동기화가 아니라 pageable 상태 업로드와 호스트 루프다.

원자료: `results/gpu-hem-optimization-20260925/end-to-end.json`, 각 실행의 `solver.log`.

### 추정과 실측 비교

변경 전에 예측한 스텝 시간은 약 8.2 s(HEM 커널 약 3.6 s)였고, 실측은 7.9–8.1 s(HEM 커널 3.9 s)였다.

## 검증하지 못한 것

- **장시간 실행.** 2스텝만 계산했다. 10 μs 이상의 과도 구간에서 외부 반복 수, 재사용 비율, 보존 잔차가 어떻게 변하는지는 확인하지 않았다.
- **새 정책으로 저장한 체크포인트의 재시작.** 캠페인 체크포인트(이전 정책)에서 수정본으로 재시작하는 경우는 확인했다. 두 수정 실행 모두 `REACTIVE_RESTART_POLICY decision=supported_current_settings physicalModelUnchanged=1`을 기록했고, 물리 모델 해시는 같았다(수치 정책 해시만 다름). 기준 실행에서는 정책이 같아 이 로그가 없다. 반대로 새 정책으로 저장한 체크포인트를 재시작하는 경우는 실행하지 않았다.
- **FMA 정책과의 조합.** `fp64FmaRnGuard` 빌드와 해석적 Jacobian을 함께 쓰는 경우는 측정하지 않았다.
- **솔버의 기본값.** `closureJacobian`의 솔버 기본값은 여전히 `finiteDifference`다. 케이스 생성기와 README 예시만 `analytic`으로 바꿨다. 기존 케이스는 설정을 직접 바꿔야 한다.

## 남은 병목과 다음 단계

후속 Nsight 분석과 추가 수정(무거운 셀 먼저 배치, 셀별 메시지 제거, 호스트 루프 병렬화; 스텝 7.8 → 6.5 → 5.3 s)은 [GPU 실질 부하 분석](gpu-load-nsight-20260925.ko.md)에 정리했다.

40 bar 100–200 μs 실행의 125 μs 중단 원인(PR 근 판정)과 검증기 수정은 [PR 근 판정 수정](eos-near-repeated-root-20260925.ko.md)에 정리했다.

- **증기 셀의 두 상 후보 탐색.** 해석적 Jacobian을 쓴 뒤에도 셀당 후보 5개를 모두 시도하는 구조는 그대로다(후보 274만 개, 안정 후보 55만 개). 대부분의 셀에서 최종 해는 단상 기체다. 최대 엔트로피 해와의 동일성을 보장하는 조건(certificate)을 두고 후보를 가지치기하는 것이 다음으로 큰 효과를 낼 수 있다.
- **HEM 레지스터·스택 압력.** 255 레지스터/스레드, 약 6 KB 스택이다. 운영 형상(NS=4, NL=1)에 맞춘 특화 빌드를 검토할 수 있다.
- **호스트 배관.** RK 단계마다 q를 다운로드하고, WALE PR 물성 계산을 위해 상태 전체(약 0.72 GB)를 pageable 메모리로 업로드한다. CFL 계산도 호스트 경로로 한다. 장치 상주 구조로 바꾸는 것은 이번 변경 범위 밖이다.
