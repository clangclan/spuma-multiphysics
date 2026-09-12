# GPU·희소 포팅 리뷰 반영 — 2026-09-12

이 문서는 `c26e9c9`의 성능 구조 개선과 당시 검사 기록이다. 후속 리뷰에서 발견한 `Flow::step() const` 컴파일 오류 수정과 검사 보강은 [오류 수정 기록](README.reactive-error-fixes.ko.md)에 구분했다. 아래 백엔드 검사 통과는 SPUMA 전체 빌드 통과를 뜻하지 않으며, 당시 그 경로에 이 컴파일 오류가 남아 있었다.

첨부 리뷰가 지목한 **CFL의 전체 장 전송, 경계 reduction의 작은 병렬성, 희소 준비 비용의 반복**을 수정했다. HLL 공통 계수와 확산 엔탈피 합도 등가식으로 정리하고, 실제 열역학 상태·3차원 면·캐시 수명 검사를 추가했다. 기본값은 계속 CPU 수송·밀집 화학이며 PR은 draft다.

## 기준과 현재 범위

| 구분 | 기준 / 증거 |
|---|---|
| 원래 실험 상태 | `50512ce…`; [이전 연구 보고서](reports/multiphase-review-fixes-20260911.md) |
| 이번 리뷰의 대상 | `fad4ecfd4d627c8903ee71dde25941a63268a767`; [최초 포팅 기록](README.reactive-gpu-sparse.ko.md) |
| 이번 개선 | 위 커밋에 대한 후속 변경. 환경·리뷰 hash·소스 hash·검사 hash는 [별도 기록](results/reactive-review-environment-20260912.json) |
| 실제 실행 가능 범위 | 별도 CPU 시험 환경과 CUDA 컴파일. 실제 GPU 실행과 SPUMA 전체 빌드는 미검증 |

기존 413종·14,922반응, 상평형 모델, 화학 허용오차, 채택 상태의 양수성·원소·질량 검사는 유지했다. 이전의 HEM 접촉면 정확도 실패, 16셀 기계적 접촉면 오차, 고압 다상 연소 미검증 상태도 그대로다. 이번 성능 구조 개선이 그 물리·수치 모델의 문제를 해결한 것은 아니다.

## 수송에서 바꾼 부분

`pintle_transport_stable_step_primitives()`는 셀별 속도·밀도와 compact 열역학 상태만 받는다. CFL 조회 때문에 전체 종 보존장을 업로드하거나 SoA로 전치하지 않는다. 실제 `Flow::stableStep()`도 이 API를 사용한다.

`pintle_transport_upload_conserved()`와 `pintle_transport_stage_resident()`는 명시적인 내용 버전을 사용한다. 새 CPU 화학 결과 또는 거부한 단계의 복원은 더 큰 버전으로 업로드한다. 동일 버전의 중복 업로드는 생략하고, 오래된 입력 버전·증가하지 않은 출력 버전·짝이 없는 RK stage 1은 거부한다. 포인터 주소는 재사용의 근거가 아니다.

버전은 호출자가 내용 변경 때 증가시켜야 하는 계약이다. 배열 내용을 hash로 비교하지 않으므로, 같은 버전으로 바뀐 내용을 넘기는 호출자 오류는 자동 검출하지 않는다.

두 RK 단계 사이 CPU `recover()`는 보존량을 바꾸지 않으므로 stage 1은 device의 동일한 보존장을 사용하고 열역학 상태만 갱신한다. 정상 스텝의 전체 보존장 전송은 다음과 같다. 오류·재시도와 초기화·출력은 제외한다.

| 전체 보존장 한 벌을 `Q`라고 할 때 | 리뷰 대상 | 개선 후 |
|---|---:|---:|
| CFL 3회에 의한 업로드 | `3Q` | 0 |
| RK에 의한 업로드 | `2Q` | `Q` |
| CPU flash를 위한 다운로드 | `2Q` | `2Q` |
| 합계 | `7Q` | `3Q` |

이는 전체 보존장 전송 바이트의 **57.1% 감소**이며 실행시간 감소율은 아니다. `gasY/gasH`와 작은 상태 배열의 전송은 남는다. 새 `PintleTransportProfile`과 `REACTIVE_TRANSFER`는 보존장·상태·원시량·기상 수송 배열의 바이트를 구분한다. 검사에서 CFL 3회와 RK 2회가 실제로 보존장 업로드 1회·다운로드 2회만 만드는지 확인했다.

확산이 켜진 정상 RK2 스텝에서 종 배열 한 벌을 `G`라고 하면 `gasY/gasH` 전송은 여전히 `4G`다. 작은 상태·원시량을 제외한 합계는 **`7Q + 4G → 3Q + 4G`**이며, 413종·417변수에서는 약 **36.5% 감소**다. Device의 보존장 저장공간도 여전히 네 벌이므로 메모리 용량 감소나 전체 PCIe 전송의 57.1% 감소로 해석하면 안 된다.

경계 수지는 `변수 × 경계 분할`마다 256개 lane으로 부분합을 만든 뒤 두 번째 reduction으로 합친다. 경계 분할은 최대 256개이며 메모리는 `8 × 변수 수 × 분할 수` byte다. 면×종 flux 배열은 만들지 않는다. 417변수의 최댓값은 약 0.82 MiB다. 실제 GPU 점유율·속도는 아직 측정하지 않았다.

CFL 최솟값은 연속 입력을 읽는 256-lane tree reduction을 사용한다. 오류인 셀은 음수 sentinel로 표시하고 이 값을 모든 reduction 단계에서 보존한다. NaN을 정상 최솟값으로 가리지 않는다. 유한한 입력이 device 계산 중 overflow를 일으키는 경우도 검사한다. C API의 실패 경로는 진행 중인 전송을 마친 뒤 반환해, 호출자가 복원하거나 해제한 host 배열을 비동기 연산이 계속 참조하지 않게 했다.

면 공통 HLL 계수는 한 번 계산한다.

```text
a = SR (uL − SL) / (SR − SL)
b = SL (SR − uR) / (SR − SL)
Fspecies = a qL + b qR
```

운동량·에너지의 압력 계수도 면별로 저장한다. 셀 부피의 역수를 재사용한다. 확산은 기존 carrier 규칙 `Jc = −sum(k≠c, Jk)`를 유지하면서 `sum(k≠c, Jk (hk − hc))`로 엔탈피 유속을 계산해 세 번째 종 순회를 제거한다. RHS와 stage 갱신은 여전히 별도 커널이므로 이웃이 읽는 입력 장을 갱신 중에 덮어쓰지 않는다.

## 희소 화학에서 바꾼 부분

최신 선형화의 `Jv`와 재사용할 전처리기 Jacobian을 서로 다른 저장공간에 둔다. `Jv`는 같은 종 배열 내용일 때만 재사용한다. 에너지·셀·적분 구간·Jacobian 정책을 바꾸는 `reset()`에서는 수치 캐시를 무효화한다. 음수 Newton 시험값의 원래 값도 키에 포함되어 마스크가 달라진 상태를 같은 상태로 취급하지 않는다.

`precSetup()`은 SUNDIALS의 `jok`를 따른다. 재사용이 허용된 경우 전처리기용 이전 `S`를 보존하되, 현재 `gamma`의 `I − gamma S`를 다시 수치 인수분해한다. 최신 `Jv`의 `S` 및 온도·압력 결합은 덮어쓰지 않는다. [SUNDIALS의 재사용 계약](https://sundials.readthedocs.io/en/v7.4.0/cvode/Usage/index.html#preconditioning)

CSC의 outer/inner 배열을 검사하고 실제 패턴이 바뀔 때만 CSR row/column, CSC→CSR 값 매핑, 전처리기 대각·값 매핑을 다시 만든다. 작은 값이나 수치적 0을 제거하지 않는다. Eigen `analyzePattern()`은 패턴 변경 때만 호출하고, `factorize()`는 필요한 수치 분해마다 호출한다. [Eigen SparseLU 문서](https://libeigen.gitlab.io/eigen/docs-nightly/classEigen_1_1SparseLU.html)

각 `Model`에 밀집·희소 작업공간을 하나씩 보관한다. context, N_Vector, constraints, CVODE, 선형 풀이 객체와 구조 정보를 재사용하되 매 독립 source 적분에서 `CVodeReInit()`으로 **BDF 이력을 초기화**한다. 다른 셀의 이력을 이어 쓰지 않는다. setup 실패로 부분 초기화된 객체는 폐기한다. 이 객체를 여러 스레드가 공유하도록 만든 것은 아니며, 향후 CPU 병렬화에서는 작업자별 `Model`이 필요하다. [CVODE 재초기화 문서](https://sundials.readthedocs.io/en/v7.4.0/cvode/Usage/index.html#cvode-reinitialization-function)

별도 `PintleChemicalProfile`을 추가해 기존 통계 ABI 크기를 유지했다. `Jv`/전처리기 callback, 실제 Jacobian 구성, 패턴 생성, 구조 분석, 수치 분해, 작업공간 생성·재초기화, LU 저장 개수를 구분한다. 시간은 Jacobian 구성의 열역학 복원·Cantera 미분·CSR 처리와 구조 분석·수치 분해·삼각 풀이의 host 경과시간이다. 전체 RHS/flash 또는 GPU 시간을 모두 측정하는 프로파일러는 아니다.

특히 `kineticsSeconds`는 미분 설정, `ddCi`, 질량 단위 변환, 부분 몰 열역학, `ddT/ddP`, rank 벡터 구성을 포함한다. 20 bar 기록의 약 2.198초를 순수 반응률 계산 시간으로 분류하지 않는다. `auto`는 앞선 희소 적분의 구조를 재사용한 기록이므로 최초 희소 적분과 같은 초기 조건의 성능 비교도 아니다.

## 실제 검사 결과

| 검사 | 결과 | 증거 |
|---|---|---|
| 수송·상주 API·CFL | 21개 통과 | [결과](results/reactive-review-transport-final-20260912.json) |
| 기존 화학 회귀 | 6개 통과 | [결과](results/reactive-review-chemical-regression-20260912.json) |
| 기존 413종 희소 화학 검사 | 4그룹 통과 | [결과](results/reactive-review-sparse-20260912.json); 1 bar의 거부·복구 포함 |
| 캐시·확대 화학·실제 상태 수송 | 6그룹 통과 | [결과](results/reactive-review-expanded-final-20260912.json) |
| CUDA 컴파일·링크 | 통과 | CUDA 13.4.59, compute_80/sm_80 |
| 실제 CUDA 실행 | 미검증 | driver/runtime 오류로 장치 생성 실패 |
| SPUMA 전체 빌드·실행 | 미검증 | 실행 환경에 SPUMA/wmake 없음 |

수송 검사에는 비축정렬 법선, 비균일 셀 연결수, 4,099개 경계면, 413종 확산, 255/256/257셀 및 65,535/65,536/65,537셀의 reduction 경계, 버전 불일치·rollback·같은 포인터의 새 내용, 오류 전파가 포함된다. 큰 셀 수 검사는 최솟값 reduction의 세 번째 단계에 집중한다. 임의 연결 그래프를 실제 닫힌 3차원 CFD 메시라고 주장하지 않는다.

실제 C++ 열역학 백엔드로 만든 413종 기상, 기체–IPA 액체 공존, 4종 PR 순수 액체 상태에서도 수송 RHS·경계 수지·CFL을 독립 NumPy 식과 대조했다. 최대 연산자 scaled error는 약 `2.47e-14`였다. 이는 실제 상태에서의 연산자 시험이며 SPUMA 전체 시간 적분 시험은 아니다.

초기 확대 검사의 [실패 기록](results/reactive-review-expanded-20260912.json)도 보존했다. 기상 체적분율 `0.99908`인 공존 상태를 임의의 `<0.999` 기준으로 잘못 제외했던 fixture를 활성 액상·질량 검사로 고쳤다. 비축정렬 반사벽의 거의 0인 에너지 순유속은 큰 항의 상쇄를 포함하므로, 새 경계 대조는 셀별 유속 규모로 정규화한 같은 `3e-12` 허용오차를 사용한다. 기존 연산자·보존성 검사의 허용오차는 완화하지 않았다.

추가 화학 검사는 1400/2000/2600 K, 서로 다른 희석률·생성물 조성, 음수 시험값 미분 마스크와 PLOG knot 양쪽을 포함한다. knot 자체의 미분 가능성을 가정하지 않는다. 세 조건 × 밀집 전체 RHS/밀집 구조화/희소의 9개 적분에서 재사용 객체와 새 객체의 온도·종 상태 차이는 기록된 정밀도에서 0이었다.

20 bar·2200 K·1 µs의 새 희소 기록은 다음과 같다.

| 항목 | 값 |
|---|---:|
| `Jv` setup / 실제 Jacobian 구성 | 588 / 589 |
| 전처리기 setup / Jacobian 재사용 | 52 / 42 |
| CSR 패턴 생성 / 구조 분석 / 수치 분해 | 1 / 1 / 52 |
| `S` 저장 항목 / `L+U` 저장 항목 | 57,132 / 83,579 |
| 밀집 기준 대비 온도 상대차 | `5.39e-11` |
| 밀집 기준 대비 질량분율 최대차 | `2.38e-11` |

이전 커밋은 전처리기 51회마다 `compute()`를 호출했다. 이번에는 구조 분석이 1회이며 동일 패턴으로 이어지는 `auto` 재적분은 0회다. 그러나 `jok` 재사용으로 Krylov/비선형 반복 경로도 바뀌므로 전체 Jacobian 구성 횟수가 같은 비율로 줄지는 않는다.

**1 bar·2200 K·20 µs의 엄격 희소 적분은 여전히 음수 미량 종 때문에 거부된다.** `auto`는 같은 초기 보존량·에너지·기구로 밀집 재적분해 독립 ReactorNet 대조를 통과했다. clipping이나 조성·에너지 보정으로 통과시키지 않았다. 새 단발 기록에서도 `auto`가 밀집보다 느렸다. 시험 일부는 겹쳐 실행했으며, JSON 시간은 반복 성능 벤치마크가 아니다. 원래 RTX 5080 환경에서의 가속률을 주장하지 않는다.

## 재현과 다음 단계

원래 환경에서는 기존 실행 잠금 규칙을 지키고 새 출력 경로를 사용한다. 아래 source 변경은 원래 Cantera/Eigen/SUNDIALS 고정 환경에서도 다시 빌드해야 한다. 이번 C++ 수치 검증은 최초 PR과 같은 별도 Cantera 3.2.0/SUNDIALS 7.4.0/Eigen 3.4.1 환경에서 수행했다.

```bash
PINTLE_REACTIVE_CACHE_TEST=1 bash tools/build_reactive_backend.sh
bash tools/build_reactive_transport.sh
research/reactive-env/bin/python tools/validate_reactive_transport.py \
  --backend cpu --output results/local-review-transport.json
research/reactive-env/bin/python tools/validate_sparse_chemistry.py \
  --thermo-dir research/reactive-thermo --output results/local-review-sparse.json
research/reactive-env/bin/python tools/validate_reactive_review.py \
  --thermo-dir research/reactive-thermo --backend cpu \
  --output results/local-review-expanded.json

PINTLE_REACTIVE_TRANSPORT_BUILD=cuda PINTLE_CUDA_ARCH=120 \
  bash tools/build_reactive_solver.sh
research/reactive-env/bin/python tools/validate_reactive_transport.py \
  --backend cuda --output results/local-review-cuda.json
research/reactive-env/bin/python tools/validate_reactive_review.py \
  --thermo-dir research/reactive-thermo --backend cuda \
  --output results/local-review-expanded-cuda.json
```

이어 실제 GPU의 Compute Sanitizer, SPUMA CPU/CUDA 동일 케이스, 단계 거부·재시작·수지, Nsight의 API별 전송·커널 시간을 확인해야 한다. 기본 경로 변경이나 병합을 대신 수행한 것은 아니다.

| 리뷰의 다음 제안 | 이번 상태 |
|---|---|
| GPU `gasY/h`, 무액상 이상기체 1변수 T 복원, GPU 진단 | 후속. 현재 CPU 생성·복원·진단과 두 번의 RK 다운로드가 남음 |
| pinned staging·이벤트 파이프라인·tiled transpose | 후속. 이번에는 불필요한 업로드/전치부터 제거 |
| 입력·출력 분리형 RHS/RK 융합 | 후속. 이웃 읽기와 rollback 버퍼 수명을 먼저 검증해야 함 |
| 제3체 공통항 분리·Woodbury·GPU 배치 화학 | 후속 연구. 넓은 미분 항을 임의 삭제하지 않음 |
| 검증된 축소 기구 패키지 | 별도 과제. 413종·300만 셀을 16 GB GPU에 수용한 상태가 아님 |

이번에 완료한 범위는 리뷰의 우선 세 개선과 수송 등가식 정리·검증 보강이다. 원래 제안의 GPU 화학·열역학 전체 이전은 계속 별도 단계로 남는다.
