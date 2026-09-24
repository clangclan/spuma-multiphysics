> 이 문서는 2026-09-12 포팅 당시의 기록이다. 현재 기본 솔버는 ReactiveFoam이며 ColdFoam은 제거했다. 최신 빌드·실행 방법은 [프로젝트 안내](README.ko.md)를 따른다.

# ReactiveFoam GPU 수송·희소 화학 포팅 — 2026-09-12

> 이 문서는 작성 시점의 구현·검증 기록이다. 2026-09-21의 실제 GPU·SPUMA 검증 및 재시작 수정 결과는 [PR #1 로컬 검증 보고서](reports/pr1-local-validation-20260921.md)를 기준으로 한다.

이 문서는 최초 포팅 커밋 `fad4ecf`의 구현·측정 기록이다. 이후 전송·reduction·희소 준비 재사용 개선과 최신 검증은 [리뷰 반영 문서](README.reactive-review.ko.md)에 분리했다.

`ReactiveFoam`에 선택 가능한 CUDA 수송 연산과 CPU 희소 화학 선형계를 추가했다. **기본값은 기존 CPU 수송·밀집 화학이다.** CUDA 소스 컴파일과 독립 연산자/화학 수치 시험은 수행했으나, 이 환경에는 실행 가능한 GPU와 SPUMA가 없어 실제 GPU 실행, 전체 솔버 빌드, 성능 향상은 검증하지 못했다. 병합 전 원래 환경에서 확인해야 하는 개발 단계다.

## 기존 진행상황과 이번 변경의 경계

시작점은 `main`의 `50512cecdc7f8047f3ccbb8c60e05a43222e1cc3`, tree `17c09d52cd349b7a63396a8d3386b1368c7160cf`다. 연결된 GitHub에서 파일별 blob SHA를 확인해 같은 tree를 복원했다. 원래 실험 호스트의 작업 디렉터리, 실행 중인 프로세스, 미커밋 변경을 확인한 것은 아니다.

| 구분 | 시작점에서 확인한 상태 | 이번 변경 / 검증 범위 |
|---|---|---|
| ColdFoam | SPUMA GPU 포트와 기존 성능 보고서 존재 | 소스·기존 결과를 변경하지 않음 |
| ReactiveFoam 수송 | CPU HLL, SSPRK2, Strang, 상수 점성·전도·확산 | 같은 연산의 CUDA 구현과 명시적 실행 선택 추가 |
| 열역학·상분배 | 호스트 Cantera/Eigen UV flash | 계속 CPU에서 수행 |
| 화학 | CPU CVODE BDF, 구조화 Jacobian, 밀집 선형계 | 이상기체·액상 없는 구성에 희소 블록 + rank-two 결합 추가 |
| 다상 정확도 | HEM 물질 접촉면 및 16셀 기계적 접촉면 이동 오차의 알려진 실패 | 해결했다고 주장하지 않으며 기존 판정 보존 |
| 실행 환경 | 기존 보고서의 Ryzen 9600X / RTX 5080 / SPUMA v2512 / CUDA 13.2 | 별도 x86-64 환경, CUDA 13.4.59 컴파일만 가능 |

이전 실험 수치는 [2026-09-11 보고서](reports/multiphase-review-fixes-20260911.md)에 그대로 있다. 이번 증거는 날짜가 붙은 새 `results/*20260912.json`에만 기록했다. 환경·소스 SHA는 [새 환경 기록](results/reactive-porting-environment-20260912.json)을 참조한다.

## 두 공유 채팅의 아이디어와 대조

[첫 번째 GPU 가속 포팅 분석](https://chatgpt.com/share/6aa4e705-b7bc-83e8-b246-28524ed275d0)과 [두 번째 분석](https://chatgpt.com/share/6aa4e72b-b6a0-83e8-a457-27baae3eed0f)의 본문을 읽고 비교했다. 첨부 HTML 보고서의 과거 실행 결과도 이번 측정과 구분했다.

| 제안 | 이번 반영 | 남은 작업 |
|---|---|---|
| C++/CUDA 유지, 수송부터 포팅 | HLL, 구배, 점성·전도·종 확산, CFL, SSPRK2 커널 | 실제 SPUMA 연동 빌드·실행·프로파일 |
| SoA, cell CSR gather, 지속 버퍼 | 장은 device SoA, 셀–면 CSR gather, 면×종 flux 저장/FP64 atomic 없음 | host flash 때문에 남은 전체 장 왕복 제거 |
| 희소 화학 + 작은 열역학 결합 | `J = S + uT vTᵀ + uP vPᵀ`의 전체 Jv, sparse LU 전처리 | 다상 `A − B solve(D,C)`의 일반 rank≤4 경로 |
| 반응 참여뿐 아니라 실제 미분 의존성 | Cantera `ddCi`, `ddT`, `ddP`; 제3체·falloff 유지, PLOG 시험 | 생성 커널의 전체 의존성·수치 검증 |
| 검증된 고정 축소 기구 패키지 | 종·반응을 삭제하지 않고 기존 413종·14,922반응으로 구현 오차 비교 | 축소 기구, 종 매핑, 적용 범위, hash, 코드 생성기 |
| 작은 축소 기구의 GPU batched dense LU도 비교 | 현재 기본 밀집 풀이를 기준 경로로 보존 | GPU dense/sparse 실측에 따른 선택 |
| 셀별 적응 적분·같은 기구의 CPU 복구 | 현재 셀별 CVODE 유지, `auto`에서 같은 초기 상태·기구로 밀집 재적분 | worker별 Cantera/CVODE 재사용, GPU 배치 적분·셀별 오차 제어 |
| 순수 이상기체의 1차원 T 역산 | 희소 경로를 구성 자체에 액상이 없는 이상기체로 제한 | GPU T 역산 및 상분배 계산 |

따라서 이번 변경은 제안의 첫 단계이며 전체 GPU 화학 포트의 완성은 아니다. 액체를 허용하는 모델에서 순간 액상량이 0이라는 이유로 순수 기체 전용 경로를 선택하지 않는다. 축소로 생기는 모델 오차와 포팅으로 생기는 구현 오차를 섞지 않았으며, 누락 종을 0으로 채우는 재구성도 도입하지 않았다.

## GPU 수송 구현

`src/reactiveTransport`는 SPUMA와 분리한 C ABI를 제공한다. CPU 연산자 시험 모드와 CUDA 모드는 같은 C++ 연산자를 사용한다. 실제 솔버의 `transportBackend cpu`는 기존 face-scatter 경로를 유지한다. `cuda`는 CUDA 빌드와 장치를 요구하며, 실패를 CPU 실행으로 숨기지 않는다.

GPU 장은 variable-major SoA다. host ABI는 기존 cell-major를 유지해 별도 버퍼로 변환한다. 메시·CSR·고정 경계·수치 작업 버퍼는 생성 시 할당하고 재사용한다. HLL 공통 면 정보만 저장하고 각 셀·보존 변수의 발산을 CSR로 모은다. 양쪽 셀은 같은 면 유속식을 평가한다. 확산의 혼합물 질량 유속 보정, 생성 엔탈피 수송, 점성 일, 기계적 평형의 체적분율 소스도 기존 식을 유지한다.

계산은 FP64이며 fast math를 사용하지 않는다. 하나의 명시적 CUDA stream에 연산을 순서대로 넣고, CPU가 결과를 필요로 하는 API 경계에서 동기화한다. 각 RK 단계 뒤 CPU flash와 CFL 검사가 있어 장 업로드·다운로드가 남는다. 전체 단계 거부 시 host 상태를 복원하고 다음 stage 0이 device 초기 복사본을 교체한다. `REACTIVE_TRANSPORT` 로그는 할당량·전송량·커널 호출·RK/CFL 횟수를 출력한다.

`maxDeviceMemoryGB`는 이 라이브러리의 지속 버퍼에 대한 **십진 GB** 한도다. SPUMA 메모리 풀, CUDA context, host 배열은 포함하지 않는다. 현재 약 네 벌의 보존장과 메시 작업공간이 필요하며 확산을 켜면 기상 조성·엔탈피 배열도 추가된다. 300만 셀·413종의 네 벌 보존장만 약 **37.3 GiB**이므로 16 GB GPU에 전체 상세 기구가 들어가는 구현이 아니다. 고정 경계 수가 크면 임시 변환 버퍼도 커진다. 검증된 축소 기구 또는 다른 전역 메모리 설계가 필요하다.

## 희소 화학 구현

순수 이상기체의 보존량은 `q[k] = rho Y[k]`, 농도는 `C[k] = q[k]/W[k]`다. 고정 부피·내부에너지에서 다음을 사용한다.

```text
S[i,j] = W[i] / W[j] * ∂omega[i]/∂C[j]  (T, p 고정)
vT[j]  = -e[j] / sum(q[k] cv[k])
vP[j]  = R T / W[j] + (p/T) vT[j]
uT[i]  = W[i] * ∂omega[i]/∂T
uP[i]  = W[i] * ∂omega[i]/∂p
J v    = S v + uT (vTᵀv) + uP (vPᵀv)
```

Cantera가 제공하는 농도 미분의 저장 항목을 모두 CSR로 옮기며 임계값으로 가지치기하지 않는다. 제3체·falloff 미분을 생략하지 않는다. `nonzeros`는 수치적으로 0인 저장 항목도 포함하는 저장 개수다. Cantera의 미분 API는 일부 속도 미분에 수치 차분을 사용할 수 있으므로 전체가 기호적으로 생성된 해석 Jacobian이라고 주장하지 않는다.

CVODE SPGMR은 위 **전체 Jv**를 사용한다. Eigen SparseLU로 `I − gamma S`를 인수분해해 전처리하며, 열역학 rank-two 항은 Krylov 곱에 남긴다. Sherman–Morrison–Woodbury 직접 풀이 또는 GPU 희소 LU를 구현한 것은 아니다. Cantera·CVODE 객체와 CSR 구성은 아직 각 셀 적분에서 생성된다.

| `chemicalLinearSolver` | 동작 |
|---|---|
| `dense` | 기본값. 기존 밀집 선형계와 기존 구조화/전체 RHS 차분 정책 |
| `sparse` | `chemicalJacobian structured`, 이상기체, 구성에 액상 없음이 필수. 적분 실패·음수 채택 상태를 거부 |
| `auto` | 가능한 구성에서 희소 시도. 실패하거나 음수/비유한 채택 상태이면 같은 초기 보존량·에너지·기구로 전체 RHS 차분 밀집 재적분. 다른 구성은 밀집 사용 |

`auto`는 성능 자동 튜닝이 아니다. 재적분 때문에 기본 경로보다 느릴 수 있다. Newton 내부의 허용된 작은 음수 시험값에만 기존 RHS 확장과 같은 0 경계를 적용하고 해당 미분 열을 마스킹한다. **채택할 화학 상태는 음수 clipping, 조성 재정규화, 에너지 재설정으로 보정하지 않는다.** 기존 질량·원소 검사를 통과해야 한다. 통계 ABI를 깨지 않도록 희소 통계는 별도 `ReactiveSparseStats`로 제공한다.

## 빌드와 선택

[기존 환경·반응 자료 준비 절차](README.reactive-phase.ko.md)를 먼저 따른다. 새 화학 빌드는 SUNDIALS SPGMR 라이브러리를 추가로 링크한다. Eigen SparseLU를 사용하므로 KLU/SuiteSparse를 추가로 요구하지 않는다.

```bash
# CPU 수송 라이브러리 + 화학 백엔드 + SPUMA 애플리케이션
bash tools/build_reactive_solver.sh

# 원래 RTX 5080 환경에서 CUDA 수송을 포함해 새로 빌드
REACTIVE_TRANSPORT_BUILD=cuda REACTIVE_CUDA_ARCH=120 \
  bash tools/build_reactive_solver.sh
```

`NVCC`로 compiler 경로를 지정할 수 있다. `REACTIVE_CUDA_ARCH` 기본값은 `80`이며 지정 아키텍처의 native code와 PTX를 포함한다. 테스트 환경의 CUDA 13.4 산출물을 원래 CUDA 13.2 환경으로 복사하지 말고 해당 환경에서 다시 빌드한다. 원래 실험 호스트에서는 기존 `/home/jsw/cae-benchmark/run.lock` 직렬 실행 규칙을 계속 적용한다.

```foam
transportBackend cuda;
maxDeviceMemoryGB 2;
chemicalJacobian structured;
chemicalLinearSolver auto;
```

새 케이스 생성기의 `--transport-backend cuda --chemical-linear-solver auto`도 위 선택을 기록한다. 예를 들어 기존 환경에서 작은 `--kind chemistry` 케이스를 별도 경로에 생성한다. 기본 생성값은 `cpu`, `dense`다. 실행 로그의 `REACTIVE_BACKENDS`는 수송·열역학·화학 실행 위치를 구분한다.

## 이번 환경에서 실제 검증한 것

| 검사 | 결과 | 증거와 한계 |
|---|---|---|
| CUDA transport 번역·링크 | 통과 | CUDA 13.4.59, `compute_80` + `sm_80`. device 실행 아님 |
| 수송 독립 연산자 대조 | 14/14 | [JSON](results/reactive-transport-portable-20260912.json). NumPy face-scatter 기준과 CPU로 실행한 C++ 커널 비교; 최대 scaled error `4.80e-16` |
| 기존 화학 회귀 | 6/6 | [JSON](results/reactive-chemical-regression-20260912.json). 기체/평형 액상/고정 액상 Jacobian, CVODE, 상 경계 fallback |
| 희소 화학 추가 검사 | 4/4 그룹 | [최종 JSON](results/reactive-sparse-chemistry-final-20260912.json). 아래의 엄격 모드 실패를 포함한 판정 |
| 실제 CUDA 실행 | 미검증 | 장치 probe가 `CUDA driver version is insufficient for CUDA runtime version`으로 실패 |
| SPUMA 전체 빌드·케이스 | 미검증 | `wmake`·SPUMA 설치 없음 |
| GPU 성능·메모리 안전성 | 미검증 | 실제 GPU 프로파일·Compute Sanitizer 미실행 |

수송 시험은 1/7/257셀, 4/8/53/413종, cyclic/slipWall/extrapolate/fixedState, 점성·전도·확산, 기계적 체적분율 소스, 두 단계 최소값 reduction, SSPRK2 및 단계 재시작을 포함한다. 이는 3차원 실제 메시·전체 PDE 시간 적분의 실행 증거는 아니다.

413종·14,922반응 시험은 2200 K, `N2O:3, IC3H7OH:1, N2:100` 조성이다. 희소 `Jv`와 전체 flash 방향 차분의 최적 오차는 존재 종 방향 약 `2.0e-9`, 부재 종 방향 약 `3.4e-11`이었다. 별도 PLOG 반응 시험의 오차는 `7.93e-8`이다. 저장 블록은 `57,132 / 170,569` 항목(약 33.5%)으로, 매우 낮은 밀도라고 볼 수 없다.

- **1 bar, 20 µs:** 엄격 희소 적분이 음수 미량 종의 채택 상태를 반환해 거부됐다. `auto`는 희소 1회 후 밀집 1회로 복구했다. 밀집 기준 대비 온도 상대차 `1.83e-9`, 질량분율 최대차 `7.14e-10`; 독립 Cantera ReactorNet과도 통과했다. 이 조건의 엄격 희소 적분은 아직 성공하지 않는다.
- **20 bar, 1 µs:** 엄격 희소와 `auto` 모두 밀집 fallback 없이 통과했다. 밀집 기준 대비 온도 상대차 `4.78e-11`, 질량분율 최대차 `3.26e-11`. 이상기체 수치 시험이며 고압 다상 연소의 물리 검증은 아니다.
- 초기 검사의 실패는 [초기 JSON](results/reactive-sparse-chemistry-20260912.json)에 보존했다. 최종 검사는 1 bar 엄격 희소 실패를 숨기지 않고 거부·자동 복구 동작을 검사한다. 정확도 허용오차를 완화한 것이 아니다.

JSON의 시간은 이 환경의 단일 실행 진단이다. 일부 회귀 시험은 서로 겹쳐 실행됐으며 성능 비교 캠페인이 아니다. 1 bar의 `auto`는 밀집보다 느렸다. 기존 GPU 속도 향상 수치를 ReactiveFoam에 전용할 수 없다.

화학 라이브러리는 실제 Cantera 3.2.0 PyPI wheel, SUNDIALS 7.4.0, Eigen 3.4.1, fmt 11.2.0 헤더로 C++ 소스를 컴파일해 호출했다. wheel에 없는 SUNDIALS Dense 선형 풀이·인수분해는 공식 7.4.0 소스에서 컴파일했다. 원래 고정 환경은 Eigen 5.0.1/SUNDIALS 7.5.0을 사용한다. 따라서 원래 conda 고정 환경이나 `build_reactive_backend.sh`의 링크 구성을 그대로 재현한 검증은 아니다. 상세 기구는 기존 공식 CRECK revision `640d5b492d849e33c08bfbbd9d3de25dd074c9a4`에서 재생성했다.

## 원래 환경에서 이어서 실행할 검사

```bash
# 새 경로를 사용한다. 검사 도구는 기존 결과 파일을 덮어쓰지 않는다.
bash tools/build_reactive_transport.sh
research/reactive-env/bin/python tools/validate_reactive_transport.py \
  --backend cpu --output results/local-transport-cpu.json

REACTIVE_TRANSPORT_BUILD=cuda REACTIVE_CUDA_ARCH=120 \
  bash tools/build_reactive_transport.sh
research/reactive-env/bin/python tools/validate_reactive_transport.py \
  --backend cuda --output results/local-transport-cuda.json

research/reactive-env/bin/python tools/validate_sparse_chemistry.py \
  --thermo-dir research/reactive-thermo --output results/local-sparse-chemistry.json
research/reactive-env/bin/python tools/validate_chemical_jacobian.py \
  --thermo-dir research/reactive-thermo --output results/local-chemical-regression.json
```

이후 CPU/dense, CUDA/dense, CPU/auto, CUDA/auto 조합으로 같은 작은 케이스·허용오차·초기 보존장을 비교한다. 전체 솔버의 CFL/거부·재시작, 경계 질량·원소·총에너지 수지, 기존 다상 회귀 결과를 확인한다. Compute Sanitizer와 Nsight로 장치 메모리·전송·커널 시간을 측정한 뒤 축소 기구/GPU batched 화학 설계를 진행한다. 실제 장치에서 검증되기 전에는 기본 실행 경로를 변경하지 않는다.
