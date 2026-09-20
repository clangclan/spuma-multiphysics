# SPUMA 실유체 최적화 v2 구현 보고서

> 이 문서는 v2 구현 당시의 기록이다. 최신 검증과 flash 기준 풀이 유지에 관한 수정은 [2026-09-21 검증 보고서](../../reports/pr1-local-validation-20260921.md)를 기준으로 한다.

작성: 2026-09-19 · 지시서: **SPUMA-RF-OPT-v2** · 기준: `3c3cd49b33ba090b746ce787ca11181f18c50a55`

## 1. 이번 변경의 범위

실제 EOS와 상평형을 보존하면서 **모델 식별, 동일 EOS 물성 조회와 온도 복원, 공통 접선, GPU 수송 메모리, 독립 CPU worker**를 개선했다. 새로운 source Jv는 진단 C API로 제공한다. 지시서 §14의 단계 구분에 따라 구현했으며 **P0~P11 전체가 완료된 상태는 아니다.** 특히 새로운 고압 상세반응 물리, 비이상 확산, 고차 압력평형 flux를 구현했다고 주장하지 않는다.

사용자 요청에 따라 검증은 작은 수학·코드 검사와 컴파일로 제한했다. 장시간 유동/반응 campaign, 전체 SPUMA 실행, 실제 GPU 실행, 성능 측정은 로컬에서 수행할 후속 항목이다. 이 보고서는 이번 변경을 설명하는 단일 보고서이며, 아래 JSON은 그 근거 데이터다.

착수 시 로컬 HEAD와 원격 PR #1 head가 위 기준과 일치했고 기존 미커밋 변경은 없었다. 이전 결과를 덮어쓰지 않고 `results/real-fluid-v2-20260919/`를 새로 만들었다. 입력·구성 파일 hash는 [baseline-manifest.json](../../results/real-fluid-v2-20260919/baseline-manifest.json)에 있다. 첨부 `report(1).html`은 과거 단상/3상 분석 자료로만 취급했으며 현재 ReactiveFoam 성능으로 사용하지 않았다.

## 2. 실제 구현한 내용

### 2.1 모델과 수치 정책을 별도로 식별

- 기존 `pintle-reactive-thermo-v3` checkpoint fingerprint 계산을 보존했다. 따라서 이전 재시작 입력의 의미를 바꾸지 않는다.
- 새로운 `physicalModelHash`는 선택 EOS, 원본 기구/액상 파일 내용, 종 순서·계수, 상 구성과 온도·압력 영역을 식별한다. Flow에서는 closure, 반응 활성화와 처방된 점성·열전도·확산 계수도 hash에 포함한다.
- `numericalPolicyHash`는 열역학 허용오차, scalar 복원 선택, 화학 Jacobian/선형해법과 Flow의 source tolerance·파속 계수·수송 backend 선택을 구분한다. 시간/격자/worker 수 등 전체 실행 입력은 별도의 case 파일 hash로 추적하며 이 hash 하나가 모든 실행조건을 뜻하지 않는다.
- 체크포인트 schema 2에 두 hash를 추가했다. 기존 schema 1의 fingerprint 검사는 유지하며, 새 물리 hash가 있으면 함께 검사한다. 수치 정책이 다르다는 이유만으로 EOS가 바뀌었다고 판정하지 않는다. 단, 기존 v3 fingerprint에 포함된 설정을 바꾸는 재시작에는 종전 제한이 여전히 적용된다.
- strict YAML policy loader가 unknown key, EOS downgrade, accepted-state clipping, energy reset, 미검증 근사표·정밀도 모드를 거부한다. 정책은 [real-fluid-optimization-v2.yaml](../../policies/real-fluid-optimization-v2.yaml)이다.
- capability는 `realEOS`, `phaseEquilibrium`, `partialMolarEnthalpy`, `thermoJvp`, `nonidealDiffusion`, `deviceClosure`, `deviceKinetics`, `mixtureLiquid`로 분리했다. 마지막 네 기능은 현재 false다. `thermoJvp`는 매끄러운 고정 활성상에서 제공하는 접선이며 상전이 전체의 도함수를 보장하지 않는다.

3개 기준 구성의 종·NASA 계수/구간·기준압력·EOS 데이터·반응 유형·액상 정의와 누락 자료를 [physical-model-manifest.json](../../results/real-fluid-v2-20260919/physical-model-manifest.json)에 기록했다. [capability-matrix.json](../../results/real-fluid-v2-20260919/capability-matrix.json)은 각 구성의 지원 상태다. 상세화학 기상 파일에는 413종 모두 명시적 PR 기상 계수가 없으며, 별도 순수 액상 데이터가 있다고 해서 전체 실가스 반응기구가 정의되는 것은 아니다. 누락 계수나 이성분 상호작용을 새로 만들어 넣지 않았다.

### 2.2 동일 EOS의 선택 물성 인터페이스

신규 `pintleRealFluid.h`는 기존 C ABI 구조체를 변경하지 않는 추가 인터페이스다. `pintle_rt_evaluate_real_fluid()`의 독립변수는 `(T,rho,Y)`이고 기상 또는 지정 순수 액상 branch를 명시한다. 직접 EOS에서 압력·내부에너지·엔탈피·엔트로피·cp/cv를 계산하고 요청 mask에 따라 다음을 추가 계산한다.

- 기준상태와 잔차: `e=e0+er`, `h=h0+hr`, `s=s0+sr`. 엔트로피 기준은 같은 `(T,rho,Y)`의 이상기체 기준상태와 혼합항이다. 실제 반환 물성은 항상 원 EOS 값이다.
- 고정 조성 압력 미분: `dp/dT|rho,Y = expansion/compressibility`, `dp/drho|T,Y = 1/(rho*compressibility)`.
- 단상 음속 항등식: `c² = p_rho + T*p_T²/(rho²*cv)`. 다상 평형 음속에는 이 식을 대입하지 않는다.
- 지정 종의 **실제 부분몰 엔탈피/W** 및 chemical potential/W. 기준상태 NASA 값만 실가스 부분몰량으로 반환하지 않는다.

Cantera의 공개 API로 요청 branch의 밀도근을 다시 확인한다. PR `getPartialMolarCp()`는 호출하지 않는다. 부분몰량을 요청하지 않으면 해당 배열을 생성하지 않는다. 선택 종을 요청하더라도 Cantera의 내부 API는 전체 종 배열을 계산할 수 있으므로 종 수와 무관한 비용이라고 주장하지 않는다. 실패 시 호출자의 결과 buffer를 갱신하지 않는다.

### 2.3 실가스 기상 후보의 온도 1변수 복원

액상 질량이 정확히 0인 기상 후보에서 `rho=sum(q)`, `Y=q/rho`를 고정하고 다음 식을 푼다.

\[
F(T)=\rho e_{EOS}(T,\rho,Y)-\varepsilon,\qquad F'(T)=\rho c_{v,EOS}(T,\rho,Y).
\]

이전 온도에서 시작해 동일 branch에서 에너지 bracket을 찾고, Newton step이 bracket 밖으로 나가면 이분법을 사용한다. EOS 압력/온도 영역, 양의 cv·등온 압축률, branch 일치와 최종 체적/에너지 잔차를 확인한다. 실패하면 **기존 동일 EOS의 p–T 복원**으로 돌아간다. `reference_pT` 정책으로 이 빠른 경로를 끌 수 있다.

이는 기상 후보의 inversion을 바꾼 것이다. 응축성 종이 있는 모델의 absent-phase 안정성 검사, 모든 기존 active-set/multistart 후보 및 최대 엔트로피 비교는 유지한다. 여러 순수 액체의 no-gas 상태는 scalar 기상 경로에 들어가지 않는다. NASA 계수합 캐시와 PR exact mixing 압축은 아직 구현하지 않았다.

### 2.4 공통 flash 잔차·작은 선형계·방향미분

고정 활성상의 잔차를 `closureResidual()`로 모아 기존 structured chemistry와 새 접선에서 공유한다. 평형은 기존 flash 잔차를 사용하고, frozen chemistry는 **액상 분율이 아닌 액상 질량**을 고정한다.

작은 선형계는 행·열 scaling과 pivoted LU로 풀며 역행렬을 만들지 않는다. 새 접선 API는 scaled reciprocal condition과 원 방정식의 선형 잔차를 반환한다. flash에서는 scaling 검사가 실패하면 기존 FP64 pivoted solve로 돌아가므로 새 screening 기준만으로 기존 엔트로피 후보를 제거하지 않는다.

`pintle_rt_thermo_tangent()`는 `F_z dz = -F_q v - F_epsilon d_epsilon`를 방향차분으로 적용한다. 로그의 기준은 `log(p/1 Pa)`, `log(T/1 K)`다. 무기상, 소멸/출현 경계, 작은 absent-phase 안정성 여유는 고정 branch 접선을 거부한다. 현재 미분은 bounded finite difference이며 analytic/AD 완료를 뜻하지 않는다.

`pintle_rt_chemical_matrix_free_jvp()`는 `Jv=R_q v+R_z dz`를 계산한다. `Ns×Ns` 행렬이나 `F_q` 전체를 만들지 않고 작은 `F_z`, 종 벡터와 방향 probe를 사용한다. 고정 branch 적용이 어려우면 같은 EOS의 전체 source 방향차분으로 복구한다. **진단 API이며 기존 CVODE 적분기 선택에는 아직 연결하지 않았다.** 기존 최신 Jv cache와 preconditioner snapshot의 분리도 유지한다. 동적-rank Woodbury·제3체 공통항 생성기는 후속 작업이다.

### 2.5 GPU 수송의 메모리와 명시적 상태 수명

- 전체 셀 수만큼 만들던 AoS↔SoA `bridge`를 `transportBridgeCells` 단위로 제한했다. 마지막 부분 tile도 처리한다. 기본은 256셀이다.
- SSPRK2는 두 SoA 보존장 사이를 교대한다. `B=A+dt L(A)`, `A=.5 A+.5(B+dt L(B))`를 gather/update 커널에서 계산한다. 이웃 읽기는 입력장, 쓰기는 별도 출력장으로 분리한다.
- 정상 RK에서 전역 `rhs`를 할당하지 않는다. 기존 RHS 진단 API를 호출할 때만 지연 할당한다.
- 생성 NASA 물성 경로에서 내부 셀 전체의 `gasY/gasH` 배열을 없앴다. 종 질량분율과 엔탈피를 필요한 면·종 계산 시 재구성한다. 고정 경계의 작은 불변 물성 배열은 유지한다. host 물성 비교 경로는 기존 전체 배열을 유지한다.
- 새 resident advance는 보존장을 자동 다운로드하지 않는다. Flow는 **CPU flash가 소비하는 시점에** `download_conserved()`를 명시적으로 호출한다. 따라서 CPU flash용 두 다운로드는 여전히 남는다.
- transport의 model hash 및 `attemptId/stageId/contentVersion`을 검사한다. stale download/완료를 거부한다. 실패 시 resident 상태를 무효화하고, 새 시도는 새 버전으로 원 상태를 업로드한다.
- `liveBytes/peakBytes`, bounded bridge, 보존장·물성·진단 RHS, D2D 바이트와 동기화 횟수를 별도 조회한다. 이는 backend allocator가 센 메모리이며 CUDA 전체 프로세스 또는 물리 VRAM residency 측정은 아니다.

수학적 주요 저장량은 생성 물성의 정상 RK에서 기존 `4Q+2G`에서 **`2Q + bounded bridge + 고정 경계/상분배/격자/기타 scratch`**로 바뀐다. 300만 셀·413종에서는 두 보존장만 **20.016 GB**이므로 16 GB GPU에 전체장을 담을 수 있다는 뜻이 아니다. host-backed streaming, domain decomposition, multi-GPU는 아직 구현하지 않았다.

전체 Strang rollback은 첫 화학 half-step 이전의 기존 host `q/states` snapshot이 계속 담당한다. 경계 적분은 성공한 스텝만 누적하고 실패 시 drift와 trial boundary를 버린다. 실패한 계산의 성능 카운터는 초기화하지 않는다. 두 device 작업장만으로 복구 메모리가 사라진다고 계산하지 않는다.

### 2.6 독립 CPU worker와 bounded batch

`thermoWorkers>1`이면 각 worker가 독립 Cantera 모델·CVODE 작업공간·수치 cache를 소유하는 persistent thread pool을 사용한다. 모델 생성 시 물리·정책 hash를 대조하고, 배치 제출 시 prototype 변경도 거부한다. 셀마다 기존 적분기와 tolerance를 사용하며 BDF 이력을 다른 셀에 넘기지 않는다.

한 배치는 `thermoBatchCells` 이하이다. 이전 phase/iteration 정보로 실행 순서를 정리하되 각 셀의 source interval과 Strang/RK 순서는 유지한다. 모든 worker가 끝나기 전에는 transport로 진행하지 않는다. 하나의 셀이라도 실패하면 **그 배치의 입력/출력은 부분 commit하지 않는다.** 앞선 배치가 성공한 후 다음 배치가 실패한 경우에는 전체 Flow rollback이 승인 snapshot을 복원한다.

배치 토큰은 시도·단계·내용 버전을 검사한다. 동기식 API이므로 실패한 시도의 worker가 다음 시도까지 남지 않는다. scratch 상한과 worker 수·배치 수·셀 수·실패 수를 계측한다. **Cantera/CVODE 내부 할당의 전체 peak는 아직 계측하지 않으며 scratch 상한에 포함되지 않는다.** 기본 worker 수는 1로 기존 순차 경로다. 다중 `mechanicalEquilibrium` worker, GPU/CPU 동시 비동기 스케줄러는 지원하지 않는다.

## 3. 간단한 검증 결과

| 검사 | 결과와 해석 |
|---|---|
| C++ thermo 빌드 | g++ 13.3, C++17, FP64, `-Wall -Wextra -Werror`, 링크 통과 |
| C++ transport 및 CUDA 컴파일 | CPU 공유 라이브러리와 nvcc 13.4.59 compute_80/sm_80 빌드·링크 통과. CUDA 실행은 수행하지 않음 |
| C ABI·문법 | C11 header 검사, 기존 thermo state 176B·transport profile 64B, Python/shell 문법 통과 |
| 동일 EOS 물성·정책·접선 | 두 작은 PR 상태에서 직접 Cantera 값과 일치, 물성 방향차분 오차 약 `1.44e-11`. 잔차 에너지가 약 −4.60/−4.89 kJ/kg로 유지됨 |
| scalar 복원 | 작은 검사에서 4회 시도·4회 수락, 온도 Newton 6회, p–T fallback 0회. 넓은 영역의 수락률이 아님 |
| matrix-free source Jv | 기존 PR 종 데이터를 쓰는 **합성 반응 harness**에서 방향차분 대비 scaled error 약 `2.56e-11`. 실제 연소기구 검증이 아님 |
| CPU 배치 | 독립 worker 2개·배치 3셀, 직렬 결과 대조·한 셀 오류 시 출력 보존·stale 거부·새 시도 복구 통과 |
| resident RK·메모리 | 413종·7셀·bridge 2셀에서 독립 NumPy flux/RK와 기록 정밀도에서 일치. 내부 gas 배열/RHS 0B, bridge 6,672B, 고정경계 gas 6,608B |
| 기존 C API | 작은 확산/경계 및 mechanical 수송 2개 확인. 전체 기존 21그룹 재실행은 생략 |
| Flow 변경 계약 | 실제 step의 비-const/버전 증가 부분 컴파일 통과. **전체 SPUMA 빌드가 아님** |

근거는 [code-smoke-final.json](../../results/real-fluid-v2-20260919/code-smoke-final.json), [legacy-api-smoke.json](../../results/real-fluid-v2-20260919/legacy-api-smoke.json), [flow-contract.json](../../results/real-fluid-v2-20260919/flow-contract.json), [code-verification.json](../../results/real-fluid-v2-20260919/code-verification.json)이다. 같은 Cantera를 이용한 직접 대조는 독립적인 실험 물성 검증으로 세지 않는다.

초기 smoke의 수송 항목은 시험 helper에 context-manager 기능이 없다는 이유로 실행 전 실패했다. `closing()`으로 수명을 명시한 뒤 같은 검사를 통과했으며 초기 JSON도 보존했다. C ABI 검사의 처음 예상값 184B는 시험 작성 오류였고 기존 정의/ctypes에 맞는 176B로 정정했다. 실제 ABI 구조를 바꾸어 맞춘 것이 아니다. Manifest 생성 시 YAML 1.1 파서가 종 이름 `NO`를 bool로 읽는 문제는 Cantera의 species 데이터 API로 변경해 해결했다.

## 4. 지시서 단계별 상태

| 단계 | 이번 상태 | 남은 부분 |
|---|---|---|
| P0 | 구현 | 로컬에서 생성 케이스/재시작을 포함한 G0 전체 확인 |
| P1 | CPU 조회·reference/residual·선택 부분몰량 구현 | analytic/AD, 생성 device EOS와 폭넓은 branch 검증 |
| P2 | scalar 동일-EOS 복원 구현 | NASA 합산 캐시, exact PR mixing 압축 및 계측 |
| P3 | 공통 잔차·scaled tangent 구현 | 접선 predictor의 실제 flash warm start 연결, 폭넓은 G3/G4 |
| P4 | 두 작업장·bounded bridge·recompute gas·transaction 구현 | CPU flash 왕복 자체 제거, pinned/event 파이프라인, 전체 host/worker peak, 분할 |
| P5 | 독립 CPU worker·bounded 동기식 batch 구현 | 실제 stiffness 기반 분배, 비동기 이종 스케줄링·전체 worker 메모리 계측 |
| P6 | matrix-free Jv **진단 API** 구현 | CVODE 연결·동적 Woodbury·제3체 공통항 최적화 |
| P7 | 미검증 근사 모드의 명시적 거부 | predictor 표·bounded-final·mixed precision과 해당 gate |
| P8 | 기존 nonideal diffusion guard 보존 | 필요한 물성/이동도 모델 정의 후 비이상 flux·CFL 구현 |
| P9 | 미구현, capability=false | 동일 모델 device kinetics와 reaction-type coverage |
| P10 | 기존 기준 경로 유지 | MUSCL·APEC/contact와 독립 정확도 campaign |
| P11 | 지시서의 별도 후속 단계 | 결합 DAE·multi-GPU·검증된 축소화학 |

전체 G0~G10 acceptance campaign은 수행하지 않았다. HEM 접촉면/조밀하지 않은 계면 이동의 기존 실패도 이번 변경으로 해결됐다고 표시하지 않는다. 고압 상세반응은 EOS·transport 자료가 확보되지 않은 채 production 지원으로 바뀌지 않는다. 근사표·FP32 전처리·축소기구는 활성화하지 않았다.

## 5. 로컬 재현과 후속 확인

원래 프로젝트 환경의 실행 잠금 규칙을 지킨 상태에서 다음과 같이 빌드한다. GPU arch는 해당 장치에 맞춘다.

```bash
bash tools/build_reactive_backend.sh
bash tools/build_reactive_transport.sh
research/reactive-env/bin/python tools/check_real_fluid_v2.py \
  --thermo-dir research/reactive-thermo \
  --output results/local-rf-v2-smoke.json

PINTLE_REACTIVE_TRANSPORT_BUILD=cuda PINTLE_CUDA_ARCH=120 \
  bash tools/build_reactive_solver.sh
```

기존 케이스에는 선택적으로 다음 설정을 추가한다. 새 case generator에도 같은 옵션을 추가했다.

```text
optimizationPolicy "/absolute/path/policies/real-fluid-optimization-v2.yaml";
thermoWorkers 2;
thermoBatchCells 64;
maxThermoBatchMemoryMB 64;
transportBridgeCells 256;
```

`thermoWorkers 1`은 기존 순차 경로, `transportGasProperties host`는 기존 host 물성 비교 경로다. `deviceNasa`는 여전히 **이상기체 확산용**이며 PR을 강제로 선택시키지 않는다. CPU shared library의 smoke 통과를 CUDA device 결과로 해석하지 않는다.

로컬에서는 우선 (1) schema 1/2 재시작과 case manifest 일치, (2) cold-pr의 `reference_pT` 대 scalar 및 상 출현/소멸, (3) 순차 대 2-worker의 동일 source interval과 실패 후 전체 rollback, (4) GPU host/recompute 물성 대조 및 Compute Sanitizer, (5) 같은 종료 물리시간의 메모리·전송·전체 wall time을 확인한다. 이후 나머지 G0~G10과 단계별 개발을 진행한다. 이번 결과에서 가속 배수나 전체 실행시간 감소율은 산출하지 않았다.
