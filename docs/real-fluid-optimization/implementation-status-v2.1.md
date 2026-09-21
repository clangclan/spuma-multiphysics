# SPUMA 실유체 최적화 v2.1 구현 보고서

> 이 문서의 단계별 표는 2026-09-19 당시 기록이다. 현재 RF21-01~06 완료 상태와 후속 범위는 [postmerge v3 상태표](../postmerge-v3/README.ko.md)를 따른다.

> 이 문서는 작성 시점의 구현·검증 기록이다. 2026-09-21의 실제 GPU·SPUMA 검증 및 재시작 수정 결과는 [PR #1 로컬 검증 보고서](../../reports/pr1-local-validation-20260921.md)를 기준으로 한다.

작성일: 2026-09-19 · 지시 패키지: **SPUMA-RF-OPT-v2.1**

착수 SHA: `0517be936c1a5f4720bd8c3703ffb38ba4289045` · 코드 결과 SHA: `e37a771005dafd2627fb9f5caefacd87f5567ca2`

브랜치: `codex/reactive-gpu-sparse-20260912` · [draft PR #1](https://github.com/clangclan/spuma-multiphysics/pull/1)

## 1. 적용 범위와 이전 상태

패키지의 현재 구현 범위인 **RF21-01~06 및 RF21-10**을 수정했다. 코드 작성, 실제 C ABI/CVODE 연결, 작은 CPU 검사, 로컬 GPU·Flow 검증을 별도로 구분했다. 기존 v2 결과와 보고서는 덮어쓰지 않았다. 새 근거는 `results/real-fluid-v21-20260919/`에 있다.

착수 때 로컬·원격 HEAD가 패키지 기준과 같았고 작업 트리는 깨끗했다. 기존 기반은 두 GPU 보존장, bounded bridge, 독립 CPU worker, 동일 EOS scalar 복원, 진단 Jv였다. **이번 변경으로 진단 Jv가 실제 CVODE의 opt-in 경로에 연결되었으며, 전송·열량 계산·통계 계약이 보강되었다.** 전체 ReactiveFoam의 GPU 포팅 완료를 뜻하지 않는다.

이 환경에는 전체 SPUMA 빌드 환경과 사용 가능한 CUDA 실행 장치가 없다. 과거 실험 환경의 실행시간을 새 구현의 성능으로 재사용하지 않았다. 검증은 사용자 요청대로 작은 수학·코드 검사와 컴파일로 제한했다. 로컬 인수 runner는 실행할 수 없는 항목을 `BLOCKED`/`SKIPPED`로 기록한다.

패키지 SHA-256과 실제 입력 파일 hash는 [baseline.json](../../results/real-fluid-v21-20260919/baseline.json)에 있다. 리뷰의 산술 파일에 적힌 원 보고서 hash와 현재 제공된 HTML의 hash는 다르다. 따라서 해당 HTML을 동일 파일이라고 간주하지 않았고, tile 수 산술은 코드에서 독립적으로 계산했다.

## 2. 작업별 결과

아래 `결과 SHA`는 보고서 자체가 아닌 해당 코드를 포함하는 최종 코드 커밋이다. 모든 행의 시작 SHA는 위 `0517be9`다. `wired_to_flow=true`는 소스의 호출 경로 연결을 뜻하며, 전체 Flow를 실행했다는 뜻은 아니다.

| 작업 ID | 시작 SHA | 결과 SHA | 변경된 실제 호출 경로 | 구현 상태 | 수행한 작은 검사 | 로컬 미실행 항목 |
|---|---|---|---|---|---|---|
| RF21-01 | `0517be9` | `e37a771` | C tangent/Jv → 공통 입력 검사; checkpoint → 공통 schema 검사; pool → 고정 오류 buffer | implemented=true; wired_to_flow=true; smoke_passed=true; local_validated=false; default_enabled=true | NaN/±Inf, 정확한 zero, 극단 scale/underflow, sentinel, schema 1/2, 실패 후 새 attempt | 실제 파일 checkpoint 재시작·쓰기 실패 |
| RF21-02 | `0517be9` | `e37a771` | worker 원자료 → 합산 API → Flow combined/attempt 로그 | implemented=true; wired_to_flow=true; smoke_passed=true; local_validated=false; default_enabled=true | worker 1/2, 마지막 작은 batch, 실패 비용 보존, 원자료 합과 aggregate 대조 | 전체 heap/RSS/VRAM peak, 큰 batch 부하 |
| RF21-03 | `0517be9` | `e37a771` | frozen 기상 후보 → 최소 e/cv/p → NASA 합산·exact PR mixing → 기존 최종 승인 | implemented=true; wired_to_flow=true; smoke_passed=true; local_validated=false; default_enabled=지원되는 scalar 경로 | PR 잔차 열량·NASA 경계·같은 EOS p/T·mixing 직접합/미분 | 상출현·근임계·넓은 물성 영역, 413종 PR 데이터 |
| RF21-04 | `0517be9` | `e37a771` | Flow 설정 → worker ChemicalODE → CVodeSetJacTimes → setup/apply; 선택적 Woodbury | implemented=true; wired_to_flow=true; smoke_passed=true; local_validated=false; default_enabled=false | 실제 짧은 PR 합성반응 적분, Jv/trace/가산성, callback 실패, 재사용·gamma·K 거부 | 상세 PR 반응기구 성능, 제3체 희소 코어 |
| RF21-05 | `0517be9` | `e37a771` | Flow → create_v21 → 바이트 예산 slot → event 완료 → CPU flash barrier | implemented=true; wired_to_flow=true; smoke_passed=true; local_validated=false; default_enabled=true | 1셀/부분 tile/고정 경계/예산/토큰, 같은 payload 제출 수 | 실제 pinned 전송·CUDA sanitizer·전송 중첩 |
| RF21-06 | `0517be9` | `e37a771` | generated gas → 셀 온도 기저 → face h 계산·오류 flag → 승인 전 검사 | implemented=true; wired_to_flow=true; smoke_passed=true; local_validated=false; default_enabled=deviceNasa 재계산 경로 | NASA7/9 경계·역온도/log·무기상·carrier/에너지 flux·overflow | 실제 GPU 성능·메모리 사용량 |
| RF21-10 | `0517be9` | `e37a771` | smoke runner → 실제 라이브러리; local runner → 선택 인수 단계 | implemented=true; wired_to_flow=runner가 실행 파일 호출; smoke_passed=true; local_validated=false; default_enabled=명시 실행 | 7개 작은 검사 그룹, 기존 413종 회귀, C/C++/CUDA 컴파일 | 전체 Flow campaign·재시작 연속성·동일 종료시간 benchmark |

| 작업 ID | 동일 모델 근거 | counter/메모리 변화 | 알려진 실패와 fallback |
|---|---|---|---|
| RF21-01 | 기존 fingerprint와 물리 hash 비교 유지 | 새 ABI만 추가; 기존 state 176B/profile 64B 유지 | 비유한·표현 불가 방향은 오류; 불완전 schema 2는 거부 |
| RF21-02 | 셀별 source·tolerance·BDF 재초기화 유지 | prototype/workers/combined 분리; 실패 비용도 유지 | 외부 라이브러리 내부 메모리는 미계측으로 표시 |
| RF21-03 | 같은 NASA 계수·PR a/b/acentric/binary-a·잔차항 유지 | 고정 조성·구간 합산; 명시적 root 재검사 감소 | 미지원 표현/Cantera 버전/모호한 branch는 같은 EOS 공개 경로 또는 기존 p–T 복원 |
| RF21-04 | 원 source와 frozen 액상 질량 유지; 허용 음수 Newton trial의 기존 RHS extension 유지 | Fz/Rz를 base별 재사용; 정상 matrix-free에 Ns²/Fq 전체 생성 없음 | 직접 전체 source 방향차분, 나쁜 K에서 identity, 적분 실패 시 원 입력의 dense 재적분 |
| RF21-05 | RK 두 stage·closure 이웃 의존성·pre-Strang rollback 유지 | 1개 bounded slot; 전송 바이트는 동일하고 제출 수 감소 | 실패 다운로드는 미승인; Flow가 전체 snapshot 복원 |
| RF21-06 | 기준 NASA 경계 규칙과 carrier 에너지 보정 유지 | 내부 전체 gasY/H는 계속 0B; 온도 기저 40B/셀 추가 | 비유한 실제 h/flux 발견 시 output 승인 거부 |
| RF21-10 | 새 물리자료·허용오차 완화 없음 | 실행한 source/binary/backend hash 기록 | 실행하지 않은 gate를 PASS로 표시하지 않음 |

커밋 구성: `5595a93`(계약·열역학·matrix-free), `8e7375d`(Woodbury), `3ef8ced`(수송), `e37a771`(Flow·회귀·runner).

## 3. 수학과 코드 변경

### 입력·재시작·worker 계약

`directionProbe`는 모든 q/v/energy/de/scale의 유한성과 scale 양수성을 먼저 검사한다. 원래 성분이 모두 0인 경우만 zero 방향이다. 방향을 extended precision으로 정규화한 뒤 표현 가능한 perturbation을 만들고, h와 h/2의 차이를 확인한다. 한쪽 probe만 가능하면 이를 기록한다. 비영 방향이 underflow로 0이 되는 경우와 결과가 표현 불가능한 경우는 실패한다. 오류 시 output 구조체·벡터·`usedFixedBranch`는 유지된다. Python wrapper도 배열 길이를 확인한다.

schema 1의 기존 fingerprint/species/closure 검사를 보존했다. schema 2에는 다섯 필수 항목과 hash 형식을 강제한다. 수치 정책이 달라도 물리 hash와 legacy fingerprint가 같고 현재 설정이 지원되면 이전/현재 policy hash를 로그와 다음 identity에 기록한다. 과거 hash만으로 과거 개별 설정을 복원한 것처럼 보고하지 않는다. v2.1 알고리즘 변경도 numerical policy hash에 반영했다.

prototype은 pool보다 오래 살아야 하며 pool 생성 후 모델·정책 변경을 허용하지 않는다. error/profile/destruction까지 호출 수명 계약을 문서화했다. 중복 batch/transport 진입은 공유 오류 문자열을 덮어쓰지 않고 상태 코드 2로 거부한다. worker의 셀 처리 예외는 고정 크기 오류 buffer에 수집하고 모든 worker가 종료 장벽에 도달한 뒤 실패를 반환한다. 전체 mesh가 여러 batch에 걸치면 기존 Flow snapshot이 최종 rollback을 담당한다.

### 동일 EOS 최소 평가

scalar 기상 후보에서 고정 조성의 NASA7/9 기준 열량을 구간별 계수합으로 만든다. 조성이 0인 종의 기여는 정확히 0이므로 합산에서 제외하며, 어떤 기여 종의 구간이 바뀌면 다시 합산한다. NASA7은 경계 등호에서 하위 구간, NASA9는 상위 구간을 사용한다.

PR의 설치된 실제 계수를 Cantera에서 읽는다. geometric mixing과 모든 `binary-a` 수정항을 보존한다. `binary-a`를 무차원 kij로 바꾸지 않았다. alpha의 square-root 부호가 일정한 구간에서는 혼합 attraction이 정확히

\[
a_\alpha(T)=C_0+C_1\sqrt T+C_2T,
\quad a_T=C_1/(2\sqrt T)+C_2,
\quad a_{TT}=-C_1/(4T^{3/2})
\]

가 된다. dense 수정항도 전부 타일 순회하며 저랭크 절단은 하지 않는다. 조성 방향 대조는 작은 직접 이중합과 독립 차분으로 검사했다. 현재 빠른 PR 구현은 확인한 **Cantera 3.2.0**에 한정하고 다른 버전은 기존 경로를 사용한다.

molar volume V와 혼합 b에 대해

\[
L=\frac{\log[(V+(1+\sqrt2)b)/(V+(1-\sqrt2)b)]}{2\sqrt2 b},\qquad
 e=e^0+(T a_T-a_\alpha)L/W,\quad c_v=c_v^0+T a_{TT}L/W
\]

를 사용하므로 실가스 잔차 에너지·열용량이 사라지지 않는다. 반복 probe에서 전체 엔트로피·chemical potential·부분몰 배열을 만들지 않는다.

cubic discriminant가 충분한 여유로 단일 실근을 나타내거나, 알려진 입력근을 제외한 두 근이 모두 `V≤b` 영역에 있음을 인증할 때만 해당 probe의 역 EOS 재검사를 생략한다. 인증은 현재 EOS·조성·밀도·온도에 국한된다. 이는 전역 상안정성 인증이 아니다. 최종 전체 물성/체적/에너지 검사, absent-phase 안정성, 기존 모든 후보와 최대 엔트로피 선택은 그대로 수행한다.

### 재사용 linearization과 실제 CVODE

새 설정은 `chemicalLinearSolver matrixFree`와 `matrixFreeWoodbury`이며 기본값 dense는 유지했다. setup이 기준 q의 내용·에너지·상태·모델/정책·generation을 확인하고 Fz를 한 번 분해하며 `Ns×m` Rz를 준비한다. apply는

\[
F_z\,\delta z=-F_q v,\qquad Jv=R_qv+R_z\delta z
\]

를 적용한다. m≤4는 현재 두 순수 액상 모델의 제한이다. 미분은 h/h2 품질 검사를 하는 차분이며 analytic/AD라고 표시하지 않는다. Cantera 상태를 소비하는 함수는 자체 q/p/T/상태를 설정한다.

첫 선택은 identity 전처리기다. 별도 Woodbury 선택은 **S=0, A=I**인 초기 코어에 U=Rz, `Vᵀv=δz(v)`를 사용한다. V 전체를 만들지 않고 필요한 방향만 적용한다.

\[
K=I-\gamma V^TU,\qquad (I-\gamma UV^T)^{-1}r=r+\gamma U K^{-1}V^Tr.
\]

K는 작은 선형계로 풀며 명시적 역행렬을 만들지 않는다. 최신 Jtimes와 preconditioner snapshot을 별도로 소유한다. gamma가 바뀌면 같은 snapshot의 K를 갱신한다. dimensionless K의 원래 조건수와 작은 크기를 검사하여, row scaling이 특이성·상쇄를 감추지 않게 했다. 불안정한 K는 identity로 복귀한다. 제3체 효율별 희소 S 생성이나 413종 production 전처리기 완성을 뜻하지 않는다.

### 전송 슬롯·온도 기저

GPU block, staging chunk, CPU batch는 각각 `transportBlockThreads`, `transportStagingBytes`, `thermoBatchCells`로 분리했다. chunk 크기는 `floor(slotBytes/(8*Nv))`이며 한 셀도 못 담거나 override가 예산을 넘으면 거부한다. 명시적인 `transportBridgeCells`는 호환 override이고 0이면 byte budget을 사용한다.

현재는 1개 slot, 기본 1 MiB 예산이다. CUDA에서는 실제 pinned host allocation과 stream event를 사용하고, host 소비 및 event 완료 뒤 slot을 재사용한다. CPU emulation에서도 같은 상태·token 검사를 거친다. 전송 overlap은 구현·실측했다고 주장하지 않는다. CPU flash용 두 전체 q 다운로드와 전체 stage barrier는 남아 있다.

온도 기저는 내부 셀당 5개 double(T²/T³/T⁴/1/T/log T), 즉 40B다. NASA9가 없으면 log를 계산하지 않는다. recompute 모드의 사전 inventory 검사에서 사용하지 않는 모든 h를 중복 계산하지 않고, face에서 실제 h/energy/carrier flux의 비유한성을 모아 stage 승인 전에 확인한다. 기본 detailed gas counter는 off이며 켜면 면마다 integer counter를 갱신하므로 성능 비교에서 같은 설정을 사용해야 한다. 이 provider는 기존 이상기체 기준 수송용이고 PR 확산 guard는 유지했다.

## 4. 수행한 검사와 결과

주 근거: [최종 smoke](../../results/real-fluid-v21-20260919/final-verification/smoke.json), [검증 manifest](../../results/real-fluid-v21-20260919/code-verification.json), [Flow 연결 검사](../../results/real-fluid-v21-20260919/flow-contract.json).

| 검사 | 관찰 결과 | 해석 |
|---|---|---|
| V1 입력·schema | NaN/Inf/zero/extreme/underflow/sentinel 통과; schema 필수키·형식/물리 hash 오류 7종 거부 | 공개 API와 실제 공통 schema 함수 검사 |
| V2 worker | worker 1/2, 실패 batch 뒤 새 attempt, 마지막 1셀 batch, 합산 대조 통과 | 실패한 연산과 시간도 집계에 남음 |
| V3 최소 열량·mixing | 같은 EOS e/cv/p 최대 scaled 차이 약 `9.74e-16`; exact mixing 직접합 약 `6.03e-15`; dT 차이 약 `3.23e-10` | 수식·코드 대조이며 실험 물성 검증이 아님 |
| V4 실제 source | 짧은 합성 PR 반응에서 matrix-free/dense 차이 약 `1.46e-8`; Jv/full-source 차이 약 `2.85e-10` | 기존 허용오차 안의 수치 harness; 실제 연소기구 검증이 아님 |
| V4 cache·실패 | 같은 base에서 apply 4회, Fz/Rz 재생성 0; callback NaN output 유지·재적분 통과 | 실제 내부 callback과 CVODE를 호출한 격리 검사 |
| V4 Woodbury | dense 선형계 residual 약 `1.72e-13`; snapshot/gamma·나쁜 K identity fallback 검사 | 실제 CVODE 선택 경로 및 작은 수학 검사 |
| V5 staging | 19셀/413종: chunk 2→11일 때 layout 33→9, memcpy 470→446; payload 301,032B 동일 | CPU 공통 커널의 제출 counter; 초기화 포함. GPU 시간 아님 |
| V6 NASA·flux | NASA7/9 h scaled 차이 약 `1.10e-12`, NASA9 역온도/log 약 `8.18e-14`; gas/혼합상/무기상 flux 오차 ≤`1.25e-13` | 캐시 경로와 새 재계산 경로에서 기존 검사 재사용 |
| V6 overflow | 유한 계수·유한 온도에서도 실제 h가 overflow하면 경계·q 승인 거부 | 유한 입력만으로 안전하다고 가정하지 않음 |
| 기존 413종 회귀 | dense/sparse/auto와 ReactorNet 대조 통과; ReactorNet Y 차이 약 `4.98e-11`, T 상대차이 `2.13e-10` | **기존 이상기체** 413종·14,922반응 회귀. PR 지원으로 분류하지 않음 |
| 컴파일·ABI | host C++ 경고 오류화 빌드, C11 ABI, CUDA compute_80/sm_80 컴파일·링크, Python/shell 문법 통과 | CUDA 실행과 전체 Flow 컴파일은 미실행 |

stable N2/PR 300 K 복원의 한 검사에서는 scalar probe 6회에 명시적 densityCalc 재검사 1회, 전체 최종 phase 평가 1회, NASA 기여 계수 방문 1회였다. 기존 probe별 4종 반복과 구별되는 실제 counter다. 최종 TP 평가 안의 implicit root 비용은 별도 `fullPhaseEvaluations`에 포함된다. 전체 EOS 비용이 1회라고 해석하지 않는다.

19셀 비교의 내부 전체 gasY/H는 계속 0B이고 고정 경계 배열만 남는다. 온도 기저는 760B였다. 이 메모리·제출 수 결과만으로 GPU speedup을 선언하지 않는다.

내부 callback 실패 검사는 실제 callback에 NaN 방향을 직접 넣고 원 입력에서 비반응 CVODE 적분을 다시 실행한 것이다. 합성 U를 사용하는 Woodbury 수학 검사와 실제 합성 PR 반응 적분도 별도로 구분했다. GPU fault 재현을 뜻하지 않는다.

초기 검사에서 발견한 큰 방향의 노름 overflow 및 검사 harness 문제와 수정 내역은 [development-failures.json](../../results/real-fluid-v21-20260919/development-failures.json)에 보존했다. 초기·중간 결과도 삭제하지 않았다. 413종 회귀는 최종 K 조건수 screening 추가 전 바이너리에서 수행했으며 dense/sparse/auto 경로의 소스는 그 변경으로 바뀌지 않았다. 해당 시점 source/binary hash를 별도로 보존했다.

## 5. 계측과 메모리 해석

[counter-and-memory-scope.json](../../results/real-fluid-v21-20260919/counter-and-memory-scope.json)에 단위·scope를 고정했다. 상세 로그는 worker 원자료, workers 합, prototype, combined를 각각 출력한다. **combined=prototype+workers**이며 세 scope를 다시 합하지 않는다. 기존 `REACTIVE_CHEMISTRY/SPARSE/CHEMICAL_PROFILE/REAL_FLUID` 로그도 combined를 소비한다. 실패/재시도/성공 attempt의 비용 delta를 따로 기록한다. nnz는 마지막 snapshot 합이며 temporal peak가 아니다. worker job elapsed 합·최대와 batch wall time도 다르다.

| 동시에 살아 있는 메모리 | 계측/산정 | 이번 취급 |
|---|---|---|
| 기본 host q/states, pre-Strang snapshot, oldStates, CPU stage/RHS | q는 `Nc*Nv*8`; CellState는 실제 구조체 크기; 수명은 Flow::step 기준 | 기존 보수적 host estimate를 유지하고 batch 직접 할당을 추가. snapshot 제거 주장 없음 |
| Flow batch staging + pool scratch | 실제 capacity와 고정 buffer 항목 합 | 함께 thermo batch budget 내에서 큰 할당 전에 검사 |
| worker Model/Cantera/CVODE/Eigen/thread stack | Model 본체만 known lower bound; 나머지 미계측 | 전체 peak나 budget 보장으로 표시하지 않음 |
| pinned host slot / device slot | 실제 할당 바이트와 별도 예산 | CUDA에서 실제 pinned, CPU에서는 일반 host emulation |
| device Q 두 장·geometry·고정 경계 | Execution allocator live/peak | CUDA process 전체 메모리와 구분 |
| NASA 온도 기저 | `40*Nc` bytes | Ns 전체 enthalpy 배열을 대체하는 작은 O(Nc) 저장 |
| source Rz·작은 factor | worker별 현재/전처리기 snapshot에 제한 | 전체 셀×종×종 저장 없음 |

300만 셀·413종이면 두 Q만 20.016 GB이고 온도 기저는 0.120 GB다. 16 GB 장치에 전체장을 담을 수 있다는 의미가 아니다. 1 MiB slot은 이 경우 314셀이다. 300만 셀에서 3회 보존장 layout 산술은 35,157회(256셀)에서 28,665회(314셀)로 변한다. 이 역시 산술이고 실측 시간이 아니다. 4/16 MiB 등 override는 로컬에서 예산과 제출 비용을 함께 비교할 수 있다.

## 6. 로컬 재검증 방법

아래 파일과 인수는 이번에 실제 작성한 것이다. 기존 로컬 환경에서 의존성과 SPUMA를 활성화한 뒤 실행한다. 출력 디렉터리는 기존 결과가 없는 새 경로를 지정한다.

```bash
PINTLE_RF21_CONTRACT_TEST=1 tools/build_reactive_backend.sh
PINTLE_REACTIVE_TRANSPORT_BUILD=cpu tools/build_reactive_transport.sh
python tools/run_real_fluid_v21_local.py \
  --stages smoke cpu-integration \
  --thermo-dir research/reactive-thermo \
  --thermo-library lib/libpintleReactiveBackend.so \
  --transport-library lib/libpintleReactiveTransport.so \
  --contract-executable bin/check-real-fluid-v21-contract \
  --output-dir results/rf21-local-cpu
```

CUDA 빌드는 기존 `tools/build_reactive_transport.sh`의 `PINTLE_CUDA_ARCH`를 실제 장치에 맞춘다. CPU .so와 CUDA .so를 서로 다른 파일로 보존하고 `--cuda-library`로 지정한다. `--stages cuda-sanitizer`는 실제 CUDA 장치와 Compute Sanitizer가 있어야 실행한다. 현재 환경의 [인수 가용성 결과](../../results/real-fluid-v21-20260919/local-gate-availability/acceptance.json)는 해당 gate를 BLOCKED로 기록한다.

Flow/benchmark는 `--local-plan` JSON의 `cases`에 기존 case 경로, `expected_end_time`, 실행 파일을 지정한다. benchmark에는 동일 `comparison_group`과 종료 물리시간이 필요하다. runner는 프로세스 성공만으로 통과시키지 않고 실제 `REACTIVE_STEP` 종료시간과 `REACTIVE_FAILURE`를 확인한다. 전체 SPUMA 빌드, checkpoint 연속 실행, 상출현·소멸·근임계, 기존 HEM 접촉면의 알려진 실패는 별도 로컬 인수 항목이다. 이번 작업으로 그 실패가 해결됐다고 표시하지 않는다.

## 7. 다음 범위와 자료 gate

RF21-07~09의 solver 기능은 이번 범위 밖이다. 후속 계약은 다음과 같다.

- **07 predictor/table:** 기존 모든 후보 탐색과 같은 EOS 최종 승인을 유지하는 predictor-only부터 시작한다. EOS/조성/branch/domain/policy hash와 fallback 비용을 기록한다. 초기값 개선을 물성 대체로 확대하지 않는다.
- **08 device closure:** 이번 최소 PR/NASA 수식을 기반으로 기존 cold-pr 단상만 먼저 대조한다. 단일근 인증과 전역 상안정성을 구분하며 불명 셀의 실제 q/energy를 CPU fallback으로 전송한다. GPU chemistry·전체 전송 제거로 표시하지 않는다.
- **09 자료:** 기존 manifest의 413종 PR 누락은 그대로다. 종별 a/b/acentric·유효영역·출처, pair별 binary-a/kij 정의와 추정 불확실성을 별도로 확보한다. trace 종이라도 임의 값을 대입하지 않는다. 비이상 확산은 구동력/mobility/binary D/실제 부분몰 h/합산 flux=0/CFL 명세가 선행한다.
- **kinetics·고차 flux·DAE/MPI:** 반응 유형별 coverage와 단위/역반응/농도·activity 정의, 같은 EOS 면 상태·공유 flux·limiter, coupled residual 또는 halo/승인-state/rollback을 각각 별도 gate로 둔다. 제3체 공통항을 일괄 rank-1로 가정하지 않는다.

`nonidealDiffusion`, `deviceClosure`, `deviceKinetics`, `mixtureLiquid`는 계속 미지원이다. EOS downgrade, accepted-state clipping, 총에너지 재설정, 액상 물리 변경은 하지 않았다.

## 8. 공식 구현 대조 근거

- [SUNDIALS 7.4 CVODE 인터페이스](https://sundials.readthedocs.io/en/v7.4.0/cvode/Usage/index.html): linear solver 연결 후 JacTimes setup/apply 및 preconditioner callback 계약.
- [Cantera 3.2 Peng–Robinson 구현](https://cantera.org/3.2/cxx/db/db8/PengRobinson_8cpp_source.html): 설치 계수·binary-a·alpha·잔차 열량 식 대조. 로컬의 동일 버전 원본 소스도 확인했다.
- [CUDA 비동기 실행 문서](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html): pinned memory·stream/event 수명. Async 호출만으로 overlap 검증을 주장하지 않는다.
