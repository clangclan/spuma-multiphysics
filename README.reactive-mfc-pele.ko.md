# MFC·PeleC 구조를 참고한 GPU 수송 물성 계산 — 2026-09-13

> 이 문서는 작성 시점의 구현·검증 기록이다. 2026-09-21의 실제 GPU·SPUMA 검증 및 재시작 수정 결과는 [PR #1 로컬 검증 보고서](reports/pr1-local-validation-20260921.md)를 기준으로 한다.

MFC와 PeleC/PelePhysics의 실제 소스를 읽고 **기상 질량분율·종 엔탈피의 device 계산, 불변 열역학 계수의 상주, CFL 전용 면 커널**을 구현했다. 현재 413종 기구의 NASA7 409종·NASA9 4종을 원래 계수와 온도 구간 그대로 지원한다. 기존 수송·화학 회귀와 새 CPU 수치 검사를 수행했으며 CUDA 컴파일·링크도 통과했다. **실제 GPU 실행과 SPUMA 전체 빌드·시간 적분은 여전히 미검증**이다.

## 작업 기준과 참고한 구조

이번 변경의 기준은 오류 수정 커밋 `69bd3f0e4a81f89c6ca083f9ea69acd6849111fd`다. 앞선 [GPU·희소 포팅](README.reactive-gpu-sparse.ko.md), [준비 비용 개선](README.reactive-review.ko.md), [컴파일 오류 수정](README.reactive-error-fixes.ko.md)은 각각 당시 기록으로 보존한다. 첨부 `report(1).html`은 원래 Ryzen 9600X·RTX 5080의 SPUMA 단상 실험이다. 이번 ReactiveFoam 검사나 GPU 가속률의 측정 결과가 아니다.

| 실제 확인한 소스 | 확인한 구조 | 이번 코드에 적용한 내용 |
|---|---|---|
| MFC `ec783a8`: [RK 데이터](https://github.com/MFlowCode/MFC/blob/ec783a899ff1d8ed0056a986117166cc5f868dc6/src/simulation/m_time_steppers.fpp), [RHS 루프](https://github.com/MFlowCode/MFC/blob/ec783a899ff1d8ed0056a986117166cc5f868dc6/src/simulation/m_rhs.fpp) | device 상태 유지, 명시적 갱신, 성분·셀 방향의 병렬 루프와 지역 스칼라 작업값 | 상주 보존장으로 물성 생성, 계수의 초기 1회 전송, 종별 SoA 쓰기, 면 임시값의 지역 변수화 |
| MFC [GPU 매크로](https://github.com/MFlowCode/MFC/blob/ec783a899ff1d8ed0056a986117166cc5f868dc6/src/common/include/parallel_macros.fpp), [기상 물성·수송](https://github.com/MFlowCode/MFC/blob/ec783a899ff1d8ed0056a986117166cc5f868dc6/src/common/m_chemistry.fpp) | 같은 루프·함수 구조를 GPU 실행에 연결하고 device 루프 안에서 물성 계산 | 기존 CPU/CUDA 공용 연산자를 확장해 동일 식을 비교 가능하게 유지 |
| PeleC `c3f925c`: [Diffusion.cpp](https://github.com/Pele-Suite/PeleC/blob/c3f925cc9f54b761b7d5a1efe9e5409c98ed3a3c/Source/Diffusion.cpp) | `ParallelFor`에서 보존량→원시량 변환과 수송계수 계산, device 파라미터 전달 | CPU가 만든 큰 `gasY/h` 배열을 받는 대신 GPU에서 재구성 |
| PeleC가 고정한 PelePhysics `6e8026f`: [Fuego.H](https://github.com/Pele-Suite/PelePhysics/blob/6e8026fc6ed121c07e32c9d3fe1d1c919bafc64b/Source/Eos/Fuego.H), 별도로 확인한 `4eef939`의 [CEPTR 열역학 생성기](https://github.com/Pele-Suite/PelePhysics/blob/4eef939b943693e97c24fb3359fcd2970a8ad4db/Support/ceptr/ceptr/thermo.py) | host/device 호출 가능한 이상기체 종 엔탈피, NASA7·NASA9 지원 | Cantera에서 추출한 SI 계수 표와 device 다항식 평가 |

참고 위치·커밋·blob SHA는 [소스 대조 기록](reports/reactive-mfc-pele-sources-20260913.json)에 있다. 기존 프로젝트의 런타임 기구 로딩을 유지하기 위해 불변 계수 표를 사용했다. MFC/PeleC의 소스 파일·메시·수치 flux·화학 반응 모델을 복사한 변경은 아니다. CFL 전용 커널은 위의 데이터 수명·작업 분리 원칙을 현재 비정렬 cell–face CSR 구조에 맞게 적용한 것이다.

## 기상 수송 물성의 device 계산

새 `pintle_rt_export_gas_thermo()`는 실제 모델에서 종 순서, `R/W`, NASA 다항식 종류, 모든 온도 구간과 계수를 내보낸다. 가변 구간은 종별 offset/count와 연속 구간 배열로 저장한다. **409×2 + 4×3 = 830개 구간**, 표 크기는 86,256 byte이며 시작할 때 한 번 전송한다. NASA9를 NASA7로 근사하거나 낮은 온도 구간을 버리지 않는다.

device 연산자는 각 셀·종에 대해 다음을 계산한다.

\[
Y_k^g=\frac{q_k-\sum_{\ell:s_\ell=k}m_\ell}{m_g},
\qquad h_k(T)=\frac{R}{W_k}\,T\,H_k(T).
\]

여기서 `q`는 기상·액상의 합계 종 질량/셀 체적이고, `m_l`, `m_g`, `T`는 CPU flash가 정한 상태다. NASA7의 `H`는 기존 7계수 엔탈피 식이며 NASA9는 역온도·로그 항까지 평가한다. [Cantera NASA9 정의](https://cantera.org/3.2/cxx/d4/db6/classCantera_1_1Nasa9Poly1.html)

- NASA7 구간 경계의 등호는 낮은 구간, NASA9는 높은 구간에 속하도록 Cantera 구현과 맞췄다. 온도 clipping이나 계수 보간은 하지 않는다. 다항식의 외삽은 Cantera와 일치시키며, 전체 솔버의 허용 온도 범위는 계속 기존 flash 설정이 결정한다.
- 기상이 없는 셀은 `Y/h=0`이며 종 확산은 기존처럼 꺼진다. 액상 분배·상 안정성·압력·음속을 GPU에서 새로 근사하지 않는다.
- 인접 CUDA lane은 같은 종의 인접 셀을 처리한다. thread마다 413종 임시 배열을 만들지 않는다. 질량분율·엔탈피 결과는 기존 SoA 배열에 쓴다.
- 음수 종, 액상 질량 초과, 불일치한 총질량/기상 질량, 비유한 엔탈피는 거부한다. 오류 표시에만 정수 atomic을 사용하며 FP64 flux atomic은 도입하지 않는다. 거부 시 호출자의 RK 출력 배열을 바꾸지 않고 상주 상태를 무효화한다.
- 고정 경계의 물성은 초기화 때 한 번 만드는 기존 불변 입력으로 유지한다. 매 단계 내부 셀의 물성은 새로 계산하므로 온도·상분배가 바뀐 캐시를 재사용하지 않는다.

`Flow::packTransport()`는 선택된 경로에서 큰 host `gasY/h` 배열과 셀별 Cantera 엔탈피 호출을 생략한다. `Flow::transportStage()`가 새 C API를 실제 두 RK 단계에 연결한다. CPU 기준 수송과 기존 host-property C API도 유지한다.

## 전송·작업공간 변화

`Q = 8 N_c (N_s+4)`, `G = 8 N_c N_s`, `L = 16 N_c` byte라고 하자. `L`은 두 액상 슬롯을 갖는 셀별 분배 배열이다. 정상 확산 RK2 스텝의 주요 장 전송은 다음과 같다. 시작·출력·거부/재시도 및 작은 상태·경계 수지·오류 flag는 제외한다.

| 항목 | 기준 `69bd3f0` | 이번 device 물성 경로 |
|---|---:|---:|
| 보존장 업로드 + CPU flash용 다운로드 | `3Q` | `3Q` |
| 내부 셀 `gasY/h` 업로드 | `4G` | **0** |
| CPU flash 액상 분배 업로드 | 큰 `Y` 배열에 반영됨 | 무액상 모델 0, 액상 모델 `2L` |
| 합계 | `3Q + 4G` | `3Q` 또는 `3Q + 2L` |

413종·417변수에서는 위 주요 장의 바이트가 무액상 약 **56.9%**, 액상 슬롯을 전송하면 약 **56.8%** 줄어든다. 전체 PCIe 트래픽이나 실행시간 감소율은 아니다. 실제 7셀 시험에서는 보존장 업로드 23,352 byte·다운로드 46,704 byte를 유지하고 `gasY/h`의 기존 92,512 byte 업로드를 제거했다. 기액 공존 조건의 분배 전송은 224 byte였다.

엄격한 오류 처리를 위해 물성 생성 후 4-byte flag를 내려받아 확인한다. 현재 구현에는 **RK 단계당 한 번의 추가 stream 동기화**가 있다. 큰 배열 전송 제거와 이 지연의 실제 손익은 GPU에서 측정해야 한다. CPU flash를 위한 두 번의 보존장 다운로드, device `gasY/h` 두 벌과 보존장 네 벌의 저장공간도 남아 있다.

`FaceSpeeds`는 CFL에 필요한 파속·확산 제한만 계산하고 면당 double 하나를 기록한다. 기존 `FaceWork`의 `unL/unR/left/right`는 flux 계산 중 지역 변수로 바꿨다. 파속 배열을 포함한 면 작업공간은 **144 → 112 byte/face**, 32 byte/face 감소다. CFL 3회·RK 2회의 계수에서 각각 전용 커널 3회·flux 면 커널 2회를 확인했다. 기존 invalid sentinel의 reduction 전파와 HLL·carrier 확산 식은 유지한다.

## 검사 결과와 시행착오

| 검사 | 결과 / 증거 |
|---|---|
| 기본 수송·RK·경계·CFL 회귀 | 21개 통과, [최종 결과](results/reactive-mfc-pele-transport-final-20260913.json) |
| NASA7/NASA9·실제 상태·전송·거부/복구 | 7그룹 통과, [CPU 결과](results/reactive-mfc-pele-gas-final-20260913.json) |
| CUDA로 컴파일한 라이브러리의 **CPU 실행 경로** | 같은 7그룹 통과, [결과](results/reactive-mfc-pele-cuda-host-20260913.json). GPU 실행으로 세지 않음 |
| 기존 화학 Jacobian·적분 회귀 | 6개 통과, [별도 결과](results/reactive-mfc-pele-chemical-20260913.json) |
| CUDA 컴파일·링크 | CUDA 13.4.59, compute_80/sm_80, FP64 fast-math 미사용 |
| 실제 CUDA 장치 생성 | driver/runtime 오류, [probe](results/reactive-mfc-pele-cuda-probe-20260913.json) |
| C 헤더·ABI·Python 문법, Flow 버전 변경 계약 | 통과. [Flow 부분 검사](results/reactive-mfc-pele-flow-contract-final-20260913.json)는 전체 SPUMA 빌드가 아님 |

413종 검사는 268개 온도 상태에 모든 내부 경계의 `nextafter` 아래/등호/위를 포함했다. `max |h−h_ref|/max(1,|h_ref|)`는 `1.65e-12`, 질량분율 최대차는 `8.67e-19`였다. 별도의 Cantera NASA9 공기 기구에서는 0이 아닌 역온도·로그 항을 확인했고 엔탈피 오차는 `8.18e-14`였다.

기상·기액 공존·순수 IPA 액체의 수송 RHS를 독립 NumPy 면 scatter와 비교한 최대 scaled error는 `1.25e-13`이었다. 고정·반사·외삽 경계를 포함한다. 실제 CPU flash를 사이에 둔 RK2 시험에서는 기준 결과와 기록 정밀도에서 차이가 없었고, 잘못된 버전·분배·계수·음수 종·overflow의 거부와 새 버전 업로드 후 복구를 확인했다. 이 연결 그래프를 실제 닫힌 SPUMA 메시나 전체 Strang 적분 검증으로 해석하지 않는다.

초기 구현은 NASA7만 지원하여 현재 기구의 N2에서 중단되었다. [초기 거부 기록](results/reactive-mfc-pele-gas-20260913.json)을 보존했다. 이후 NASA9의 원래 구간·계수를 지원하여 같은 실제 기구를 통과시켰다. 지원 검사를 끄거나 기구를 줄여 통과시킨 것은 아니다.

소스·검사·라이브러리 hash와 최종 판정은 [환경 및 검증 기록](results/reactive-mfc-pele-environment-20260913.json)에 있다. 이전 NASA7 전용 중간 빌드의 수송·Flow 부분 검사도 보존하며, 최종 빌드의 증거는 파일명에 `final`을 붙여 구분했다. 화학 JSON의 실행시간은 회귀 검사의 단발 진단이며 수송 성능 비교로 사용하지 않는다.

## 선택과 재현

`transportGasProperties`는 다음 세 값을 받는다. 기본 CPU 수송·밀집 화학은 그대로다.

| 값 | 동작 |
|---|---|
| `auto` — 기본값 | CUDA 수송·확산에서 전체 기구가 지원되면 device 물성, 지원하지 않는 표현은 사유를 출력하고 host 물성 |
| `host` | 기존 CPU 물성 생성·전송을 사용해 대조 |
| `deviceNasa` | CUDA 수송·확산과 지원되는 전체 기구를 요구. 불일치하면 시작 실패 |

설정 예:

```text
transportBackend cuda;
transportGasProperties deviceNasa;
```

위 설정은 기존 `molecularDiffusivity`가 양수인 케이스에 적용한다. 생성기에는 `--transport-gas-properties`를 추가했다. 선택 결과는 `REACTIVE_BACKENDS`, 표·분배 바이트와 커널 횟수는 `REACTIVE_DEVICE_PROPERTIES`, 큰 장 전송은 `REACTIVE_TRANSFER`에 기록한다.

원래 고정 의존성 환경에서 기존 공유 실행 잠금 규칙을 지키고 새 결과 경로를 사용한다.

```bash
bash tools/build_reactive_backend.sh
bash tools/build_reactive_transport.sh
research/reactive-env/bin/python tools/validate_device_gas_transport.py \
  --thermo-dir research/reactive-thermo --backend cpu \
  --output results/local-device-gas-cpu.json

PINTLE_REACTIVE_TRANSPORT_BUILD=cuda PINTLE_CUDA_ARCH=120 \
  bash tools/build_reactive_solver.sh
research/reactive-env/bin/python tools/validate_device_gas_transport.py \
  --thermo-dir research/reactive-thermo --backend cuda \
  --output results/local-device-gas-cuda.json
```

이어서 실제 GPU의 Compute Sanitizer, SPUMA의 `host`/`deviceNasa` 동일 케이스, 상 출현·소멸·rollback·재시작, Nsight 전송·동기화·커널 시간을 확인해야 한다. 이번 CPU 수치 환경은 Cantera 3.2.0/SUNDIALS 7.4.0/Eigen 3.4.1이며 원래 실험 호스트와 다르다.

PeleC의 [reactor 호출](https://github.com/Pele-Suite/PeleC/blob/c3f925cc9f54b761b7d5a1efe9e5409c98ed3a3c/Source/React.cpp)과 고정 PelePhysics의 [GPU CVODE 구성](https://github.com/Pele-Suite/PelePhysics/blob/6e8026fc6ed121c07e32c9d3fe1d1c919bafc64b/Source/Reactions/ReactorCvode.cpp)도 확인했다. 배치별 상태 순서, device kinetics/Jacobian, 선형 풀이 라이브러리와 stream 수명을 함께 맞춰야 하므로 이번에는 GPU 화학으로 이식하지 않았다. 현재 희소 화학은 계속 CPU이며, 무액상 1변수 온도 복원·전체 GPU flash·GPU 배치 화학·검증된 축소 기구는 후속 단계다. 기존 HEM 접촉면 정확도와 고압 다상 연소의 미검증 상태도 별개로 남는다.
