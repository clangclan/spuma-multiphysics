# ReactiveFoam

**`ReactiveFoam`**은 반응·상전이·수송 모델을 선택할 수 있는 범용 압축성 다중물리 연구 솔버다. 소스 파일·API·빌드 환경변수도 `reactive` 계열 이름을 사용한다. 기존 명령과 외부 라이브러리의 변경 사항은 [명칭 전환 안내](docs/generic-naming.ko.md)에 정리했다.

N₂O/IPA 상분배, 화학종 수송과 압축성 총에너지를 계산하며, `constant/reactiveProperties`의 `physics`에서 화학반응·상전이·점성·열전도·종 확산을 선택한다. `combustion`은 `chemistry`의 별칭이다. 상전이를 끄면 액상 질량을 보존하여 수송하고 증발·응축 교환만 제외한다. EOS 복원과 필요한 에너지 결합은 유지한다.

## 빌드와 실행

SPUMA 환경과 Cantera/Sundials 의존성을 먼저 준비한다. 설치 및 반응기구 준비 방법은 [상세 사용 설명](README.reactive-phase.ko.md#환경과-빌드)을 따른다. 아래 경로는 설치 위치에 맞게 지정한다.

```bash
export REACTIVE_SPUMA_ENV=/path/to/spuma-env.sh
export REACTIVE_PREFIX="$PWD/research/reactive-env"
export REACTIVE_RUN_LOCK=/tmp/reactivefoam-run.lock
# 기본 CPU 수송. CUDA를 빌드하려면 다음 환경변수를 지정한다.
# export REACTIVE_TRANSPORT_BUILD=cuda
flock "$REACTIVE_RUN_LOCK" ./Allwmake
source ./env.sh

flock "$REACTIVE_RUN_LOCK" "$REACTIVE_PREFIX/bin/python" tools/prepare_reactive_case.py cases/frozen-example \
  --thermo-dir research/reactive-thermo --kind acoustic --cells 32 \
  --chemistry off --phase-change frozen --transport-backend cpu
flock "$REACTIVE_RUN_LOCK" ReactiveFoam -case cases/frozen-example
```

`./Allwmake`는 열역학 백엔드, 수송 라이브러리, `bin/ReactiveFoam`을 빌드한다. CUDA 빌드는 같은 실행파일에서 CPU/CUDA 수송을 선택할 수 있다. `REACTIVE_CUDA_ARCH`는 대상 GPU의 compute capability에 맞춰 지정할 수 있다.

## 물리 연산 선택과 검증

비반응 HEM 평형에서는 정확한 입력 중복 제거를 기본 사용한다. PR/NASA 온도 후보의 CUDA 경로는 선택 옵션이다. [상평형 탐색 최적화·GPU 범위와 설정](docs/closure-acceleration.ko.md), [검증 및 성능](reports/closure-acceleration-20260921.md)을 참고한다.

[연산 선택 및 의존관계](README.reactive-phase.ko.md#실행할-연산-선택), [selection 검증 보고서](reports/reactive-physics-selection-20260921.md), [검증 집계](results/physics-selection-20260921/validation-summary.json)를 참고한다.

```bash
# 이 검증 도구는 내부에서 REACTIVE_RUN_LOCK을 획득한다.
"$REACTIVE_PREFIX/bin/python" tools/validate_reactive_physics.py \
  --thermo-dir research/reactive-thermo --output cases/physics-check --backends cpu
# CUDA 수송을 빌드했다면 --backends cpu cuda로 두 경로를 검사한다.
```

현재는 고정 메시·단일 MPI rank·FP64 연구 솔버다. 이 개발 브랜치는 CUDA WALE 응력·열/종 혼합과 실험적인 표면장력·flashing 결합을 포함한다. 추가로 `capillaryGeometry cartesianImplicit`를 선택하면 액체 체적분율에서 C² 계면을 GPU로 복원하고 공유 면 유량을 실제 RK 수송에 연결한다. [구현과 검증 한계](reports/implicit-static-drop-20260923.ko.md)에 정적 액적 결과를 기록했다. TCI, 난류 통계·1차 분열 정확도, 액적 슬립과 MPI 영역 분할은 검증 또는 구현이 남아 있다. [구현 및 검증 현황](docs/spray-physics/implementation-status.ko.md)을 참고한다. 기본 HEM 접촉면 정확도와 고압 반응 모델에도 알려진 한계가 있다. 전체 범위와 제약은 [상세 사용 설명](README.reactive-phase.ko.md)에 명시한다.

## 저장소 구성

- `src/reactiveFoam`: 솔버와 다중물리 연산 선택
- `src/reactiveThermo`: EOS, 상평형·동결 복원, 화학반응
- `src/reactiveTransport`: CPU/CUDA 보존량 수송
- `tools`: 케이스 준비, 실행·검증, 공통 필드 분석
- `reports`, `results`: 날짜와 커밋에 대응하는 검증 기록. ColdFoam 관련 과거 결과는 현재 ReactiveFoam의 기능·성능 근거가 아니다. 과거 소스는 Git 이력에서 확인할 수 있다.

SPUMA/OpenFOAM에서 파생한 코드는 GPL-3.0-or-later 조건을 따른다. [LICENSE](LICENSE)를 확인한다.

90도 액체 N₂O 충돌·평형 flashing은 55 bar(g) 공급, 1 atm/40 bar(abs) 환경에서 검증한다. 최신 메시 시리즈는 한 변 80 mm, 지름 5 mm 입구, 입구 z=40 mm의 40³·80³·160³ 셀이다. [40³/80³ 적응 시간 간격 결과](reports/impinging-cube80-z40-adaptive-1us-20260923.ko.md), [160³ 상세 GPU 계측](reports/impinging-cube80-n160-gpu-profile-20260924.ko.md), [배치·정확 재사용 최적화](reports/hem-utilization-optimization-20260924.ko.md), [현재 솔버의 정밀도 실험](reports/current-solver-precision-20260924.ko.md)을 참고한다. 정밀도 실험 후에도 기본 산술은 FP64다. 현재 [과냉각 액체 프로필](docs/benchmarks/impinging-n2o-supercooled.ko.md)은 148 K까지 기체–액체 상평형을 계산하고 고체·승화를 제외한다. 삼중점 아래 액체는 준안정 PR 연장 모델이다.
