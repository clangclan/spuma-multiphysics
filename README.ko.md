# ReactiveFoam

이 저장소의 기본 솔버는 **`ReactiveFoam`**이다. 다중물리 연산 선택을 개선한 `selection` 커밋(`96a781c`, PR #2)을 포함한 `main`을 기준으로 개발한다. ColdFoam 코드와 전용 빌드·실행·벤치마크 도구는 제거했다.

N₂O/IPA 상분배, 화학종 수송과 압축성 총에너지를 계산하며, `constant/reactiveProperties`의 `physics`에서 화학반응·상전이·점성·열전도·종 확산을 선택한다. `combustion`은 `chemistry`의 별칭이다. 상전이를 끄면 액상 질량을 보존하여 수송하고 증발·응축 교환만 제외한다. EOS 복원과 필요한 에너지 결합은 유지한다.

## 빌드와 실행

SPUMA 환경과 Cantera/Sundials 의존성을 먼저 준비한다. 설치 및 반응기구 준비 방법은 [상세 사용 설명](README.reactive-phase.ko.md#환경과-빌드)을 따른다. 아래 경로는 설치 위치에 맞게 지정한다.

```bash
export PINTLE_SPUMA_ENV=/path/to/spuma-env.sh
export PINTLE_REACTIVE_PREFIX="$PWD/research/reactive-env"
export PINTLE_RUN_LOCK=/tmp/reactivefoam-run.lock
# 기본 CPU 수송. CUDA를 빌드하려면 다음 환경변수를 지정한다.
# export PINTLE_REACTIVE_TRANSPORT_BUILD=cuda
flock "$PINTLE_RUN_LOCK" ./Allwmake
source ./env.sh

flock "$PINTLE_RUN_LOCK" "$PINTLE_REACTIVE_PREFIX/bin/python" tools/prepare_reactive_case.py cases/frozen-example \
  --thermo-dir research/reactive-thermo --kind acoustic --cells 32 \
  --chemistry off --phase-change frozen --transport-backend cpu
flock "$PINTLE_RUN_LOCK" ReactiveFoam -case cases/frozen-example
```

`./Allwmake`는 열역학 백엔드, 수송 라이브러리, `bin/ReactiveFoam`을 빌드한다. CUDA 빌드는 같은 실행파일에서 CPU/CUDA 수송을 선택할 수 있다. `PINTLE_CUDA_ARCH`는 대상 GPU의 compute capability에 맞춰 지정할 수 있다.

## 물리 연산 선택과 검증

비반응 HEM 평형에서는 정확한 입력 중복 제거를 기본 사용한다. PR/NASA 온도 후보의 CUDA 경로는 선택 옵션이다. [상평형 탐색 최적화·GPU 범위와 설정](docs/closure-acceleration.ko.md), [검증 및 성능](reports/closure-acceleration-20260921.md)을 참고한다.

[연산 선택 및 의존관계](README.reactive-phase.ko.md#실행할-연산-선택), [selection 검증 보고서](reports/reactive-physics-selection-20260921.md), [검증 집계](results/physics-selection-20260921/validation-summary.json)를 참고한다.

```bash
# 이 검증 도구는 내부에서 PINTLE_RUN_LOCK을 획득한다.
"$PINTLE_REACTIVE_PREFIX/bin/python" tools/validate_reactive_physics.py \
  --thermo-dir research/reactive-thermo --output cases/physics-check --backends cpu
# CUDA 수송을 빌드했다면 --backends cpu cuda로 두 경로를 검사한다.
```

현재는 고정 메시·단일 MPI rank·FP64 연구 솔버다. 이 개발 브랜치에는 실험적 CUDA WALE 운동량 응력·총에너지 결합을 추가했다. SGS 열/종 유속·TCI 및 난류 통계 검증은 남아 있다. 표면장력 기초 함수와 액적 식별은 독립 모듈이며 실제 계면 수송·1차 분열은 아직 연결하지 않았다. 액적 합체·슬립과 MPI 영역 분할도 지원하지 않는다. [구현 및 검증 현황](docs/spray-physics/implementation-status.ko.md)을 참고한다. 기본 HEM 접촉면 정확도와 고압 반응 모델에도 알려진 한계가 있다. 전체 범위와 제약은 [상세 사용 설명](README.reactive-phase.ko.md)에 명시한다.

## 저장소 구성

- `src/reactiveFoam`: 솔버와 다중물리 연산 선택
- `src/reactiveThermo`: EOS, 상평형·동결 복원, 화학반응
- `src/reactiveTransport`: CPU/CUDA 보존량 수송
- `tools`: 케이스 준비, 실행·검증, 공통 필드 분석
- `reports`, `results`: 날짜와 커밋에 대응하는 검증 기록. ColdFoam 관련 과거 결과는 현재 ReactiveFoam의 기능·성능 근거가 아니다. 과거 소스는 Git 이력에서 확인할 수 있다.

SPUMA/OpenFOAM에서 파생한 코드는 GPL-3.0-or-later 조건을 따른다. [LICENSE](LICENSE)를 확인한다.

90도 액체 N₂O 충돌·평형 flashing 입력은 [벤치마크 구성](docs/benchmarks/impinging-n2o-setup.ko.md)에 있다. 55 bar(g) 공급과 1 atm/40 bar(abs) 환경, 각 256k 정육면체 메시를 제공한다. 현재 입력은 [과냉각 액체 N₂O 프로필](docs/benchmarks/impinging-n2o-supercooled.ko.md)로, 148 K까지 기체–액체 상평형을 계산하고 고체·승화는 제외한다. 두 환경의 GPU 10스텝은 재시도·CPU fallback 없이 완료됐고, 이전 고체 프로필 대비 실행 시간은 각각 92.5%, 68.8% 감소했다. 삼중점 아래 액체는 준안정 PR 연장 모델이다.
