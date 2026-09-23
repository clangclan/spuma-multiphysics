# 액주·액막 분열 구현: 첫 단계 결과

**2026-09-23 후속 변경:** [표면장력 선택과 난류 열·종 수송](physics-switches.ko.md)에 이어 [GPU 표면장력–flashing 결합](capillary-flashing.ko.md)을 추가했다. [리뷰 반영](../../reports/capillary-WALE-review-20260923.ko.md)으로 단일 액체 모세관 경로의 WALE PR 엔탈피도 GPU로 옮겼으며 면 일률의 속도·color 가중치를 통일했다. 표면장력 기본값은 `false`이며 켜면 실험적인 diffuse-interface 경로를 사용한다. 정적 액적 기생 유속 수렴은 아직 통과하지 못했다. 아래는 9월 22일의 첫 단계 기록이며 최신 결합 상태와 구분한다.

2026-09-22. 우선 목표는 사용자가 선택한 **액주·액막 분열과 액적 생성**이다. 성능 최적화 작업은 중단한 상태로 유지했다. 조사는 GPT-5.6 Sol 에이전트 3개가 수행했다.

작업 경로는 `spuma-multiphysics-spray-physics`, 브랜치는 `codex/reactive-spray-physics`다. `main`의 `9ff04ff`에 이미 수용한 compact-active-state GPU 소스 48개를 옮긴 뒤 물리 기능을 추가했다. [기준 소스](../../results/spray-physics/gpu-baseline-source.json)와 [기준 실행 파일](../../results/spray-physics/gpu-baseline-runtime.json)을 기록했다. 기존 `spuma-multiphysics-gpu-hem`의 중단된 최적화 후보는 승격하지 않았다.

## 구현한 범위

| 기능 | 이번 변경 | 남은 범위 |
|---|---|---|
| WALE | 현재 속도 구배에서 GPU `nut` 계산, `rho*nut` 난류 응력과 총에너지 응력 일을 수송에 연결, 점성 CFL 반영 | 난류 통계·격자/필터 수렴 검증, 열·종 SGS와 TCI |
| 표면장력 | 별도 기하학적 color 상태, 계면 법선·면적밀도, 표면 응력과 에너지 플럭스, 모세관 시간 제한, Laplace 압력 차 함수 | 실제 계면 수송·곡률·압력 결합 및 솔버 연결 |
| 액적 | 기하학적 color의 연결 성분에서 입구 연결 액주/액막과 분리 성분, 체적·등가직경·중심·평균속도 진단 | 물리 계산이 실제 단절을 만드는 1차 분열 해석 |

과거 SPUMA/OpenFOAM WALE 구현과 실행 기록은 존재한다. 다만 이번 작업 전 **현재 ReactiveFoam 수송 경로에는 WALE가 연결되어 있지 않았다.** 그 구분과 근거는 [기능 감사](capability-and-acceptance.md)에 남겼다.

## WALE 선택 방법과 제약

`constant/reactiveProperties`에서 다음을 지정한다. 기본값은 `physics.turbulence none`이며 기존 실행 결과를 유지한다.

```foam
transportBackend cuda;
waleCw 0.325;
physics
{
    chemistry false;
    turbulence WALE;
}
```

`WALE-stress-v1`은 비반응 HEM 또는 동결 상 수송의 **운동량 SGS 응력** 모델이다. 화학반응 또는 `mechanicalEquilibrium`과 함께 선택하면 명시적으로 거부한다. `physics.viscosity false`는 분자 점성을 끄며, WALE는 별도 선택이다. 실제 솔버에서는 CUDA 수송을 요구하고 CPU로 자동 대체하지 않는다. CPU 라이브러리 경로는 연산자 검증용으로 제공한다.

필터 폭은 셀 체적의 세제곱근이다. 분자 점성이 0이어도 WALE를 사용하면 직교 메시와 면 중심 skew 검사를 적용한다. 따라서 현재 R04 메시에서 사용 가능하다고 아직 판정하지 않았다. 지원 경계는 기존 slipWall·extrapolate·fixedState이며, 경계 `nut`는 소유 셀 값을 사용한다. no-slip 벽 모델이나 SGS 입구 조건은 추가하지 않았다.

진단 필드 `waleNut`는 저장 시점의 현재 상태에서 다시 계산한다. 로그는 모델·필터·누락된 SGS 열/종 및 SGS k 모델을 표시한다. 물리 모델 해시에 WALE와 계수를 포함하므로 다른 설정의 체크포인트를 그대로 재시작할 수 없다. 기존 체크포인트 이진 형식은 유지한다.

## 검증 결과

- WALE 독립 텐서식, 회전 불변성, 순수 전단의 0 극한, 벽 근처 스케일링, 극단값 검증 통과. RTX 5080 실제 CUDA에서 34개 fixture 통과.
- 64셀 주기 격자의 독립 NumPy 기준과 CPU/GPU 수송 비교 통과. 최대 `nut` 오차 `5.76e-20 m²/s`, 응력 RHS 차이 오차 `9.01e-8`. 총에너지·운동량 보존, 음의 SGS 운동에너지 변화율, 점성 시간 제한, 잘못된 입력·응력 일 overflow의 출력 보존 확인.
- 기존 수송 회귀 **CPU 29개 + CUDA 29개 통과**. 최종 바이너리 해시는 각 결과에 기록했다.
- 실제 ReactiveFoam **64셀, 30 ns × 2스텝** CUDA 수송·GPU HEM 실행 통과. WALE 비활성 결과는 보존장 배열에서 기준 솔버와 비트 단위 일치. `Cw=0`도 비활성과 일치. WALE 활성 시 양의 `waleNut`과 결과 변화 확인.
- 중간 체크포인트에서 재시작한 보존장 배열은 연속 실행과 비트 단위 일치. WALE 설정 변경 재시작 및 비직교 메시 거부 확인.
- 시간 진행 중 GPU HEM 로그의 `cpuFallbacks=0`, `deviceFailures=0` 확인. 초기 상태 생성과 메시·입출력 등 호스트 작업까지 GPU로 옮겼다는 의미는 아니다.
- 표면장력 기초 함수는 CPU와 실제 CUDA `sm_120`에서 부호·단위·해석식·오류 경로·overflow 검증 통과. 액적 진단은 6개 테스트 통과.

[수송 CPU](../../results/spray-physics/wale-transport-cpu.json), [수송 GPU](../../results/spray-physics/wale-transport-cuda.json), [기존 수송 CPU](../../results/spray-physics/transport-off-cpu-final.json), [기존 수송 GPU](../../results/spray-physics/transport-off-cuda-final.json), [실제 솔버](../../results/spray-physics/wale-solver-integration.json), [계면 함수](../../results/spray-physics/interface-primitives.json).

솔버 실행 로그와 필드는 `logs/wale-solver-integration-v3`에 있다. 이 작은 시험의 30 ns는 **R04 1M 메시에서 30–50 ns가 가능하다는 검증이 아니다.** R04 전체 성능 측정은 재실행하지 않았다.

## 1차 분열까지 남은 작업

열역학적 `alphaLiquid`를 그대로 기하학적 계면으로 사용하면 안 된다. 현재 HEM에는 계면 수송과 Laplace 압력 차를 유지하는 압력·계면력 이산화가 없다. 공통 압력을 사용하는 모든 one-fluid 모델이 공간적인 압력 점프를 표현할 수 없다는 뜻은 아니다. 첫 단계에서는 계면 기초 함수만 독립적으로 구현했다.

다음 구현 순서는 기하학적 계면과 물질 재고의 일관된 보존 수송, 서로 다른 상 EOS와 모세관 압력 결합, 표면 에너지 수지, 정적 액적·모세관파·액적 진동 검증, Rayleigh–Plateau 액주 분열과 액적 진단 연결이다. [상세 계면 설계](resolved-interface-design.md)와 [수용 기준](capability-and-acceptance.md)을 따른다. **실제 액주·액막 분열과 액적 생성은 아직 완성되지 않았다.**
