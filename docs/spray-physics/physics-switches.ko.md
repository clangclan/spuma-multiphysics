# 표면장력 선택과 난류 열·종 수송

2026-09-23. 기존 과냉각 액체 N₂O HEM 경로에 선택 제어와 WALE scalar 수송을 추가했다. 고체·승화·연소는 다시 활성화하지 않았다. 이 변경은 **표면장력을 포함한 flashing 계면 솔버 완성 보고가 아니다.**

## 표면장력 끄기

`constant/reactiveProperties`의 기존 `physics` 블록에 다음 항목을 둔다.

```foam
physics
{
    surfaceTension false;
}
```

기본값도 `false`다. 상변화·분자 점성·열전도·WALE와 독립적인 선택이다. 끄면 모세관 곡률·응력·일·표면에너지·모세관 시간 제한을 요구하지 않는다. 별도 모세관 작업 공간도 만들지 않는다. 옵션 생략과 명시적 `false`는 같은 물리 모델 해시를 사용하므로 기존 체크포인트와 호환된다.

`surfaceTension true`는 새 [diffuse-interface GPU 결합 경로](capillary-flashing.ko.md)를 선택한다. 하나의 액상, 비반응 HEM, CUDA 수송·열역학, `closureCpuFallback false`, 양의 `surfaceTensionCoefficient`가 필요하다. 독립 액체 질량과 벌크·표면 총에너지를 수송한다. 실제 flashing 결합은 실행되지만 정적 액적의 기생 유속 수렴 기준은 아직 통과하지 못했다.

독립 모듈 `pintleInterfaceGeometry.h`에서는 `InterfaceOptions::enableSurfaceTension`이 모세관 경로를 제어한다. 비활성일 때 곡률과 capillary workspace 포인터는 null로 둘 수 있다. 계면 수송에 별도로 필요한 color·기하 계산은 표면장력 비용과 분리한다. 이 모듈은 현재 ReactiveFoam 실행 파일에 연결되어 있지 않다. 주어진 면 체적 플럭스의 donor-cell 수송은 비발산 이동 시험용이며, 일반 압축성 유동의 체적분율 팽창항이나 상변화 질량 전달을 대신하지 않는다.

## WALE 열·종 혼합 선택

```foam
transportBackend cuda;
waleCw 0.325;
turbulentPrandtl 0.85;
turbulentSchmidt 0.7;
physics
{
    chemistry false;
    phaseChange true;
    turbulence WALE;
    turbulentHeatFlux true;
    turbulentSpeciesMixing true;
    surfaceTension false;
}
```

두 scalar 스위치는 기본적으로 꺼져 있다. 각각 끄면 해당 물성 배열과 GPU 작업 공간을 만들지 않는다. 분자 점성·열전도·종 확산 선택과 독립적이다. `turbulentPrandtl` 또는 `turbulentSchmidt`는 해당 스위치가 켜졌을 때만 유효한 양수여야 한다. scalar 선택과 계수는 재시작 물리 모델 해시에 반영한다.

열 플럭스는 `-rho*nut*cp/Prt * grad(T)`다. 명시적 확산 시간 제한은 실제 면 전도계수를 인접 셀 각각의 `rho*cv`로 나눈 값에 기반한다. 균일 상태에서는 `nut*cp/(Prt*cv)`가 된다. 같은 이유로 종·운동량의 WALE 확산 제한도 실제 면 계수와 양쪽 밀도를 사용한다. 셀별 `nut`의 최댓값만 사용하면 밀도 차이가 큰 액체·기체 경계에서 제한을 과소평가할 수 있으므로 800:1 밀도비를 시험한다.

종 혼합은 전체 상을 합한 보존 종 질량분율을 사용한다. 종 플럭스의 합이 0이 되도록 보정하고 `sum(h_eff,k*J_k)`를 같은 면의 총에너지 플럭스에 더한다. PR 모델에서 이 SGS 혼합을 활성화해도 비이상 분자 확산을 구현했다고 표시하지 않는다. 기존 `physics.speciesDiffusion`의 비이상 EOS 제한은 유지한다.

`h_eff,k`는 이미 복원된 압력·온도·상 분할에서 같은 EOS의 기상 부분 엔탈피와 응축상 엔탈피를 종 질량으로 가중한 값이다. 공통 압력 경로는 `sum(q_k*h_eff,k)=rho*e+p`를 확인한다. 표면장력 OFF의 기존 물성 준비는 CPU이고 열·종 플럭스와 WALE는 CUDA다. 후속 [단일 액체 모세관 경로](capillary-flashing.ko.md)는 이 엔탈피 준비도 GPU PR 함수로 수행하며 상별 압력 일 `p_g*alpha_g+p_l*alpha_l`을 검사한다. `REACTIVE_WALE_SCALARS hostEnthalpyCells/hostPropertySeconds`와 `REACTIVE_WALE_PR`로 호스트와 장치 물성을 구분한다. 단계별 `nestedScalarPropertySeconds`는 포장·GPU 호출 대기를 포함하는 중첩 물성 시간이며 기존 단계 시간에 이미 포함되므로 총시간에 다시 더하지 않는다.

동결 응축상에 종 SGS를 적용하면 별도 응축상 질량장도 같은 플럭스로 움직여야 한다. 그 결합이 없는 구성은 명시적으로 거부한다. 반응/기계적 환경과 WALE의 기존 제한도 유지한다. 공간과 상에 따라 달라지는 N₂O/공기 분자 점성·열전도 상관식은 이번 상수계수 연산자 시험에 포함되지 않는다.

## 검증과 남은 범위

- [실제 솔버 시험](../../tools/validate_physics_switches.py): 64셀, 30 ns × 2스텝, 실제 CUDA 수송과 GPU HEM, CPU closure fallback 0. 표면장력 생략/false의 보존장·물리 해시·GPU 메모리 동등성, scalar off의 추가 물성 작업 0, 점성·열전도·WALE 열/종 동시 활성, 주기 경계 보존, 재시작 일치, 잘못된 표면장력 활성화 거부를 확인한다.
- [SGS 연산자 시험](../../tools/validate_wale_scalars.py): 독립 NumPy 플럭스, 질량·에너지 보존, 열 확산 CFL, 이전 단계 엔탈피 재사용 차단, CPU/CUDA 수송 경로를 검사한다.
- [엔탈피 시험](../../tools/validate_total_species_enthalpies.py): PR 기체·순수 액체·동결 및 평형 2상·148 K 기체·이상기체에서 엔탈피 항등식과 실패 시 입력/출력 보존을 검사한다.
- [계면 모듈 시험](../../tools/validate_interface_geometry.py): 법선·곡률, 보존적 color 수송, 표면장력 on/off, CPU/CUDA 경로를 검사한다. donor-cell 수송은 PLIC가 아니며 모듈의 비용은 전체 솔버 성능 수치가 아니다.

실제 솔버는 **8개 구성·35개 검사**를 통과했다. N₂O 기액 혼합물의 두 스텝에서는 GPU flash 후보 평가 320회, CPU fallback/장치 실패 0, 최대 전역 보존 잔차 `7.96e-13`을 기록했다. 열만 켠 경로와 종 혼합만 켠 경로도 각각 시험했다. [솔버 결과](../../results/physics-switches-20260923/solver.json), [CPU SGS](../../results/physics-switches-20260923/wale-scalars-cpu.json), [CUDA SGS](../../results/physics-switches-20260923/wale-scalars-cuda.json), [엔탈피 항등식](../../results/physics-switches-20260923/total-species-enthalpies.json)에 입력·실행 파일 해시와 수치를 남겼다. 기존 수송 회귀는 [CPU 29개](../../results/physics-switches-20260923/reactive-transport-cpu.json)와 [CUDA 29개](../../results/physics-switches-20260923/reactive-transport-cuda.json)를 통과했다.

[독립 계면 결과](../../results/spray-physics/interface-geometry.json)는 CPU/CUDA 수치 일치와 표면장력 on/off 커널·작업 공간 및 작은 격자 반복 실행 시간을 기록한다. 공통 color/기하 메모리는 off에서도 필요하므로 표면장력 전용 메모리와 구분한다. 두 백엔드의 시간 측정 격자 크기가 서로 다르므로 이 결과로 CPU 대비 GPU 가속 배율을 계산하지 않는다.

위 선택 제어 시험은 기존 off 경로의 회귀 기록이다. 후속 GPU 결합의 실제 실행·보존·재시작 결과와 아직 남은 정확도 기준은 [결합 모델 안내](capillary-flashing.ko.md), [정적 액적 검토](capillary-static-balance-review.md)를 참고한다. 짧은 통합 실행을 액주 분열이나 장시간 모세관파 정확도 검증으로 해석하지 않는다.
