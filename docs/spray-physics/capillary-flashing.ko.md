# GPU 표면장력–flashing 결합

`physics.surfaceTension true`는 독립 액체 질량 수송, 계면 기하, 모세관 응력,
곡률을 반영한 상평형, 표면에너지 회계를 함께 활성화한다. 상수 표면장력을
사용하는 1차 diffuse-interface 모델이다. PLIC VOF 또는 이미 검증된 액주 분열
모델이라는 의미는 아니다. 실제 시험 결과는 별도 검증 기록과 함께 판단한다.

## 사용 조건

하나의 액상 N₂O, 비반응 HEM, 고정 직교 메시, CUDA 수송과 CUDA 열역학을 사용한다.
고체상 및 승화는 포함하지 않는다. 기존 148 K 과냉각 액체 설정을 사용한다.
`physics.phaseChange false`는 액체 질량을 보존하는 frozen-phase 시험,
`true`는 즉시 UV 상평형에 의한 flashing/응축 계산이다. 유한 속도 핵생성 모델은 아니다.

```foam
transportBackend cuda;
closureBackend cuda;
closureCpuFallback false;
closureScalarBackend cpu;
closureJacobian finiteDifference;
thermoExactReuse false;
surfaceTensionCoefficient 0.0020577273399028486; // N/m, constant
capillaryCfl 0.25;
capillaryGeometryTolerance 1e-10;
physics
{
    chemistry false;
    phaseChange true;
    surfaceTension true;
    viscosity true;
    heatConduction true;
    turbulence WALE;
    turbulentHeatFlux true;
    turbulentSpeciesMixing true;
    speciesDiffusion false;
}
dynamicViscosity 1.821e-5;
thermalConductivity 0.02587;
molecularDiffusivity 0;
turbulentPrandtl 0.85;
turbulentSchmidt 0.7;
```

예시의 점성·열전도 상수는 기존 연산 결합 시험용 값이다. 실제 액체/기체 혼합물의
온도·조성 의존 수송 물성으로 검증된 값으로 해석하지 않는다. PR EOS의 분자 확산은
비이상 열역학 인자가 구현되지 않아 꺼둔다. 종의 난류 혼합은 별도 WALE SGS 연산이다.
frozen-phase + SGS 종 혼합은 아직 입력 단계에서 거부한다.

표면장력 예시 값은 CoolProp의 N₂O 포화 표면장력을 293.15 K에서 평가한 상수다.
[CoolProp N₂O 자료](https://coolprop.org/fluid_properties/fluids/NitrousOxide.html)는
[Mulero 등(2012)](https://doi.org/10.1063/1.4768782)의 상관식을 인용한다.
이 버전은 냉각 중에도 이 상수를 유지한다. σ(T)를 그대로 넣으면 표면 내부에너지와
표면 엔트로피 회계를 함께 바꾸어야 하므로 온도 의존 σ 또는 Marangoni 효과를 주장하지 않는다.

## 보존 상태와 GPU 경로

- `rhoLiquid0`: 총 N₂O 중 액체의 질량/체적. 화학종·운동량과 같은 접촉 유량으로 수송한다.
- `interfaceColor`: 액체 질량과 액체 EOS 밀도에서 얻은 `Vl/(Vl+Vg)`.
  분모는 UV 복원의 유한 체적 잔차를 포함한 상 체적 합이다. 질량을 클리핑하거나 재분배하지 않는다.
- `surfaceEnergyDensity`: `sigma*|grad(interfaceColor)|`, J/m³.
- `rhoTotalEnergy`: 벌크 내부에너지 + 운동에너지 + 표면에너지.
- `interfaceCurvature`: 곡률, 1/m.

각 RK 단계에서 GPU가 기하 및 응력·에너지 유량을 계산한다. UV 복원에 넘기는
에너지는 총에너지에서 운동에너지와 표면에너지를 뺀 값이다. `J=σ κ`일 때
`p_g=p_bar-c J`, `p_l=p_bar+(1-c)J`로 PR 밀도·에너지·화학퍼텐셜을 평가한다.
곡률이 있는 flash는 GPU 유한차분 Jacobian을 사용한다. 상변화는 총 화학종·운동량·
총에너지를 유지하면서 액체 질량만 갱신한다. 기하와 UV 복원을 반복 결합하며,
실패한 스텝은 q·열역학 상태·기하를 되돌리고 시간 간격을 줄인다.

계면 기하, 모세관 유량/CFL, 곡률 UV flash는 CUDA 실행이다. 파일 입출력, 배치 포장,
외부 반복 제어와 WALE 종 혼합의 부분 엔탈피 준비는 호스트에서 수행한다.
호스트 엔탈피 준비는 동일 PR EOS를 사용하며 `REACTIVE_WALE_SCALARS`에 계수된다.
GPU UV 실패를 CPU flash로 대체하지 않는다.

상변화 후 RK1의 액체 질량을 GPU에 다시 올릴 때 RK0 보존 상태는 유지한다.
재시작은 추가 액체 질량과 표면에너지 의미를 물리 모델 해시에 포함하며, 상수 σ가
바뀌면 기존 체크포인트를 거부한다. 기존 HEM 체크포인트의 에너지 의미를 자동 변경하지 않는다.

## 끄기와 검증 도구

`physics.surfaceTension false` 또는 항목 생략 시 기존 HLL/HEM 경로를 사용한다.
추가 액체 질량 슬롯, 계면 기하 커널, 모세관 작업 메모리는 생성하지 않는다.
`REACTIVE_CAPILLARY_PROFILE`에서 실제 할당 및 커널 횟수를 확인한다.

- `tools/validate_capillary_flash.py`: 실제 CUDA 곡률 UV 복원·화학퍼텐셜·실패 원자성.
- `tools/capillary_transport_test.cpp`: CPU/CUDA 모세관 수송, 전역 보존, RK1 교체, SGS 액체 유량.
- `tools/prepare_capillary_case.py`, `tools/validate_capillary_solver.py`: 평면/액적/파/flash 실제 솔버 케이스와 수치 측정.
- `tools/prepare_capillary_impingement.py`: 기존 256,000셀 분사 메시를 재사용하여 점성·열전도·WALE 혼합·표면장력·flashing 케이스 생성.

시간 간격은 음향·확산·모세관 제한 중 최솟값이다. `maxDeltaT=30 ns`는 상한이며
안정성 제한을 무시하여 30 ns 이상으로 고정하지 않는다.
