# 90도 충돌 N₂O 분사: 공급 상태와 flashing 범위

## 고정된 조건

- 두 분사류: 순수 액체 N₂O
- 공급 온도: `293.15 K` (`20 °C`)
- 공급 압력: `55 bar(g)`이며 기준 압력은 `1.01325 bar(abs)`
- 따라서 공급 절대압: `56.01325 bar(abs)` (`5,601,325 Pa`)
- 외부 조건: `1.01325 bar(abs)`와 `40 bar(abs)`
- 압력차: 각각 정확히 `55 bar`와 `16.01325 bar`

게이지 값을 절대압으로 바꾼 것 외에 압력이나 온도를 바꾸지 않았다.

## 물성 기준과 재현성

두 모델을 분리해 계산했다.

1. **현재 ReactiveFoam 모델**: 저장소의 4종 Peng–Robinson(PR) gas와 pure-PR liquid, exact host equilibrium closure. 사용한 physical model fingerprint는 `b90c87d0ebb97f92af562c63708d3f5b33ac259f855db7c0ee7f164f8dfc7db1`이다.
2. **참조 모델**: CoolProp `NitrousOxide`의 Helmholtz EOS. 이 구현은 N₂O에 대해 Lemmon–Span의 *Short Fundamental Equations of State for 20 Industrial Fluids*를 사용한다. [NIST의 원 논문 페이지](https://www.nist.gov/publications/short-fundamental-equations-state-20-industrial-fluids)는 30 MPa 이하에서 대표 밀도 불확도 0.2%, 증기압 불확도 0.2%를 제시한다. [CoolProp N₂O 페이지](https://coolprop.org/fluid_properties/fluids/NitrousOxide.html)는 EOS 출처와 REFPROP 비교 범위를 명시한다. NIST의 [포화 물성 설명](https://www.nist.gov/publications/thermophysical-properties-selected-fluids-saturation)과 [REFPROP 페이지](https://www.nist.gov/srd/refprop)는 포화선·quality 계산의 의미를 확인하는 1차 자료다.

실행 환경은 `/home/jsw/문서/analysis/spuma-multiphysics/research/reactive-env/bin/python`이고 저장소 `tools/reactive_backend.py`로 동일 ABI를 호출했다. 입력 identity는 다음과 같다.

| 입력 | SHA-256 |
|---|---|
| `lib/libreactiveBackend.so` | `44224acb75f87352ed76eebd67f7dcc06f94b2ec6fbdd070d42186733a09cb0b` |
| `tools/reactive_backend.py` | `b7a53297e402b9ae48c700c2b37484df56fa27e6ab6a15cbaf7ff8b1db8e1c76` |
| `cold-pr-config.yaml` | `1262c7293e0cd73e99a1d94aaa7942d2675fd00a6f6beab94ad91a9abe76a23c` |
| `cold-pr.yaml` | `7ea020c480bff27404a9ccb051c4940bac71631257e159084b93df2092a6ecdd` |

PR 포화압은 gas와 pure-liquid chemical potential의 차를 압력에 대해 Brent 해법으로 0으로 만들었다. 지정 압력의 포화온도는 이 계산을 온도에 대해 다시 풀었다. 참조 값은 같은 입력에서 CoolProp `PropsSI`로 계산했다.

수치는 다음 명령으로 재현하며 결과 JSON은 식, 상태, round-trip residual과 모든 물성 입력 hash를 보존한다.

```bash
PYTHONPATH=tools /home/jsw/문서/analysis/spuma-multiphysics/research/reactive-env/bin/python \
  tools/probe_impinging_n2o_flash.py \
  --configuration /home/jsw/문서/analysis/runs/r04_frozen_14k_20260921/case/thermo/cold-pr-config.yaml \
  --output results/benchmarks/impinging-n2o-thermodynamics.json
```

## 공급 상태

| 293.15 K | 현재 PR | 참조 Helmholtz EOS |
|---|---:|---:|
| N₂O 포화압 | 50.769211 bar | 50.525093 bar |
| 56.01325 bara까지의 과냉 압력 여유 | 5.244039 bar | 5.488157 bar |
| 공급 액체 밀도 | 750.480 kg/m³ | 796.018 kg/m³ |

두 EOS 모두 이 조건을 포화선 위 압력의 **압축 액체**로 판정한다. 현재 PR에서 `liquidFractions=(1,0)`로 상태를 만든 뒤 exact UV recovery를 수행하면 `p=56.01325 bar`, `T=293.15 K`, N₂O 액상 질량분율 `1.0`, 기상 체적분율 `0.0`을 그대로 돌려준다. 따라서 현 HEM `fixedState` 입력으로는 열역학적으로 허용된다.

다만 PR 액체 밀도는 참조보다 `5.72%` 낮다. 포화압 차이는 `+0.483%`지만, 밀도 차이는 같은 면적·속도에서 질량유량과 운동량유량을 직접 바꾼다. 이 비교는 현재 PR을 검증된 정밀 N₂O 액체 EOS로 승격하지 않는다.

## 감압 시 평형 flashing

두 제한 계산을 사용했다.

\[
x_h=\frac{h_{in}-h_l(p_{out})}{h_v(p_{out})-h_l(p_{out})},\qquad
x_s=\frac{s_{in}-s_l(p_{out})}{s_v(p_{out})-s_l(p_{out})}.
\]

`x_h`는 단열 throttle의 등엔탈피 평형 vapor mass quality이고, `x_s`는 가역 단열 평형 경로의 등엔트로피 quality다. vapor volume fraction은

\[
\alpha_g=\frac{x/\rho_v}{x/\rho_v+(1-x)/\rho_l}
\]

로 계산했다.

| 외부 절대압 | 모델 | 포화온도 | `x_h` | `αg,h` | `x_s` | `αg,s` |
|---:|---|---:|---:|---:|---:|---:|
| 40 bar | 현재 PR | 282.999 K | 0.1342 | 0.5254 | 0.1220 | 0.4982 |
| 40 bar | 참조 | 283.137 K | 0.1174 | 0.4967 | 0.1060 | 0.4681 |
| 1.01325 bar | 현재 PR | 184.275 K | 0.5848 | 0.99837 | 0.4372 | 0.99705 |
| 1.01325 bar | 참조 | 184.684 K | 0.5681 | 0.99816 | 0.4296 | 0.99679 |

현재 PR의 등엔탈피 상태를 `(p, Tsat, liquid fraction=1-xh)`로 다시 만들고 UV flash했다. 40 bar에서는 액상 질량분율 `0.865850`, 기상 체적분율 `0.525400`; 1 atm에서는 액상 질량분율 `0.415161`, 기상 체적분율 `0.998368`을 회복했다. 압력·온도 상대/절대 오차는 각각 `9e-12`와 `4e-10 K` 이하였다.

따라서 두 외부압 모두 equilibrium flashing 가능성이 명확하다. 40 bar에서도 질량으로 약 11–13%가 증기지만 큰 밀도비 때문에 기상이 부피의 약 절반을 차지한다. 1 atm에서는 질량 quality가 약 57–58%이고 부피는 사실상 모두 기상이다.

이 수치는 nozzle 내부의 순간 상분배 예측값이 아니다. 실제 노즐은 총엔탈피를 정압 엔탈피와 운동에너지로 나누며, nucleation delay와 metastable liquid, cavitation, 체류시간, 열전달 및 two-phase choking에 따라 평형 flash보다 늦게 기화할 수 있다. 표의 등엔탈피·등엔트로피 결과는 두 개의 평형 기준이다. 특히 1 atm의 PR 포화온도 `184.275 K`는 구성의 하한 `182.34 K`보다 약 `1.93 K` 높을 뿐이므로, 수치 상태가 더 냉각되면 모델 범위를 즉시 벗어난다.

## 현 `fixedState` 경계가 뜻하는 것

`ReactiveFoam.C`의 현재 경로는 다음과 같다.

1. 패치에서 `p,T,U,Y,liquidFractions`를 읽는다.
2. `reactive_rt_make_state`로 conserved ghost state를 만들고, HEM이면 exact UV recovery를 수행한다.
3. 회복된 `p,T`가 요청값과 `1e-6` 상대 오차 안에서 일치하지 않으면 경계를 거부한다.
4. 각 face에서 이 ghost state를 고정된 HLL right state로 사용한다. 내부 owner state가 left state다.

그러므로 `U=(0 0 0)`인 고압 `fixedState`도 압력 불연속을 푸는 Riemann flux를 통해 유입과 가속을 만들 수 있다. 이는 매 시각 같은 정적 ghost state를 공급하는 무한 저장조형 경계다. 다음 요소는 포함하지 않는다.

- `p=56.01325 bar`를 stagnation pressure로 해석하고 출구 정압·속도·질량유량을 함께 푸는 조건
- 실제 2×2 mm 유로 안의 수축, vena contracta, discharge coefficient, 벽 마찰 및 reservoir depletion
- single/two-phase choking 또는 유한속도 cavitation·nucleation
- 비반사 outlet 또는 외부 압력만을 특징량으로 강제하는 pressure outlet

따라서 zero-velocity reservoir ghost는 **압력차가 유동을 생성하는 smoke benchmark**에는 사용할 수 있지만, 계산된 질량유량을 실제 injector 유량이나 choking 결과로 해석할 수 없다. 단순 비압축 Bernoulli 규모 `sqrt(2 Δp/ρPR)`는 1 atm 케이스 약 `121 m/s`, 40 bar 케이스 약 `65 m/s`이지만, flashing과 압축성 때문에 경계에 이 값을 몰래 지정해서는 안 된다.

물리 벤치마크로 올리려면 다음 중 하나를 명시해야 한다.

1. 저장조와 2×2 mm 유로를 계산영역에 포함하고, 멀리 있는 저장조 면에 `fixedState U=0`을 둔다. 이 경우 유로가 가속을 해상하지만 현 HEM의 instantaneous equilibrium 가정을 함께 인정해야 한다.
2. 외부 nozzle 계산/상관식에서 검증한 mass flux와 exit thermodynamic state를 얻어 입구에 지정한다. 이 경우 pressure-driven reservoir benchmark와 구분한다.
3. stagnation-state/characteristic reservoir와 two-phase choking을 푸는 새 경계 모델을 구현한다.

현재 직접 2×2 mm patch에 `fixedState U=0`을 두는 선택은 1번의 유로를 생략한 축약형이다. 결과 이름과 보고서에 `fixed-ghost pressure-driven inflow; no nozzle/choking model`이라고 남겨야 한다.

## 256k one-step의 1 atm recovery 진단

1 atm case는 초기 `dt=25.884037 ns`에서 다섯 번 반감한 뒤 `dt=0.808876 ns`로 1 step을 성공했다. 성공 step의 `CPU fallback=0`, 최저 온도는 `212.616 K`였다. 실패 로그 240건은 각 시도에서 48 cell이 실패한 결과이며, `detailed:true`로 `q`, internal-energy density, 전체 guess를 보존한 레코드는 재시도별 한 개씩 정확히 5개다.

이 5개를 로그의 `q`, internal energy, guess와 equilibrium flag 그대로 동일 host PR backend의 `reference` recovery에 재생했다. host도 5개 모두 실패했으므로 이 현상을 CUDA 산술 문제로만 분류할 수 없다. 다만 실패 원인은 두 부류로 나뉘다.

| retry | `dt` (ns) | exact host replay | 같은 `q,E`, 포화선 근처 guess | 판정 |
|---:|---:|---|---|---|
| 0 | 25.8840 | 실패 | `p=4.00010 bar`, `T=207.080 K` 회복 | host/device 공통 탐색 초기화 문제 |
| 1 | 12.9420 | 실패 | `p=2.39282 bar`, `T=193.874 K` 회복 | host/device 공통 탐색 초기화 문제 |
| 2 | 6.47101 | 실패 | `p=1.54124 bar`, `T=182.520 K` 회복 | host/device 공통 탐색 초기화 문제; `Tmin`보다 `0.180 K`뿐 |
| 3 | 3.23550 | 실패 | 실패 | 구성된 fluid temperature 범위 밖 |
| 4 | 1.61775 | 실패 | 실패 | 구성된 fluid temperature 범위 밖 |

원래 guess인 `p=1.01325 bar`, `T=293.15 K`에서는 PR liquid root가 없어 액상 후보가 첫 반복 전에 종료되었고, 순수 기상 후보는 모델 하한 `182.34 K`까지 내려간 뒤 line search에 실패했다. `q,E`를 고정하고 guess의 `p,T`만 1 atm PR 포화선 근처로 바꾸는 존재성 probe에서 retry 0–2는 잔차 `1e-9` 이하의 동일-EOS 상태를 회복했다. 이 probe는 원래 재생이 아니며 solver 설정 제안도 아니다. 상태가 존재함을 구분하기 위한 진단이다.

retry 3–4에서는 보존 밀도를 맞춘 `Tmin`의 PR gas branch가 N₂O 응축에 대해 안정적이었지만, 그 내부에너지 밀도가 목표보다 각각 `148,038 J/m³`, `26,352 J/m³` 높았다. 즉 현 fluid-only PR/NASA closure로는 더 낮은 온도가 필요하지만, 이는 N₂O triple-point 근처의 구성 하한 아래이며 solid closure도 없다. 따라서 이 두 중간상태는 현 EOS 허용 범위 밖으로 판정한다.

성공 `dt`가 짧아진 것은 이 문제가 timestep에 민감함을 보여주지만, 거부된 큰-step RK 중간상태가 물리적으로 표현 가능했음을 입증하지 않는다. 메시의 직교성이 정상이므로, 이 제한된 증거에서는 메시 skew를 원인으로 보지 않았다. 재현 명령과 레코드별 입력·탐색 요약·해시는 `results/benchmarks/impinging-n2o-recovery-diagnosis.json`에 있다.

```bash
PYTHONPATH=tools /home/jsw/문서/analysis/spuma-multiphysics/research/reactive-env/bin/python \
  tools/diagnose_impinging_n2o_recovery.py \
  --failures logs/impinging-n2o-validation-v1/ambient_1atm/reactiveFailures.jsonl \
  --configuration /home/jsw/문서/analysis/runs/impinging_n2o_90deg_20260922/thermo/cold-pr-config.yaml \
  --library lib/libreactiveBackend.so \
  --solver-log logs/impinging-n2o-validation-v1/ambient_1atm/solver.log \
  --seed-temperature 184.27467115507127 \
  --output results/benchmarks/impinging-n2o-recovery-diagnosis.json
```
