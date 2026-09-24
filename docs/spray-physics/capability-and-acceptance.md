# 액주·액막 1차 분열: 기능 경계와 수치 수용 기준

이 문서는 `main`의 `9ff04ff`와 현재 작업 트리를 대상으로 한 기능 감사와, 직접 해상하는 액주·액막의 1차 분열(primary breakup)을 수용하기 위한 작은 회귀 계획이다. 목표는 기하학적 액체 계면이 표면장력·공기역학적 응력 아래에서 끊어지고, 끊어진 연결 성분을 액적으로 식별하는 것이다. 액적 parcel 생성, 2차 분열, 충돌·합체, 증발 parcel 모델은 이 범위가 아니다.

## 현재 주장할 수 있는 범위

| 항목 | 확인된 상태 | 의미 |
|---|---|---|
| `ReactiveFoam` 보존 수송 | 있음 | 셀 중심 질량·운동량·총에너지와 HEM/동결 상 재고를 수송한다. |
| `alphaLiquid*` | 있음, 그러나 열역학적 양 | EOS/flash가 복원한 셀 평균 상 체적분율이다. 기하학적 계면 재구성용 VOF 표지가 아니다. |
| 기하학적 액체 `alpha`의 유계 보존 이류·재구성 | 기준 커밋에는 없음 | 이것이 없으면 계면 곡률, 액주 목 반경, 액막 두께 및 위상 변화(topology change)를 해석할 수 없다. |
| 표면장력 | 기준 커밋에는 없음 | 곡률, 균형력 압력 결합, 모세관 시간 간격 제한이 모두 필요하다. |
| WALE | CUDA stress-v1 통합 및 연산자·작은 유동/재시작 검증 완료 | 새 stress-v1 경로는 운동량 SGS 응력 범위다. SGS 열유속·화학종 유속과 turbulence–chemistry interaction(TCI)은 포함하지 않는다. 아래 게이트를 통과하기 전 물리 검증 완료로 세지 않는다. |
| 1차 분열 | 미완료 | 별도 breakup source가 필요한 것이 아니라, 검증된 기하학적 계면 수송과 표면장력이 분리를 직접 만들고 보존 법칙이 그 운동을 이어가야 한다. |
| 연결 성분 진단 | 구현됨 | `tools/resolved_drops.py`가 명시적인 기하학적 `alpha`, 셀 체적·중심, 면 연결, 속도 및 입구 셀을 받아 액주/액막과 분리 액적을 구분한다. 이는 식별 도구이며 물리 모델이 아니다. |
| Lagrangian parcel | 없음 | 연결 성분을 parcel로 변환하지 않고 parcel 수·직경분포를 생성했다고 주장하지 않는다. |

현재 HEM `alphaLiquid*`를 기하학적 `alpha`로 바꾸어 부르는 것은 허용하지 않는다. 두 양은 액체의 존재량이라는 공통점만 있고 계면 법선·곡률·연결성에 필요한 의미가 다르다. 새 진단도 `alpha_semantics == "geometric_volume_fraction"`를 요구하고 `hem_flash_phase_fraction` 같은 입력을 거부한다.

### 과거 WALE의 실제 출처와 한계

작업 공간에는 현재 `ReactiveFoam`과 별개인 과거 SPUMA/OpenFOAM 다상 솔버의 WALE 실행 근거가 있다.

- 보관된 `WALE.C`: `/home/jsw/문서/analysis/reports/spuma_gpu_analysis_20260910/evidence/multiphase/source/spuma/src/TurbulenceModels/turbulenceModels/LES/WALE/WALE.C`, SHA-256 `f84f709b8e4865fbd34a478ff2ad143ee2c87f68fd2c5964936e8512314198ea`. `Ck=0.094`, `Cw=0.325`, `cubeRootVol` 계열 OpenFOAM 구현이다.
- 보관된 실행 설정: `/home/jsw/문서/analysis/reports/spuma_gpu_analysis_20260910/evidence/multiphase/source/production/constant/momentumTransport`, SHA-256 `ec49bf0f1da59b67dc6a944fdc88199d0bd0cf1c89c973cfff4253ec22eb6be8`, `simulationType LES`, `model WALE`, `delta cubeRootVol`을 선택한다.
- 반복 실행 로그 `timed_cpu_r1/r2/r3/solver.log`는 각각 `Selecting LES turbulence model WALE`를 기록하고 12단계 뒤 정상 종료한다. SHA-256은 순서대로 `79ab281d…`, `db03a39f…`, `d106fad5…`이다. 위치는 `/home/jsw/문서/analysis/benchmarks/native_20260910/pintle/`이다.
- 원본 수집 manifest에도 위 소스와 설정의 원래 위치·크기·해시가 남아 있다: `/home/jsw/문서/analysis/reports/spuma_gpu_analysis_20260910/evidence/multiphase/source_manifest.json`.

이 근거는 과거 솔버에서 WALE가 선택되고 실행되었다는 것만 보인다. WALE 물리 정확도, 새 GPU 커널과의 동등성, 현재 `ReactiveFoam` 보존 플럭스와의 결합을 검증하지 않는다. 새 구현의 식은 [Nicoud와 Ducros의 WALE 원 논문](https://doi.org/10.1023/A:1009995426001)과 대조해야 한다.

## 연결 성분 진단의 계약

`tools/resolved_drops.py`는 JSON 명령행 도구이자 import 가능한 Python 모듈이다. 필수 입력은 다음과 같다.

```json
{
  "alpha_semantics": "geometric_volume_fraction",
  "alpha_threshold": 0.5,
  "alpha": [1.0, 0.8, 0.0],
  "cell_volumes": [1e-12, 1e-12, 1e-12],
  "cell_centers": [[0, 0, 0], [1e-4, 0, 0], [2e-4, 0, 0]],
  "edges": [[0, 1], [1, 2]],
  "periodic_edges": [],
  "velocity": [[1, 0, 0], [2, 0, 0], [0, 0, 0]],
  "inlet_cells": [0]
}
```

`alpha_i >= alpha_threshold`인 셀만 그래프 정점이 되고 `edges`와 `periodic_edges`가 같은 방식으로 연결한다. 레이블 `0`은 선택되지 않은 셀이고, 연결 성분은 최소 셀 번호 순서로 `1, 2, …`의 결정적 레이블을 얻는다. 활성 입구 셀을 하나라도 포함한 성분은 공급부에 붙은 액주·액막이며, 나머지는 분리 성분이다. 입구가 여러 개여도 같은 규칙을 적용한다.

성분 (C)마다 다음을 계산한다.

\[
V_C=\sum_{i\in C}\alpha_i V_i,\qquad
d_{eq}=\left(\frac{6V_C}{\pi}\right)^{1/3},
\]

\[
\boldsymbol{x}_C=\frac{1}{V_C}\sum_{i\in C}\alpha_iV_i\boldsymbol{x}_i,
\quad
\boldsymbol{M}_{V,C}=\sum_{i\in C}\alpha_iV_i\boldsymbol{u}_i,
\quad
\overline{\boldsymbol{u}}_C=\boldsymbol{M}_{V,C}/V_C.
\]

`volume_velocity_moment`는 체적-속도 모멘트이고 `mean_velocity`는 액체 체적 가중 평균이다. 밀도를 받지 않으므로 물리적 운동량이라고 부르지 않는다. 실제 운동량에는 일관된 액체 밀도장이 추가로 필요하다. `input_liquid_volume`, `labeled_liquid_volume`, `unlabeled_liquid_volume`을 함께 내어 임계값 아래에서 빠진 액체량을 숨기지 않는다.

주기 경계 쌍은 연결성만 합친다. 주기 이음매를 가로지르는 중심 좌표를 자동으로 펴지 않으므로 해당 성분의 기하 중심이 필요하면 입력 중심을 먼저 같은 주기 영상으로 unwrap해야 한다. 음수·1 초과·NaN `alpha`, 0 이하 체적, 비유한 좌표·속도, 길이 불일치, 범위 밖/자기 자신 면, 비정수 셀 번호, 알 수 없는 JSON 키를 거부한다.

실행 예와 현재 단위 회귀는 다음과 같다.

```bash
python3 tools/resolved_drops.py input.json > drops.json
python3 -m unittest tools/test_resolved_drops.py -v
```

단위 회귀는 정확한 부피·등가 직경·중심·체적-속도 모멘트·평균 속도, 여러 독립 레이블, 비활성 입구, 주기 이음매 병합, HEM 의미 거부와 잘못된 메시 입력을 검사한다.

## 작은 수치 수용 사다리

아래 순서를 건너뛰지 않는다. 앞 단계 실패 상태에서 뒤 단계가 그럴듯한 액적 그림을 만드는 것은 수용 근거가 아니다.

### 1. 기하학·보존 계약

작은 정렬/비정렬 메시에서 평면, 구와 얇은 액막의 기하학적 `alpha`를 만들고, 재구성된 체적과 법선·곡률을 해석값에 대해 메시 세분한다. 정지 및 병진 이류 뒤 (0\le\alpha\le1), 주기 경계 총 액체 체적 보존, 한 주기 뒤 형상 오차 감소를 요구한다. 이 단계의 `alpha`만 이후 액적 진단에 전달한다.

WALE는 임의로 준 속도구배 묶음에 대해 CPU/GPU의 (\nu_t)와 SGS 응력을 비교한다. 균일 속도에서는 정확히 0, 단순 전단에서 WALE 연산자가 0, 벽 근처 적절한 구배에서는 원 논문의 (\nu_t=O(y^3)) 거동을 확인한다. 이 검사는 모델 식과 커널 배선을 확인하며 난류 분무 정확도를 주장하지 않는다.

### 2. 표면장력 선행 게이트

1. **정적 3D 구:** 반지름 (R)인 구에서 압력 점프 목표는 정확히 (\Delta p=2\sigma/R)다. 유속의 감쇠, 액체 체적, 압력 점프 오차를 세 메시에서 기록한다. [Popinet 2009](https://doi.org/10.1016/j.jcp.2009.04.042)은 균형력 VOF가 정적 액적의 압력-표면장력 평형을 기계 정밀도까지 회복하는 검증을 제시한다. 비정렬 메시에서는 그 수치를 그대로 약속하지 않고, 오차가 세분에 따라 0으로 가는지 확인한다.
2. **점성 모세관파:** 파장당 (N=16,32,64,128), 초기 상대 진폭 0.01의 2D 문제를 Prosperetti 초기값 해와 같은 시각에서 비교한다. 정규화 RMS는
   \(E_N=\sqrt{\sum_j(a_N(t_j)-a_{ref}(t_j))^2/n}/a_0\)로 고정한다. [Basilisk의 공개 회귀](https://basilisk.fr/src/test/capwave.c)는 같은 정의와 해상도를 쓰고 (2/N^2)의 2차 기준선을 명시한다. 따라서 수용선은 새로 맞추지 않고 `E_N <= 2/N^2` 및 세분 시 비증가로 둔다.
3. **3D 액적 진동:** 작은 (n=2) 변형의 주파수 목표는 Rayleigh–Lamb 식
   \[
   \omega_n^2=\frac{n(n+1)(n-1)(n+2)\sigma}
   {[(n+1)\rho_l+n\rho_g]R^3}
   \]
   이다. 공개된 [3D 검증 설정과 Prosperetti 초기값 비교](https://basilisk.fr/sandbox/jieyun/test/oscillation_3d_ebit.c)를 그대로 재현한다. 세 메시 결과에서 오차가 감소하고, 세 점 Richardson 외삽값과 최세망 차이로 계산한 이산화 불확도 안에 해석 주파수가 들어와야 한다. 임의의 추가 백분율 허용치는 두지 않는다.

명시적 표면장력 시간 적분은 가장 작은 셀의 모세관파에 제한된다. 구현은 [공개 표면장력 회귀의 제한식](https://basilisk.fr/src/tension.h)처럼 사용한 (\rho,\sigma,\Delta_{min})에서 계산한 제한을 로그에 기록하고 위 테스트에서 이를 위반하지 않아야 한다.

### 3. 1차 분열의 해석 기준: Rayleigh–Plateau

[Rayleigh의 원 논문](https://doi.org/10.1112/plms/s1-10.1.4)에 따라, 저밀도 비점성 주변 기체 속 반지름 (R)의 비점성 액주에서 축대칭 섭동 (a(t)e^{ikz})의 성장률은

\[
s^2=\frac{\sigma}{\rho_lR^3}
q(1-q^2)\frac{I_1(q)}{I_0(q)},\qquad q=kR.
\]

따라서 (0<q<1)은 불안정, (q=1)은 중립, (q>1)은 안정이다. 최대 성장 모드 (q=0.697)의 파장은 (\lambda/R=2\pi/q=9.014613\), 무차원 성장률은

\[
s\sqrt{\rho_lR^3/\sigma}=0.3433387.
\]

첫 CI 회귀는 한 파장 주기 축대칭 영역, (a_0/R=0.01), 반지름당 (N=16,32,64)에서 진폭이 작을 때까지만 실행한다. `log(a/a0)`의 기울기를 같은 무차원 시간 구간에서 구한다. 다음을 모두 만족해야 한다.

- (q=0.697)은 양의 성장, (q=1)은 0, (q=1.1)은 지수 성장을 보이지 않아 해석 안정성 경계를 재현한다.
- (q=0.697)의 성장률 절대오차는 세분할 때마다 감소한다.
- 세 해상도의 Richardson 외삽값과 최세망 값의 차이, 그리고 선형 회귀의 기울기 불확도를 합친 구간 안에 `0.3433387`이 들어간다. 이는 계산 자체가 정한 이산화·측정 불확도이며 임의의 백분율 허용치가 아니다.
- 이 구간의 액체 체적 수지 오차와 `alpha` 최솟값·최댓값을 함께 보고한다.

이 선형 회귀가 primary-breakup 구동 물리를 가장 작게 검증한다. [Popinet의 공개 Plateau–Rayleigh 문제](https://basilisk.fr/src/test/plateau.c)는 이어지는 비선형 목 가늘어짐에서 (r_{min}\propto(t_0-t)^{2/3}), (u_{max}\propto(t_0-t)^{-1/3}) 및 위상 분리를 확인한다. 이 문제의 18단계 적응 세분과 다중 decade fit은 긴 release 검증으로 두고 매 커밋에서 실행하지 않는다.

### 4. 분리 액적 생성과 진단 연결

CI용 위상변화 smoke test는 같은 한 파장 축대칭 문제를 반지름당 32셀에서 첫 단절 직후까지만 실행한다. solver가 쓴 **기하학적** `alpha`와 메시 연결을 `resolved_drops.py`에 전달한다.

- 단절 전에는 모든 선택 액체가 한 연결 성분이다.
- 단절 뒤에는 레이블 수가 증가하고, 주기 쌍을 넣거나 뺀 결과가 의도한 위상과 일치한다.
- 각 시각에 `sum(component.liquid_volume) + unlabeled_liquid_volume == input_liquid_volume`가 부동소수점 합산 오차 안에서 성립한다.
- 입구가 있는 유한 액주 회귀에서는 입구 접촉 성분과 분리 성분이 나뉘고, 분리 성분마다 (V,d_{eq},\boldsymbol{x},\overline{\boldsymbol{u}})가 유한하다.

이 smoke test는 분리가 실제 계산장에서 발생하고 진단이 이를 찾는다는 연결 증거다. 32셀/반지름에서 나온 액적 직경이나 satellite 수를 물리적으로 수렴한 분무 통계로 쓰지 않는다. 적어도 세 해상도의 직경·생성 시각·체적 수렴과 실제 물성/경계 검증이 끝난 뒤에만 정량 분열 예측을 주장한다.

## 완료 정의

“액주·액막 1차 분열과 액적 생성 구현”은 다음 항목이 모두 충족될 때만 쓴다.

1. HEM flash와 구별되는 기하학적 `alpha`가 보존적으로 수송되고 계면 재구성·곡률에 연결된다.
2. 표면장력 힘과 압력 결합이 정적 구, 모세관파, 액적 진동 게이트를 통과한다.
3. WALE를 켠 경우 SGS 응력이 실제 운동량 플럭스에 들어가며 CPU/GPU 식·보존 회귀를 통과한다. stress-v1에는 없는 SGS 열/종/TCI를 별도로 명시한다.
4. Rayleigh–Plateau 안정성 경계와 최대 성장률이 위 불확도 기반 기준을 통과한다.
5. 계산이 만든 단절을 연결 성분 진단이 입구 연결 액주/액막과 분리 액적으로 나누고 체적 수지를 닫는다.

이 완료 정의도 Lagrangian spray, 미해상 액적의 subgrid 생성, 2차 분열, 충돌·합체 또는 실제 R04 인젝터 검증을 포함하지 않는다.
