# 90도 충돌 N₂O 벤치마크 수용 기준

## 목적과 현재 범위

이 벤치마크는 같은 공급 상태의 N₂O 두 유동을 서로 직각인 경계에서 압력차로 출발시키는 작은 3차원 HEM 문제다. 첫 산출물은 케이스 생성, 메시·경계 확인, 그리고 각 외부압에서 256,000셀 GPU 실행이 한 단계를 보존적으로 완료하는지 확인하는 것이다. 이번 범위는 정상 충돌류, 액막 형성, 분열, 액적 생성 또는 실제 노즐 유량을 검증하는 production run이 아니다.

현재 `ReactiveFoam`은 공통 속도·온도의 equilibrium HEM flash와 압축성 보존 수송을 갖는다. 기하학적 VOF 이류·계면 재구성·곡률·표면장력이 없으므로 `alphaLiquid*`로 액주 표면, 충돌 액막 또는 액적을 직접 해상했다고 해석할 수 없다. WALE stress-v1을 선택하더라도 이 계면 물리의 부재는 바뀌지 않는다.

공급 상태와 두 EOS의 별도 계산은 [공급 상태와 flashing 범위](impinging-n2o-thermodynamics.md)에 기록한다. 그 0D 계산은 열역학 입력과 평형 한계를 확인하지만 공간 유동 수용을 대신하지 않는다.

## 고정 입력과 압력 정의

모든 압력은 solver 사전에 **절대압 Pa**로 쓴다. 사용자가 지정한 `55 bar(g)`의 기준은 `1.01325 bar(abs)`이므로 두 외부압 케이스 모두 공급압이 같다.

| 항목 | 값 |
|---|---:|
| 공급 온도 | `293.15 K` |
| 공급 게이지압 | `55 bar(g)` |
| 게이지 기준 절대압 | `1.01325 bar(abs)` |
| 공급 절대압 | `56.01325 bar(abs) = 5,601,325 Pa` |
| 저압 외부 | `1.01325 bar(abs) = 101,325 Pa` |
| 고압 외부 | `40 bar(abs) = 4,000,000 Pa` |
| 저압 케이스 공급-외부 차압 | `55 bar = 5,500,000 Pa` |
| 고압 케이스 공급-외부 차압 | `16.01325 bar = 1,601,325 Pa` |
| 공급/외부 압력비 | 각각 `55.28077967`, `1.40033125` |

40 bara 케이스에서 `55 bar(g)`를 chamber 기준 게이지압으로 다시 더하지 않는다. 그렇게 하면 사용자가 지정한 공급 상태와 다른 `95 bara`가 된다.

두 입구의 `fixedState`는 순수 N₂O, `T=293.15 K`, `p=5,601,325 Pa`, `U=(0 0 0)`인 동일한 평형 ghost state다. x=0 면의 외향 법선은 -x이고 y=0 면의 외향 법선은 -y이므로, 낮은 내부압에 대한 Riemann 해가 각각 +x와 +y 방향 유입을 만든다. `U=0`에 +x/+y 속도를 숨겨 넣지 않는다.

이 경계는 고정된 정적 상태와 내부 상태 사이의 HLL flux다. 다음 의미를 갖지 않는다.

- `56.01325 bara`를 전압력으로 놓고 정압·속도·질량유량을 함께 푸는 stagnation reservoir 경계
- 2×2 mm 유로 내부의 수축, 마찰, vena contracta, discharge coefficient 또는 choking 해
- 지정 질량유량 또는 지정 출구속도
- 비반사 pressure outlet

따라서 여기서 얻은 입구 flux는 이 축약형 fixed-ghost 문제의 결과다. 실제 N₂O 노즐 유량으로 보고하려면 저장조와 유로를 계산영역에 포함하거나, 별도 검증한 nozzle/choking 모델에서 출구 상태와 질량 flux를 받아야 한다.

외부압도 경계 종류와 분리해서 기록한다. 외부 셀을 101,325 Pa 또는 4,000,000 Pa 공기로 초기화한 뒤 외부 면을 `extrapolate`로 두면 그 압력은 **초기 조건**이지 계속 유지되는 ambient reservoir가 아니다. 외부 면을 `fixedState`로 두면 압력은 유지되지만 여전히 비반사 경계가 아니다. 생성 manifest는 모든 비입구 patch의 실제 종류를 명시해야 하며, 장시간 충돌 해석 전에 경계 반사·역류 민감도를 따로 확인해야 한다.

## 형상과 메시 수용

도메인과 노즐은 다음 SI 좌표를 정확히 사용한다.

- 도메인: `[0,20] × [0,20] × [0,10] mm`
- x 입구: `x=0`, `5<=y<=7 mm`, `4<=z<=6 mm`; 압력 구동 방향 +x
- y 입구: `y=0`, `5<=x<=7 mm`, `4<=z<=6 mm`; 압력 구동 방향 +y
- 각 입구 면적: `2×2 mm² = 4e-6 m²`
- 두 중심선: `(x,6,5) mm`와 `(6,y,5) mm`
- 기하학적 교점: `(6,6,5) mm`; 각 입구 중심에서 6 mm

기준 격자는 `h=0.25 mm`의 균일 정육면체다.

| 단계 | h | `(Nx, Ny, Nz)` | 셀 수 | 입구당 face 수 | 용도 |
|---|---:|---:|---:|---:|---|
| 기준 | `0.25 mm` | `(80,80,40)` | `256,000` | `8×8=64` | 생성 및 각 압력의 1-step GPU smoke |
| 세분 후보 1 | `0.2 mm` | `(100,100,50)` | `500,000` | `10×10=100` | 이후 공간 민감도 |
| 세분 후보 2 | 정확히 `1/6 mm` | `(120,120,60)` | `864,000` | `12×12=144` | 이후 공간 민감도 |

세 격자 모두 사용자 범위 `100,000~1,000,000`셀 안에 있고 도메인·노즐 모서리를 정확히 맞춘다. 마지막 격자는 누적 반올림으로 끝점이 어긋나지 않도록 `20/120 mm`로 생성한다. 1,024,000셀 같은 100만 초과 후보는 사용하지 않는다.

생성 검사는 적어도 다음을 기계적으로 확인해야 한다.

1. 셀 수, 축별 분할 수, 도메인 부피 `4e-6 m³`, 양의 셀 체적과 정육면체 직교성을 확인한다.
2. 두 입구 patch의 면적 합이 각각 `4e-6 m²`, face 수가 위 표와 같고, 면 중심 범위가 지정 사각형 안에 있는지 확인한다.
3. x 입구의 모든 외향 단위 법선은 `(-1,0,0)`, y 입구는 `(0,-1,0)`이고, 같은 face가 두 patch 또는 벽 patch에 중복되지 않아야 한다.
4. 나머지 경계 face의 합과 전체 외부 face 수가 맞고 patch 이름·경계 종류가 manifest와 `reactiveProperties`에서 일치해야 한다.
5. 초기 내부장은 전 셀 `293.15 K` 공기이며 케이스별 외부 절대압을 사용한다. 초기 N₂O 보존 질량은 0이어야 한다.

교점 `(6,6,5) mm`는 이 격자들의 grid plane 위에 있으므로 “교점 셀” 하나를 고르면 방향에 따라 다른 셀이 선택된다. 충돌부 통계는 노즐 폭 `D=2 mm`로 정한 대칭 control volume

\[
5\le x\le7,\quad5\le y\le7,\quad4\le z\le6\ \mathrm{mm}
\]

에서 적분한다. 이 영역은 세 격자 모두 정확히 맞는다.

## 시간 간격과 실행 단계

### CFL 목표

시간 간격은 공급 차압에서 추정한 임의 속도로 고정하지 않는다. `ReactiveFoam`이 각 셀에서 사용하는 음향·대류 제한을 따른다. 점성/확산을 끈 정육면체에서 그 규모는

\[
\Delta t_c \le C_{CFL}\frac{V_c}
{\sum_{f\in c}A_f\left(|\boldsymbol{u}_f\cdot\boldsymbol{n}_f|+
1.1c_{frozen,f}\right)},
\]

이며 실제 코드는 `0.95*C_CFL`, 모든 face, 경계 ghost 상태, `maxDeltaT`, 남은 종료시간의 최솟값을 쓴다. 초기·최종 상태의 `soundFrozen`, 실제 `dt`, `maxCo`, `maxDeltaT`, retry 횟수를 결과에 보존한다. 허용되는 제어 범위는 코드 계약인 `0<maxCo<=0.5`이고, 첫 기준은 `maxCo=0.25` 이하로 둔다. `dt`가 작다는 사실만으로 flashing이나 충돌이 정확하다고 판정하지 않는다.

### 이번 실행: 256k 한 단계씩

두 외부압에 대해 각각 기준 256,000셀 케이스를 GPU HEM으로 **정확히 한 accepted step** 실행한다. 이 단계의 수용 목적은 다음뿐이다.

- 256k 메시·경계와 GPU transport/HEM closure가 실제 연결된다.
- 초기 checkpoint와 한 단계 뒤 checkpoint가 완전하게 기록된다.
- step이 retry 없이 끝나고 모든 보존량·열역학 상태가 유한하다.
- 로그가 요청한 두 절대압, `phaseChange=equilibrium`, 실제 backend와 accepted-step cap을 식별한다.
- solver 내장 수지가 통과한다. 현재 계약은 경계 보정 상대 질량 `<1e-7`, 총에너지 `<1e-9`, 운동량 `<1e-9`, 비반응 species `<1e-9`다.

한 CFL step의 정보는 압력파와 경계 인접 첫 셀의 갱신이다. 입구 중심에서 교점까지 6 mm이므로 이 결과를 “두 제트가 충돌했다”, “flashing plume이 형성됐다” 또는 “40 bar에서 충돌이 약해졌다”는 증거로 쓰지 않는다. 4,000셀 축소 smoke는 이번 수용 자료에 포함하지 않는다.

### 이후의 충돌 실행

실제 충돌 수용은 step 수를 미리 고정하지 않고 다음 사건으로 종료시간을 정한다.

1. x와 y 입구의 N₂O-rich front가 각각 중심선 6 mm를 이동한다.
2. 두 front가 위 control volume에 동시에 들어온다.
3. 교점 전후의 운동량 재배향과 압력 상승이 여러 출력 시각에서 관찰된다.
4. 이 사건 뒤 경계 반사가 control volume에 돌아오기 전 분석 구간을 확보한다.

`fixedState U=0`은 제트 속도를 주지 않으므로 `6 mm/U_in`을 사전에 확정할 수 없다. 실행에서 측정한 입구 질량가중 속도와 front 도달시각을 함께 보고한다. 시간 민감도는 같은 격자에서 `maxCo`를 절반으로 줄여 확인하고, 공간 민감도는 256k/500k/864k 순서로 확인한다. 세분 오차가 감소하지 않으면 충돌 위치·최대압력·기상량에 대한 수렴 주장을 보류한다.

## 필수 측정량과 보존 수지

### 경계와 control volume

두 입구와 나머지 외부 patch를 분리해 다음 signed flux를 시간 적분한다.

- 총질량과 N₂O species 질량
- x/y/z 운동량
- 총에너지

solver의 부호로 영역 보존량 `Q`와 누적 외향 경계 flux `B`는

\[
Q(t)-Q(0)+B(t)=0
\]

이어야 한다. 유입은 `B<0`이다. 현재 전역 수지는 모든 patch의 합을 검사하므로, 두 노즐의 대칭성과 외부 역류를 평가하려면 patch별 flux 출력이 별도로 필요하다.

전체 영역과 충돌 control volume에서 다음을 기록한다.

- `M_total`, `M_N2O`, 공기 species 질량, 총에너지, 운동량 벡터
- 입구별 `mdot_N2O`, 운동량 flux와 에너지 flux
- 중심선의 N₂O 질량분율, 속도, 압력, 온도와 equilibrium/frozen 음속
- control volume의 N₂O 질량, 평균·최대압력, 평균 운동량과 각 방향 운동량 flux
- x↔y 교환 후의 질량분율·압력·속도 성분 대칭 오차
- `min/max(p,T,rho)`, Mach, recovery residual, retry와 fallback 수

40 bara 케이스는 초기 공기 질량과 내부에너지가 1 atm 케이스보다 크므로 원시 영역 총량을 직접 비교하지 않는다. 초기값을 뺀 변화량과 경계 유입량, N₂O 질량으로 정규화한 양을 함께 사용한다.

N₂O front는 HEM 혼합 셀에서 날카로운 계면이 아니다. 하나의 임계값으로 도달시각을 확정하지 말고 N₂O 질량분율 여러 등치면과 control-volume 적분을 함께 보고한다. 이는 종 수송 front이며 VOF 액주 표면이 아니다.

### 기화 질량과 체적분율은 다른 양이다

셀 `i`의 보존된 총 N₂O 질량밀도를 `q_N2O,i`, HEM 복원이 정한 N₂O 액상 질량밀도를 `m^l_N2O,i`, 셀 체적을 `V_i`라 하면

\[
M_{N2O}=\sum_iq_{N2O,i}V_i,\quad
M_l=\sum_im^l_{N2O,i}V_i,\quad
M_v=M_{N2O}-M_l.
\]

`M_v/M_N2O`가 vapor **mass quality**다. 반면

\[
V_l=\sum_i\alpha_{l,N2O,i}V_i,
\qquad V_g=\sum_i\alpha_{g,i}V_i
\]

는 상이 차지하는 체적이다. 밀도비 때문에 vapor mass quality와 gas volume fraction은 크게 다를 수 있다.

특히 이 문제의 ambient air 셀은 N₂O가 없어도 `alphaGas≈1`이다. 따라서 `sum(alphaGas*V)` 또는 `alphaGas` 증가를 “기화한 N₂O 질량”으로 사용할 수 없다. `alphaLiquidN2O`도 HEM의 열역학적 셀 평균 상 체적분율이지 기하학적 액주 `alpha`가 아니다.

일반 field 출력은 equilibrium 모드의 per-cell N₂O 액상 질량밀도를 쓰지 않는다. `rhoLiquid*` field는 frozen 모드의 보존 재고다. 그러나 schema-3 `reactiveState.bin`은 conserved `q` 뒤에 각 셀의 전체 `PintleThermoState`를 저장하며, 그 구조체에 `liquidMass[0]`가 들어 있다. 따라서 이 케이스의 N₂O가 liquid index 0임을 thermo identity에서 확인한 뒤

\[
m^v_{N2O,i}=q_{N2O,i}-\mathrm{state}_i.\mathrm{liquidMass}[0]
\]

로 기상 N₂O 질량밀도를 직접 얻을 수 있다. 새 field를 만들거나 checkpoint를 다시 flash할 필요가 없다.

postprocessor는 임의의 native 구조체 파일처럼 읽지 않는다. `reactiveCheckpointComplete` manifest의 `reactiveState.bin` SHA-256을 먼저 검증하고, schema-3 magic, endian marker, FP64 크기, 셀 수, conserved-variable 수, `sizeof(PintleThermoState)`, 예상 파일 끝을 확인해야 한다. 같은 checkpoint의 physical-model identity와 종·액체 순서도 고정한다. 각 셀에서 `liquidMass[0]`가 backend 허용오차 안에서 `0`과 `q_N2O` 사이이고 모든 값이 유한한 경우에만 `M_l`, `M_v`를 집계한다. 이 검증에 실패하면 `alphaGas`로 대체하지 않고 분석을 실패시킨다.

checkpoint 분석이 가능해도 “현재 영역에 존재하는 증기 질량”과 “누적 기화량”은 다르다. 후자는 증기 N₂O의 입·출구 flux 또는 상간 질량전달 이력을 함께 알아야 한다. HEM은 매 복원에서 순간 평형을 강제하므로 유한속도 nucleation/cavitation 시간을 측정한 값도 아니다.

## 0D flash, 짧은 실행, 물리 검증의 분리

| 증거 | 확인하는 것 | 확인하지 못하는 것 |
|---|---|---|
| 공급점 `(293.15 K, 56.01325 bara)` 0D 복원 | fixed ghost가 허용되는 압축 액체 평형 상태인지, EOS branch와 보존 상태가 유한한지 | 노즐 질량유량, choking, 출구속도 |
| 공급 엔탈피에서 40/1.01325 bara 0D 평형 flash | HEM이 허용하는 등엔탈피 평형 quality의 한계, PR과 참조 EOS 차이 | 유동 셀의 실제 상분배, nucleation delay, 충돌 |
| 256k 1-step GPU 실행 두 건 | 생성된 3D 메시와 실제 GPU HEM 수송·경계·보존의 연결 | front 도달, 충돌, steady jet, 물리적 flashing 양 |
| 사건까지 이어진 256k/500k/864k 실행 | 이 수치모델 안의 front 도달, 충돌부 운동량과 격자 민감도 | 실험 정확도, 액막·액적 예측 |
| 외부 물성·노즐/충돌 실험 대조 | 모델의 물리 예측 범위 | 기하학적 분열 검증은 VOF·표면장력 추가 전 불가 |

같은 PR/HEM backend로 계산한 0D 값과 flow 값의 일치는 내부 일관성 검사다. 독립 물성 검증은 Helmholtz EOS/실험 자료와의 오차를 별도로 유지한다. 0D 등엔탈피 quality를 각 공간 셀의 정답으로 강제하지 않는다. 유동에서는 총엔탈피가 정적 엔탈피와 운동에너지로 나뉘고 경계 수치 flux, 압력파 및 혼합이 함께 작용한다.

## 단계별 완료 정의

### A. 생성 완료

- 두 케이스가 정확한 절대압·온도·조성으로 생성된다.
- 256k 메시의 좌표, 셀 수, patch face 수·면적·법선 검사가 모두 통과한다.
- 요청 상태와 실제 backend가 회복한 경계 `p,T,alphaGas,liquidMass`를 manifest에 기록한다.
- case 입력, thermo 구성, solver와 두 shared library의 SHA-256을 보존한다.

### B. 현재 요청의 1-step GPU 완료

- 1 atm 및 40 bara 케이스가 각각 256k에서 정확히 한 accepted step을 완료한다.
- retry 0, finite state, 완전한 checkpoint, solver 내장 질량·에너지·운동량·species 수지를 만족한다.
- 실제 `dt`와 상태 범위, GPU HEM profile을 결과에 남긴다.
- 결과 설명에 `setup/runtime smoke only; no collision or flow validation`을 포함한다.

### C. 유동 충돌 수용

- 두 front의 독립 도달과 control volume 내 동시 존재를 보인다.
- patch별 질량·운동량·에너지 수지가 닫힌다.
- x/y 대칭 오차와 시간·공간 세분 민감도를 보고하고, 256k/500k/864k에서 관심량 오차가 감소한다.
- 외부 경계 반사 이전 구간과 이후 구간을 분리한다.

### D. flashing 수용

- 총/액상/기상 N₂O 질량을 위 정의대로 분리한다.
- 0D 평형 한계, 독립 EOS, 격자·시간 민감도와 비교한다.
- instantaneous HEM, 공통 속도, metastability·nucleation 부재를 결과 한계에 남긴다.

### E. 액막·분열·액적 수용

현재 모델로는 이 단계에 도달할 수 없다. 기하학적 액체 VOF, 유계 이류와 재구성, 곡률·표면장력 및 해당 물리 게이트를 구현한 뒤 별도의 수용 문서를 적용해야 한다. HEM `alphaLiquid` 등치면이나 N₂O 종 front를 액막·액적이라고 이름 붙이지 않는다.
