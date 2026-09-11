# N₂O/IPA 상변화·반응 열유동 연구 솔버

`pintleReactiveFoam`은 N₂O/IPA의 액체–증기 상분배, 반응 종 수송, 압축성 총에너지 방정식을 함께 계산하는 **직렬 HEM 기준 솔버**다. 기존 `spumaPintleColdFoam`과 별도 실행 경로이며 SPUMA 메시·입출력을 사용한다. 기체 반응과 액체 상평형의 보존적 결합을 구현했지만, **고압 액체 주입부터 연소까지의 예측 솔버가 완성·검증된 상태는 아니다.** 물질 접촉면 압력 시험도 현재 실패한다.

검증 수치와 미해결 항목은 [개발·검증 보고서](reports/reactive-phase-development-20260911.md), 재현 가능한 검사 코드는 `tools/validate_reactive_thermo.py`, `tools/validate_reactive_runtime.py`, `tools/run_reactive_campaign.py`에 있다. 모든 구현·실행 검증은 MAIN이 수행했다.

## 계산 모델

공통 압력·온도·속도의 균질 평형 모델(HEM)을 사용한다. N₂O와 IPA는 각각 순수 액상이며 기체 혼합물과 공존할 수 있다. `q0 … q(N−1)`는 **각 화학종의 액상과 기상 질량을 합친 셀 체적당 질량**이다. 기상 질량분율이나 체적분율이 아니다. `rhoMomentum`와 `rhoTotalEnergy`는 운동량 및 생성·열·운동에너지를 포함하는 총에너지의 체적 밀도다.

- 매 단계 보존량으로부터 일정 부피·내부에너지(UV) flash를 수행한다. 공존상은 화학퍼텐셜을 일치시키고, 없는 상의 안정성 조건을 검사한다. 여러 초기 추정에서 찾은 허용 해 중 엔트로피가 가장 큰 해를 선택한다. 전역 최대해를 보증하는 알고리즘은 아니다.
- 잠열과 반응열은 같은 열역학 에너지 기준에 들어 있다. 별도 `Gamma*L`나 `Qdot`를 총에너지에 다시 더하지 않는다.
- 전체 반응기구의 강직 적분은 CVODE BDF를 사용한다. 각 RHS에서 flash를 풀고 기상 체적분율을 곱한 반응률을 적용한다. 화학종·원소·총질량 검사를 통과한 상태만 채택한다.
- 대류는 모든 보존량에 같은 HLL 면 유속, 공간 1차, SSPRK2 시간 적분을 사용한다. 화학은 Strang 분할이다. 고정 조성·상분배 음속으로 파속을 제한하고 평형 음속을 별도로 계산한다. 상태 또는 CFL 검사가 실패하면 전체 단계를 복원하고 시간 간격을 줄인다.
- 상수 Newtonian 점성·점성 일·Fourier 열전도, 이상기체에서 공통 계수 Fick 종 확산과 종 엔탈피 수송을 제공한다. 종 확산은 기상에만 작용하며 총 확산 질량 유속은 0이다. 수송계수는 사용자가 지정하며, 현재 솔버가 Cantera 수송계수를 자동 적용하지는 않는다.

## 제공되는 세 가지 구성

| 파일 | 모델 | 사용 범위와 중요한 제한 |
|---|---|---|
| `cold-pr-config.yaml` | 4종 PR 기체 + 순수 액체 N₂O/IPA, 반응 없음 | 차가운 고압 상평형·감압 연구. 축소 연소기구가 아님 |
| `chemistry-config.yaml` | 413종·14,922반응 이상기체, 액상 없음 | 기상 화학·충격파 수치 비교. 고압 반응의 실험 정확도 보증 없음 |
| `reactive-dilute-config.yaml` | 전체 이상기체 화학 + PR 순수 액체 | 압력 상한 2 bar의 결합 구현 시험. 고압 N₂O 분사에 적용 불가 |

이 세 구성은 **하나의 계산 도중 자동 전환되지 않는다.** 기체 EOS가 바뀌면 포화 조건도 달라진다. 반응 중간체에 임계 물성을 임의 배정해 전체 PR 반응기구를 만들지 않았다. 전체 고압 반응 모델의 공통 에너지·fugacity·속도식 정합성은 후속 연구가 필요하다.

설정의 압력·온도 범위는 수치 도메인 제한이며 실험 검증 범위와 다르다. 저온 확장한 표준 열역학 계수 및 출처는 생성 manifest에 기록된다. 300 K 미만의 반응 중간체 열역학과 저온 반응률은 검증되지 않았다.

## 환경과 빌드

기존 SPUMA 환경과 별도로 Cantera 3.2, Sundials, CoolProp 환경을 만든다. 기존 DeepFlame 환경은 사용하지 않는다. 아래 명령은 저장소 루트에서 실행한다. `env.sh`의 `PINTLE_SPUMA_ENV`가 해당 시스템의 SPUMA 환경을 가리켜야 한다.

```bash
micromamba create -y -p "$PWD/research/reactive-env" -c conda-forge \
  --strict-channel-priority python=3.12 cantera=3.2 libcantera-devel=3.2 \
  coolprop=6.7 scipy numpy pyyaml
flock /home/jsw/cae-benchmark/run.lock bash tools/build_reactive_solver.sh
```

실제 검사에 사용한 패키지의 고정 목록은 `tools/reactive-env-linux64.lock`이다. `micromamba create -p <새 경로> --file tools/reactive-env-linux64.lock`으로 복원할 수 있으나, 이 목록은 **linux-64/x86-64-v4 CPU용**이다. 다른 CPU·SPUMA 빌드의 호환성은 별도로 확인해야 한다. 백엔드는 호스트 C++ 라이브러리이며 GPU 가속 화학 계산이나 MPI 성능을 주장하지 않는다.

## 반응 자료 준비

원본 반응기구는 배포 소스에 포함하지 않는다. 고정된 공식 CRECK revision에서 내려받아 출처와 SHA-256을 저장한다. 원본 transport 자료에 중복이 있어 변환 정책을 명시적으로 지정한다. 첫 항목 선택은 transport 레코드에만 적용하며, 화학종·반응·속도상수는 삭제하지 않는다.

```bash
research/reactive-env/bin/python tools/fetch_reactive_mechanism.py \
  --revision 640d5b492d849e33c08bfbbd9d3de25dd074c9a4 \
  --download-only --output research/creck-source
research/reactive-env/bin/python tools/convert_reactive_mechanism.py \
  --source research/creck-source --output research/creck-converted \
  --transport-policy first-occurrence
research/reactive-env/bin/python tools/prepare_reactive_thermo.py \
  --mechanism research/creck-converted/mechanism.yaml --output research/reactive-thermo
```

생성된 구성은 기구 파일의 **절대 경로**를 사용한다. 디렉터리를 옮기거나 파일을 재생성하면 보존장 재시작 식별값이 달라질 수 있다. 외부 YAML 종·반응 import 및 PR 임계물성의 암묵적 데이터베이스 조회는 차단한다. 자체 완결된 기구와 모든 EOS 계수를 제공해야 한다.

## 실행과 재시작

먼저 실제 주입기와 분리된 검증 케이스를 사용한다. 생성기는 기존 경로를 덮어쓰지 않는다.

```bash
flock /home/jsw/cae-benchmark/run.lock research/reactive-env/bin/python \
  tools/prepare_reactive_case.py cases/reactive-example \
  --thermo-dir research/reactive-thermo --kind acoustic --cells 32 --mach 2
source env.sh
flock /home/jsw/cae-benchmark/run.lock pintleReactiveFoam -case cases/reactive-example
```

`constant/reactiveProperties`의 `thermoConfiguration`과 `initialization conserved`를 사용한다. 시간 디렉터리에 전체 `q*`, `rhoMomentum`, `rhoTotalEnergy`, 추정치 `p/T`, 출력용 `U`, 그리고 `reactiveStateIdentity`가 필요하다. 재시작은 이 보존장을 읽어 상분배를 다시 계산한다. `p/T`로 총에너지를 재생성하지 않는다. 출력은 시작 시간에도 생성하므로 원본에서 복사한 별도 케이스로 작업한다.

원시량 초기화는 `initialization primitive`와 시간 0에서만 지원한다. `Y0 … Y(N−1)`는 전체 화학종 질량분율, `liquidFraction0/1`은 각 응축성 종의 질량 중 액상에 배정한 비율이다. 백엔드의 초기 flash가 원시 입력의 압력·온도를 바꿀 수 있다. 생성기처럼 먼저 평형 상태를 구해 보존량으로 저장하는 경로를 권장한다.

경계 유속은 `reactiveProperties/boundaryConditions`에 직접 지정한다. 지원 항목은 `slipWall`(단열), `extrapolate`, `fixedState`(평형인 정적 `p,T,U,Y,liquidFractions`), 평행 이동 cyclic이다. `fixedState`는 고정 reservoir를 사용하는 Riemann 경계이며 전압력·전온도 경계나 비반사 출구가 아니다. 출력 필드의 `calculated` 패치 값은 물리적 경계 상태를 표시하지 않는다.

고정 메시·직렬·FP64만 지원한다. 수송은 직교 메시, 점성 구배는 면 중심 skew가 없는 메시가 필요하다. 회전 cyclic, AMI, MPI, 동적 메시, function objects, 실행 중 사전 변경 및 다른 `stopAt` 설정은 거부한다. `maxDeltaT>0`, `0<maxCo<=0.5`, 종료시간이 시작시간보다 커야 한다. `maxHostMemoryGB`는 배열 크기 추정 한도이며 실제 프로세스 메모리 사용량 측정값이 아니다.

## 검사 실행

```bash
flock /home/jsw/cae-benchmark/run.lock research/reactive-env/bin/python \
  tools/validate_reactive_thermo.py --thermo-dir research/reactive-thermo \
  --output results/my-thermo-check.json
research/reactive-env/bin/python tools/validate_reactive_runtime.py \
  --thermo-dir research/reactive-thermo --output cases/my-runtime-check
research/reactive-env/bin/python tools/run_reactive_campaign.py \
  --thermo-dir research/reactive-thermo --output cases/my-reactive-campaign
```

마지막 두 도구는 내부에서 실행 잠금을 획득하므로 외부 `flock`으로 다시 감싸지 않는다. 기본 캠페인은 알려진 물질 접촉면 정확도 실패를 포함하며 정상적으로 **종료 코드 1**을 반환할 수 있다. `campaign.json`에서 실패 항목을 확인한다. 점성·전도·확산 검사의 큰 상수 계수는 연산자 검증용이며 실제 유체의 물성값이 아니다.

실제 분사·연소계 적용 전에 고압 반응 EOS, 접촉면 압력 보존, 유한속도 상변화·상간 열교환, 액적 슬립·분열·합체, 액체 혼합·용해, 표면장력, 난류·복사·벽 열전달 및 실험 자료에 대한 검증을 추가해야 한다. 세부 우선순위와 초음속 적용 판정은 보고서에 있다.
