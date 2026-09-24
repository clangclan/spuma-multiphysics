# Cartesian implicit 계면의 실제 GPU 솔버 통합

2026-09-23. Draft PR #6의 후속 변경이다. **실제 GPU 솔버 통합은 구현했지만 정적 액적의 엄격한 세 메시 감소 기준은 아직 통과하지 못했다.** 기본 diffuse 모델은 유지한다. 메인 에이전트가 구현과 테스트 코드를 작성했고, GPT-6 Sol xHigh 서브 에이전트는 읽기 전용 검토와 지정된 검증 커맨드를 수행했다.

## 이산화와 열역학

셀의 액체 체적분율만으로 하나의 연속 C² tensor cubic B-spline 계면을 복원한다. 구의 중심·반경·해석 곡률은 solver 입력에 없다. 같은 계면에서 셀 체적, 면적, 곡률 적분, 공유 면의 액체 점유 면적과 conormal 선적분을 계산한다. reduced-pressure HLLC와 기하 면 운동량·일 항은 기존 진단용 경로를 재사용하며 실제 RK 수송에 연결했다.

체적 역문제는 GPU Gauss–Newton/PCG로 풀고, 전체 3차 혼합 차분을 정규화한다. 먼저 이 차분의 영공간인 10개 일반 이차 다항식에서 같은 체적 역문제를 푼다. 이는 구의 중심·반경을 맞추는 방식이 아니며 타원체·평면 등의 일반적인 이차 계면을 포함한다. 모든 순수/혼합 셀의 hard volume tolerance를 만족하는 후보만 채택하고, 만족하지 못하면 전체 spline 자유도로 넘어간다. 이차 계면도 처음 허용오차에 들어온 순간에 멈추지 않고 체적 잔차를 더 줄이며 최종 후보를 재인증한다. 전체 역문제에서는 이 10개 모드를 coarse preconditioner로 재사용한다.

전체 spline 복원의 근접 단계에서는 정규화 항을 줄이고 체적 오차도 감소하도록 요구한다. 체적 잔차가 정체되면 정규화 강도를 낮춰 같은 penalized 해 주변의 반복을 피한다. PCG가 반복 상한에 도달한 상태는 선형 수렴으로 기록하지 않는다. 작은 10×10 coarse Cholesky와 스칼라 제어는 호스트에서 수행하며 형상 적분·Jacobian·벡터 연산·축약은 GPU에서 수행한다.

초기 p와 T가 주어지면 GPU에서 기상 압력 `pg=p-cJ`, 액상 압력 `pl=p+(1-c)J`, `J=σ κ`에 맞는 밀도·벌크 에너지를 생성한다. 종 질량분율과 액상 질량분율을 유지하고 c/J는 계면과 반복 정합한다. 이전 common-pressure 초기 상태를 curved UV로 바로 회복할 때 생긴 최대 21.56 Pa 변화는 이 초기화 시험에서 4.66e-10 Pa로 줄었다. 실패 셀이 하나라도 있으면 초기화 배치의 질량·에너지·상태는 모두 커밋하지 않는다.

## 재사용·실패·재시작

이미 적분한 계수장이 새 체적분율을 논리 좌표와 물리 좌표 양쪽의 허용오차 안에서 만족하면 계면을 재사용한다. 이는 새로운 물리 근사를 넣는 것이 아니라 동일 계수장에 대한 반복 적분과 상수 배율 재정규화를 생략한다. RK 시작 시 계수를 보관하고 실패 시 복원한다. 체크포인트의 `implicitSurface.bin`에는 계수와 Cartesian 메시 정보가 포함되며 기존 원자적 체크포인트 manifest로 검증한다. 복원 계면의 체적 제약을 다시 확인하며 이 모델은 동일 수치 정책으로만 재시작한다.

주기 경계는 면적·conormal 모멘트뿐 아니라 bicubic 값·1/2차 미분 trace도 검사한다. 일반적인 비주기 계수장에서 임의의 주기 계면을 자동 생성하지는 않는다. 고정 유입 저장소는 순수 상만 허용하며 ghost 총에너지를 bulk+kinetic으로 해석한다. 혼합상의 curved ghost 경계는 지원하지 않는다.

## 설정

```foam
transportBackend cuda;
closureBackend cuda;
closureCpuFallback false;
thermoExactReuse false;
capillaryGeometry cartesianImplicit;
capillaryQuadratureTolerance 1e-9;
capillaryVolumeTolerance 1e-9;
capillaryFitIterations 40;
capillaryFitLinearIterations 1600;
physics { surfaceTension true; }
```

기본 `capillaryGeometry diffuse`는 유지된다. `physics.surfaceTension false`일 때 남아 있는 implicit 옵션과 초기 color 필드는 비활성화되며 Cartesian/implicit workspace는 생성하지 않는다. 별도로 geometry 모드를 바꾸거나 초기 color 파일을 준비할 필요가 없다. 초기 sharp volume seed가 있으면 `capillaryInitialColorField interfaceColor`를 지정할 수 있다. 이 필드는 초기 기하 추정이며, 보존 질량분율과 EOS에 맞지 않는 color를 강제로 유지하지 않는다.

## 검증 범위와 제한

실제 solver의 독립적인 비정렬 구면 fixture는 R=0.74 mm, 4 mm 정육면체, 중심 (2.071, 1.947, 2.037) mm, σ=0.01 N/m, 기상 압력 3 MPa, 270 K다. 해석 구면은 초기 체적분율과 기준 압력차를 만드는 데만 사용한다. 같은 60 ns에서 서로 다른 메시를 비교한다. 이 시험은 동결 상·비점성 정적 평형이며 난류·flashing·1차 분열 정확도 시험을 대신하지 않는다.

초기 전역 복원은 아직 비싸다. 초기 n16 실패 실험은 반복 적분 때문에 240초 제한에서 중단됐으며, 그 원인을 고친 뒤 새 사례로 검증했다. 100k–1M 메시 성능을 측정하거나 권장하는 결과가 아니다. 체적 제약 또는 적분·경계 검증에 실패하면 계산을 중단하며 CPU로 대체하거나 표면장력을 몰래 끄지 않는다.

계면 질량과 총에너지는 HLLC finite-volume 플럭스로 수송한다. swept PLIC, 재메시, MPI, 파단·합체 정확도, 긴 모세관파/액적 진동, 계면이 전역 영역에 균일하게 퍼진 HEM 혼합물은 이번 검증 범위가 아니다. 기존 diffuse flashing/WALE 경로와 새 implicit 정적 평형 검증을 구분해야 한다.

## 일반 계면 복원 검증

회전 타원체(n=12)는 이차 영공간에서 체적 오차 최대 3.33e-16으로 수렴했고, 곡률은 구처럼 일정하게 강제되지 않았다. 3차 항으로 변형한 닫힌 계면(n=12)은 이차 후보가 거부되고 전체 spline 복원으로 수렴했다. 후자의 최대 체적분율 오차는 5.03e-8, 요구 허용오차는 1e-7이며 순수 셀 누출과 미해결 적분 셀은 0이었다. 전체 Gauss–Newton 17회, PCG 27,200회, GPU probe 시간은 78.16초였다. 이 두 시험의 정방향 fixture는 같은 sharp 적분기를 사용했으므로 구현 일관성 검증이며 독립 적분 정확도 증거로 간주하지 않는다.

## 실제 GPU 솔버의 최종 결과

같은 물리 조건, 같은 60 ns 종료시각의 결과다. 해석 곡률을 runtime에 주입하지 않았고, UV와 수송의 CPU fallback 및 GPU 실패는 모두 0이었다. 각 메시의 4·5·7 accepted step은 재시도 없이 완료됐다.

| 메시 | 최대 유속 (m/s) | 체적 가중 L2 유속 (m/s) | 면적 가중 곡률 L2 오차 (1/m) |
|---|---:|---:|---:|
| 16³ | 7.81534e-14 | 9.54202e-15 | 6.08927e-9 |
| 24³ | 2.64055e-13 | 3.24353e-14 | 1.91525e-9 |
| 32³ | 4.55540e-13 | 3.79047e-14 | 7.19187e-9 |

세 메시의 Laplace 압력차 절대 오차는 모두 2.76881e-10 Pa다. 3 MPa의 FP64 간격 4.65661e-10 Pa보다 작으므로 이 압력차 측정의 분해능 이내다. 이를 근거로 유속·곡률의 엄격한 공간 수렴까지 통과했다고 해석하지 않는다. 최대 유속과 L2 유속이 메시 세분화에 따라 증가해 **기존 `staticVelocityGatePassed=false`, CLI exit 2**이며 기준을 완화하지 않았다. 실제 solver의 정상 종료와 물리 정확도 gate는 별개다. [원시 결과](../results/implicit-static-drop-20260923/actual-solver-series.json).

측정한 총에너지·종 질량·액체 질량 변화는 0, 전역 운동량 변화는 최대 5.23e-23 kg·m/s다. 곡률은 초기화 이후 체적 제약을 만족하는 같은 계면에서 유지됐다. 최대 accepted time step은 각 18.75·12.50·9.38 ns다. 30 ns 상한보다 짧은 이유는 이번 4 mm 작은 영역과 음향 CFL이며, 사용자 분사 벤치마크의 메시 문제나 30–50 ns 가능 여부를 판정한 결과는 아니다.

## 압력 합산과 재시작 검증

작은 모세관 힘을 MPa 압력과 직접 더한 뒤 여러 면을 합산하면 유효 자릿수를 잃는다. 최종 운동량 RHS는 셀 압력 `p_c`를 먼저 뺀 면 유량을 합산하고, `p_c * sum(sign*A*n)/V`를 별도로 복원한다. 면적벡터 합에는 보상 합산을 적용한다. 따라서 입력으로 저장된 면적·법선이 미세하게 비폐합이어도 기존 공유 면 보존식과 실수 산술에서 동일하다. 이 항을 0으로 강제하거나 입력 메시를 묵시적으로 정규화하지 않는다. 경계 유량은 원래 전체 압력을 사용한다.

CPU와 CUDA 공개 ABI 시험에서 정확히 폐합된 평면 계면은 큰 균일 압력 offset을 추가해도 RHS가 비트 단위로 동일했다. 면 하나의 면적에 상대 1e-9 오차를 주면 해당 셀의 압력 힘 2 N/m³가 남고 이웃 셀과 상쇄되어 전역 운동량 변화율은 1.52e-40이었다. 계수 조회 오류의 resident 상태 보존, 재구성 후 rollback 계수·기하 일치, 복원 입력 버퍼 즉시 소멸도 확인했다. 복원·rollback GPU 복사를 동기화하고 상위 solver에서 rollback 오류를 검사한다. [CPU](../results/implicit-static-drop-20260923/transport-cpu.log), [CUDA](../results/implicit-static-drop-20260923/transport-cuda.log).

16³ 실제 solver의 첫 체크포인트에서 이어 계산한 60 ns 결과는 연속 계산과 보존장, 기하 필드 및 `implicitSurface.bin` 모두 비트 단위로 일치했다. [재시작 결과](../results/implicit-static-drop-20260923/restart-validation.json). 실제 solver의 의도적인 재구성 실패→여러 번 재시도→실패 checkpoint 경로 전체를 통과시킨 시험은 이번 자료에 없다.

## 회귀와 기하 연산자 검증

- 기존 35개와 implicit OFF 검사 6개를 포함한 [물리 선택 41개 검사](../results/implicit-static-drop-20260923/physics-switches.json)가 통과했다. 표면장력 OFF에서는 implicit workspace가 없다.
- 기존 diffuse 4096셀의 점성·열전도·WALE 열/종 혼합·표면장력·flashing 계산을 저장된 t=0 상태에서 60 ns까지 이어 실행했다. 보존장 시계열은 이전 기준과 비트 단위로 일치했다. GPU PR 물성 86,016셀, 실패 0, 내부 CPU 엔탈피 준비 0, GPU flash CPU fallback 0이다. 이 결과는 **기존 diffuse 모델의 회귀**이며 새 implicit 모델의 flashing·분열 정확도 검증은 아니다. [결과](../results/implicit-static-drop-20260923/diffuse-full-physics-regression.json).
- [Sharp 적분/Jacobian](../results/implicit-static-drop-20260923/sharp-volume-validation.json): 비해석 seed 379셀, 유한차분 방향 30개, 최대 Jacobian 오차 1.13e-9, CPU/CUDA 기하·Jacobian 차이 0.
- [공유 계면 폐합](../results/implicit-static-drop-20260923/sharp-face-validation.json): 복원된 4096셀·12,288면의 체적 기하 항등식 오차 9.01e-15, traction 항등식 오차 2.02e-14. CUDA 면 적분 차이는 4.45e-16 이하.
- [CUDA 선형 연산자](../results/implicit-static-drop-20260923/linear-operators-cuda.log): adjoint 차이 4.27e-14, 이차 영공간의 3차 차분 norm 4.43e-15.
- [회전 타원체](../results/implicit-static-drop-20260923/ellipsoid-fit-cuda.json) 및 [비이차 계면](../results/implicit-static-drop-20260923/nonquadratic-fit-cuda.json)은 형상 계수 seed 없이 현재 역복원 코드로 검증했다.

최종 판단은 Draft 유지다. 위 수치 감소가 유의미해도 전체 정적 수렴·동적 계면·장시간 상변화·1차 분열의 미검증을 대신하지 않는다. 실행 파일과 자료 해시는 [delivery manifest](../results/implicit-static-drop-20260923/delivery.json)에 보존한다.

## 실제 solver 시험 재현

CUDA 라이브러리와 solver를 빌드하고 환경을 설정한 뒤 실행한다. `PINTLE_SPUMA_ENV`는 해당 빌드와 호환되는 SPUMA 환경을 가리켜야 한다. 각 출력 경로는 새 디렉터리여야 한다. 검증 도구 자체가 GPU 실행 잠금을 잡으므로 외부 `flock`으로 이 도구를 다시 감싸지 않는다.

```bash
for n in 16 24 32; do
  "$PINTLE_REACTIVE_PREFIX/bin/python" tools/prepare_implicit_case.py \
    --n "$n" --configuration examples/impinging-n2o-supercooled/cold-pr-148K-config.yaml \
    --output "logs/implicit-reproduction/n$n"
done
"$PINTLE_REACTIVE_PREFIX/bin/python" tools/validate_capillary_solver.py \
  logs/implicit-reproduction/n16 logs/implicit-reproduction/n24 logs/implicit-reproduction/n32 \
  --run --timeout-seconds 180 --sigma .01 --surface-energy-field surfaceEnergyDensity \
  --require-static-convergence --output logs/implicit-reproduction/series.json
```

현재 결과의 예상 strict 검증 종료코드는 2다. JSON은 정상적으로 작성되며 이를 계산 실패나 통과로 바꾸어 보고하지 않는다.
