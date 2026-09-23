# 정적 액적 기하학적 균형 제안 반영

작성일: 2026-09-23. 대상: Draft PR #6.

**이번 변경은 GPU 기하 면 유량과 실제 수송 커널을 통한 진단을 구현한다. 실제 솔버의 정적 액적 수렴 문제는 아직 해결되지 않았다.** 해석적 구면을 공급한 연산자 시험은 통과했지만, 보존된 액체량으로부터 모든 면의 호환 기하를 재구성하고 이동·상변화까지 연결하는 모델은 준비되지 않았다. 따라서 기존 `surfaceTension true`의 diffuse v1 실행 모델을 묵시적으로 교체하지 않았고 PR은 Draft로 유지한다.

## 입력 리뷰와 판단

사용자가 제공한 `SPUMA_static_drop_geometric_balance_proposal_20260923.zip`의 문서·검산 코드와 「GPU 정적 액적 해법」 후속 대화를 검토했다. 압축파일 SHA-256은 `80c54ddd896a1be96cc4de5f591e9004ba67566f042404d129f69608374a66e0`이며 내부 manifest 7개 항목이 일치했다. [출처 기록](../results/geometric-balance-20260923/proposal-provenance.json).

제안은 다음의 서로 다른 양을 구분한다.

- 셀의 액체 체적분율 `c`: EOS 상별 압력 분할과 reduced-pressure HLLC에 사용한다.
- 면의 액체 점유면적 `A_l`: 압력의 실제 면적 적분에 사용한다.
- 같은 계면과 면의 교선에서 적분한 co-normal traction `T` [N]: 모세관 면 힘에 사용한다.

같은 매끄러운 계면에서 계산하면 `sum(A_l n)+N=0`, `sum(T)+sigma*integral(kappa n dA)=0`이다. 일정 곡률·Laplace 평형에서는 셀 힘이 상쇄된다. 다만 재구성한 계면의 곡률 적분이 초기 압력차와 맞아야 하므로, 면 점유율과 traction을 공유하는 것만으로 임의 재구성의 정적 균형이 보장되지는 않는다. 보존성과 정적 균형을 구분해야 한다는 점은 [Saini et al. (2025)](https://doi.org/10.1016/j.jcp.2025.114348)과도 일치한다. 압력 보정형 비압축성 알고리즘의 균형 조건을 현재 압축성 HLLC/UV 경로로 그대로 이전할 수도 없다. [François et al. (2006)](https://doi.org/10.1016/j.jcp.2005.08.004).

## 반영한 코드

| 위치 | 구현 |
|---|---|
| `src/reactiveInterface/pintleGeometricCapillary.h` | CPU/CUDA 공통 기하 면 유량. 기존 HLLC를 재사용하고 압력 복원과 응력을 동시에 교체 |
| `src/reactiveTransport/pintleGeometricTransport.h` | 공급 기하를 이용하는 별도 일회성 진단 ABI, 셀·면 SI 단위와 체적/힘 폐합 출력 |
| `src/reactiveTransport/pintleTransportKernels.h` | 실제 `Faces`/`Rhs`에서 기하 경로 실행, 벌크/표면에너지 분리, GPU 셀 폐합 진단 |
| `src/reactiveTransport/pintleReactiveTransport.cpp` | 별도 메모리·스트림의 진단 실행, 검증 완료 후 출력, 실패·재시도와 resident 상태 격리 |

운동량 복원은 면적당 `(DeltaP*A_l*n-T)/A_face`다. 기존 보간 응력을 추가로 더하지 않는다. HLLC에는 `E_total-sigma*A_interface/V`를 전달한다. 이후 기존 gather가 총에너지 밀도에 곱하는 이류 계수에서 표면에너지 몫을 한 번 제거하고, 동일 면 속도의 traction work와 별도로 공급받은 표면에너지 이류량을 더한다. 공급받은 이류량의 물리적 정확성까지 검증한 것은 아니다.

새 API `pintle_transport_geometric_diagnostic_v1`은 다음 계약을 갖는다.

- 닫힌 internal/cyclic 면 그래프, 비점성·비확산 단일 액체 재고 레이아웃만 허용한다. WALE가 설치된 객체와 진행 중인 RK 시도는 거부한다.
- 기존 q·RK 스냅샷·기하·WALE 캐시·버전 토큰을 수정하지 않는다. 진단 오류도 기존 resident 상태를 무효화하지 않는다. 같은 handle의 동시 호출은 기존 mutex로 배제한다.
- GPU 실패를 CPU로 대체하지 않는다. 입력 검증이나 커널이 실패하면 공개 출력 배열을 덮어쓰지 않는다.
- `volumeMismatch`, `volumeClosure`, `tractionClosure`, `surfaceEnergy`를 기록한다. 유한한 재구성 오차를 사후 상쇄하거나 숨기지 않는다.
- 이 API는 시간 적분 모델을 설치하지 않는다. 해석적 반경·중심을 생산 솔버로 전달하는 설정도 추가하지 않았다.

## 시험과 결과

RTX 5080, CUDA sm_120에서 실행했다. 기존 전체 물리 경로의 회귀와 새 기하 연산자의 검증 범위를 분리했다.

### 해석 기하를 공급한 면 연산자

[시험 자료와 재현 명령](../results/geometric-balance-20260923/face-operator-tests.json)은 CPU 16³·24³·32³·48³·64³ 및 CUDA 16³·24³·32³의 원점/비정렬 중심 구면을 포함한다. 실제 reduced-pressure HLLC와 공유 면 조립을 통과시킨 정규화 최대 힘 잔차는 대략 `1.6e-11 ~ 3.3e-11`이다. 압력 3 MPa를 포함한 부동소수점 조립 오차가 남으므로 RHS가 정확히 0이라는 뜻은 아니다.

압력 점유면적은 disk/rectangle 적분, traction은 circle-arc co-normal 적분으로 계산한다. 힘을 압력 힘의 음수로 지정하거나 셀 잔차를 빼지 않는다. 셀 color와 밀도·음속·벌크 에너지는 시험용 입력이다. 이 결과는 PR/UV나 실제 VOF 재구성 수렴을 검증하지 않는다.

음성 대조와 에너지 시험도 포함한다.

- 압력차와 HLLC의 jump를 함께 1% 높이면 reduced pressure는 계속 같지만, 실제 구면 traction과의 불일치로 힘 잔차 `9.90099e-3`이 남는다.
- face jump를 유지하고 한쪽 셀 압력만 바꾸면 Riemann 질량 유량과 접촉 속도가 0이 아니게 된다.
- 면 점유면적을 셀 color 평균으로 대체하면 균형이 깨진다.
- 서로 다른 좌우 상태의 면 방향 반전, HLL 대체 경로, 이동 면의 일률과 공급한 표면에너지 이류, 잘못된 입력에서 출력 불변을 검사했다.

### 실제 transport ABI와 기존 솔버 회귀

`tools/validate_geometric_transport.py`는 공개 API를 통해 실제 `Faces`/`Rhs`를 실행한다. 기하의 셀 적분은 별도 공간 적분으로 계산하지만, 면 검산과 같은 해석적 disk/arc 기본 함수를 재사용한다. 독립된 두 EOS/기하 라이브러리의 대조로 해석하지 않는다.

공개 API 시험은 balanced/wrong-jump 상태, 오류 출력 불변, resident 토큰 보존, RK 도중 거부, 재시도, 움직이는 비균일 에너지 상태를 포함한다. CPU·CUDA 각각 64개 검사를 통과했으며 기록된 수치 지표는 두 backend에서 비트 단위로 일치한다. 16³ 균형 상태의 최대 운동량 RHS 성분은 `3.93845e-6 N/m³`, 압력/traction 힘 밀도로 정규화한 값은 `2.83349e-11`이다. 잘못된 압력차에서는 `1081.08 N/m³`가 남는다. [CPU 결과](../results/geometric-balance-20260923/transport-api-cpu.json), [CUDA 결과](../results/geometric-balance-20260923/transport-api-cuda.json).

[기존 물리 선택 회귀 35개](../results/geometric-balance-20260923/physics-switches.json)가 통과했다. [4096셀 전체 물리 계산](../results/geometric-balance-20260923/full-physics-regression.json)은 점성·열전도·WALE 열/종 혼합·표면장력·flashing을 켜고 60 ns까지 진행했다. GPU PR 물성 86,016셀 평가, 실패 0, 내부 셀 CPU 엔탈피 평가 0, GPU flash CPU fallback 0을 기록했다. 보존장 시계열은 이전 기준 계산과 [비트 단위로 일치](../results/geometric-balance-20260923/full-physics-parity.json)했다. 이 회귀는 기존 diffuse v1 모델을 사용하며 새 geometric 모델의 통합 시험이 아니다.

### 체적분율에서의 수치 재구성 탐색

`tools/geometric_reconstruction_probe.py`는 구면의 합성 VOF 셀 체적을 입력으로 받아 column height, 체적 평균 보정, 공유 C2 spline을 만든다. 재구성에는 구 중심·반경·곡률을 전달하지 않는다. 같은 표면에서 aperture·traction·면적·곡률 적분을 구하면 제한된 패치에서 두 기하 폐합은 약 `1e-14` 수준으로 맞는다. [전체 탐색 결과](../results/geometric-balance-20260923/geometric-reconstruction-probe.json).

| 해상도 | 평가 셀 / 내부 후보 열 | 최대 정규화 힘 잔차 | 최대 액체 체적분율 불일치 |
|---:|---:|---:|---:|
| 16 | 36 / 64 | 1.011e-5 | 2.213e-5 |
| 24 | 184 / 256 | 5.053e-6 | 8.005e-6 |
| 32 | 400 / 576 | 3.388e-6 | 4.728e-6 |
| 48 | 1060 / 1600 | 2.682e-6 | 1.913e-6 |

이 표는 단일 높이함수 패치의 일부 셀만 평가한다. 수평 면을 통과하는 교선, 패치 연결부와 닫힌 전체 구면은 제외되며 선택에도 참조 기하를 사용한다. CPU/SciPy 탐색이고 GPU 생산 재구성은 아니다. 따라서 이 감소 추세를 실제 정적 액적의 메시 수렴 통과로 사용할 수 없다. 체적 오차가 작아져도 보존된 액체량과 정확히 일치하지 않아 UV 입력으로 바로 채택할 수 없다.

## 실제 솔버 활성화까지 남은 일

1. mesh face의 꼭짓점·다각형과 셀 clipping 정보를 포함하는 기하 계약이 필요하다. 현재 `PintleTransportFace`의 면적·법선·거리만으로는 일반 메시의 교선을 정할 수 없다.
2. 모든 교차면과 방향 전환을 덮고 셀별 액체 체적을 만족하는 공유 계면을 재구성해야 한다. 서로 다른 셀 PLIC의 교선·법선을 평균하는 것만으로는 두 셀의 표면 발산식이 동시에 보장되지 않는다.
3. 같은 재구성에서 curvature, aperture, traction, cell area를 얻고 실제 수치 곡률을 사용한 전체 구면의 최종 RHS·최대/L2 유속 수렴을 통과해야 한다.
4. 이동 계면의 swept liquid/species/bulk/surface-energy 수송과 면적–일률 수지를 연결해야 한다. 상변화로 계면 속도가 유체 속도와 달라질 때의 면적 변화도 필요하다. 단순 총에너지 보존만으로 이 항등식이 보장되지는 않는다.
5. 기하 면적 정의가 바뀐 checkpoint를 명시적으로 버전 구분하고, frozen 시험 이후 PR/UV·flashing·동적 파/액적 시험을 수행해야 한다.

이번 변경은 고도의 최적화나 256k/1M 성능 재측정을 수행하지 않는다. 현재 production 경로를 바꾸지 않은 상태에서 대규모 계산을 반복해도 정적 수렴 결론이 달라지지 않기 때문이다. 실제 static gate를 통과했다는 결과는 없으며, 기존 미통과 기준을 완화하지 않았다.
