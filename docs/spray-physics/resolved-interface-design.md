# Resolved interface 설계: 액주·액막 primary breakup의 첫 계약

## 결정

첫 morphology 모델은 `ResolvedInterfaceV1` (`RIF1`)이다. 보존 상태에 별도 color/volume field `c`를 두며 `c=1`은 물질 A(첫 경로에서는 액체), `c=0`은 물질 B(기체)다. `c`는 `PintleThermoState::alphaLiquid`, HEM flash 결과, 또는 `alphaEnvironment`의 별칭이 아니다. HEM의 액상 체적분율은 증발·응축으로도 바뀌고, mechanical environment 체적분율은 각 환경 안의 HEM을 포함할 수 있으므로 둘 다 기하 계면을 정의하지 못한다.

첫 실제 계산 경로는 **비반응·비상변화·비혼화성·압축성 2물질**이다. A/B는 서로 분리된 보존 질량과 frozen phase inventory, 각자의 EOS를 가지며 한 속도를 공유한다. color face flux와 A/B inventory flux는 같은 기하 재구성에서 나와야 한다. 계면 압축이나 regularization이 `c`만 바꾸어 물질 및 에너지 flux와 어긋나는 방식은 허용하지 않는다. 이 경로에서 액주와 액막의 변형, ligament 형성, pinch-off와 해상 액적 생성을 검증한 뒤에만 증발과 반응을 붙인다.

현 `pintle_rt_recover_mechanical`은 한 셀 안의 두 환경 압력이 같아질 때까지 에너지를 분할한다. 따라서 이 폐쇄식에 상별 압력 차만 임시로 넣으면 다음 recovery에서 지워진다. 이는 모든 one-fluid 공통 압력 모델이 셀 사이의 공간적인 압력 점프를 표현할 수 없다는 주장은 아니다. 여기서 설계한 상별 에너지 경로는 다음 결합 중 하나를 요구한다. 다른 one-fluid 경로를 채택하려면 그 압력·계면력·상변화·에너지 이산화를 별도로 검증해야 한다.

1. cold immiscible 전용 EOS recovery가 각 물질 압력을 유지하고 계면 Riemann/pressure coupling에서 점프를 적용한다.
2. mechanical closure를 `p_A-p_B=σ κ`로 확장하고, 에너지 분할·음속·dilatation 관계를 같은 제약에서 다시 유도한다.

`phasePressures()`는 체적 평균 압력 `p̄=c p_A+(1-c)p_B`와 점프를 동시에 만족하는 값

\[
p_A=p̄+(1-c)\Delta p,\qquad p_B=p̄-c\Delta p
\]

을 제공한다. 이는 새 closure가 사용할 수 있는 국소 계약이며, 기존 common-pressure closure를 우회하는 수단이 아니다.

## 구현된 host/device 원시 연산

[`pintleResolvedInterface.h`](../../src/reactiveInterface/pintleResolvedInterface.h)는 OpenFOAM과 thermo 타입이 없는 CPU/CUDA 공용 값 API다. 잘못된 model ID, `σ<0`, NaN/Inf, 범위 밖 color, 양수가 아닌 밀도·cell length, 단위가 아닌 face normal은 상태를 부분 변경하지 않고 오류를 반환한다. `σ=0`은 유효한 비활성 모델이다.

입력 `∇c`의 단위는 `m⁻¹`이다. A에서 B로 향하는 법선과 계면 면적 밀도를

\[
\mathbf n=-\frac{\nabla c}{|\nabla c|},\qquad a=|\nabla c|\quad[m^{-1}]
\]

로 정의한다. `a h`가 dimensionless geometry tolerance 이하면 비활성이다. 이 계산은 분석용 gradient 원시 연산이다. PLIC, height-function, least-squares mesh gradient 또는 곡률 재구성을 구현하지 않는다.

상수 표면장력의 보존형 capillary stress는

\[
\mathbf C=σa(\mathbf I-\mathbf n\mathbf n)\quad[Pa],
\qquad
\partial_t(ρ\mathbf u)+\nabla\cdot(ρ\mathbf u\mathbf u+p\mathbf I-\mathbf C)=0
\]

이다. 따라서 face의 운동량 flux 추가분은 `-C·n_f`, bulk total-energy flux 추가분은 `-u_f·(C·n_f)`이다. 한 face flux를 owner와 neighbour에 반대 부호로 넣으면 운동량과 bulk-energy 교환은 보존형이다. 상수 `σ`에서 surface free-energy density는 `e_s=σa [J/m³]`이고 해석할 총량은

\[
E_{all}=\int_V ρE\,dV+\int_V σa\,dV
\]

이다. 실제 이산 시간 적분에서 이 합을 보존하려면 color advection에 따른 면적 변화와 stress work를 같은 단계에서 결합해야 한다. 현재 원시 연산만으로 에너지 일관성이 완성됐다고 주장하지 않는다. `σ(T,Y)`와 Marangoni stress는 surface entropy와 조성 excess 모델을 정하기 전까지 지원하지 않는다.

법선이 A에서 B로 향하고 `κ=∇·n`일 때 A 구의 곡률은 양수이고

\[
p_A-p_B=σ κ
\]

이다. `laplacePressureJump()`는 이 부호와 단위를 검사하는 분석 helper일 뿐 mesh curvature가 아니다.

명시적 capillary 제한은 [Basilisk `tension.h`](https://basilisk.fr/src/tension.h)의 최소 파장 식과 같은

\[
\Delta t_{cap}=C_{cap}\sqrt{\frac{ρ_m h^3}{πσ}},\qquad
ρ_m=\frac{ρ_A+ρ_B}{2},\quad 0<C_{cap}\le1
\]

이다. SI 차원은 `sqrt((kg/m)m³/(N/m))=s`다. 전역 timestep은 convective, viscous, species/thermal diffusion 제한과 이 값을 함께 최소화해야 한다.

## 근거와 선택 범위

- [Basilisk geometric VOF](https://basilisk.fr/src/vof.h)는 보존적 color advection과 같은 기하 flux에 묶인 phase tracer flux를 보여 준다. 본 설계에서 color와 물질 inventory flux를 같이 만들도록 요구한 근거다.
- [Basilisk interfacial-force source](https://basilisk.fr/src/iforce.h)는 압력 gradient와 계면력에 같은 face gradient를 써야 well balancing된다고 명시한다. 그래서 단순 cell-centered CSF를 현 solver에 더하지 않고, pressure/curvature pair의 후속 구현을 요구한다.
- [Basilisk compressible surface tension](https://basilisk.fr/src/compressible/tension.h)는 평균 압력에서 Laplace jump를 만족하는 phase pressure를 재구성하며 phase energy flux도 수정한다. 본 pressure helper와 energy-coupling 요구가 이를 따른다.
- [OpenFOAM multiphase interface source](https://api.openfoam.com/2506/multiphaseInterSystem_8C_source.html#l01075)는 interface pair별 `σ`, curvature와 face-normal gradient를 조립한다. 구현을 복사하지 않았으며, face level pressure/capillary 결합이 필요하다는 비교 근거로만 사용했다.
- Jain, Mani, Moin의 [conservative diffuse-interface method](https://arxiv.org/abs/1911.03619) ([JCP DOI](https://doi.org/10.1016/j.jcp.2020.109606))는 compressible two-phase에서 phase mass, momentum, total energy, bounded volume fraction과 일정한 계면 두께를 함께 다룬다. diffuse-interface를 선택한다면 color regularization flux만 따로 가져오지 말고 전체 보존식을 채택해야 한다.
- [PeleMP spray equations](https://amrex-combustion.github.io/PeleMP/Equations.html)는 dilute Lagrangian point-droplet 가정이다. 이는 primary breakup을 해상하거나 초기 액적 분포를 생성하는 모델이 아니므로 현 우선 경로에 쓰지 않는다.

## 시험과 주장 가능한 범위

`python3 tools/validate_interface_primitives.py`는 CPU와 `sm_120` CUDA에서 같은 원시 연산을 실행한다. 시험은 다음을 검증한다.

- 평면 분석 gradient의 법선, 면적 밀도, tangential stress, face traction/work와 surface energy 단위
- color orientation 반전 시 `n⊗n` stress 불변
- 반지름 2 mm 구의 **분석 곡률 입력**에서 `σ=0.072 N/m`일 때 `Δp=72 Pa`인 부호
- volume-average pressure와 phase pressure jump의 두 항등식
- capillary timestep 식과 CPU/CUDA 일치
- 잘못된 `σ`, color, density, cell length, face normal과 유한 입력의 산술 overflow에서 결과를 보존하는 fail-closed 동작

실행기는 결과와 컴파일러 정보, 이 모듈·시험·실행기·설계 문서의 SHA-256을
`results/spray-physics/interface-primitives.json`에 원자적으로 기록한다.

이 시험은 정적 구형 액적, Laplace equilibrium, spurious current, capillary wave, oscillating drop 또는 breakup을 계산하지 않는다. 그런 주장은 mesh geometry와 pressure-balanced discretization을 연결한 뒤 다음 순서로 얻는다.

1. bounded conservative color advection + material inventory consistency: translation, rotation, vortex reversal, compression/expansion.
2. mesh normal/curvature와 pressure-gradient matched face force: planar interface zero force, static circle/sphere Laplace jump와 spurious-current mesh convergence.
3. surface-energy work: inviscid capillary wave dispersion과 bulk+surface energy, oscillating-drop frequency/decay.
4. cold injector: liquid-column/film mass, ligament and detached-component topology, cells-per-diameter/thickness convergence.
5. shear breakup: Weber/Ohnesorge/Reynolds similarity, breakup length, fragment-size distribution와 domain/mesh/time convergence.
6. 해상 한계 아래 연결 성분만 보존적인 Eulerian-to-droplet 변환 계약에 넘긴다. 질량·조성·운동량·bulk+surface energy와 생성 통계를 기록하며 임의 parcel 초기 분포로 primary breakup을 대체하지 않는다.
