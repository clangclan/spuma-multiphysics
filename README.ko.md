# Pintle용 SPUMA GPU 다중물리 솔버

실제 3,094,455셀 Pintle 체크포인트에서 실행하는 압축성·비등온 3상 VOF 개발판이다. 액체 IPA, Peng–Robinson N₂O, 이상기체 air의 공통 속도·온도를 풀고 WALE LES를 사용한다. N₂O와 air는 별도 체적분율로 수송하고 가스–가스 계면 압축은 끈다. 분자 확산, 증발·응축, 화학종 수송, 연소는 구현하지 않았다. 여기서 3상은 수치적으로 구분한 세 재료를 뜻한다. 고정 격자·global Euler·단일 NVIDIA GPU가 현재 검증 범위다.

핵심 CUDA 연산과 물성 포트는 메인 에이전트가 작성했다. 별도 GPT 5.6 Sol xHigh 에이전트가 작성한 벤치마크 도구는 메인이 전체 검토하고 실제 실행 중 발견한 CLI·진단 문제를 수정했다. 원본 Pintle 및 SPUMA 설치는 수정하지 않고 이 디렉터리에 별도 소스·라이브러리·실행파일을 둔다.

별도 반응·상변화 연구 솔버는 [ReactiveFoam 문서](README.reactive-phase.ko.md)에 설명한다. [2026-09-12 GPU 수송·희소 화학 포팅](README.reactive-gpu-sparse.ko.md)은 선택 가능한 새 경로이며, 아래 ColdFoam의 기존 GPU 성능 측정과 구분한다.

ReactiveFoam의 최신 변경은 [포팅 리뷰 반영 기록](README.reactive-review.ko.md)에 있다.

## 실행

이 컴퓨터의 SPUMA v2512 `5916a466`, NVIDIA HPC SDK 26.5, CUDA 13.2, RTX 5080(sm_120), FP64 환경을 사용한다.

```bash
cd /home/jsw/문서/analysis/spuma-multiphysics
./Allwmake
source ./env.sh
# 필요할 때에만 새 경로로 준비. 기존 디렉터리 덮어쓰기를 거부한다.
python3 tools/prepare_case.py cases/pintle45us-new
# 물성 함수 CPU/GPU 대조
bin/pintleThermoCheck -case cases/pintle45us -pool fixedSizeMemoryPool -poolSize 1
# 같은 SPUMA 수치법 안에서 host MULES/물성과 GPU 경로 비교
python3 tools/benchmark.py --case cases/pintle45us \
  --mode reference gpu mixed --thermo native device device \
  --repeats 3 --steps 12 --output benchmarks/new-run
```

`benchmark.py`는 입력을 실행하지 않고 매회 독립 복사한다. 45 μs부터 dt=30 ns로 진행하며 첫 2스텝과 마지막 출력 스텝을 제외한다. 각 실행의 9스텝 산술평균을 구하고 3회 중앙값을 비교한다. 기본 smoother는 실제 GPU 구현인 `multicolorGaussSeidel`이다. `--smoother prepared`는 준비된 CPU smoother를 보존한다. 허용오차는 바꾸지 않는다. 출력에는 실행파일·설정 hash, 단조시계 시간, 단계별 시간, GPU/RSS 표본, 최종 내부 필드 비교가 남는다.

GPU 물성은 `thermoType.type pintleDeviceHeRhoThermo`로 선택한다. 이때 `thermoType.device false`는 SPUMA의 자동 타입명 변환을 끄는 설정이며, 선택된 로컬 클래스가 실제 물성을 GPU에서 평가한다. 벤치마크 도구가 IPA·N₂O 모델 이름까지 함께 설정한다.

## GPU 구현

- `pintleLimiter.C`: 고정 Euler·rho=1·Sp=Su=0 제한기 특수화. cell CSR gather로 원자 연산을 피하고, 제한량 합산과 ratio 계산을 융합한다. 버퍼를 재사용하며 반복 내부의 불필요한 device synchronization을 없앴다. 결합 경계의 min 동기화는 유지한다.
- 3상 correction flux `limitSum`, divergence+source explicit update, phase source 계산을 융합했다. 상별 순차 갱신은 보존한다.
- `pintleIpa.H`, `pintlePengRobinsonGas*`: native 상관식을 FP64로 GPU에서 직접 계산한다. 보간표를 사용하지 않는다. 공통 T에서 phase energy를 재생성하며 upstream device thermo의 잘못된 pressure pointer도 로컬 포트에서 수정했다.
- 표면장력은 alpha가 변경될 때만 무효화해 같은 압력 반복에서 재사용한다. 현재 Pintle의 경계조건을 기준으로 검증했다.
- `pintleTemperatureIncrement.C`: 동일한 T 선형계를 증분으로 풀고 수렴 실패를 중단한다. `diagnostics yes`로 원래 선형계의 잔차를 출력할 수 있다.
- `pintleLimiter mixed`는 ratio 저장만 FP32로 줄이고 아래 방향 반올림으로 FP64 제한을 완화하지 않는다. 보존량·flux·EOS·선형계는 FP64다. FP16/BF16/Tensor Core 경로는 없다.

## 수치적 범위와 주의점

이 구현의 CPU/GPU 대조 기준은 **같은 SPUMA 애플리케이션의 기존 MULES 경로**다. reference도 SPUMA GPU 유한체적 연산과 GPU smoother를 사용하며, 비교 시 CPU 경로로 바뀌는 것은 MULES와 지정한 물성 평가다. 원본 Foundation 14 Pintle과 완전한 이산화 동등성을 달성한 상태는 아니다.

원본에 맞춘 항목은 momentum/temperature `contErr` 보정, phase별 열확산 계수, 모든 alpha의 subcycle old-time 보존, phase-sum drift 보정, 공통 T에서 물성 갱신, vDot를 dgdt로 읽는 재시작 처리다. 남은 차이는 Foundation 14 MULES의 첫 반복 및 lambda 갱신 규칙, 일부 비결합 경계의 total flux/limitSum 처리, 에너지 기준값이다. SPUMA phase energy는 p,T에서 재생성한다. 원본의 압력 증분으로 갱신된 phase rho/psi 등 열역학 체크포인트 상태도 그대로 복원하지 않고 공통 p,T에서 재초기화한다. 압력 최소 반복 수도 기존 F14 벤치의 0에 비해 1이다.

`dgdt`는 작은 차이의 상쇄에 민감하다. p/U/T/alpha가 가까워도 raw dgdt의 상대차는 클 수 있으므로 이를 숨기거나 모든 필드가 동등하다고 판정하지 않는다. 긴 물리 시간의 안정성, 원본 F14와의 체계적 수렴 비교, MPI 결합 경계 실행 검증은 후속 과제다. 사용자 지정 IPA 물성 계수는 현재 명시적으로 거부한다.

## 재현 자료

- `cases/pintle45us/preparation.json`: 원본 및 복사 변환, source hash.
- `logs/thermo-check.log`: 24,511 p,T 점의 native CPU 물성 대조.
- `benchmarks/oracle-device4/stdout.log`: 실제 mesh에서 limiter/limitSum/explicitSolve 대조.
- `benchmarks/profile-device2.{nsys-rep,sqlite}`: CPU smoother 병목 프로파일.
- `upstream-files.json`, `src/coldFoam/multiphaseMixtureThermo/thermo-upstream.json`: SPUMA 원본 추적.

검증·성능 집계는 `reports/validation.md`에 기록한다. 빌드 산출물과 큰 케이스는 Git에서 제외한다. SPUMA/OpenFOAM에서 파생한 코드는 GPL-3.0-or-later 조건을 따른다.

[성능·필드 검증](reports/validation.md)에서 12스텝×3회 결과를 확인할 수 있다. GPU FP64는 중앙값 1.8629 s/step으로, 같은 SPUMA의 CPU MULES·CPU 물성 경로 대비 12.68배 빠르다. FP32 ratio 저장은 유의한 추가 이득이 없어 FP64를 기본값으로 유지한다.

[추가 실행 검증](reports/runtime-checks.md)에 선택 커널 메모리 검사와 프로파일을 기록했다. 메모리 접근 및 명시적 CUDA API 검사는 0건으로 끝났지만, 공용 템플릿의 중복 커널 등록에 대한 extended CUDA 진단은 미해결이다. [원본 무변경·실행파일 확인](reports/integrity.json)도 함께 남긴다.
