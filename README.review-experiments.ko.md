# 최적화 리뷰 후속 실험

검토 계획은 `reports/review-experiment-plan.md`에 있다. 이 실험은 GPU alpha 수송의 시간 이산화 제한, 접촉각 캐시의 속도 의존성, 온도 보정의 실제 잔차, 자동 회귀 판정과 alpha source 초기화 비용을 다룬다.

## 실행 환경

기존 빌드와 같은 SPUMA v2512, FP64/Int32, NVHPC CUDA 환경이 필요하다. 이번 실행 환경은 RTX 5080, CUDA 13.2, NVHPC 26.5다. 다른 설치 위치에서는 먼저 `PINTLE_SPUMA_ENV`를 해당 SPUMA 환경 스크립트의 절대 경로로 지정한다. 전용 실행 잠금 경로도 `PINTLE_RUN_LOCK`으로 바꿀 수 있다. 환경 스크립트가 SPUMA의 컴파일러·라이브러리·GPU 아키텍처 설정을 제공해야 한다.

```bash
export PINTLE_SPUMA_ENV=/absolute/path/to/spuma-env.sh
export PINTLE_RUN_LOCK=/absolute/path/to/benchmark.lock
source ./env.sh
bash src/coldFoam/Allwmake
bash tools/build_async_smoother.sh
bash tools/build_regression_check.sh
```

Python 도구에는 Python 3와 NumPy가 필요하다. `wmake`, `blockMesh`, GPU 실행파일과 라이브러리가 위 환경에서 검색되어야 한다. 전처리는 일부 SPUMA 유틸리티의 GPU 초기화 누락을 피하기 위해 자체 검사 프로그램이 실제 격자 좌표를 읽어 수행한다.

## 외부 restart가 필요 없는 회귀 case

```bash
python3 tools/prepare_minicase.py benchmarks/my-mini-input
python3 tools/check_review_regressions.py \
  --source benchmarks/my-mini-input --output benchmarks/my-regressions
python3 tools/benchmark_review.py \
  --source benchmarks/my-mini-input --output benchmarks/my-mini-pair \
  --steps 8 --repeats 1 --mini --replay
python3 tools/accept_results.py benchmarks/my-mini-pair/benchmark.json \
  --output benchmarks/my-mini-pair/acceptance.json
python3 tools/check_acceptance_policy.py benchmarks/my-mini-pair/benchmark.json \
  --output benchmarks/my-policy-controls
```

모든 output은 새 디렉터리여야 한다. 입력 case는 실행하지 않고 매번 복사한다. 1,152셀의 작은 case에는 IPA·N2O·air, 압력 차이, 비영 속도와 속도 의존 접촉각이 들어 있다. 합성 수치 회귀 case이며 실험으로 검증한 물리 초기조건은 아니다. 작은 case의 시간은 성능 근거로 사용하지 않는다.

검사 프로그램은 `nSweeps=-1`의 실제 잔차가 default norm에서 0.25, norm none에서 0.5임을 구분한다. `-24`는 충분히 수렴한다. 접촉각 검사에서는 alpha와 시각을 고정하고 U를 바꾼 뒤 비캐시 결과와 비교한다. 원래 라이브러리를 별도 보관했다면 `check_review_regressions.py --legacy-lib /absolute/path/to/original/lib`로 두 기존 오류도 재현할 수 있다.

## 실제 Pintle 비교

```bash
# 두 설정 모두 async smoother. source 초기화만 on/off.
python3 tools/benchmark_review.py --output benchmarks/my-fusion12 \
  --steps 12 --repeats 4

# 기존 multicolor 설정과 async + source fusion의 120-step 비교.
python3 tools/benchmark_review.py --output benchmarks/my-long120 \
  --steps 120 --repeats 1 --combined --replay
python3 tools/analyze_review_qoi.py benchmarks/my-long120/benchmark.json \
  --output benchmarks/my-long120-qoi
```

실제 Pintle 비교에는 별도로 준비한 `cases/pintle45us`가 필요하며, 저장소의 소스만으로 해당 생산 격자와 restart를 복원할 수는 없다. 12스텝 측정에서는 처음 두 스텝과 마지막 출력 스텝을 제외한다. 120스텝은 45→48.6 μs, 고정 dt=30 ns이며 alpha subcycle은 15 ns다. 긴 비교의 한 쌍만으로 성능 배율을 확정하지 않는다.

`pintleFuseAlphaSource` 기본값은 false다. true이면 Sp의 불필요한 zero fill과 Su의 중간 배열 초기화를 source kernel 안으로 옮긴다. 상별 갱신 순서와 FP64 연산을 유지하며 두 설정은 같은 바이너리를 사용한다. 선택하지 않은 limiter ratio 정밀도의 배열은 필요할 때 할당한다.

## 합격 정책과 해석 범위

`policies/numerical-acceptance-v1.json`은 실험 전에 정한 개발용 회귀 기준이다. p/T/rho 양성, 내부 셀 U·alpha의 유한성, alpha 범위·합, 스텝별 총질량 balance와 주요 필드 차이를 검사한다. 실패 시 CLI는 1을 반환한다. 로그·최종 필드·실행 정의의 SHA-256도 검증한다. 이 정책은 물리 모델 인증이나 전체 경계 필드 검사에 해당하지 않는다.

`--replay`는 마지막으로 저장된 `solveAlphas` 호출의 실제 dgdt/divU/Sp/Su/old alpha/flux divergence 및 갱신 전후 alpha를 남긴다. 분석기는 IPA→N2O→air 순서로 source를 재계산하고 explicit update를 검증한다. 출력 디렉터리의 최종 `dgdt.*`는 이후 압력 풀이에서 바뀔 수 있으므로 replay의 dgdt와 구분한다. 모든 과거 호출을 저장하거나 직접 검증하는 기능은 아니다.

QoI 후처리는 실제 셀 체적으로 오차를 가중하고, 저장된 p/T로 재구성한 상별 밀도와 내부에너지를 사용한다. 이는 솔버 마지막 압력 보정 직후의 선형화된 내부 밀도와 다를 수 있다. 계면 지표는 `sum(V*|grad(alpha)|)`인 확산 계면 진단값이다. 에너지 값은 재구성한 sensible internal + kinetic energy inventory 비교이며, 경계 열·일·질량 유입출을 모두 평가한 에너지 보존 검사가 아니다.
