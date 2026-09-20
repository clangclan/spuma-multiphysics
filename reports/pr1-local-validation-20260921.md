# PR #1 로컬 검증 — 2026-09-21

기준 main: `50512cecdc7f8047f3ccbb8c60e05a43222e1cc3`. 최초 검사 PR HEAD: `ad2e178d6a9409094a891466a683051514f39bf1`.

현재 상태: PR의 로컬 회귀 인수 완료. 기존 main에서 재현되는 HEM 접촉면 정확도와 SPUMA 확장 CUDA 진단 한계는 아래에 명시한다. [최종 인수 요약](../results/pr1-local-20260921/acceptance-summary.json)이 최초 실패와 재검사 결과의 우선순위를 정한다.

## 확인하고 수정한 문제

- HEM 케이스 생성기의 schema 2 identity에 필수 `closure HEM`이 빠져 있었다. 생성 파일을 수정하고 실제 솔버의 정상 시작 및 누락 시 거부 검사를 추가했다.
- RF21 재계산 검사에서 새 `ctypes.CDLL` 인스턴스의 v2.1 생성자 반환형이 지정되지 않았다. 64비트 포인터가 C int로 잘려 CPU/GPU 검사가 SIGSEGV로 끝났다. 사용하는 모든 인스턴스에 C ABI 시그니처를 연결했다.
- 오류 저장소가 고정 char 배열로 바뀐 뒤 희소 화학 회귀 harness가 문자열 연산을 그대로 사용해 컴파일에 실패했다. char 데이터로 명시적으로 변환했다.
- 로컬 인수 runner가 성공 종료한 자식의 결과 파일이나 필수 검사 그룹 누락을 통과시킬 수 있었다. 필수 결과와 그룹 집합을 확인하고, 누락·skip을 거부하는 회귀 검사 4개를 추가했다.
- 솔버가 체크포인트의 해시를 따옴표 없는 `word`로 저장했다. 숫자로 시작하는 해시는 OpenFOAM의 숫자 토큰으로 해석되어 실제 재시작이 실패했다. 네 해시 필드를 인용 문자열로 저장하고, CPU/CUDA 연속 실행·분할 재시작·백엔드 전환 재시작으로 검사했다.
- 기울어진 메시의 확산 검사는 대조군의 확산계수만 변경하고 이전 모델의 schema 2 물리 해시를 남겨 거부됐다. 별도 초기 모델의 올바른 식별값을 생성하도록 검사 fixture를 수정했다. 솔버의 물리 모델 변경 거부 조건은 유지한다.

- OpenFOAM `write()`의 실패 반환값을 무시해 저장할 수 없는 체크포인트 경로에서도 성공 종료했다. scalar/vector 필드 및 identity 쓰기의 실패를 명시적으로 보고하고 종료하도록 수정했다. CPU/CUDA에서 필드·identity 출력 경로를 막는 오류 주입 검사를 추가했다.
- 기본 CPU의 기액 반응 결합 실행에서 main 대비 약 7배의 성능 회귀를 발견했다. flash의 작은 선형계에 새 scaling을 적용한 반올림 차이가 미량 종 음수 검사에 걸려 전체 source의 밀집 재적분을 유발했다. flash에는 기존 FP64 pivoted LU를 유지하고, tangent/Jv API의 scaled factor는 그대로 둔다. 분리 실험에서는 추가 재적분 4회가 0회로 줄고 주요 최종장이 main과 정확히 일치했다. numerical policy 버전을 변경해 재시작 로그에서 이 변경을 식별한다.

물성 계수, EOS, 반응기구, 보존량 및 기존 수치 허용오차는 변경하지 않았다.

## 실행 환경

현재 호스트는 Intel Core Ultra 7 270K Plus / RTX 5080이다. NVIDIA HPC SDK 26.5 / CUDA 13.2, native `sm_120` 및 기존 Cantera 환경을 사용했다. 예전 Ryzen 호스트에서 만든 SPUMA 바이너리는 AVX-512 명령에서 SIGILL을 발생시켰다.

기존 설치를 보존하고 `/home/jsw/cae-gpu-pr1-compatible/spuma`에 소스를 복사하여 `-tp=x86-64-v3`로 필요한 SPUMA 라이브러리를 재빌드했다. NVIDIA 전용 `-tp`는 CUDA `.cu`의 nvcc 호출에서 제외했다. `symmetryPlaneFvPatchFields.C` 한 파일은 NVIDIA O3 optimizer의 SIGSEGV 때문에 O1로 컴파일했다. 이 공통 SPUMA 빌드를 main과 PR 비교 양쪽에서 사용한다.

Cantera의 설치 자료 검색에서 한글 경로가 제거되는 문제는 ASCII 경로를 가리키는 `CANTERA_DATA`로 해결했다. 반응기구 파일을 바꾼 것은 아니다.

```bash
export PINTLE_SPUMA_ENV=/home/jsw/cae-gpu-pr1-compatible/env.sh
source ./env.sh
```

모든 계산 캠페인은 기존 `/home/jsw/cae-benchmark/run.lock`으로 직렬 실행한다. 빌드, 사례 및 원로그는 새 경로에 보존하며 기존 실험은 덮어쓰지 않는다.

## 독립 API·GPU 결과

첫 수정 후 14개 실행 명령이 모두 성공했다. 최종 flash 수정 후 관련 API 8개 명령도 [다시 전부 성공](../results/pr1-local-20260921/final-api/api-run.json)했다. 변경하지 않은 수송 라이브러리의 21개 연산자 검사와 sanitizer 결과는 기존 근거를 유지한다. [실행 명령과 종료 코드](../results/pr1-local-20260921/fixed/api-run.json)에 전체 인수가 있다.

- RF21 CPU 7개 그룹, CUDA 3개 대상 그룹 통과.
- 수송 CPU/CUDA 각각 21개 연산자 검사 통과.
- 기존 열역학, 화학 Jacobian, 상세 희소 화학, workspace 수명·오류 복구, v2 호환성 검사 통과.
- 실제 RTX 5080에서 RF21 memcheck, initcheck, synccheck, racecheck가 각각 종료 코드 0 및 오류 0으로 완료.
- 인수 결과 누락 거부 회귀 4개 통과.

초기 실패와 수정 후 결과는 각각 `results/pr1-local-20260921/` 및 `fixed/`에 구분했다. 초기 RF21 실패 출력의 부분 `passed` 필드는 프로세스 SIGSEGV 뒤의 성공 근거로 사용하지 않는다.

## 전체 솔버 인수

[최종 전체 비교](../results/pr1-local-20260921/verified-final/matrix.json)와 기액 결합의 추가 두 번 반복을 합쳐 81회 실행, 최종장 비교 63회, 점성·전도·확산 계수 비교 12회가 통과했다. 균일 유동, 32/64셀 음향파, 충격파, 감압 상변화, 접촉면, 수송, 413종 반응·기액 결합·반응 충격파 및 합성 PR matrix-free/Woodbury 경로를 포함한다. CPU/CUDA, worker 1/2, 작은 마지막 batch, bridge, host/device 기상 물성을 비교했다.

최대 최종장 정규화 오차는 `9.316e-8`(기준 `1e-6`), 최대 종 질량분율 차이는 `3.266e-10`(기준 `1e-7`)였다. 기존 HEM 접촉면만 main과 같은 물리 정확도 실패로 유지하며, 그 사례도 최종장 일치와 수지는 통과해야 한다. `matrix.json`의 `passed`는 새 회귀가 없다는 판정이며 모든 물리 모델이 검증됐다는 뜻이 아니다.

| 검사 | 최종 결과 |
|---|---|
| HEM runtime: 입력 거부·schema·쓰기 실패·재시작·기울어진 메시 확산 | CPU/CUDA 각각 14개 통과 |
| Mechanical runtime: 지원 범위·폐쇄식·재시작 | CPU/CUDA 각각 8개 통과 |
| Mechanical 물리·보존량: 독립 해석식, 접촉면·음향파·균일 유동 | CPU/CUDA 각각 10개 통과 |
| 부분 batch 후 전체 단계 복구·schema 1·백엔드 전환 재시작 | 음향파·합성 PR 반응 각각 6개 통과 |
| 로컬 인수 실행기 | smoke, CPU integration, CUDA sanitizer, Flow, benchmark 5단계 통과 |
| 수송 Compute Sanitizer | memcheck/initcheck/synccheck/racecheck 각각 오류 0 |
| 전체 앱 Compute Sanitizer | 전체 커널 메모리 및 명시적 API 검사 오류 0; 커널 필터 없음 |

[최종 runtime 결과](../results/pr1-local-20260921/verified-final/)와 [최종 rollback 결과](../results/pr1-local-20260921/rollback-final/)에 세부 판정이 있다. 화학을 켜면 첫 source half-step이 worker stage 1을 사용하므로 복원 실패 주입은 stage 2에서 해야 한다. 최초 시험용 훅이 이를 반영하지 못한 실패를 보존했고, 수정한 훅이 실제 이전 batch의 반영과 화학 half-step 이후에 실패하는 것을 확인했다. 그 후 전체 상태 복구는 기준 실행과 일치했다.

## 같은 종료시간 성능 회귀

기액 결합 4셀·413종·`t=1e-7 s`를 같은 호스트에서 직렬로 세 번 실행했다. 아래는 준비·분석을 제외한 솔버 프로세스 시간이다. 기준 main은 같은 SPUMA와 컴파일 조건을 사용한다.

| 경로 | 3회 중앙값 | 추가 밀집 재적분 |
|---|---:|---:|
| main CPU | 8.338 s | 매회 0 |
| 최종 PR CPU | 8.185 s | 매회 0 |
| 최종 PR CUDA | 8.787 s | 매회 0 |
| 최종 PR CUDA + CPU worker 2 | 6.833 s | 매회 0 |

수정 전 CPU는 55.062 s, 해당 main은 8.287 s였다. 최종 기본 CPU는 동일 사례에서 main 수준으로 복구됐다. 전체 비교에는 기본 CPU 실행이 main의 2배를 넘으면 원인 확인·반복을 요구하는 회귀 기준도 추가했다. 이 작은 사례 결과를 대규모 GPU 가속률로 해석하지 않는다.

## 남아 있는 기존 한계

- HEM 접촉면 압력 peak 오차는 main 약 6.49%로 1% 기준을 충족하지 않는다. PR CPU/CUDA에서도 동일하게 재현된다. 16셀 mechanical 접촉면의 기존 이동 속도 오차도 해결 완료로 분류하지 않으며 이번 인수의 접촉면 정확도 검사는 32/64셀을 사용한다.
- 전체 앱의 **확장 CUDA 진단을 모두 켠 memcheck는 15건으로 실패**한다. 공용 SPUMA `Foam::cuda` 템플릿의 중복 커널 등록 진단이며, main에서도 동일한 15개 이름이 나온다. `--report-api-errors explicit`에서 전체 커널 메모리 및 명시적 API 검사는 오류 0이다. [기준 대조와 두 실행 원로그](../results/pr1-local-20260921/flow-sanitizer/validation.json)를 남겼으며, 무필터 확장 진단까지 깨끗하다고 주장하지 않는다.
- 상세 413종 고압 PR 물성 자료, 비이상 확산, GPU chemistry, RF21-07~09, production 규모 성능·전체 heap/RSS/VRAM peak 및 실험 정확도 검증은 이번 변경의 완료 범위가 아니다.

## 재현과 근거

[source·binary·물성 입력 SHA-256](../results/pr1-local-20260921/provenance.json), [환경과 빌드 근거](../results/pr1-local-20260921/environment/), [검증 범위](../results/pr1-local-20260921/validation-scope.json)를 기록했다. 초기 실패 파일은 최종 성공 파일로 덮어쓰지 않았다. 특히 `verified-final/run.json`의 최초 화학 failure-injection 실패는 `rollback-final`의 수정 후 결과로 대체하며, 확장 CUDA 진단 실패는 위 기존 한계 판정을 유지한다.

```bash
export PINTLE_SPUMA_ENV=/home/jsw/cae-gpu-pr1-compatible/env.sh
source ./env.sh
# 각 output에는 새 경로를 지정한다. 내부 lock을 쓰는 검사는 외부 flock으로 감싸지 않는다.
research/reactive-env/bin/python tools/validate_reactive_backends.py \
  --thermo-dir research/reactive-thermo --baseline-root research/main-baseline \
  --output cases/new-backend-matrix
research/reactive-env/bin/python tools/validate_reactive_runtime.py \
  --thermo-dir research/reactive-thermo --transport-backend cuda --thermo-workers 2 \
  --output cases/new-runtime
# test-only shim은 운영 solver에 링크하지 않는다.
g++ -std=c++17 -shared -fPIC tools/test_reactive_batch_failure.cpp -ldl \
  -o lib/libpintleTestBatchFailure.so
research/reactive-env/bin/python tools/validate_reactive_rollback.py \
  --thermo-dir cases/new-backend-matrix/synthetic-pr --kind chemistry \
  --output cases/new-chemical-rollback
```
