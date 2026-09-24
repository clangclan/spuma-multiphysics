# 범용 솔버 명칭과 기존 환경 전환

실행 파일은 계속 `ReactiveFoam`이다. 특정 분사기 형상과 관계없는 소스 파일, C++ 타입·네임스페이스, C API, 라이브러리, 환경변수에서 `pintle` 접두사를 제거했다. 물리식, 자료형 정밀도, 구조체 배치, 체크포인트 형식과 기본 물리 옵션은 바꾸지 않았다.

| 기존 이름 | 현재 이름 |
|---|---|
| `pintleReactiveThermo.cpp/.h` | `reactiveThermo.cpp/.h` |
| `pintleReactiveTransport.cpp/.cu/.h` | `reactiveTransport.cpp/.cu/.h` |
| `pintleDeviceFlash.h` 등 일반 모듈 | `reactiveDeviceFlash.h` 등 |
| `PintleThermoState`, `PintleDevicePR` | `ReactiveThermoState`, `ReactiveDevicePR` |
| `pintle_rt_*`, `pintle_transport_*`, `pintle_gpu_hem_*` | `reactive_rt_*`, `reactive_transport_*`, `reactive_gpu_hem_*` |
| `libpintleReactiveBackend.so` | `libreactiveBackend.so` |
| `libpintleReactiveTransport.so` | `libreactiveTransport.so` |
| `PINTLE_SPUMA_ENV` | `REACTIVE_SPUMA_ENV` |
| `PINTLE_REACTIVE_PREFIX` | `REACTIVE_PREFIX` |
| `PINTLE_REACTIVE_TRANSPORT_BUILD` | `REACTIVE_TRANSPORT_BUILD` |
| `PINTLE_GPU_ROOT` | `REACTIVE_PROJECT_ROOT` |
| `PINTLE_RUN_LOCK`, `PINTLE_CUDA_ARCH` | `REACTIVE_RUN_LOCK`, `REACTIVE_CUDA_ARCH` |
| 나머지 `PINTLE_*` 빌드 옵션 | `REACTIVE_*` |

같은 이름을 두 번 겹쳐 쓰지 않도록 `pintleReactive*` 파일은 `reactive*`로 줄였다. 전체 파일 이동 목록은 [검증 자료](../results/generic-naming-20260924/renames.json)에 있다.

## 빌드와 실행

기존 체크아웃에서 업그레이드하면 OpenFOAM의 `.dep`에 옛 헤더명이 남아 있을 수 있다. 한 번 `wclean`한 뒤 전체를 다시 빌드한다. 새 체크아웃은 바로 `./Allwmake`를 실행할 수 있다.

```bash
export REACTIVE_SPUMA_ENV=/path/to/spuma-env.sh
export REACTIVE_PREFIX=/path/to/reactive-env
export REACTIVE_TRANSPORT_BUILD=cuda
export REACTIVE_CUDA_ARCH=120  # 실제 장치에 맞춘다.
export REACTIVE_RUN_LOCK=/tmp/reactivefoam-run.lock
source ./env.sh
flock "$REACTIVE_RUN_LOCK" wclean src/reactiveFoam
flock "$REACTIVE_RUN_LOCK" ./Allwmake
flock "$REACTIVE_RUN_LOCK" ReactiveFoam -case /path/to/case
```

C 심볼 이름이 달라졌으므로 외부 C/C++/Python 클라이언트와 실험용 HEM 라이브러리도 새 헤더/API로 다시 빌드한다. Python 도구와 `dlsym` 로더는 저장소 안에서 함께 갱신했다. 예전 바이너리 심볼을 위한 별칭 라이브러리는 제공하지 않는다.

케이스에서 `closureLibrary` 또는 `closureScalarLibrary`에 라이브러리 경로를 직접 지정했다면 새 파일명으로 바꾼다. 생략한 경우에는 새 기본 라이브러리를 사용한다. 수치 정책이 다른 실험용 라이브러리로 바꾸면서 체크포인트의 해시를 편집해서는 안 된다.

## 보존한 과거 식별자

열역학 모듈의 fingerprint·physical model·case physics·numerical policy 생성에 들어가는 네 종류의 문자열은 기존 체크포인트와의 호환성을 위해 바이트 그대로 유지했다. 이는 형상 또는 물리 모델의 이름이 아니라 이미 사용된 저장 형식의 식별자다. 체크포인트 magic, version, 구조체 배치, `reactiveState*` 파일명도 같다.

날짜가 붙은 `reports/`·`results/`의 실행 로그, 명령, 해시, 절대 경로, 실제 과거 분사기 이름은 원자료다. 과거 로그의 라이브러리명까지 바꾸면 실제 실행 기록과 달라지므로 보존했다. 보고서의 저장소 소스 링크 대상만 이동된 파일로 연결했다. 과거 명령을 현재 소스에서 다시 실행할 때는 위 표로 환경변수와 API를 전환한다. `tools/legacy_artifacts.py`는 과거 라이브러리 경로와 커널 네임스페이스가 기록된 자료도 현재 분석 도구가 읽도록 한다.

최근 N₂O 메시·GPU 계측·HEM 최적화·정밀도 실험의 소스와 선별된 원자료도 같은 PR에 반영했다. 전체 결과는 [명칭 전환 검증](../results/generic-naming-20260924/validation.json)에 기록한다. 재생성 가능한 실행 파일·공유 라이브러리·대형 메시·체크포인트·프로파일러 바이너리는 로컬 실행 폴더에 보관한다.
