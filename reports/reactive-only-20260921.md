# ReactiveFoam 기본 솔버 정리 — 2026-09-21

`selection` 커밋 `96a781c`를 포함하는 `main` (`49d476c`)에서 ColdFoam을 제거했다. `src/reactiveFoam`, `src/reactiveThermo`, `src/reactiveTransport`의 파일 내용은 selection과 동일하다.

## 변경

- ColdFoam 소스·물성 및 전용 검사 실행파일, VOF 케이스 생성기·가스 실험·정밀도/정수/비동기 smoother 벤치마크를 제거했다.
- `Allwmake`는 ReactiveFoam 백엔드·수송 라이브러리·실행파일을 빌드한다. 기본은 CPU 수송이며 `PINTLE_REACTIVE_TRANSPORT_BUILD=cuda`로 CUDA 수송을 빌드한다.
- `README.md`, `README.ko.md`의 첫 진입점을 ReactiveFoam으로 통일하고, 연산 선택·빌드·실행·현재 한계를 연결했다.
- `tools/benchmark.py`는 현재 검증 도구가 사용하는 공통 환경·필드 입출력 함수만 유지했다. 남긴 함수/클래스 16개의 AST는 기존 구현과 같다. ColdFoam 생성기에 있던 공통 OpenFOAM 헤더 함수를 이 모듈로 옮겼다.
- ReactiveFoam 결과 수집기의 ColdFoam 회귀 입력 및 과거 바이너리·소스 불변 조건을 제거했다. 기존의 실패한 물리 정확도 결과는 그대로 보고한다.
- 과거 보고서·결과는 해당 커밋의 기록으로 보존하고 `reports/README.md`에서 현재 기능과 구분했다. 이전 소스는 Git 이력에서 조회할 수 있다.

## 검증

ColdFoam 빌드 산출물이 없는 별도 worktree에서 수행했다. SPUMA v2512/NVHPC 26.5/CUDA 13.2, RTX 5080, 기존 Cantera/Sundials 환경과 `reactive-thermo-v4` 입력을 사용했다. 실행·빌드는 공통 잠금으로 직렬화했다.

| 확인 | 결과 |
|---|---|
| 새 `Allwmake` CUDA 빌드 | 통과 |
| 새 `Allwmake` 기본 CPU 빌드 | 통과 |
| README의 CPU 동결 음향파 예제 | 32셀 생성·실행 통과 |
| CUDA 빌드에서 CPU/CUDA 옵션·초기화·동결 상 수송·재시작 | 44개 통과 |
| 기본 CPU 빌드의 옵션·초기화·동결 상 수송·재시작 | 27개 통과 |
| 기존 HEM 런타임 | CPU 14개, CUDA 14개 통과 |
| 기존 mechanicalEquilibrium 런타임 | CPU 8개, CUDA 8개 통과 |
| 공통 필드 I/O·실유체 검증 근거 단위 테스트 | 8개 통과 |
| 남은 Python 모듈 import | 31개 통과 |
| 결과 수집기 | ColdFoam 입력 없이 기존 자료 수집 성공; 기존 campaign 실패 판정 유지 |
| 소스·문서 | selection 솔버 코어 동일, 기본 문서 링크·셸/Python 구문·diff 공백 확인 |

[검증 집계](../results/reactive-only-20260921/validation-summary.json)와 같은 디렉터리의 개별 결과·실행 명령을 보존한다. 전체 검증 명령은 `PINTLE_SPUMA_ENV=/home/jsw/cae-gpu-pr1-compatible/env.sh`, `PINTLE_REACTIVE_PREFIX=/home/jsw/문서/analysis/spuma-multiphysics/research/reactive-env`를 사용했다.

이번 변경은 새로운 표면장력·액적 모델·난류 모델·MPI를 추가하지 않는다. 이전 selection 검증과 현재 솔버의 물리 모델 한계가 유지된다.
