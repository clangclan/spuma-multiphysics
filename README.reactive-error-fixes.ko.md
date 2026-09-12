# ReactiveFoam 컴파일 오류 수정·검사 보강 — 2026-09-12

리뷰가 지적한 `Flow::step() const`의 일반 멤버 변경 오류를 수정했다. 실제 소스에서 추출한 선언·증가식은 수정 전 컴파일 실패, 수정 후 통과했다. 희소 LU 패턴 변경과 CVODE 작업공간 수명 검사도 보강했다. **SPUMA 전체 빌드와 실제 GPU 실행은 아직 미검증이며 PR은 draft다.**

## 이전 작업과 이번 변경

| 구분 | 상태 |
|---|---|
| 원래 실험 | `50512ce…`, Ryzen 9600X·RTX 5080 환경. 이번 검사를 그 환경의 실행 결과로 취급하지 않음 |
| 최초 GPU·희소 포팅 | `fad4ecf`; [최초 기록](README.reactive-gpu-sparse.ko.md) |
| 이전 리뷰 개선 | `c26e9c9965145a4d0f1bf0253a61f9d2a5c6b2e0`; [전송·희소 준비 비용 개선](README.reactive-review.ko.md) |
| 이번 수정 | 위 커밋의 `step()`에서 `const` 제거, 회귀 검사·문서 보강 |
| 검증 환경 | 별도 Cantera 3.2.0 / SUNDIALS 7.4.0 / Eigen 3.4.1, g++ 13.3.0. SPUMA/`wmake` 없음 |

리뷰·소스·검사 결과의 SHA-256은 [환경 기록](results/reactive-error-fixes-environment-20260912.json)에 남겼다. 수송·화학 수치 백엔드는 `c26e9c9`와 같다. GPU 열역학·배치 화학·축소 기구 도입은 이번 수정 범위에 포함되지 않는다.

## 컴파일 오류와 수정

`transportVersion`은 일반 멤버인데 `const`인 `step()`이 세 번 증가시켰다. 함수 본문은 실행 시 선택하는 CPU/CUDA 분기와 관계없이 컴파일되므로 전체 솔버의 빌드를 막는 오류다. 시간 적분과 실행 버전을 변경하는 함수라는 계약에 맞게 `step()`에서 `const`를 제거했다. 호출부의 `Flow` 객체는 비상수이므로 호출 방식을 바꿀 필요가 없다.

새 [검사 도구](tools/validate_reactive_flow_contract.py)는 실제 솔버 파일에서 멤버 선언, 함수 선언, 증가식을 추출해 최소 C++ 단위로 컴파일한다. 별도로 수작업한 동일 모양의 예제만 검사하지 않는다. `mutable`로 계약을 우회하는 변경도 거부한다.

| 검증 | 결과 |
|---|---|
| 수정 전 소스의 변경 계약 | 세 증가식 모두 `increment ... in read-only object`; [원본 실패 기록](results/reactive-const-before-20260912.json) |
| 수정 후 동일 검사 | 컴파일 성공; [결과](results/reactive-const-after-20260912.json) |
| 전체 `Flow` 본문·SPUMA 헤더·링크 | 미검증 |

이전 백엔드 검사와 CUDA 수송 라이브러리 컴파일은 `pintleReactiveFoam.C`를 컴파일하지 않았다. 따라서 그 검사들의 통과로 이번 오류를 발견할 수 없었다. 새 최소 검사도 **전체 솔버 빌드를 대체하지 않는다.**

## 보강한 수치 검사

`tools/test_reactive_sparse_cache.cpp`는 실제 백엔드 구현을 포함하는 별도 시험 라이브러리다. 운영 라이브러리에 시험용 API나 오류 주입 분기를 추가하지 않았다.

| 추가 검사 | 내용과 결과 |
|---|---|
| LU 패턴·대각 삽입 | 합성 3×3 블록을 413종 크기에 배치하고 나머지 대각은 실제 코드가 삽입. 대각 누락 → 같은 `nnz`의 다른 패턴 → 같은 패턴의 값 변경·0 교차를 검사 |
| LU 독립 대조 | 매 단계에서 `I − gamma S` 자체, 밀집 LU 풀이, 독립 `Ax − b` 잔차 대조. 상대 풀이 오차·잔차 `<1e-12`. 구조 분석 정확히 2회, 수치 분해 3회 |
| 적분 구간 왕복 | `1e-9 → 1e-7 → 1e-9`초, 같은 초기 상태·작업공간 |
| 허용오차 왕복 | 기본 `(rtol, atol)=(1e-8,1e-14)` → `(1e-7,1e-13)` → `(1e-9,1e-15)` → 기본값 |
| 조성·상태 왕복 | A: 1400 K·1 bar·N2 희석 25 → B: 2600 K·20 bar·N2 희석 200 및 CO2/H2O → A |
| 새 객체와 비교 | 밀집 전체 RHS·밀집 구조화·엄격 희소 각각 8회, 총 24회. 종 질량분율·온도 차이는 기록된 정밀도에서 0 |
| 실제 CVODE 실패 후 복구 | 실제 화학 RHS를 평가한 뒤 시험용 비선형 RHS callback이 `-1` 반환. 밀집·희소에서 CVODE 실패 코드 확인, 호출자 배열·상태·drift 보존, 실패 객체 소멸 확인 |
| `auto` 실패 복구 | 실패한 희소 객체를 폐기하고 원래 초기 상태로 밀집 재적분. 이후 source는 새 희소 객체 생성. 각각 새 객체 풀이와 차이 0 |

오류 주입은 SUNDIALS의 공개 `CVodeSetNlsRhsFn()`을 사용한다. 임의의 잘못된 입력을 API 입구에서 거부한 것을 CVODE 복구 시험으로 세지 않았다. 음수 RHS 반환에 따른 중단은 [SUNDIALS의 callback 계약](https://sundials.readthedocs.io/en/v7.4.0/cvode/Usage/index.html#ode-right-hand-side)에 따른다. 실제 화학기구에서 자연 발생하는 모든 실패를 재현했다는 뜻은 아니다.

### 검사 과정에서 수정한 가정

초기 수명 검사는 source당 재초기화가 항상 한 번이라고 가정해 실패했다. [초기 결과](results/reactive-error-review-20260912.json)와 [조건을 명시한 진단 기록](results/reactive-lifetime-diagnostic-20260912.json)을 보존했다.

원인은 밀집 구조화 경로의 완화된 허용오차 조건에서 기존 채택 상태 검사가 작동하여, 같은 작업공간을 전체 RHS Jacobian으로 다시 초기화한 것이었다. 검사에서는 `integrationFallbacks`를 별도로 읽어 **기본 재초기화 + 실제 재적분 횟수**와 정확히 일치하는지 확인하도록 고쳤다. 새로운 객체를 생성해 재사용을 피하거나 수치 허용오차·채택 상태의 양수성 기준을 완화하지 않았다. 최종 24회 기록은 객체 생성 3회, 재초기화 22회, 밀집 재적분 1회다.

수명 그룹만 다시 실행한 [최종 결과](results/reactive-lifetime-final-20260912.json)는 통과했다. 초기 실행에서 나머지 7그룹은 통과했으며, 변경하지 않은 그룹을 다시 실행하지 않았다. [최종 집계](results/reactive-error-fixes-summary-20260912.json)는 각 그룹의 결과 파일을 연결한다. 이는 한 번의 실행에서 8그룹 모두 통과한 기록과 구분된다.

## 기존 성능 설명의 범위

- 보존장 전송의 `7Q → 3Q`는 그 부분만 57.1% 감소다. 확산의 `gasY/gasH`를 포함하면 `7Q+4G → 3Q+4G`이며 413종·417변수에서는 작은 상태 배열을 제외하고 약 36.5% 감소다. 보존장 device 저장공간은 계속 네 벌이다.
- 내용 버전은 호출자가 변경 때 증가시켜야 한다. 같은 버전의 다른 내용을 hash로 검출하는 기능은 없다.
- `kineticsSeconds`에는 반응 미분과 단위 변환·부분 열역학·rank 벡터 구성 등이 포함된다. 순수 반응률 계산 시간으로 해석하지 않는다.
- 20 bar 기록과 구조가 준비된 `auto` 기록은 단발 진단이다. 반복 성능 측정이나 GPU 가속률이 아니며, 1 bar 조건의 엄격 희소 거부·느린 `auto`도 그대로 남는다.

## 재현과 남은 검증

새 출력 경로를 사용한다. 원래 실험 호스트에서는 기존 공유 실행 잠금 규칙을 따른다.

```bash
python3 tools/validate_reactive_flow_contract.py \
  --output results/local-flow-contract.json
PINTLE_REACTIVE_CACHE_TEST=1 bash tools/build_reactive_backend.sh
research/reactive-env/bin/python tools/validate_reactive_review.py \
  --thermo-dir research/reactive-thermo --backend cpu \
  --output results/local-error-review.json
# 보강한 그룹만 별도 실행할 경우
research/reactive-env/bin/python tools/validate_reactive_review.py \
  --thermo-dir research/reactive-thermo --backend cpu \
  --checks sparse-cache-callbacks workspace-lifetime CVODE-failure-recovery \
  --output results/local-workspace-review.json
```

후속 필수 확인은 원래 고정 의존성 환경의 SPUMA 전체 CPU/CUDA 빌드, 실제 GPU 실행·Compute Sanitizer, 닫힌 메시의 화학 반스텝·RK·flash·거부/rollback·재시작 결합 시험이다. 이번 결과는 소스의 특정 컴파일 오류 및 CPU 백엔드의 수명 계약을 확인한 범위로 한정한다.
