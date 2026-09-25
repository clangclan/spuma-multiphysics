# 무거운 셀 한 launch 실행과 모세관 루프 GPU 상주 (2026-09-26)

기준은 main 3c7060f(PR #8, `closureHemPath direct`)다. [200–500 μs 분석](benchmark-200to500us-ideas-20260926.ko.md)의 1번(무거운 셀 모으기)과 5번(모세관 루프 장치 상주)을 구현했다. 두 변경 모두 결과를 바꾸지 않아야 한다는 조건을 두었다.

## 결과

40 bar 160³ 사례에서 체크포인트부터 10스텝을 진행하고, 2–10스텝 평균을 냈다. 최종 `reactiveState.bin`을 기준 실행과 바이트 단위로 비교했다.

| 시작 | 경로 | s/스텝 | 복원 | HEM 커널 | 최종 상태 |
|---|---|---:|---:|---:|---|
| 500 μs | 기존 `direct` (1M 배치 4개) | 2.410 | 1.614 | 0.936 | 기준 |
| 500 μs | `direct`, 전체 셀 한 launch | 2.179 | 1.384 | 0.658 | 비트 동일 |
| 500 μs | **`resident`** | **1.712** | 0.943 | 0.653 | **비트 동일** |
| 200 μs | 기존 `direct` | 1.874 / 1.935 (재실행) | 1.109 / 1.143 | 0.467 | 기준 |
| 200 μs | `direct`, 전체 셀 한 launch | 1.974 | 1.179 | 0.456 | 비트 동일 |
| 200 μs | **`resident`** | **1.547** | 0.753 | 0.453 | **비트 동일** |

- `resident` 경로는 500 μs에서 **−29%**, 200 μs에서 **−18~20%**다. 캠페인 시작(PR #7 이전 `reference` 7.78 s)과 비교하면 200 μs에서 약 5배 빠르다.
- 같은 빌드를 다시 돌리면 약 3% 흔들렸다(1.874 → 1.935 s).
- 모든 실행에서 CPU fallback과 장치 실패는 0이었다.
- 200 μs에서는 한 launch만 쓴 `direct`가 오히려 약 0.04 s 느렸다. 액체 셀이 적어 꼬리가 짧고, 호스트 쪽 비용이 흔들린 몫이 더 컸다. `resident`에서는 이 호스트 비용이 사라진다.

## 1. 무거운 셀을 한 launch에 (`closureHemBatchCells`)

- **이전:** 직접 HEM은 1M 셀 배치로 나눠 순서대로 돌았다. 액체 셀이 배치 2·3에 흩어져 있어, 복원마다 꼬리를 두 번 기다렸다.
- **변경:** HEM 용량 상한 1,048,576을 없앴다. HEM 순서 배열의 상한은 2³²−1이며, resident 기하 정렬은 `INT_MAX` 셀 이하를 명시적으로 검사한다. 실제 용량은 메모리 예산으로 제한된다. `closureHemBatchCells` 기본은 `direct`에서 `thermoBatchCells`, `resident`에서 전체 셀이다.
  - 일반 핸들은 생성 시 호스트 배치용 장치 staging을 할당하고, resident 전용 핸들은 모델·phase cache·셀별 카운터만 할당한다. 일반 핸들의 모세관 입력은 첫 사용 시 할당되므로 솔버가 그 크기를 메모리 예산에 미리 예약한다.
  - 풀의 배치 용량은 그대로 `thermoBatchCells`다. 장치 실패 뒤 풀로 진단을 다시 돌릴 때는 풀 용량 단위로 나눠 호출한다.
- **효과 (500 μs Nsight):** 반복 0이 356 ms(배치 4개)에서 215 ms(한 launch)로 줄었다. 액체 꼬리가 기체 셀과 겹친다.
- **남은 병목:** 215 ms는 이제 꼬리가 아니라 기체 셀 처리량(FP64)이다. 액체 활성 셀을 정렬 맨 앞에 두는 키 변경도 시험했지만, 1.712 → 1.714 s로 효과가 없어 되돌렸다(`b500-res7-resident`).

## 2. 모세관 루프 GPU 상주 (`closureHemPath resident`)

호스트는 외부 반복 제어만 맡는다. 셀별 배열은 모두 수송 라이브러리 안의 장치 버퍼에 있다(`src/reactiveTransport/reactiveCapillaryResident.h`).

| 단계 | 장치에서 하는 일 | 호스트 코드의 같은 연산 |
|---|---|---|
| begin | 상태·q·호스트 `internalEnergy`를 1회 업로드한다. 들어온 상태의 색을 계산한 뒤 seed의 액체량을 q로 바꾸고, 기하를 만든다. | `refreshInterface`, `capillarySeeds` |
| select | before 배열을 저장하고 dirty를 판정한다(flashed 값과 비트 비교, 반복 0은 전부). HEM 입력 `bulk−surface`와 모세관 입력을 만든다. dirty 셀을 heavy-first 안정 정렬(CUB radix)한다. | dirty 목록, pack |
| HEM | `reactive_gpu_hem_run_resident_v1`이 장치 포인터와 순서 배열로 기존 `hemKernel`을 돌리고, 결과를 셀 위치에 쓴다. 카운터와 성공 수만 내려받는다. | 직접 HEM 배치 |
| update | flashed 입력을 기록하고, 새 상태로 색·기하·jump를 다시 만든다. 잔차는 비음수 double의 비트 max로 구한다. | commit, `liquidColor`, `interfaceGeometry`, 잔차 |
| relax | `.5*(before+candidate)`로 기하를 다시 만든다. | 동일 |
| commit | 수렴한 상태를 장치에서 연속 배열로 모은 뒤 한 번에 내려받는다. 색·표면 에너지·곡률도 받고, 기하 shadow를 갱신해 뒤따르는 호스트 요청이 재사용하게 한다. | `states`·계면 배열 |

**비트 동일을 지키는 방법:**
- 장치 연산은 모두 호스트 식을 그대로 옮긴 단일 IEEE 연산이다(뺄셈, 곱, 나눗셈, abs, max). 수송 TU는 `--fmad=false`라 FMA로 합쳐지지 않는다.
- nvc++에서 FMA가 섞일 수 있는 `internalEnergy`는 호스트가 계산해 올린다. 이 함수의 마지막 연산은 나눗셈 결과를 빼는 것이라, 뒤따르는 `−surface`는 합쳐질 수 없다.
- 정렬 순서는 셀 계산에 영향을 주지 않는다.

**호스트 경로로 되돌아가는 경우:**
- 대상: 호스트 경로가 거부할 입력(색 범위, 상 부피 합, 비유한 jump·잔차), 장치 HEM 실패, 40회 미수렴.
- 동작: `resident`는 q·상태·논리적 반복/flash 셀 카운터를 바꾸지 않은 채 false를 반환한다. 풀 경로가 같은 복원을 처음부터 다시 풀어 기존 진단·롤백을 낸다. 실제 실행한 HEM의 시간·전송·장치 실패 카운터는 버리지 않고 누적하며, 스텝 로그에 `residentFallbacks`를 추가한다. 장치 셀 실패는 풀 재실행이 성공해도 원래 시도를 롤백한다. CUDA API 오류는 풀 재실행 없이 기존 예외 처리로 즉시 롤백한다.

**검증 (500 μs, 10스텝, 모두 기준과 비트 동일):**

| 실행 | 확인한 것 |
|---|---|
| `resident` | 기본 동작 |
| `REACTIVE_CAPILLARY_RESIDENT_FALLBACK=0` | begin 뒤 매번 호스트 경로로 전환(20/20회). 외부 반복·flash 셀 수가 `resident`와 같다. |
| `REACTIVE_CAPILLARY_RESIDENT_FALLBACK=1` | 반복 0의 장치 HEM까지 끝난 뒤 전환. 카운터 복원을 확인했다. |
| `REACTIVE_WALE_STATES_CHECK=1` | WALE 상태 재사용 epoch가 상주 commit 뒤에도 맞다. |

**500 μs 복원 1회 (Nsight, 473 ms):**

| 항목 | 시간 [ms] |
|---|---:|
| HEM 반복 0 | 215 |
| HEM 반복 n (dirty 약 4,400셀, 액체 꼬리) | 114 |
| begin (상태 721 MB 업로드 포함) | 48 |
| commit (상태 721 MB + 계면 3×33 MB 다운로드) | 65 |
| 기하·select·update·입력 준비 | 약 30 |

이전 `direct` 경로의 복원은 약 800 ms였다. 호스트 루프(색 3회, 기하 3회, 잔차 2회, dirty·pack·commit)와 HEM 입출력 복사가 빠졌다.

**중간에 고친 것.** 첫 구현은 commit에서 `cudaMemcpy2DAsync`(184 B 간격 → 176 B)를 pageable 메모리로 직접 받았다. 이 복사가 2.7 GB/s(264 ms)라 오히려 느려졌다(`b200-res4-resident` 1.952 s). 장치에서 연속 배열로 모은 뒤 1D로 받도록 바꿨다(`res5`).

## 일반성 점검 후 수정 (res9)

벤치마크 조건에 묶인 부분 세 가지를 고쳤다. 500 μs 10스텝에서 모두 기준과 비트 동일했다.

| 문제 | 수정 | 검증 |
|---|---|---|
| `direct`의 기본 launch를 전체 셀로 바꿔, 큰 격자에서 호스트 경로 staging(셀당 약 0.5 KB)이 메모리를 넘을 수 있었다. | `direct` 기본은 원래대로 `thermoBatchCells`다. `resident`는 셀당 카운터만 두는 전용 핸들(`reactive_gpu_hem_create_resident_v1`)을 쓰고 기본이 전체 셀이다. 값을 줄이면 launch만 나뉜다. 장치 메모리는 추정 대신 핸들 조회값(`reactive_gpu_hem_device_bytes_v1`)으로 예산에 넣는다. | `direct` 2.398 s. `resident` 1M 분할 1.872 s, 전체 1.747 s |
| `resident`에서 호스트 경로로 되돌아갈 때 직접 HEM 핸들로 전체 셀 staging을 새로 할당할 수 있었다. | 되돌아가면 미리 할당된 풀(같은 커널)로 다시 푼다. 장치 실패였다면 재실행이 성공해도 롤백한다(`direct`와 같은 규칙). | 반복 1 강제 전환 20/20회 |
| heavy-first 정렬 키가 화학종 4개에서만 동작했다(`ns==4`, 기존 코드). 다른 화학종 수에서는 정렬이 꺼졌다. | 화학종 수에 무관한 키(`reactiveHemBucket.h`)로 바꿨다. 순서는 활성 액체 수, 존재 화학종 수, 지배 화학종(인덱스 7 이상은 한 칸) 순이다. 셀별 계산은 독립이라 결과에는 영향이 없다. | 성능 차이 없음 |

## 메모리

`resident`는 셀당 약 0.5 KB의 장치 버퍼를 수송 예산(`maxDeviceMemoryGB`)에서 할당한다. 4.1M 셀에서 약 2.1 GB다. 대신 직접 HEM의 호스트 경로 staging은 할당하지 않는다. main 반영 전 500 μs 이후 짧은 실행에서 장치 전체 사용량을 샘플링한 최대값은 `direct` 8,742 MiB, `resident` 10,526 MiB, resident 1M 분할 10,328 MiB였다. resident 증가는 약 1.74 GiB다. 이는 장치 전체의 샘플링 최대값이며 순간 피크를 보장하지 않는다. launch 크기를 줄여도 전체 셀의 모세관 작업 버퍼는 유지된다.

## 사용법

```
closureSearch   stableGasPrune;
closureHemPath  resident;      // pool(기본) | direct | resident
// closureHemBatchCells: resident 기본 전체 셀, direct 기본 thermoBatchCells
```

`resident`의 조건:
- 모세관, strict CUDA HEM, `closureCpuFallback false`
- diffuse 기하(`cartesianImplicit` 불가), 비기계(mechanical off)

## 남은 것

- **반복 0의 기체 셀 처리량(215 ms/복원).** FP64 한계라 공기 셀 안정성 검사 생략(결과 불변 여부는 재생으로 확인 필요)이나 혼합 정밀도가 다음 후보다.
- **반복 n의 액체 꼬리(114 ms/복원).** 정체 seed 조기 종료 진단, 후보 병렬 재측정이 남아 있다.
- **복원마다 상태 721 MB 업로드와 다운로드(약 0.11 s/복원).**
  - PCIe x16 복구로 줄어든다.
  - RK 두 단계가 같은 시작 상태에서 복원하므로, 장치에 그 상태를 남겨 두면 업로드를 줄일 수 있다. 다만 호스트 상태와 같다는 것을 보장할 규약이 필요하다.
- **CFL(0.22 s/스텝)과 `Advance`(0.13 s/스텝).** 분석 보고서의 3·4번이다.

## 자료

- 비교표: `results/capillary-resident-20260926/comparison.txt`
- 스텝 로그: `results/capillary-resident-20260926/*-steps.log`
- 스크립트: `results/capillary-resident-20260926/cmp.py`, `bench500.sh`
- 실행: `runs/gpu_opt_validation_20260925/b{200,500}-res*`, Nsight `p200-res4-resident-nsys`, `p500-res6-resident-nsys`

## main 반영 전 독립 리뷰와 재검증

초기 제출본을 그대로 병합하지 않고 다음 결함을 수정했다.

1. **실패/시간 카운터 유실:** resident를 포기할 때 `directHemProfile`을 되돌리고 실패 배치를 합산 전에 반환하여, 실제 GPU 셀 실패와 버려진 연산 시간이 로그에서 사라졌다. 논리적 반복·flash 셀 카운터만 복구하고 실제 실행 비용·실패는 누적하도록 고쳤다. 시험용 라이브러리로 transient GPU 셀 실패를 주입해 재현 경로를 검증했다.
2. **부분 할당 후 잘못된 버퍼 재사용:** resident 작업 공간을 모두 할당하기 전에 cells/species 태그가 설정되어, 메모리 부족 후 재시도가 미완성 버퍼를 정상 공간으로 볼 수 있었다. 완성된 공간만 공개하고 실패 시 새 할당을 전부 회수한다. 1·5·16개 화학종에서 중간 예산 초과를 두 번씩 일으켜 메모리 사용량이 원래대로 돌아오는지 확인했다.
3. **지연 할당 예산 누락:** direct HEM의 첫 모세관 배치에서 추가되는 입력 버퍼를 수송 메모리 예산 계산 전에 예약한다.
4. 원본 벤치마크 실행 스크립트가 솔버 실패를 성공 종료처럼 처리하던 부분을 고쳤다. 관련 분석의 고정 dt 표현과 액체 질량 단위(μg → mg)도 바로잡았다.

검증 결과는 `results/capillary-resident-20260926/main-review/summary.json`과 같은 디렉터리의 개별 기록에 있다.

| 검증 | 결과 |
|---|---|
| 기존 main 스냅샷 및 pool/direct/resident 정상·분할·WALE 검사·반복 0/1 재실행 | 500 → 500.15 μs 최종 상태 비트 동일 |
| GPU 셀 실패 주입, pool/direct/resident | 각 1회 실패와 1회 롤백 기록, 복구 후 상태 비트 동일 |
| CUDA API 오류 주입, pool/direct/resident | 각 1회 롤백, 복구 후 상태 비트 동일 |
| 위 솔버 실행 | 총 15개, CPU fallback 0 |
| resident 버퍼/정렬/기하 계약 | 1·5·16종, 35셀 워프 경계, 변경 셀 0, 캐시, 부분 할당 실패 재시도: 12개 항목 통과 |
| Compute Sanitizer memcheck | 같은 계약 검증에서 오류 0 |
| 곡면/동결/평형 flash | 20개 항목 통과 |
| WALE PR epoch v1/v2 | 각각 73개 항목 통과 |
| CPU/CUDA 수송 연산 | 각각 29개 항목 통과 |

최종 빌드로 200·500 μs에서 각각 10스텝을 다시 실행했다. 기존 direct 실행의 최종 체크포인트와 바이트 단위로 일치했고, 첫 스텝을 제외한 평균은 다음과 같다.

| 시작 | 기존 direct [s/스텝] | 최종 resident [s/스텝] | 감소 |
|---|---:|---:|---:|
| 200 μs | 1.874 | 1.512 | 19.3% |
| 500 μs | 2.410 | 1.720 | 28.6% |

`closureHemPath` 기본값은 계속 `pool`, `closureSearch` 기본값은 계속 `reference`다. resident 최적화는 명시적으로 선택한다. 200–500 μs 장기 캠페인은 기존 main의 direct 경로로 완료한 계산이며, 새 resident 경로의 300 μs 장기 검증으로 해석하면 안 된다. 다종 버퍼 계약 검증도 해당 종수의 EOS·충격파·연소 물리 검증을 대신하지 않는다. 이 리뷰는 연소 GPU 지원이나 정적 액적 수렴 문제를 해결했다고 주장하지 않는다.
