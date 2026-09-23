# 고체·승화 제외 및 과냉각 액체 전환 — 2026-09-22

사용자 지정에 따라 현재 N₂O 충돌 벤치마크에서 고체상·승화를 제외하고 **기체 + 과냉각 액체 N₂O**만 계산하도록 바꿨다. 동일한 256k 메시에서 GPU 10스텝 실행 시간은 대기압 **151.16 → 11.38 s**, 40 bar **65.79 → 20.51 s**로 줄었다. 두 환경 모두 재시도·CPU fallback·GPU 복원 실패가 없었다.

## 변경 범위

- 응축상 슬롯을 액체 N₂O 하나로 줄였다. 고체 후보, 기체–고체·액체–고체 화학퍼텐셜 비교, 승화·융해잠열 계산이 현재 모델에서 제외된다.
- 액체 N₂O PR 가지의 최소 온도를 기존 182.34 K에서 **148 K**로 낮췄다. N₂/O₂/N₂O의 준비된 148 K NASA9 구간은 그대로 사용한다. 질량·에너지를 보정하거나 온도를 clipping하지 않는다.
- 증발·응축과 기체–액체 HEM 상평형은 유지한다. 고체를 억제한 유체 상평형이며, 액체만 강제로 유지하는 동결 상분배는 아니다.
- 고체 프로필에서 필요했던 GPU 유한차분 Jacobian을 제외하고 기존 **분석 Jacobian**을 다시 사용한다. CPU 복원은 기존 유한차분 방식이다.
- 저온 기체 종의 물성 범위 검사를 고체 활성화 여부에서 분리했다. `enforce-species-temperature-bounds: true`를 CPU/CUDA에 전달하고 물리 해시에 반영한다. 따라서 IPA가 양수인 상태의 저온 외삽을 허용하지 않는다.
- 반응·IPA 액상·고체 입자 해석은 사용하지 않는다. 기구는 비반응이며 초기장과 경계의 IPA 질량은 0이다. WALE stress-v1, 메시, 공급압, 배압, CFL 설정은 유지했다.

이전 고체 코드와 입력은 과거 결과 재현용으로 남겨두지만 **현재 프로필의 `solidN2O=false`, 응축상 수는 1**이다. 고체 물성이나 안정성을 계산하지 않는다. 기존 2 bar 고체 적용 제한은 이 프로필에 적용되지 않으며, 기체·액체는 설정된 PR 모델 영역을 따른다.

## 물리적 의미

삼중점 아래 액체는 고체 생성이 억제된 **준안정 과냉각 액체 근사**다. PR의 액체 root를 연속적으로 사용하되, 기계적으로 허용되는 root가 없거나 148 K 하한 밖이면 계속 거부한다. 고체의 진짜 안정성, 핵생성 시간, 동결·융해열을 예측하지 않는다. 따라서 이전 고체 모델과 저온 분율·온도가 달라지는 것은 모델 변경의 결과다.

[Cantera PR API](https://cantera.org/stable/cxx/d3/ddc/classCantera_1_1PengRobinson.html)는 기체/액체 EOS 가지를 지정하는 기능을 제공한다. 강제 준안정 EOS 상태와 상경계 밖 결과의 해석에 대해서는 [NIST REFPROP 문서](https://www.nist.gov/system/files/documents/2018/05/23/refprop10a.pdf)를 참고한다. 이번 저온 결과는 과냉각 PR 연장의 수치적 작동 검증이며, 삼중점 아래 실험 물성 검증은 아니다.

이전 승화압 정확도 9.39%/허용 8% 미달 기록은 [고체 보고서](impinging-n2o-solid-recovery.ko.md)에 그대로 보존한다. 해당 게이트를 완화하거나 통과로 변경한 것이 아니라 사용자 요청으로 현재 모델에서 고체 물리를 제외했다.

## 실제 GPU 10스텝 비교

80×80×40, 0.25 mm 정육면체 셀 256,000개와 55 barg·293.15 K 공급을 사용했다. 각 환경에서 한 번씩 10스텝으로 제한했다. 다음 시간은 초기화·출력 등을 포함한 프로세스 실행 시간이며 반복 평균은 아니다. 고체 모델을 다시 실행하지 않고 보존된 결과와 비교했다.

| 환경 | 이전 고체 포함 | 현재 과냉각 액체 | 속도 향상 | 시간 감소 | 현재 최저 T | 현재 dt 범위 |
|---|---:|---:|---:|---:|---:|---:|
| 1 atm | 151.162 s | **11.375 s** | **13.29배** | **92.47%** | 165.637 K | 25.884–26.136 ns |
| 40 bar(abs) | 65.786 s | **20.505 s** | **3.21배** | **68.83%** | 222.335 K | 25.636–25.855 ns |

현재 GPU HEM 카운터는 다음과 같다.

| 환경 | flash 후보 수: 이전 → 현재 | 분석 Jacobian | 유한차분 Jacobian | GPU kernel 합계 |
|---|---:|---:|---:|---:|
| 1 atm | 2,424,495 → 793,340 | 449,773 | **0** | 9.453 s |
| 40 bar(abs) | 2,368,755 → 788,975 | 11,383,841 | **0** | 18.538 s |

두 케이스 모두 CPU fallback 0, device failure 0, 재시도 0이다. 대기압 한 스텝 시간은 0.404–1.813 s로, 이전 최대 72.03 s가 사라졌다. 단, 40 bar는 여전히 많은 반복을 수행하며 이 작업에서 추가 탐색 최적화를 진행하지 않았다. 위 향상은 **물리 모델 축소와 분석 Jacobian 복귀의 합산 효과**로 해석한다.

최종 시각은 대기압 0.26101 μs, 40 bar 0.25833 μs다. dt가 30 ns보다 작은 이유는 현재 셀 크기·CFL 제한이며 재시도 때문이 아니다. 메시 품질 검사도 통과했다. 발달한 충돌류·액막 분열·액적 생성·1 ms 계산을 검증한 결과는 아니다.

## 검증과 저장

수치·단위 검사 **27/27**, 기존 유체 CPU/CUDA 복원 회귀검사 **18/18**이 통과했다. 단위 검사는 160/170/182/190 K 유체 공존, 160 K 순수 액체, 가열에 의한 증발·냉각에 의한 액체 증가, 공기 혼합, 공급 액체 상태, 148 K/IPA 범위 위반 거부와 CPU/GPU 일치를 포함한다. CPU/GPU 상태 오차는 음속을 포함해 최대 4.22e-9(스케일 정규화)였다. State ABI는 176바이트로 유지했고, 새 프로필은 이전 고체 프로필과 다른 물리 해시를 사용한다.

전체 GPU 스텝의 질량·에너지·종·원소 보존 수지 기준 1e-8 및 체크포인트·상 질량 검사를 통과했다. 대기압 최대 에너지 수지 잔차는 5.80e-12였다. 입력·기구·바이너리·소스 해시는 [전달 manifest](../../results/n2o-supercooled-20260922/delivery.json)에 있다.

- [단위·CPU/GPU 검사](../../results/n2o-supercooled-20260922/unit.json)
- [기존 유체 복원 검사](../../results/n2o-supercooled-20260922/legacy-cuda-recovery.json)
- [256k GPU 10스텝 검사](../../results/n2o-supercooled-20260922/gpu-ten-steps.json)
- [이전 모델과의 시간·카운터 비교](../../results/n2o-supercooled-20260922/comparison.json)

현재 케이스: `/home/jsw/문서/analysis/runs/impinging_n2o_supercooled_20260922/benchmark`. 실제 실행 결과는 `logs/n2o-supercooled-v1/ten-steps`에 있으며, 기존 고체 케이스·결과는 덮어쓰지 않았다.

## 재현 입력

저장소의 [과냉각 액체 예제](../../examples/impinging-n2o-supercooled/README.ko.md)를 사용한다. 빌드는 기존 CUDA 설정을 유지하며, 새 device 모델 이미지를 사용하므로 host backend와 CUDA 라이브러리를 함께 빌드한다.

```bash
export PINTLE_SPUMA_ENV=/home/jsw/cae-gpu-pr1-compatible/env.sh
export PINTLE_REACTIVE_PREFIX=/home/jsw/문서/analysis/spuma-multiphysics/research/reactive-env
export PINTLE_REACTIVE_TRANSPORT_BUILD=cuda
export PINTLE_CUDA_ARCH=120
flock /home/jsw/cae-benchmark/run.lock ./Allwmake

"$PINTLE_REACTIVE_PREFIX/bin/python" tools/prepare_impinging_n2o.py /path/to/new-benchmark \
  --configuration examples/impinging-n2o-supercooled/cold-pr-148K-config.yaml --cell-mm 0.25

"$PINTLE_REACTIVE_PREFIX/bin/python" tools/validate_impinging_n2o.py \
  --benchmark /path/to/new-benchmark --output /path/to/new-check --steps 10 --timeout 60
```

원본 182.34 K 기구에서 준비하려면 `extend_n2o_gas_thermo.py --configuration <원본 설정> --supercooled-liquid --output <새 폴더> --library lib/libpintleReactiveBackend.so`를 사용한다. 이전 고체 예제는 현재 기본 벤치마크가 아니다.
