# Pintle GPU 개발판 검증·성능 기록

3,094,455셀의 실제 Pintle 체크포인트에서 45.00→45.36 μs, dt=30 ns로 계산했다. 각 설정은 12스텝×3회이며, 각 실행에서 준비 영향을 받는 첫 2스텝과 마지막 출력 스텝을 제외한 9스텝의 산술평균을 구한 뒤 실행별 중앙값을 비교한다.

주요 상태·출력에 대한 **개발용 수치 screen: 통과**. 이 판정은 아래에 명시한 메인 에이전트의 허용치를 사용한다. 장시간 물리 모델 검증이나 Foundation 14와의 동등성 판정은 아니다.

[집계 JSON](validation.json) · [전체 원자료](/home/jsw/문서/analysis/spuma-multiphysics/benchmarks/validated-v2/benchmark.json)

## 성능

| 설정 | 계산 s/step 중앙값 | 실행별 범위 | 기준 대비 배율 | 전체 실행 s | GPU 사용률¹ | GPU 메모리 최대 MiB¹ |
|---|---:|---:|---:|---:|---:|---:|
| native-reference | 23.6238 | 23.5541–24.7746 | 1.00× | 290.67 | 35.9% | 9546 |
| device-gpu | 1.8629 | 1.8620–1.9608 | 12.68× | 36.86 | 83.5% | 9719 |
| device-mixed | 1.8697 | 1.8674–1.9689 | 12.64× | 36.86 | 83.5% | 9731 |

¹ `nvidia-smi` 장치 전체 표본이며 데스크톱 등 다른 프로세스를 포함한다. SM 점유율 측정치는 아니다.

`native-reference`도 SPUMA GPU 유한체적 연산과 multicolor smoother를 사용한다. CPU MULES·CPU 물성과 GPU limiter·GPU 물성을 동시에 바꾸는 비교이며, 전체 CPU 솔버 대 GPU 솔버의 비율이 아니다. 압력·온도·운동량 선형해법 허용오차는 동일하다.

## 단계별 계산 시간

| 설정 | alpha | 운동량 | 물성·온도 | 압력 | 난류 |
|---|---:|---:|---:|---:|---:|
| native-reference | 15.7128 | 0.4428 | 6.8332 | 0.7096 | 0.0152 |
| device-gpu | 0.6168 | 0.3713 | 0.1513 | 0.6951 | 0.0155 |
| device-mixed | 0.6197 | 0.3715 | 0.1515 | 0.6982 | 0.0154 |

단계 값도 실행별 평균의 중앙값이며 단위는 s/step이다. 전체 시간에는 위 단계 밖의 Courant·상태 진단 등도 포함된다.

## 최종 필드 차이

`max_scaled = max(abs(candidate-reference)) / max(1,max(abs(reference)))`. 아래 값은 동일 반복의 reference와 비교한 모든 반복 중 최댓값이다.

| 필드 | GPU FP64 | FP32 ratio 저장 | 개발용 허용치 |
|---|---:|---:|---:|
| p | 2.980e-10 | 4.187e-10 | 1e-05 |
| U | 8.313e-10 | 8.180e-10 | 1e-06 |
| T | 2.066e-10 | 2.018e-10 | 1e-06 |
| rho | 6.608e-10 | 1.170e-09 | 1e-06 |
| alpha.ipa | 4.454e-10 | 1.207e-09 | 1e-06 |
| alpha.n2o | 2.433e-08 | 2.985e-08 | 1e-06 |
| alpha.air | 2.433e-08 | 2.985e-08 | 1e-06 |

상태 건전성 screen은 검사한 최종 내부 필드가 모두 finite, p/T/rho>0, alpha∈[−1e−8,1+1e−8], phase-sum 오차≤1e−9, 스텝별 상대 질량 수지 오차≤1e−6이다. 미세한 alpha 음수는 원본 입력에도 있으며 0으로 잘라내 결과를 꾸미지 않았다.

- 최대 phase-sum 오차: 2.878e-13
- 최대 스텝별 상대 질량 수지 오차: 5.051e-08

## dgdt를 별도로 해석해야 하는 이유

raw dgdt는 압력 방정식에서 작은 차이의 상쇄와 phase 존재 여부에 민감하다. 아래 차이를 일반 필드 screen에서 제외한 사실을 명시한다. 모든 내부 필드가 동일하다는 결론은 내리지 않는다.

| raw 필드 | GPU FP64 max_scaled | FP32 ratio max_scaled |
|---|---:|---:|
| dgdt.ipa | 8.547e-01 | 8.547e-01 |
| dgdt.n2o | 6.778e-03 | 6.592e-03 |
| dgdt.air | 1.000e+00 | 1.010e+00 |

보조 진단으로 첫 반복의 최종장에서 압축성 source 성분 `S_i=alpha_i*(sum_j(alpha_j*dgdt_j)-dgdt_i)`를 계산했다. 이는 순간적인 대수식의 차이이며 상별 순차 업데이트를 재실행한 검증은 아니다.

- device-gpu/repeat-01: max|dt·ΔS|=2.181e-12, source 상대 L2=1.293e-09
- device-mixed/repeat-01: max|dt·ΔS|=2.970e-12, source 상대 L2=5.226e-09

## 단위 연산·프로파일 증거

- 물성: 193개 압력×127개 온도=24,511점, 12개 물성 값. native CPU 대비 최대 상대오차 7.015e−13, invalid=0.
- 실제 mesh 첫 alpha solve: limiter scaled 오차≤4.756e−14, 3상 limitSum 차이 0, explicitSolve 최대 절대차 2.221e−16.
- 기존 CPU smoother 2step Nsight trace: 68,804회 device synchronization, 69,188회 kernel launch, CPU↔GPU unified-memory 전송 약 39.07 GB. 동기화 시간에는 계산·메모리 이동 대기가 포함되므로 전체를 제거 가능한 overhead로 해석하지 않는다.
- GPU smoother 적용 후 준비 구간을 제외한 10초 trace의 unified-memory 양방향 전송은 합계 364.97 MB다. 위 trace와 수집 구간이 다르므로 두 전송량의 비율을 성능 배율로 해석하지 않는다. [추가 프로파일·메모리 검사 기록](runtime-checks.md)을 함께 참조한다.

## 기존 Foundation 14 기록과의 관계

당일 기존 CPU6 기록은 12.945791 s/step, 부분 CUDA 연산 기록은 11.806176 s/step이었다. 둘 다 동일 Pintle mesh·시점에서 12스텝×3회, 실행별 3–11번 스텝 평균의 중앙값이다. 이 개발판과는 MULES 반복·경계 flux 처리·열역학 재시작·압력 최소 반복 수가 다르므로 참고용 처리시간 비교만 가능하다. 동일 수치 결과의 가속 배율로 주장하지 않는다.

[기존 원자료](/home/jsw/문서/analysis/benchmarks/native_20260910/analyzed-runs.json)

## 제외한 실행과 남은 검증

`validated-v1`은 재시작 index=964 때문에 step 8에 출력하고 마지막 상태를 저장하지 못했다. 그 실행의 시간·필드 결과는 최종 표에서 제외했다. `writeControl runTime`으로 수정한 독립 확인과 `validated-v2`는 마지막 45.36 μs 출력 파일을 확인했다.

연소·화학종·증발/응축은 미구현이다. 장시간 안정성, 다중 GPU/MPI 실행, Foundation 14 수치법과 체크포인트 상태의 엄밀한 재현은 남아 있다. FP32 mode는 ratio 저장만 줄이며 FP16/BF16/Tensor Core를 사용하지 않는다. 성능 차이가 작거나 반복 범위가 겹치면 FP64를 기본값으로 유지한다.
