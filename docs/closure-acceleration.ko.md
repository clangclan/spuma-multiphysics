# 상평형 탐색 재사용과 CUDA 열량 후보

비반응 HEM 평형 모드의 반복 계산을 줄인다. EOS, 보존량, 상 안정성 검사, 후보 엔트로피 비교, 음속 검사는 유지한다. 전체 다상 flash나 화학반응을 GPU로 옮긴 구현은 아니다.

## 실행 옵션

`constant/reactiveProperties`:

```foam
thermoExactReuse true;
closureScalarBackend cpu;   // cpu 또는 cuda
thermoWorkers 24;
thermoBatchCells 4096;
```

`thermoExactReuse`를 생략하면 솔버는 비반응 HEM 평형 모드에서 켠다. 반응, 동결상, mechanical 모드에서는 기본적으로 끈다. 명시적으로 이들 모드에서 가속을 켜면 지원 범위 오류를 보고한다. `thermoExactReuse false`로 기준 경로와 비교할 수 있다. 검증용 `prepare_reactive_case.py`는 명시적인 기준 설정을 생성하므로 가속 케이스에는 `--thermo-exact-reuse`를 전달한다.

CUDA 온도 후보는 기본적으로 꺼져 있다. 활성화하려면 CUDA로 수송 라이브러리를 빌드하고 `closureScalarBackend cuda`로 지정한다. 필요하면 `closureScalarLibrary "/absolute/path/libreactiveTransport.so";`를 지정한다. 기본 라이브러리 이름은 실행 환경의 로더 검색 경로를 사용한다. 실행 관리기의 기존 수송 라이브러리 스냅샷에도 새 CUDA 함수가 포함된다. CPU 빌드 라이브러리로 CUDA를 요청하면 명시적으로 실패한다.

```bash
REACTIVE_TRANSPORT_BUILD=cuda REACTIVE_CUDA_ARCH=120 ./Allwmake
"$REACTIVE_PREFIX/bin/python" tools/prepare_reactive_case.py \
  cases/closure-example --thermo-dir research/reactive-thermo \
  --kind uniform --cells 32 --thermo-workers 8 \
  --thermo-exact-reuse --closure-scalar-backend cuda --transport-backend cuda
```

GPU 아키텍처와 열역학 입력 경로는 실제 환경에 맞춘다. `thermoWorkers 1`도 가속 옵션을 켜면 같은 배치 API를 사용한다.

## 정확한 탐색 재사용

1. 배치 안의 q 전체, 내부에너지, 초기 `ReactiveThermoState` 전체를 바이트 단위로 비교한다. 해시가 같아도 원본을 다시 비교한다. 반올림 키, 유사 상태, 질량 floor를 사용하지 않는다.
2. 같은 입력 하나를 먼저 계산한 뒤 성공한 결과만 원래 셀 순서로 복사한다. 대표 입력이 실패하면 나머지 동일 입력도 각각 실행해 전역 셀 대응·실패 내용·상세 기록 상한을 보존한다.
3. 키와 결과는 배치 사이에 보관하지 않는다. 기존 모델·정책 해시와 attempt/stage/content token 검사를 통과해야 한다. 실패한 배치는 호출자 상태를 변경하지 않는다.
4. 한 번의 복원 내부에서는 같은 순수 액상의 정확히 같은 p/T 물성을 16개 슬롯에 보관한다. 액상마다 독립적이며 실패는 저장하지 않는다. 화학퍼텐셜이 필요한 요청과 불필요한 요청을 구분한다. 재사용할 때도 Cantera의 온도·밀도를 복원해 공개 객체 상태를 유지한다. 복원 호출마다 비운다.
5. 화학퍼텐셜 출력 버퍼는 모델/작업자별로 한 번 할당해 반복적인 임시 벡터 할당을 줄인다.

배치 추가 버퍼는 기존 `maxThermoBatchMemoryMB` 예산에 포함된다. CUDA 입력/출력은 최대 배치 크기에 맞춰 한 번 할당하며 65,536셀 이하로 제한한다. CUDA 메모리는 수송의 `maxDeviceMemoryGB` 예산에서 차감한다. 프로파일은 배열 바이트를 보고하며 CUDA 드라이버·Cantera 내부 할당을 전부 측정한다고 주장하지 않는다.

## CUDA 경로의 정확한 범위

- 응축성 종의 총 inventory가 **정확히 0**인 기상만 대상이다. 아주 작은 양이라도 있으면 CPU 후보 탐색을 유지한다.
- 호스트가 기존 `ReactiveFixedCaloric`으로 조성과 NASA7/9 계수 및 PR 혼합 다항식을 준비한다. 모든 binary-a 보정항을 유지한다.
- GPU가 FP64로 PR/NASA e, cv, p와 보간·Newton 온도 복원을 계산한다. `--fmad=false --prec-div=true --prec-sqrt=true`로 빌드한다.
- 준비한 NASA 온도 구간이나 PR alpha 부호 구간을 넘는 상태, 지원하지 않는 물성, 수치적으로 수렴하지 않는 상태는 CPU 기준 경로로 넘기며 수를 보고한다. 구간 밖으로 계수를 외삽하지 않는다.
- GPU 결과는 최종 평형 상태가 아닌 **온도·압력 후보**다. 호스트의 기존 EOS 분지, 완전한 체적·에너지 잔차 검사 및 상 안정성/엔트로피 선택을 거쳐 승인한다. 잘못된 후보는 거부하고 원 입력과 원 추정값으로 CPU 복원을 수행한다.
- CUDA 호출 자체의 오류는 배치 오류로 보고하며 몰래 CPU 계산으로 바꾸지 않는다. 이는 수치 후보 거부와 다르다.

`REACTIVE_RUNTIME`에 실제 라이브러리 경로·SHA-256, `exactBatchReuse`, `deviceCaloricCandidates`, 적용 범위를 남긴다. `deviceFullClosure`와 `deviceChemistryIntegration`은 **false**를 유지한다.

## 계측

`REACTIVE_CLOSURE_PROFILE`은 중복 셀 수, 실패 대표 재실행, 액상 물성 cache hit/miss, GPU 제출/수렴/호스트 승인/거부/미지원 상태, 커널 실행 횟수, 전송량, 메모리, 분류/준비/GPU 경과 시간을 출력한다. GPU 커널·복사는 timing-enabled CUDA event로 별도 측정한다. 이 시간은 전체 배치 시간에 포함되므로 합산해서 중복 계산하지 않는다. `gpuSubmitted - gpuConverged`는 device 수치 경로 미수렴/영역 이탈이며 `gpuRejected`는 수렴 후보의 호스트 승인 실패다.

`fullPhaseEvaluations`는 기존처럼 실패를 포함한 함수 호출 수이며 cache hit도 포함한다. 실제 생략된 액상 물성 작업은 `liquidCacheHits`로 구분한다.

새 설정은 수치 정책 해시에 포함된다. 기존 체크포인트의 물리 모델 식별과 보존 기준은 유지하고 기존 정책 전환 로그를 출력한다. 프로파일·설정 API는 version/size를 검사하는 additive ABI이며 기존 상태와 프로파일 구조체 크기는 변경하지 않는다.

## 검증 및 성능 해석

[검증 보고서](../reports/closure-acceleration-20260921.md)와 [원시 결과](../results/closure-acceleration/)를 참조한다. 중복이 많은 입력에서는 이득이 크지만, 모두 다른 상태에서는 재사용 이득이 작다. 작은 기상 배치에서는 CPU보다 GPU가 느렸다. 따라서 CUDA 경로는 선택 옵션이며 R04 다상 flash의 GPU 가속 완료나 전체 메시 성능 개선율을 주장하지 않는다. 30–50 ns 목표 CFL/메시 문제도 이 변경으로 해결되지 않는다.
