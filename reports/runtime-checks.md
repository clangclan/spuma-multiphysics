# 추가 실행 검증

[기계 판독 기록](runtime-checks.json) · [성능·필드 비교](validation.md)

## 실제 Pintle mesh의 연산 대조

FP64 제한기 대조는 `benchmarks/oracle-device4`의 첫 alpha solve에서 실행했다. stock SPUMA MULES 대비 최대 scaled 차이는 4.756e−14, 3상 limitSum 차이는 0, explicitSolve 최대 절대차는 2.221e−16이었다.

FP32 ratio 저장 경로도 `benchmarks/memcheck-mixed-explicit1`에서 같은 대조를 수행했다. 최대 limiter scaled 차이는 4.481e−8, limitSum 차이는 0, explicitSolve 최대 절대차는 3.331e−16이었다. 이는 첫 solve의 연산 대조이며 전체 시간 이력의 오차 상한을 뜻하지 않는다. 12스텝 최종 필드 비교는 별도 성능 보고서에 있다.

온도 증분 해법의 원래 선형계에 대한 backward error는 두 outer iteration 모두 2.35e−17 이하였고, 최대 `|residual/diagonal|`는 4.014e−10 K였다. 이 값은 선형계 풀이의 정확도를 점검하며 물리 모델의 정확도를 검증하지 않는다.

## Compute Sanitizer

도구 버전은 2026.1.1.0, build 37663202다. 실제 3,094,455셀 mesh를 mixed/device 설정으로 1스텝 실행했다.

```bash
source ./env.sh
compute-sanitizer --tool memcheck --error-exitcode 97 \
  --report-api-errors explicit --print-limit 0 \
  --kernel-name kns=pintle --kernel-name kns=multiphaseMixtureThermo \
  bin/spumaPintleColdFoam \
  -case benchmarks/memcheck-mixed-explicit1 -noFunctionObjects \
  -pool fixedSizeMemoryPool -poolSize 10
```

정상 종료, `ERROR SUMMARY: 0 errors`. 이 판정은 이름에 `pintle` 또는 `multiphaseMixtureThermo`가 포함된 커널의 메모리 접근 검사와 명시적 CUDA API 실패 검사에 한정한다. `explicit`은 CUDA 내부 logging을 제외하고 직접 호출 API의 실패 보고를 유지한다. 전체 SPUMA 커널, 미초기화 읽기, 일반 race 검사를 모두 통과했다는 뜻은 아니다. [NVIDIA 검사 범위·API 보고 모드](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#cuda-api-error-checking)

**중복 커널 등록 진단은 미해결이다.** 기본 `extended` 보고 모드의 첫 실행은 오류 요약 106건으로 종료됐다. 출력된 100건은 모두 duplicate-entry kernel 등록 진단이며 나머지 6건은 출력 한도에 걸렸다. 따라서 첫 실행을 메모리 검사 통과로 계산하지 않았다. 위 재실행은 내부 logging을 제외한 검사 결과이며 중복 등록 원인을 수정한 결과가 아니다.

로컬 `libOpenFOAM.so`와 `libfiniteVolume.so`에서 동일한 `Foam::cuda::lambdaKernel` 인스턴스가 `FUNC WEAK DEFAULT`로 export된 것을 확인했다. 여러 DSO의 템플릿 심볼과 CUDA 등록 충돌이 원인일 가능성이 크지만 무해한 진단으로 확정하지 않았다. 실행 명령·반환 코드·원로그 경로는 JSON에 보존했다.

## 준비 이후 GPU 프로파일

`profile-gpu-steady12`는 별도 12스텝 실행에서 `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --delay=20 --duration=10 --kill=none --stats=true`를 사용했다. 프로파일링 실행의 시간은 성능 표에 포함하지 않았다.

| 10초 trace 항목 | 값 |
|---|---:|
| unified-memory CPU→GPU | 182,484,992 B |
| unified-memory GPU→CPU | 182,484,992 B |
| kernel 수 | 264,296 |
| kernel 시간 합 | 6.6328 s |
| device synchronization 호출 수 | 263,241 |
| 해당 API의 host 시간 합 | 8.1700 s |

각 시간은 서로 겹치므로 더해서 총 시간을 계산할 수 없다. host 동기화 시간에는 GPU 작업 완료를 기다리는 시간이 포함된다. kernel 시간 합도 SM 점유율이나 GPU 사용률과 동일하지 않다.

CPU smoother를 사용한 초기 2스텝 trace에서는 unified-memory 양방향 이동이 약 39.07 GB였다. 수집 구간과 단계가 다르므로 39.07 GB와 364.97 MB의 비율을 성능 향상률로 사용하지 않는다. 실제 성능은 프로파일러 없이 실행한 12스텝×3회 결과로 평가했다. GPU smoother 적용 후에도 SPUMA 공용 연산에서 많은 작은 커널과 동기화가 남아 있어 후속 최적화 대상이다.

전송량은 SQLite의 `CUPTI_ACTIVITY_KIND_MEMCPY`를 `copyKind`별로 집계했다. unified-memory 방향은 같은 파일의 `ENUM_CUDA_MEMCPY_OPER`에서 확인했다. 원본 `.nsys-rep`, SQLite, 실행 로그는 `benchmarks/profile-gpu-steady12*`에 보존했다.

## 최종 빌드 확인

최종 정리에서 C++ 파일의 빈 줄에 남은 공백만 제거한 뒤 `./Allwmake`를 다시 실행했다. 실행파일 hash는 같고 공유 라이브러리 hash는 달라졌으므로 [무결성 기록](integrity.json)에 벤치마크 빌드와 최종 빌드를 구분했다. 성능·sanitizer 원자료는 벤치마크 빌드에 해당한다.

최종 빌드에서도 24,511점 물성 검사를 다시 통과했고, 실제 mesh 4스텝과 첫 solve의 FP64 연산 대조를 완료했다. 앞서 실행한 GPU 4스텝 최종장 대비 주요 상태 필드의 최대 scaled 차이는 2.50e−10, 체적분율은 7.82e−9 이하이며 모든 검사 필드는 finite였다. [최종 빌드 필드 대조](final-build-check.json)에 세부 값을 보존했다. 추적한 원본 Pintle·SPUMA 파일 306개는 모두 원래 SHA-256과 일치한다.
