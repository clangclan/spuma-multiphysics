# AVX-512 SIGILL 우회 실행

현재 Core Ultra 7 270K Plus는 AVX2를 지원하지만 AVX-512는 지원하지 않는다. `/home/jsw/cae-gpu`의 기존 SPUMA 라이브러리를 불러오면 지원하지 않는 CPU 명령 때문에 `SIGILL`로 종료된다. 이미 준비된 **`/home/jsw/cae-gpu-pr1-compatible` 호환 빌드**를 사용하면 된다.

이 빌드는 CPU 코드를 `-tp=x86-64-v3`로 컴파일한 것이다. NVIDIA 문서에서 이 대상은 AVX2 수준이며, AVX-512를 포함하는 `x86-64-v4`와 구분된다. CUDA 대상 `sm_120`은 유지하므로 GPU 계산을 CPU로 대체하는 방법이 아니다. [NVIDIA 컴파일러 옵션 설명](https://docs.nvidia.com/hpc-sdk/compilers/hpc-compilers-ref-guide/#tp-target), [기존 호환 빌드 기록](../reports/pr1-local-validation-20260921.md).

새 Bash 터미널에서 다음처럼 환경을 지정한다.

```bash
cd /home/jsw/문서/analysis/spuma-multiphysics-gpu-opt
export REACTIVE_SPUMA_ENV=/home/jsw/cae-gpu-pr1-compatible/env.sh
export REACTIVE_PREFIX=/home/jsw/문서/analysis/spuma-multiphysics/research/reactive-env
export REACTIVE_RUN_LOCK=/home/jsw/cae-benchmark/run.lock
source ./env.sh

blockMesh -help
./bin/ReactiveFoam -help
ldd ./bin/ReactiveFoam | grep -E 'libOpenFOAM|libfiniteVolume|libmeshTools|libcudart'
```

`libOpenFOAM.so`, `libfiniteVolume.so`, `libmeshTools.so`가 모두 `cae-gpu-pr1-compatible/spuma/platforms/…/lib`에서 로드되는지 확인한다. **`REACTIVE_SPUMA_ENV`를 지정하지 않으면 저장소의 `env.sh`와 Python 실행 도구가 구 설치를 기본 선택한다.** 과거 이름인 `PINTLE_SPUMA_ENV` 대신 현재 이름을 사용하고, Python 도구를 시작하기 전에 export한다.

같은 환경에서 준비된 케이스를 실행하는 명령은 다음과 같다. `/path/to/prepared-case`를 실제 케이스 경로로 바꾼다.

```bash
flock "$REACTIVE_RUN_LOCK" ./bin/ReactiveFoam -case /path/to/prepared-case
```

소스 변경으로 솔버를 다시 빌드해야 한다면 같은 셸에서 `REACTIVE_TRANSPORT_BUILD=cuda`, `REACTIVE_CUDA_ARCH=120`을 지정하고 `flock "$REACTIVE_RUN_LOCK" ./Allwmake`를 실행한다. `-tp=x86-64-v3`는 **빌드 옵션**이므로 구 라이브러리에 실행 시 변수만 붙여 AVX-512를 끄는 방식은 작동하지 않는다. 호환 빌드의 nvcc 규칙은 NVIDIA CPU 컴파일러 전용 `-tp` 옵션을 제외한다.

2026-09-25에 현재 최적화 워크트리의 **동일한 솔버 바이너리**로 확인했다. 구 환경에서는 `blockMesh -help`와 `ReactiveFoam -help` 모두 SIGILL(signal 4), 호환 환경에서는 둘 다 종료 코드 0이었다. 따라서 시작 단계의 AVX-512 장애는 기존 호환 환경 선택으로 우회할 수 있다. **이번 확인은 라이브러리 로딩과 실행 진입 검사이며, 최적화된 솔버의 전체 스텝·보존·재시작 검증을 대신하지 않는다.**

## 검증 기록 (2026-09-25)

위 절차를 다시 확인한 결과다. 원자료는 `results/gpu-hem-optimization-20260925/end-to-end.json`에 있다.

- **CPU:** `/proc/cpuinfo`에는 `avx`, `avx2`, `avx_vnni`만 있고 avx512 계열은 없다. `nvc++ -tp=help`의 설명도 `x86-64-v3`는 AVX2까지, `x86-64-v4`는 일부 AVX-512를 포함한다고 되어 있다.
- **구 설치:** `foamDictionary`, `blockMesh`, `ReactiveFoam` 모두 `Foam::readDouble`의 `kmovd`에서 SIGILL(종료 코드 132)로 끝난다. 구 `libOpenFOAM.so`를 역어셈블하면 AVX-512 명령(zmm, mask 레지스터 k1–k7, `kmov`)이 43,577개 나온다.
- **호환 설치:** 호환 빌드의 `libOpenFOAM`, `libfiniteVolume`, `libmeshTools`, `libPstream`(dummy), 그리고 워크트리의 `ReactiveFoam`, `libreactiveBackend.so`, `libreactiveTransport.so`에서 AVX-512 명령은 0개다. libc, libcrypto, lapack, libgfortran, libnvcpumath에도 AVX-512 코드가 있지만, 실행 중에 CPU 기능을 확인해 코드 경로를 고르는 라이브러리라 이 CPU에서는 실행되지 않는다.
- **소스 동일성:** 구 설치와 호환 설치의 SPUMA 소스는 `doc` 디렉터리를 빼면 같고, 빌드 ID도 `5916a466-20260728`로 같다. wmake 규칙의 차이는 `General/cuda`의 `$(filter-out -tp=%,…)` 한 줄뿐이다. 따라서 구 헤더로 빌드한 솔버를 호환 라이브러리에 연결해도 ABI 차이는 없다.
- **환경:** 호환 `env.sh`를 불러온 뒤에는 `PATH`와 `LD_LIBRARY_PATH`에 구 설치(`/home/jsw/cae-gpu/`) 경로가 남지 않는다. 반면 호환 빌드에는 공유 라이브러리 9개(OpenFOAM, fileFormats, surfMesh, meshTools, finiteVolume, blockMesh, extrudeModel, dynamicMesh, dummy Pstream)와 유틸리티 `blockMesh` 하나만 있다. `checkMesh`는 없으며, `tools/prepare_impinging_n2o.py`는 이 경우 `/opt/openfoam14`를 대신 쓴다.
- **명령 수정:** 원래 명령에 있던 `rg`는 이 머신에 설치돼 있지 않아 `grep -E`로 바꿨다.
- **실제 스텝 실행:** 호환 환경에서 160³·40 bar 100 μs 체크포인트로부터 2스텝을 끝까지 계산했다(재시도 0, 보존 잔차 약 2×10⁻¹¹). `main` 빌드는 스텝당 20.0 s였다. 최적화 워크트리 빌드는 `closureJacobian analytic`일 때 7.9–8.1 s였다. 즉 이 우회로 시작 단계뿐 아니라 GPU 전체 스텝과 체크포인트 쓰기까지 동작한다. 장시간 실행과 재시작 검증은 별도 과제다.
