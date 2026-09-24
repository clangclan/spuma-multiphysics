# 범용 명칭 전환 검증

[전환 안내](../../docs/generic-naming.ko.md), [최종 집계](validation.json), [파일 이동 목록](renames.json).

열역학 백엔드·CUDA 수송 라이브러리·ReactiveFoam을 다시 빌드했다. 공개 C 심볼 65개(백엔드)·51개(수송)는 이름만 바뀌고 누락이 없었다. CPU/CUDA 수송 각 29개, WALE 응력 11개, WALE scalar 13개, 물리 옵션 41개, 곡면 flash 20개, 정확 재사용 10개 검사와 CPU/CUDA 계면 primitive 검증을 통과했다.

기존 40³ N₂O 케이스의 1 μs 체크포인트를 이름 변경 전·후 바이너리에서 각각 읽고 한 스텝씩 진행했다. 1기압과 40 bar 모두 시간 간격·재시도 이력과 최종 `reactiveState.bin` SHA256이 일치했다. 재시작 파일은 수정하지 않았다. CPU 폴백 및 GPU 실패는 0이었다. `restart/`에는 각 실행의 입력 해시와 로그를 보존했다.

새 API로 다시 빌드한 FP32 선형계 HEM sidecar도 실제 10,756개 상태 배치에서 기존 FP32 후보의 모든 출력 필드와 일치했다. 이 실행은 이름 변경 회귀 검사이며 새 속도 개선을 주장하지 않는다.

중간 검증에서 두 문제를 발견해 해결했다. OpenFOAM의 기존 의존성 캐시는 옛 헤더명을 참조하여 `wclean`이 필요했다. 저장된 case physics 해시의 문자열도 이름을 바꾸면 재시작이 거부되어, 네 종류의 저장 형식 식별자를 `reactiveIdentityTags.h`에 모아 원래 바이트로 보존했다. 첫 WALE 검증 호출은 상위/하위 도구가 같은 lock을 중복 획득하여 대기 시간 초과했으며, 도구 내부 lock만 사용한 재실행이 통과했다. 최종 통과 자료를 이 폴더에 모았고 초기 실패와 진단 로그도 아래 원본 폴더에 남겼다.

원본: `/home/jsw/문서/analysis/runs/generic_naming_20260924`. 이전 바이너리·대형 필드·실험 라이브러리는 그 폴더에 있다. 이전 소스는 `7516dd0` 커밋이며, 현재 소스로 다시 빌드한 뒤 아래 도구에 이전 바이너리·기존 케이스를 명시하여 재현한다.

```bash
"$REACTIVE_PREFIX/bin/python" tools/validate_naming_migration.py \
  --previous-binaries /path/to/previous-binaries \
  --configuration /path/to/cold-pr-config.yaml \
  --cases /path/to/ambient_1atm /path/to/ambient_40bar_abs \
  --output /path/to/new-validation
```

`previous-binaries`에는 이름 변경 전 `ReactiveFoam`, `libpintleReactiveBackend.so`, `libpintleReactiveTransport.so`를 둔다. 기존 케이스에는 통과한 `transient-validation.json`과 완전한 schema-3 체크포인트가 있어야 한다. GPU 실행 lock은 도구 내부에서 획득한다.
