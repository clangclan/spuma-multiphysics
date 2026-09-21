# ReactiveFoam 후속 개선 상태

기준은 ColdFoam을 제거하고 다중물리 selection을 포함한 `main`의 `7d95bd50b5934f41099218d308a57559d2e2f2bd`다. [baseline.json](baseline.json)은 PR #1과 selection의 검증 파일을 SHA-256으로 고정한다. 기존 검증 JSON·로그를 수정하거나 최초 실패를 삭제하지 않는다.

RF21-01~06은 PR #1에서 구현·로컬 회귀 인수를 마쳤다. 여기에는 계약 검사, worker/attempt 비용 합산, 같은 EOS 열량 계산, CVODE matrix-free/Woodbury 선택 경로, bounded pinned/event slot, device NASA 수송 물성이 포함된다. 2026-09-19 보고서의 `local_validated=false`는 당시 환경의 기록이며 현재 상태가 아니다. 실제 근거와 제외 범위는 [PR #1 보고서](../../reports/pr1-local-validation-20260921.md)를 따른다.

`regression_passed`는 기준 동작의 보존이고 `physics_accuracy_passed`는 물리 문제별 별도 판정이다. 기존 HEM 접촉면 정확도 실패, 상세 고압 PR 반응 자료 부족, 비이상 확산 미지원은 남아 있다. SPUMA 확장 CUDA 진단의 기존 15개 중복 등록 이름은 [known-cuda-diagnostics.json](known-cuda-diagnostics.json)에 고정했다. 임의의 CUDA 오류를 이 예외로 분류하지 않는다.

지원 기능의 정적 목록은 [feature-registry.json](feature-registry.json)에 있다. 구현·Flow 연결·지원 모델·GPU 검사·물리 정확도·성능 측정·기본 활성화를 각각 기록한다. 실제 활성 설정은 실행 시작 로그와 바이너리 해시로 확인한다.

## 이번 변경

R04 명세와 RF3-00/01의 공통 선행 조건인 실패 입력 보존, 명시적인 실제 backend 식별, 승인 상태 관리와 재시작을 다룬다. R04 파일의 과거 “코드 수정 금지”는 이번 구현 요청의 제약으로 적용하지 않았다. 원본 R04 케이스와 과거 시험 결과는 보존한다.

| 항목 | 구현 범위 |
|---|---|
| UV 복원 | 선택 가능한 `boundaryFallback`: 미량 기상 재고를 로그 좌표로 표현하고 경계에서 기준·보조 탐색의 안정한 후보를 엔트로피로 비교 |
| 진단 | 전역/배치/worker ID, RK 단계·시각·dt·retry, q/운동량/에너지·원래 seed, 후보 잔차와 EOS branch, 실제 solver/backend SHA-256 |
| 실패 배치 | 모든 실패 셀의 요약, 최대 16개 상세 탐색 기록, 생략 수. 작업 순서와 원래 셀 ID를 구분 |
| 승인 | 전역 보존 검사까지 CPU/CUDA attempt를 미승인 상태로 유지. 실패 시 q·상태·시각·누적 경계 유량을 복구 |
| 체크포인트 | schema 3의 정확한 q·열역학 seed·보존 기준·누적 유량·시각·스텝/재시도 수. 숨김 디렉터리에서 작성·fsync 후 원자적 교환 |
| 실행 | solver/library를 명시해 고정 사본으로 실행. 실제 스텝 진행과 완료 체크포인트로 성공 판정. 감시는 자동 재시작하지 않음 |

`boundaryFallback`은 비반응 HEM 상평형·종 확산 없음에서만 허용한다. 생성에너지, EOS·종 재고, 안정성 조건, `1e-9` 체적/에너지 및 `1e-7` 화학퍼텐셜 기준을 바꾸지 않는다. 기준 `reference`는 기본값이며 기존 화학·동결상·mechanical 경로를 유지한다. 아직 모든 물리 상태의 복원 성공을 보장하지 않는다. 특히 293.15 K/4 MPa에서 N₂O와 IPA를 모두 액체로 초기화한 일부 고밀도 혼합 입력은 별도 실패 사례로 남는다. 지원 branch의 부재와 비선형 탐색 한계를 구분하는 추가 분석이 필요하다.

## 설정과 재현

```foam
// constant/reactiveProperties
recoveryMode boundaryFallback; // 기본 reference
recoveryDiagnostics true;      // 후보별 상세 기록은 기본 false
checkpointFirstStep true;     // 기본 true
checkpointEverySteps 5;        // 기본 0: 스텝 주기 저장 없음
checkpointOnFailure true;      // 기본 true
```

실패는 케이스의 `reactiveFailures.jsonl`에 기록된다. 요약 레코드만 있는 셀은 독립 재생 입력으로 간주하지 않는다. 원래 R04 tracer와 새 상세 레코드는 다음 도구로 전체 메시 없이 재생한다.

```bash
python tools/replay_reactive_recovery.py tests/fixtures/r04-recovery.jsonl \
  --config /absolute/path/cold-pr-config.yaml \
  --library /absolute/path/libpintleReactiveBackend.so \
  --mode boundaryFallback --output replay.json
```

새 체크포인트는 `reactiveCheckpointComplete`의 파일 해시와 정확한 이진 상태를 확인한 뒤 읽는다. schema 1/2 입력은 계속 읽지만 과거 누적 경계 유량이 없으므로 그 재시작 시점부터 보존 기준을 시작한다. schema 3은 원래 기준을 이어간다. 이진 상태는 현재 FP64/endianness/C ABI 구조에 한정하며 다른 구조는 명시적 변환이 필요하다. Linux `renameat2(RENAME_EXCHANGE)`와 같은 파일시스템의 원자적 rename을 사용한다. 저장공간에는 새 스냅샷과 기존 시간 디렉터리가 동시에 들어갈 여유가 필요하다.

```bash
python tools/reactive_run.py launch \
  --case /absolute/path/case --output /absolute/path/new-run-directory \
  --solver /absolute/path/ReactiveFoam \
  --backend-library /absolute/path/libpintleReactiveBackend.so \
  --transport-library /absolute/path/libpintleReactiveTransport.so \
  --spuma-env /absolute/path/spuma-env.sh --end-time 1e-8 --cpu-list 0-23
python tools/reactive_run.py watch /absolute/path/new-run-directory/status.json
```

`--end-time`은 controlDict의 명시적 endTime과 일치해야 한다. 실행 디렉터리는 새로 만들며 기존 결과를 덮어쓰지 않는다. 종료 코드 0, 목표 시각 도달, 완료 체크포인트가 모두 있어야 `completed`다. 프로세스 생존이나 첫 스텝 성공은 연속 진행 인수의 대체가 아니다.

감시 화면은 실제 프로세스 CPU 코어 사용량, RAM·GPU 메모리, 최근 승인 시점과 열역학 잔차를 함께 표시한다. 리소스는 5초마다 표본을 기록하며 프로세스 종료 직전의 짧은 최대값은 놓칠 수 있다. `cpuCores=23`은 약 23개 코어 사용이며 worker별 작업 시간 합계와 다르다.

완료한 R04 실행은 다음 명령으로 스텝·수지·자원 예산과 모든 최종 셀의 유한성/종 재고, 체크포인트 해시를 확인한다. 기본 인수 조건은 1,042,500셀, worker 24개, 해당 프로세스에서 연속 20스텝 이상, 재시도 0회다. `passed`는 시작 구간의 수치 인수이며 분무 정확도 인증이 아니다.

```bash
python tools/validate_reactive_run.py /absolute/path/new-run-directory \
  --output full-mesh-validation.json
```

## 현재 검증

[검증 보고서](../../reports/reactive-recovery-20260921.md): 회귀 512개 통과, R04 1,042,500셀에서 재시작 후 21스텝 연속 승인·재시도 0회·12 ns 정상 종료. 최종 빌드는 같은 체크포인트의 보존량/수지 이력을 그대로 읽고 12.1 ns까지 한 스텝 더 진행했다. binary/ASCII 및 ASCII gzip 재시작도 검사했다. 이 OpenFOAM은 binary 압축 요청을 경고 후 해제한다. P2 최적화 OFF/ON 비교와 확장 물리 인수는 수행하지 않았다.

## 후속 범위

| v3 기능 | 현재 상태 |
|---|---|
| RF3-00 registry/baseline | 기존 인수·제약을 위 기록으로 고정. 현재 변경의 검사 결과는 별도 보고 |
| RF3-01 승인 상태·checkpoint | 이번에 개선. device reduction/primitive packing/D2H 제거는 미구현 |
| RF3-02/03 device PR closure/HEM | 미구현, 실제 closure는 CPU |
| RF3-04 device kinetics/integration | 미구현, 실제 화학은 CPU |
| RF3-05 다중 slot·중첩 | 미구현, 기존 bounded single slot 유지 |
| RF3-06/07 고압 상세 물성·비이상 수송 | 자료와 모델 gate 필요, 기존 guard 유지 |
| RF3-08/09 비직교·벽/경계·고차·접촉면 | 후속 구현. 기존 직교 HLL 범위를 넓혔다고 표시하지 않음 |
| RF3-10 이후 비평형·표면장력·액적·LES/TCI·MPI | 별도 모델·인수 필요. 이번 변경에서 지원하지 않음 |

P2 exact-state cache와 성능 최적화는 R04 G0–G3 인수 이후의 작업이다. 현재 검증은 수치 일관성 검사이며 분무 실험 검증이 아니다.
