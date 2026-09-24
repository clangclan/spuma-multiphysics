# 160³ · 40 bar · 100 μs GPU 계측

2026-09-24 캠페인은 검증된 FP64 1 μs 체크포인트에서 시작한다.
환경압은 40 bar(abs), 공급압은 55 bar(g)이며, 80 mm 정육면체에
160³ = 4,096,000개 셀, 지름 5 mm 입구 두 개와 z = 40 mm 위치를 유지한다.
1기압 케이스는 이번 실행에 포함하지 않는다.

`tools/run_long_gpu_campaign.py`는 다음을 순차 실행한다.

1. 1 μs 상태에서 독립된 1스텝 Nsight Systems 및 Nsight Compute 진단.
2. 본 해석을 10 μs까지 진행한 뒤 해당 체크포인트에서 같은 진단.
3. 50 μs, 100 μs까지 이어서 각각 진단.

진단에서 생성된 상태는 본 해석에 사용하지 않는다. 본 해석의 구간 끝에서는
지정 종료 시간에 맞춰 마지막 시간 간격이 잘린다. 나머지 스텝은 기존 CFL에
따라 갱신되며, 정밀도·물리 옵션·허용 오차는 변경하지 않는다.
점성, 열전도, WALE, 난류 열·종 혼합, 표면장력과 flashing을 켠다.
고체상·승화·연소·분자 종 확산은 기존 설정대로 끈다.

전체 실행에서 0.5초마다 NVML 전력·온도·클록·클록 제한 이유·VRAM·PCIe와
프로세스/CPU 통계를 저장한다. 솔버 로그는 매 스텝의 시간 간격, 보존 오차,
연산별 시간과 GPU HEM 통계를 기록한다. 표준 출력 버퍼링 때문에 터미널의
최신 스텝 표시는 실제 계산보다 늦을 수 있다.

Nsight Systems는 CUDA 호출·커널·전송·할당 및 GPU 하드웨어 시계열을 기록한다.
Nsight Compute는 HEM, WALE 물성, 면 유량, 보존량 갱신, 계면 구배·곡률의
첫 번째/세 번째 호출을 선택하여 레지스터, 점유율, 스케줄러·워프 정체,
FP64/FP32 명령, 연산 파이프라인과 캐시/DRAM 카운터를 기록한다.
선택된 커널의 수치는 전체 해석의 시간 가중 평균이 아니다.
카운터 replay가 시스템 메모리에 GPU 메모리를 백업할 수 있으므로
진단 실행의 시간과 메모리 사용량을 본 해석의 성능으로 해석하면 안 된다.

실행 파일·라이브러리·도구의 복사본과 SHA-256을 저장한다. 단일 GPU lock을
보유하며 실행한다. 본 해석이 실패하면 그 상태에서 자동 재시도하거나 물리를
완화하지 않는다. 진단 도구만 실패하면 오류를 기록하고 검증된 본 해석을
계속한다. 각 본 해석 구간의 제한은 24시간, Nsight Systems 진단은 1시간,
Nsight Compute 진단은 2시간이다.

실행 디렉터리:

```text
/home/jsw/문서/analysis/runs/n160_40bar_100us_gpu_20260924
```

- `status.json`: 현재 작업, 시각, GPU 상태, 검증 완료 시각, 오류.
- `production/40bar/`: 본 해석 구간과 체크포인트, 보존 검증, 원시 로그.
- `diagnostics/40bar-*us/`: 각 시각의 trace, 원시 카운터와 분석 CSV/JSON.
- `timing-summary.csv`: 본 해석의 비중첩 스텝 시간 집계.
- `REPORT.md`: 구간 완료 시 자동 갱신되는 진행 보고서.
- `worker.log`, `worker.pid`: 독립 실행 작업의 로그와 PID.

터미널 다시 열기:

```bash
/home/jsw/문서/analysis/spuma-multiphysics/research/reactive-env/bin/python \
  /home/jsw/문서/analysis/spuma-multiphysics-spray-physics/tools/run_long_gpu_campaign.py \
  watch /home/jsw/문서/analysis/runs/n160_40bar_100us_gpu_20260924
```

상태창의 Ctrl+C는 상태창만 종료한다. 전체 캠페인 중단은 `worker.pid`에
기록된 작업에 SIGTERM을 보낸다. 최신 체크포인트와 로그는 보존된다.
본 해석 결과와 모든 진단 결과가 확인되기 전에는 100 μs 완료로 간주하지 않는다.
