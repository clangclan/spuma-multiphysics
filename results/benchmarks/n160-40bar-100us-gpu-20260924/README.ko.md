# 160³ · 40 bar · 100 μs 검증 자료

검증된 1 μs FP64 체크포인트에서 100 μs까지 이어서 수행한 본 해석과
1·10·50·100 μs의 독립 GPU 진단 결과다. 1기압 케이스는 포함하지 않는다.

- [상세 분석 보고서](../../../reports/n160-40bar-100us-gpu-analysis-20260925.ko.md)
- [실행 구성과 모니터 사용법](../../../docs/long-gpu-campaign.ko.md)
- `analysis.json`: 시간 회계, 구간별 장치 통계, 최종 물리량, 해시 검증 결과.
- `validation-manifest.json`: 실행 바이너리 해시와 11개 작업 및 4개 카운터 검증 요약.
- `steps.csv`: 본 해석 1,931스텝의 시간·HEM 계산량·보존 잔차.
- `time-breakdown.csv`: 본 해석 프로세스 시간에 대한 비중첩 비용 비율.
- `profiled-kernels.csv`: Nsight Systems의 커널별 호출 수와 시간.
- `selected-kernel-counters.csv`: Nsight Compute 선택 호출의 상세 카운터.
- `evolution.png`: 시간에 따른 비용·시간 간격·GPU 카운터 변화.

본 해석 3개 구간과 진단 8회 모두 통과했다. 본 해석은 23,603.32초,
재시도·GPU 실패·CPU 폴백은 0건이다. 분석 도구가 최종 체크포인트를 다시
읽고 해시·상 재고·보존 기록을 확인했다.

전체 메시, 체크포인트, 원시 NVML JSONL과 Nsight 바이너리/SQLite는 대용량
로컬 실행 자료다. 저장 위치는 아래와 같으며 이 Git 증거 묶음에는 분석에
사용한 요약·시계열·해시·검증 결과를 보관한다.

```text
/home/jsw/문서/analysis/runs/n160_40bar_100us_gpu_20260924
```

GPU 커널 시간은 상위 복원 시간과 중첩된다. NCU replay 시간은 본 해석 시간과
직접 비교할 수 없다. 물리 모델의 공간 수렴 및 분열 검증은 이 자료의 범위가 아니다.
