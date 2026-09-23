# 저온·고체 N₂O 입력

현재 벤치마크는 [과냉각 액체 입력](../impinging-n2o-supercooled/README.ko.md)을 사용한다. 이 폴더는 이전 고체 모델의 재현용 기록이다.

`cold-pr-148K.yaml`은 실제 256k GPU 10스텝 검증에 사용한 기구와 동일한 파일이다. 설정의 `mechanism`만 같은 폴더 기준 상대경로로 바꿨다. `prepare_impinging_n2o.py`가 이를 절대경로로 정규화해 독립 케이스를 만든다. 직접 backend API에 전달하려면 `mechanism`을 절대경로로 지정해야 한다.

저장소 루트에서 준비한 Python/SPUMA 환경을 사용한다.

```bash
"$PINTLE_REACTIVE_PREFIX/bin/python" tools/prepare_impinging_n2o.py /path/to/new-solid-benchmark \
  --configuration examples/impinging-n2o-solid/cold-pr-148K-config.yaml --cell-mm 0.25
```

이 모델은 비반응 평형이며 액체 N₂O와 고체 N₂O가 기존 두 응축상 슬롯을 사용한다. IPA 액상은 포함하지 않는다. 고체는 기체와 공존하는 148–183 K·2 bar 이하 상태만 지원하며, 승화압 정확도 검증은 아직 미달이다. [구현·검증·계산 비용](../../docs/benchmarks/impinging-n2o-solid-recovery.ko.md)을 확인한다.
