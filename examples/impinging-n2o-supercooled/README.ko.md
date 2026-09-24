# 과냉각 액체 N₂O 입력

현재 90도 충돌 벤치마크 입력이다. 응축상은 액체 N₂O 하나이며 고체 생성·승화·융해잠열은 제외한다. 148 K까지 PR 액체 가지를 연장하고 증발·응축을 유지한다. 삼중점 아래 실제 고체 평형을 나타내지는 않는다.

기구는 이전 저온 확장과 동일하다. 설정은 액체 최소 온도 148 K, 단일 응축상, `enforce-species-temperature-bounds: true`를 사용한다. 양의 IPA 질량은 기존 IPA 물성 범위 밖에서 거부한다. 이 케이스의 초기장·경계·기구는 IPA가 생성되지 않는 비반응 조건이다.

```bash
"$REACTIVE_PREFIX/bin/python" tools/prepare_impinging_n2o.py /path/to/new-supercooled-benchmark \
  --configuration examples/impinging-n2o-supercooled/cold-pr-148K-config.yaml \
  --cells-per-axis 40 --domain-mm 80 --nozzle-diameter-mm 5 --inlet-z-mm 40
```

설정의 기구 상대경로는 생성기가 절대경로로 변환한다. backend API에 직접 사용할 때는 `mechanism`을 절대경로로 지정한다. [현재 솔버 안내와 검증 결과](../../README.md)를 참고한다.

생성한 케이스는 `physics.surfaceTension false`를 명시한다. 표면장력은 별도 옵션이며 `tools/prepare_capillary_impingement.py`로 CUDA 모세관·flashing 파생 케이스를 만들 수 있다. 단일 액체·비반응 HEM·CUDA 수송/복원 및 CPU fallback 비활성화 조건이 필요하다. 정적 액적의 메시 수렴 문제는 아직 해결되지 않았다. [실행 예제](../../README.md#n₂o-충돌-케이스-시작하기), [모세관 모델](../../docs/spray-physics/capillary-flashing.ko.md), [선택 방법과 검증 범위](../../docs/spray-physics/physics-switches.ko.md)를 참고한다.
