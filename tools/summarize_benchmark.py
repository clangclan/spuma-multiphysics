#!/usr/bin/env python3
"""Main-agent engineering screen; not a physical-model validation certificate."""
import argparse
import json
import math
import re
import statistics
from pathlib import Path

import numpy as np
import benchmark as b

STATE_FIELDS = ('p', 'U', 'T', 'rho', 'alpha.ipa', 'alpha.n2o', 'alpha.air')
THRESHOLDS = {k: (1e-5 if k == 'p' else 1e-6) for k in STATE_FIELDS}


def write_markdown(result, output):
    rows = result['timing']
    definition = result['run_definition']
    lines = [
        '# Pintle GPU 개발판 검증·성능 기록', '',
        f"3,094,455셀의 실제 Pintle 체크포인트에서 {float(definition['start_s'])*1e6:.2f}→"
        f"{float(definition['end_s'])*1e6:.2f} μs, dt={float(definition['delta_t'])*1e9:g} ns로 계산했다. "
        f"각 설정은 {definition['steps']}스텝×{definition['repeats']}회이며, "
        '각 실행에서 준비 영향을 받는 첫 2스텝과 마지막 출력 스텝을 제외한 '
        f"{definition['steps']-3}스텝의 산술평균을 구한 뒤 실행별 중앙값을 비교한다.", '',
        f"주요 상태·출력에 대한 **개발용 수치 screen: {'통과' if result['screen_pass'] else '실패'}**. "
        '이 판정은 아래에 명시한 메인 에이전트의 허용치를 사용한다. '
        '장시간 물리 모델 검증이나 Foundation 14와의 동등성 판정은 아니다.', '',
        f"[집계 JSON](validation.json) · [전체 원자료]({result['benchmark']})", '',
        '## 성능', '',
        '| 설정 | 계산 s/step 중앙값 | 실행별 범위 | 기준 대비 배율 | 전체 실행 s | GPU 사용률¹ | GPU 메모리 최대 MiB¹ |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        lines.append(f"| {r['configuration']} | {r['median_mean_measured_wall_s_per_step']:.4f} | "
                     f"{r['min_mean_measured_wall_s_per_step']:.4f}–{r['max_mean_measured_wall_s_per_step']:.4f} | "
                     f"{r['whole_step_speedup_vs_spuma_stock_limiter_path']:.2f}× | "
                     f"{r['median_total_elapsed_s']:.2f} | {r['median_gpu_utilization_percent']:.1f}% | "
                     f"{r['max_gpu_memory_mib']:.0f} |")
    lines += ['', '¹ `nvidia-smi` 장치 전체 표본이며 데스크톱 등 다른 프로세스를 포함한다. SM 점유율 측정치는 아니다.', '',
              '`native-reference`도 SPUMA GPU 유한체적 연산과 multicolor smoother를 사용한다. '
              'CPU MULES·CPU 물성과 GPU limiter·GPU 물성을 동시에 바꾸는 비교이며, '
              '전체 CPU 솔버 대 GPU 솔버의 비율이 아니다. 압력·온도·운동량 선형해법 허용오차는 동일하다.', '',
              '## 단계별 계산 시간', '',
              '| 설정 | alpha | 운동량 | 물성·온도 | 압력 | 난류 |', '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        s = r['stage_median_s']
        lines.append(f"| {r['configuration']} | " + ' | '.join(f'{s[k]:.4f}' for k in ('alpha', 'momentum', 'energy', 'pressure', 'turbulence')) + ' |')
    lines += ['', '단계 값도 실행별 평균의 중앙값이며 단위는 s/step이다. '
              '전체 시간에는 위 단계 밖의 Courant·상태 진단 등도 포함된다.', '',
              '## 최종 필드 차이', '',
              '`max_scaled = max(abs(candidate-reference)) / max(1,max(abs(reference)))`. '
              '아래 값은 동일 반복의 reference와 비교한 모든 반복 중 최댓값이다.', '',
              '| 필드 | GPU FP64 | FP32 ratio 저장 | 개발용 허용치 |', '|---|---:|---:|---:|']
    for field in STATE_FIELDS:
        values = []
        for config in ('device-gpu', 'device-mixed'):
            values.append(max(c['max_scaled'][field] for c in result['comparison_screens'] if c['candidate'].startswith(config+'/')))
        lines.append(f'| {field} | {values[0]:.3e} | {values[1]:.3e} | {THRESHOLDS[field]:.0e} |')
    lines += ['', '상태 건전성 screen은 검사한 최종 내부 필드가 모두 finite, p/T/rho>0, alpha∈[−1e−8,1+1e−8], '
              'phase-sum 오차≤1e−9, 스텝별 상대 질량 수지 오차≤1e−6이다. '
              '미세한 alpha 음수는 원본 입력에도 있으며 0으로 잘라내 결과를 꾸미지 않았다.', '',
              f"- 최대 phase-sum 오차: {max(s['alpha_sum_max_abs_error'] for s in result['state_screens']):.3e}",
              f"- 최대 스텝별 상대 질량 수지 오차: {max(s['max_step_relative_mass_imbalance'] for s in result['state_screens']):.3e}", '',
              '## dgdt를 별도로 해석해야 하는 이유', '',
              'raw dgdt는 압력 방정식에서 작은 차이의 상쇄와 phase 존재 여부에 민감하다. '
              '아래 차이를 일반 필드 screen에서 제외한 사실을 명시한다. 모든 내부 필드가 동일하다는 결론은 내리지 않는다.', '',
              '| raw 필드 | GPU FP64 max_scaled | FP32 ratio max_scaled |', '|---|---:|---:|']
    for field in ('dgdt.ipa', 'dgdt.n2o', 'dgdt.air'):
        values = [max(c['raw_dgdt_max_scaled'][field] for c in result['comparison_screens'] if c['candidate'].startswith(config+'/'))
                  for config in ('device-gpu', 'device-mixed')]
        lines.append(f'| {field} | {values[0]:.3e} | {values[1]:.3e} |')
    lines += ['', '보조 진단으로 첫 반복의 최종장에서 압축성 source 성분 '
              '`S_i=alpha_i*(sum_j(alpha_j*dgdt_j)-dgdt_i)`를 계산했다. '
              '이는 순간적인 대수식의 차이이며 상별 순차 업데이트를 재실행한 검증은 아니다.', '']
    for source in result['effective_source_diagnostics']:
        lines.append(f"- {source['candidate']}: max|dt·ΔS|={source['max_abs_dt_times_source_difference']:.3e}, "
                     f"source 상대 L2={source['rel_l2']:.3e}")
    lines += ['', '## 단위 연산·프로파일 증거', '',
              '- 물성: 193개 압력×127개 온도=24,511점, 12개 물성 값. native CPU 대비 최대 상대오차 7.015e−13, invalid=0.',
              '- 실제 mesh 첫 alpha solve: limiter scaled 오차≤4.756e−14, 3상 limitSum 차이 0, explicitSolve 최대 절대차 2.221e−16.',
              '- 기존 CPU smoother 2step Nsight trace: 68,804회 device synchronization, 69,188회 kernel launch, '
              'CPU↔GPU unified-memory 전송 약 39.07 GB. 동기화 시간에는 계산·메모리 이동 대기가 포함되므로 전체를 제거 가능한 overhead로 해석하지 않는다.',
              '- GPU smoother 적용 후 준비 구간을 제외한 10초 trace의 unified-memory 양방향 전송은 합계 364.97 MB다. '
              '위 trace와 수집 구간이 다르므로 두 전송량의 비율을 성능 배율로 해석하지 않는다. '
              '[추가 프로파일·메모리 검사 기록](runtime-checks.md)을 함께 참조한다.', '',
              '## 기존 Foundation 14 기록과의 관계', '',
              '당일 기존 CPU6 기록은 12.945791 s/step, 부분 CUDA 연산 기록은 11.806176 s/step이었다. '
              '둘 다 동일 Pintle mesh·시점에서 12스텝×3회, 실행별 3–11번 스텝 평균의 중앙값이다. '
              '이 개발판과는 MULES 반복·경계 flux 처리·열역학 재시작·압력 최소 반복 수가 다르므로 '
              '참고용 처리시간 비교만 가능하다. 동일 수치 결과의 가속 배율로 주장하지 않는다.', '',
              '[기존 원자료](/home/jsw/문서/analysis/benchmarks/native_20260910/analyzed-runs.json)', '',
              '## 제외한 실행과 남은 검증', '',
              '`validated-v1`은 재시작 index=964 때문에 step 8에 출력하고 마지막 상태를 저장하지 못했다. '
              '그 실행의 시간·필드 결과는 최종 표에서 제외했다. `writeControl runTime`으로 수정한 독립 확인과 '
              '`validated-v2`는 마지막 45.36 μs 출력 파일을 확인했다.', '',
              '연소·화학종·증발/응축은 미구현이다. 장시간 안정성, 다중 GPU/MPI 실행, '
              'Foundation 14 수치법과 체크포인트 상태의 엄밀한 재현은 남아 있다. '
              'FP32 mode는 ratio 저장만 줄이며 FP16/BF16/Tensor Core를 사용하지 않는다. '
              '성능 차이가 작거나 반복 범위가 겹치면 FP64를 기본값으로 유지한다.', '']
    (output/'validation.md').write_text('\n'.join(lines))


def effective_source(directory):
    alpha = np.stack([b.read_internal_field(directory/f'alpha.{p}').values[:, 0]
                      for p in b.PHASES])
    dgdt = np.stack([b.read_internal_field(directory/f'dgdt.{p}').values[:, 0]
                     for p in b.PHASES])
    return alpha * (np.sum(alpha*dgdt, axis=0)[None, :] - dgdt)


def summarize(report, output):
    data = json.loads(report.read_text())
    expected = {'native-reference', 'device-gpu', 'device-mixed'}
    if {r['configuration'] for r in data['runs']} != expected:
        raise ValueError('This Pintle summary requires native-reference, device-gpu and device-mixed configurations')
    expected_runs = {(config, repeat) for config in expected for repeat in range(1, data['repeats']+1)}
    if (len(data['runs']) != len(expected_runs)
        or {(r['configuration'], r['repeat']) for r in data['runs']} != expected_runs
        or len(data['comparisons']) != 2*data['repeats']):
        raise ValueError('Missing or duplicate benchmark runs/comparisons')
    rows, screens, sources = [], [], []
    for timing in data['timing_summary']:
        runs = [r for r in data['runs'] if r['configuration'] == timing['configuration']]
        def median(key):
            return statistics.median(r[key] for r in runs)
        stages = sorted({k for r in runs for k in r['pintle_timing_measured_stage_statistics']})
        rows.append({
            **timing,
            'median_total_elapsed_s': median('elapsed_wall_s'),
            'median_peak_rss_mib': median('peak_process_tree_rss_mib'),
            'median_gpu_utilization_percent': statistics.median(
                r['gpu_measured_window'][0]['mean_utilization_percent'] for r in runs),
            'max_gpu_memory_mib': max(r['gpu_measured_window'][0]['peak_memory_used_mib'] for r in runs),
            'stage_median_s': {k: statistics.median(r['pintle_timing_measured_stage_statistics'][k]['mean_s']
                                                   for r in runs) for k in stages},
        })
    for run in data['runs']:
        health = run.get('field_health', {})
        fields = health.get('fields', {})
        state = []
        for line in Path(run['stdout_log']).read_text().splitlines():
            if line.startswith('PINTLE_STATE '):
                state.append({k: float(v) for k, v in b.TIMING_VALUE_RE.findall(line)})
        reasons = []
        if not run['completed_ok']: reasons.append('solver did not complete')
        if set(b.CORE_FIELDS+b.PHASE_FIELDS)-set(fields): reasons.append('missing field')
        if health.get('field_errors'): reasons.append('field parser error')
        if not all(v['finite'] for v in fields.values()): reasons.append('nonfinite field')
        if any(fields.get(k, {}).get('min', -1) <= 0 for k in ('p', 'T', 'rho')):
            reasons.append('nonpositive thermodynamic state')
        if any(fields.get(f'alpha.{p}', {}).get('min', -1) < -1e-8 or
               fields.get(f'alpha.{p}', {}).get('max', 2) > 1+1e-8 for p in b.PHASES):
            reasons.append('alpha outside roundoff allowance')
        sum_error = health.get('alpha_sum', {}).get('max_abs_error_from_one')
        if sum_error is None or not math.isfinite(sum_error) or sum_error > 1e-9:
            reasons.append('alpha sum error')
        mass_error = max((abs(s['massBalanceRelative']) for s in state), default=math.inf)
        if any(s.get('invalid', 1) != 0 or not all(math.isfinite(v) for v in s.values()) for s in state):
            reasons.append('invalid step state')
        if len(state) != run['steps'] or not math.isfinite(mass_error) or mass_error > 1e-6:
            reasons.append('mass diagnostic absent or outside screen')
        screens.append({'configuration': run['configuration'], 'repeat': run['repeat'],
                        'screen_pass': not reasons, 'reasons': reasons,
                        'alpha_sum_max_abs_error': sum_error,
                        'max_step_relative_mass_imbalance': mass_error,
                        'first_step_relative_mass_imbalance': state[0]['massBalanceRelative'] if state else None})
    comparison_screen = []
    for comparison in data['comparisons']:
        metrics = comparison.get('fields', {})
        passed = comparison['comparison_status'] == 'computed' and not comparison.get('field_errors')
        checked = {}
        for name, limit in THRESHOLDS.items():
            value = metrics.get(name, {}).get('max_scaled')
            checked[name] = value
            passed = passed and value is not None and math.isfinite(value) and value <= limit
        comparison_screen.append({'candidate': comparison['candidate'], 'screen_pass': bool(passed),
                                  'max_scaled': checked,
                                  'raw_dgdt_max_scaled': {k: v['max_scaled'] for k, v in metrics.items() if k.startswith('dgdt.')}})
        # Compute this auxiliary diagnostic for the first repeat only. It is
        # an instantaneous bounded-alpha algebraic source, not a replay of the
        # sequential phase transport solve or a raw-dgdt equivalence test.
        if comparison.get('repeat') == 1 and passed:
            left = effective_source(Path(comparison['reference_time_directory']))
            right = effective_source(Path(comparison['candidate_time_directory']))
            delta = right-left
            sources.append({'candidate': comparison['candidate'],
                            'source_formula': 'alpha_i*(sum_j(alpha_j*dgdt_j)-dgdt_i)',
                            'max_abs_source_difference_per_s': float(np.max(np.abs(delta))),
                            'max_abs_dt_times_source_difference': float(np.max(np.abs(delta)))*float(data['delta_t']),
                            'rel_l2': float(np.linalg.norm(delta))/max(float(np.linalg.norm(left)),1e-300),
                            'acceptance_pass': None})
    result = {
        'benchmark': str(report.resolve()), 'timing': rows, 'state_screens': screens,
        'run_definition': {'steps': data['steps'], 'repeats': data['repeats'], 'delta_t': data['delta_t'],
                           'start_s': data['source_start_time_text'], 'end_s': data['runs'][0]['end_time']},
        'comparison_screens': comparison_screen, 'effective_source_diagnostics': sources,
        'engineering_thresholds': {'state_max_scaled': THRESHOLDS, 'alpha_roundoff_allowance': 1e-8,
                                   'alpha_sum_max_abs_error': 1e-9, 'max_step_relative_mass_imbalance': 1e-6},
        'screen_pass': bool(screens) and bool(comparison_screen)
                       and all(s['screen_pass'] for s in screens+comparison_screen),
        'scope': 'Main-agent engineering screen of this short SPUMA benchmark. Thresholds are not user-supplied. Raw dgdt equivalence, Foundation14 equivalence and long-time physical validity are not asserted.',
    }
    output.mkdir(parents=True, exist_ok=True)
    b.atomic_json(output/'validation.json', result)
    write_markdown(result, output)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    summary = summarize(args.benchmark, args.output)
    print(json.dumps({'engineering_screen_pass': summary['screen_pass'], 'report': str(args.output/'validation.json')}))
