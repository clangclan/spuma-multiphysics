#!/usr/bin/env python3
"""Prepare the requested 80 mm cube / 5 mm circular-port GPU cost matrix."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import re
import benchmark as common
from prepare_impinging_n2o import prepare
from prepare_capillary_impingement import prepare as capillary
from check_impinging_n2o_mesh import check


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path)
    ap.add_argument('--configuration',type=Path,default=common.PROJECT_ROOT/'examples/impinging-n2o-supercooled/cold-pr-148K-config.yaml')
    ap.add_argument('--grids',nargs='+',type=int,choices=(40,80,160),default=[40,80,160])
    ap.add_argument('--inlet-z-mm',type=float,default=40.)
    ap.add_argument('--end-time',type=float,help='Run to this physical time instead of stopping after one step')
    ap.add_argument('--max-delta-t',type=float,default=1e-6,
                    help='Safety ceiling in seconds; current-state CFL selects the actual step (default: 1e-6)')
    ap.add_argument('--max-co',type=float,default=.25)
    ap.add_argument('--thermo-batch-cells',type=int,default=1048576,
                    help='CUDA HEM scheduling capacity (default: 1048576, capped to mesh cells)')
    ap.add_argument('--disable-thermo-exact-reuse',action='store_true',
                    help='Evaluate every cell independently for a reference run')
    a=ap.parse_args()
    if a.end_time is not None and (not math.isfinite(a.end_time) or a.end_time<=0):ap.error('Positive finite end time required')
    if not math.isfinite(a.max_delta_t) or a.max_delta_t<=0:ap.error('Positive finite max delta t required')
    if not math.isfinite(a.max_co) or not 0<a.max_co<=.5:ap.error('Require 0 < maxCo <= 0.5')
    if not 1<=a.thermo_batch_cells<=1048576:ap.error('Require 1 <= thermo batch cells <= 1048576')
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    result=dict(domainM=[.08]*3,nozzleDiameterM=.005,inletZmm=a.inlet_z_mm,cases=[],
        requestedEndTime=a.end_time,
        timeStepControl=dict(mode='current-state-wave-diffusion-capillary-CFL',maxCo=a.max_co,maxDeltaT=a.max_delta_t),
        scope='Transient GPU benchmark to requested end time' if a.end_time is not None else 'One initial accepted GPU step',
        generatorHashes={name:common.sha256(Path(__file__).with_name(name)) for name in
            ('prepare_impinging_resolution_series.py','prepare_impinging_n2o.py','impinging_cartesian_mesh.py',
             'prepare_capillary_impingement.py','check_impinging_n2o_mesh.py')})
    with common.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for n in a.grids:
            folder=root/f'n{n}';base=folder/'base'
            prepare(base,a.configuration,cells_per_axis=n,nozzle_diameter_mm=5.,domain_mm=80.,inlet_z_mm=a.inlet_z_mm)
            checked=check(base);common.atomic_json(folder/'mesh-validation.json',checked)
            cases=capillary(base,folder/'full-physics',.0020577273399028486,1)
            for name in cases:
                case=Path(name);prop=case/'constant/reactiveProperties'
                text=prop.read_text()
                # Resource guards, not additional allocations or changed physics.
                text=re.sub(r'maxDeviceMemoryGB\s+[^;]+;', 'maxDeviceMemoryGB 12;',text)
                text=re.sub(r'maxHostMemoryGB\s+[^;]+;', 'maxHostMemoryGB 32;',text)
                batch=min(a.thermo_batch_cells,n**3)
                batch_budget=max(128,batch*.002)
                reuse=not a.disable_thermo_exact_reuse
                for key,value in (('thermoBatchCells',str(batch)),('maxThermoBatchMemoryMB',f'{batch_budget:.17g}'),
                                  ('thermoExactReuse',str(reuse).lower())):
                    text,matches=re.subn(r'\b'+key+r'\s+[^;]+;',f'{key} {value};',text)
                    if matches!=1:raise ValueError(f'Expected one {key} in {prop}')
                prop.write_text(text)
                path=case/'benchmark-definition.json';definition=json.loads(path.read_text())
                definition['resourceLimits']=dict(maxDeviceMemoryGB=12,maxHostMemoryGB=32)
                definition['thermoScheduling']=dict(batchCells=batch,exactReuse=reuse,maxBatchMemoryMB=batch_budget,
                    scope='Exact batch-local keys include conserved input, complete seed, color and pressure jump')
                controls=case/'system/controlDict';text=controls.read_text()
                text=re.sub(r'\bmaxDeltaT\s+[^;]+;',f'maxDeltaT {a.max_delta_t:.17g};',text)
                text=re.sub(r'\bmaxCo\s+[^;]+;',f'maxCo {a.max_co:.17g};',text)
                if a.end_time is not None:
                    text=re.sub(r'\bendTime\s+[^;]+;',f'endTime {a.end_time:.17g};',text)
                    text=re.sub(r'\bmaxAcceptedSteps\s+[^;]+;','maxAcceptedSteps 0;',text)
                    definition.update(requestedEndTime=a.end_time,maxAcceptedSteps=0)
                controls.write_text(text)
                definition.update(maxDeltaT=a.max_delta_t,maxCo=a.max_co,
                    timeStepControl='current-state-wave-diffusion-capillary-CFL')
                definition['inputHashes']={str(p.relative_to(case)):common.sha256(p)
                    for d in ('0','constant','system') for p in sorted((case/d).rglob('*')) if p.is_file()}
                common.atomic_json(path,definition)
                (case/'impinging_n2o.foam').touch()
                result['cases'].append(name)
            if a.end_time is not None:
                path=folder/'full-physics/capillary-benchmark.json';metadata=json.loads(path.read_text())
                metadata.update(requestedEndTime=a.end_time,maxAcceptedSteps=0)
                common.atomic_json(path,metadata)
            common.atomic_json(root/f'prepared-{n}.json',result)
            print(json.dumps(dict(grid=n,cells=n**3,cases=cases,meshPassed=checked['passed'])),flush=True)
    common.atomic_json(root/'benchmark-series.json',result)
    mesh_rows=[]
    for n in a.grids:
        g=json.loads((root/f'n{n}/base/benchmark-matrix.json').read_text())['geometry']
        mesh_rows.append(f"- {n}³: {n**3:,}셀, 셀 변 {80/n:g} mm, 입구 면적 오차 {g['nozzleAreaRelativeError']:+.2%}")
    termination=(f'종료 시간 {a.end_time:g} s, 스텝 수 제한 없음.' if a.end_time is not None else '초기 1스텝만 실행.')
    (root/'README.ko.md').write_text(f'''# N₂O 충돌 — 80 mm 정육면체 GPU 벤치마크

실행 대상은 각 메시 폴더 아래 `full-physics/ambient_1atm` 및
`full-physics/ambient_40bar_abs`다. `base`는 준비용 중간 입력이며 물리 설정이 다르다.

영역 80×80×80 mm, 원형 공급구 지름 5 mm, 90도 분사축 교점 (6,6,{a.inlet_z_mm:g}) mm.
원 안에 중심이 있는 경계 면을 입구로 사용한다. 셀 절단 없이 원을 근사하므로
입구 면적에 격자 근사 오차가 있다.

{chr(10).join(mesh_rows)}

두 공급구 모두 20°C 액체 N₂O, 55 bar(g)=56.01325 bar(abs).
외부는 20°C 공기(N₂:O₂=79:21 몰비), 1 atm 또는 40 bar(abs).
공급구와 외부는 기존 정지 저장조 fixedState 경계, 공급판은 slipWall을 유지한다.

상평형 flashing, 과냉각 액체, 점성, 열전도, WALE 응력, 난류 열·종 혼합,
상수 표면장력 0.0020577273399028486 N/m을 켰다. 기존 diffuse 계면 모델을 유지한다.
고체·승화·연소·분자 종 확산은 껐다. 수송·열역학·WALE PR 물성은 CUDA,
CPU fallback은 금지한다. CPU 초기 상태 준비·메시·출력 작업은 존재한다.

열역학 배치는 최대 {a.thermo_batch_cells:,}셀(메시 셀 수로 제한),
정확한 배치 내 중복 입력 재사용은 {str(not a.disable_thermo_exact_reuse).lower()}다.
재사용 키는 종 밀도·bulk energy·전체 seed·색 함수·압력 점프를 포함한다.
회귀용 기준 설정은 `--thermo-batch-cells 4096 --disable-thermo-exact-reuse`로 만든다.

{termination} `maxDeltaT={a.max_delta_t:g} s`, `maxCo={a.max_co:g}`.
솔버는 현재 유속·음속·확산·모세관 제한을 매 스텝 다시 계산한다.
`maxDeltaT`는 상한이며, CFL·단계 재검사·복원 재시도에 따라 실제 dt를 줄인다.
발달한 충돌류·액적 분열의 검증 여부는 실제 결과를 확인해야 한다.
''')


if __name__=='__main__':main()
