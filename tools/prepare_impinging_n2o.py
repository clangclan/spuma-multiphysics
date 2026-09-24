#!/usr/bin/env python3
"""Generate two reproducible 90-degree liquid-N2O HEM collision benchmark cases."""
from __future__ import annotations
import argparse
import fcntl
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np
import yaml
import benchmark as common
from real_fluid_backend import RealFluidBackend

ATM=101325.0
SUPPLY=ATM+55e5
BACK_PRESSURES={'ambient_1atm':ATM,'ambient_40bar_abs':40e5}
PATCHES=('inletX','inletY','inletPlates','ambient')
BREAKS=((0.,5.,7.,20.),(0.,5.,7.,20.),(0.,4.,6.,10.)) # mm

def mesh_definition(spacing_mm):
    if not math.isfinite(spacing_mm) or spacing_mm<=0:raise ValueError('Positive finite cell size required')
    counts=[]
    for axis in BREAKS:
        ns=[int(round((b-a)/spacing_mm)) for a,b in zip(axis,axis[1:])]
        if any(n<1 or abs(n*spacing_mm-(b-a))>1e-9 for n,a,b in zip(ns,axis,axis[1:])):
            raise ValueError('Cell size must divide every nozzle/domain segment exactly')
        counts.append(ns)
    total=math.prod(sum(ns) for ns in counts)
    if not 100000<=total<=1000000:raise ValueError('Benchmark requires 100,000 to 1,000,000 cells')
    vertices=[(x,y,z) for z in BREAKS[2] for y in BREAKS[1] for x in BREAKS[0]]
    idx=lambda i,j,k:i+4*(j+4*k)
    blocks=[];faces={key:[] for key in PATCHES}
    for k in range(3):
        for j in range(3):
            for i in range(3):
                v=[idx(i,j,k),idx(i+1,j,k),idx(i+1,j+1,k),idx(i,j+1,k),
                   idx(i,j,k+1),idx(i+1,j,k+1),idx(i+1,j+1,k+1),idx(i,j+1,k+1)]
                blocks.append('hex ('+' '.join(map(str,v))+') ('+' '.join(str(counts[a][c]) for a,c in enumerate((i,j,k)))+') simpleGrading (1 1 1)')
                if i==0:faces['inletX' if j==1 and k==1 else 'inletPlates'].append([v[q] for q in (0,4,7,3)])
                if j==0:faces['inletY' if i==1 and k==1 else 'inletPlates'].append([v[q] for q in (0,1,5,4)])
                if i==2:faces['ambient'].append([v[q] for q in (1,2,6,5)])
                if j==2:faces['ambient'].append([v[q] for q in (3,7,6,2)])
                if k==0:faces['ambient'].append([v[q] for q in (0,3,2,1)])
                if k==2:faces['ambient'].append([v[q] for q in (4,5,6,7)])
    body='scale 0.001;\nvertices (\n'+'\n'.join('('+' '.join(map(str,v))+')' for v in vertices)+'\n);\nblocks (\n'+'\n'.join(blocks)+'\n);\nedges ();\nboundary (\n'
    for name,patch in faces.items():
        body+=name+' { type '+('wall' if name=='inletPlates' else 'patch')+'; faces (\n'+'\n'.join('('+' '.join(map(str,f))+')' for f in patch)+'\n); }\n'
    body+=');\nmergePatchPairs ();\n'
    return body,{'cells':total,'shape':[sum(n) for n in counts],'cellEdgeM':spacing_mm*1e-3,
      'cellVolumeM3':(spacing_mm*1e-3)**3,'domainM':[.02,.02,.01],
      'nozzleShape':'square','nozzleSideM':.002,'facesPerNozzle':int(round(2/spacing_mm))**2,
      'inletCentersM':[[0,.006,.005],[.006,0,.005]],'nominalDirections':[[1,0,0],[0,1,0]],
      'nominalIntersectionM':[.006,.006,.005],'angleDegrees':90}

def put(case,relative,body,klass='dictionary'):
    path=case/relative;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(common.header(path.name,klass)+body)

def uniform(case,name,value,dimensions):
    vector=isinstance(value,(tuple,list,np.ndarray));entry='('+' '.join(f'{v:.17g}' for v in value)+')' if vector else f'{value:.17g}'
    put(case,'0/'+name,f'dimensions [{dimensions}];\ninternalField uniform {entry};\nboundaryField {{ '+
        ' '.join(p+' { type '+('slip' if vector and p=='inletPlates' else 'zeroGradient')+'; }' for p in PATCHES)+' }\n',
        'volVectorField' if vector else 'volScalarField')

def system(case,end_time=1e-3):
    put(case,'system/controlDict',f'''application ReactiveFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {end_time:.17g};
deltaT 3e-8; maxDeltaT 3e-8; maxCo .25;
writeControl runTime; writeInterval 1e-5; writeFormat binary; writePrecision 17;
writeCompression off; timeFormat general; timePrecision 15; runTimeModifiable false; functions {{}};
''')
    put(case,'system/fvSchemes','ddtSchemes { default Euler; } gradSchemes { default Gauss linear; } divSchemes { default none; } laplacianSchemes { default Gauss linear uncorrected; } interpolationSchemes { default linear; } snGradSchemes { default uncorrected; }\n')
    put(case,'system/fvSolution','solvers {}\n')

def prepare(output,configuration,spacing_mm=.25,temperature=293.15,turbulence='WALE',
            *, cells_per_axis=None, nozzle_diameter_mm=5., domain_mm=80., inlet_z_mm=40.):
    out=Path(output).resolve();source=Path(configuration).resolve()
    if cells_per_axis is None:
        body,geometry=mesh_definition(spacing_mm)
    else:
        from impinging_cartesian_mesh import definition, write_mesh
        body=None;geometry=definition(cells_per_axis,nozzle_diameter_mm,domain_mm,inlet_z_mm)
    if out.exists():raise ValueError('Use a new output directory')
    if turbulence not in ('none','WALE'):raise ValueError('Unknown turbulence selection')
    if not math.isfinite(temperature) or temperature<=0:raise ValueError('Invalid temperature')
    settings=yaml.safe_load(source.read_text());mechanism=Path(settings['mechanism'])
    if not mechanism.is_absolute():mechanism=source.parent/mechanism
    if any('mechanism' in c for c in settings['condensables']):raise ValueError('This benchmark requires a shared mechanism file')
    out.mkdir(parents=True);thermo=out/'thermo';thermo.mkdir()
    shutil.copy2(mechanism,thermo/'cold-pr.yaml');settings['mechanism']=str(thermo/'cold-pr.yaml')
    local_config=thermo/'cold-pr-config.yaml';local_config.write_text(yaml.safe_dump(settings,allow_unicode=True,sort_keys=False))
    mesh=out/'mesh';system(mesh)
    if body is not None:put(mesh,'system/blockMeshDict',body)
    else:
        write_mesh(mesh,geometry)
        common.atomic_json(mesh/'mesh-generation.json',geometry)
    env=common.sourced_environment()
    check_mesh=['checkMesh','-case',str(mesh),'-allGeometry','-allTopology']
    check_env=env
    if not shutil.which('checkMesh',path=env['PATH']):
        # The minimal local SPUMA build ships blockMesh only. A standard CPU
        # OpenFOAM reader can independently check the same native mesh files.
        foam_env=Path('/opt/openfoam14/etc/bashrc')
        if not foam_env.is_file():raise RuntimeError('checkMesh missing; build it in the selected OpenFOAM environment')
        check_mesh=['bash','--noprofile','--norc','-c',
            'source "$1" ParaView_TYPE=none >/dev/null && exec checkMesh -case "$2" -allGeometry -allTopology',
            'mesh-check',str(foam_env),str(mesh)]
        import os
        check_env=os.environ.copy()
    commands=[] if body is None else [(['blockMesh','-case',str(mesh)],'blockMesh.log',env)]
    commands.append((check_mesh,'checkMesh.log',check_env))
    for command,log,selected_env in commands:
        with (mesh/log).open('w') as stream:subprocess.run(command,env=selected_env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=180)
    text=(mesh/'checkMesh.log').read_text()
    if 'Mesh OK.' not in text:raise RuntimeError('Mesh checks failed: '+str(mesh/'checkMesh.log'))
    if int(re.search(r'^\s*cells:\s+(\d+)',text,re.M).group(1))!=geometry['cells']:raise RuntimeError('Unexpected cell count')
    boundary=(mesh/'constant/polyMesh/boundary').read_text()
    for patch in ('inletX','inletY'):
        count=int(re.search(r'\b'+patch+r'\s*\{[^}]*\bnFaces\s+(\d+);',boundary,re.S).group(1))
        if count!=geometry['facesPerNozzle']:raise RuntimeError('Unexpected inlet face count')
    definitions={}
    with RealFluidBackend(local_config) as b:
        if b.names!=['N2','O2','N2O','IC3H7OH']:raise ValueError('Benchmark requires the versioned four-species cold model')
        solid=[phase for phase in b.condensed if phase['kind']=='solid']
        if solid:
            expected=[('liquid','N2O'),('solid','N2O')]
            actual=[(phase['kind'],phase['species']) for phase in b.condensed]
            if actual!=expected:raise ValueError(f'Solid-enabled benchmark requires liquid/solid N2O slots, got {actual}')
        elif not b.condensed or b.condensed[0]['kind']!='liquid' or b.condensed[0]['species']!='N2O':
            raise ValueError('Benchmark requires liquid N2O in condensed slot 0')
        inlet_mass,inlet_energy,inlet=b.make_state(temperature,SUPPLY,{'N2O':1},(1,0))
        recovered=b.recover(inlet_mass,inlet_energy,inlet)
        if abs(recovered.p/SUPPLY-1)>1e-6 or abs(recovered.T/temperature-1)>1e-6 or abs(recovered.liquidMass[0]/recovered.rho-1)>1e-10:
            raise ValueError('Requested inlet is not an equilibrium pure liquid in this EOS')
        ambient_y=b.mole_to_mass({'N2':.79,'O2':.21});inlet_y=b.vector({'N2O':1})
        physical='closure=HEM;chemistry=0;viscosity=0;conductivity=0;commonD=0'
        if turbulence=='WALE':physical+=f';turbulence=WALE-stress-v1;Cw={.325:.17g};filter=cubeRootVolume;sgsScalarClosure=none;sgsK=none'
        numerical=f'chemicalRtol={1e-8:.17g};chemicalAtol={1e-14:.17g};waveFactor={1.1:.17g};transportBackend=cuda;transportGasProperties=auto'
        b.check(b.lib.reactive_rt_set_case_context(b.handle,physical.encode(),numerical.encode()))
        for name,back in BACK_PRESSURES.items():
            case=out/name;system(case)
            # Ordinary copies keep each case independent when opened/modified by the user.
            shutil.copytree(mesh/'constant/polyMesh',case/'constant/polyMesh')
            if body is not None:put(case,'system/blockMeshDict',body)
            q,e,state=b.make_state(temperature,back,ambient_y,(0,0));state=b.recover(q,e,state)
            uniform(case,'p',state.p,'1 -1 -2 0 0 0 0');uniform(case,'T',state.T,'0 0 0 1 0 0 0')
            uniform(case,'U',(0,0,0),'0 1 -1 0 0 0 0')
            for k,v in enumerate(q):uniform(case,f'q{k}',v,'1 -3 0 0 0 0 0')
            uniform(case,'rhoMomentum',(0,0,0),'1 -2 -1 0 0 0 0');uniform(case,'rhoTotalEnergy',e,'1 -1 -2 0 0 0 0')
            ytext=lambda a:'('+' '.join(f'{v:.17g}' for v in a)+')'
            inlet_dict=f'type fixedState; p {SUPPLY:.17g}; T {temperature:.17g}; U (0 0 0); Y {ytext(inlet_y)}; liquidFractions (1 0);'
            ambient_dict=f'type fixedState; p {back:.17g}; T {temperature:.17g}; U (0 0 0); Y {ytext(ambient_y)}; liquidFractions (0 0);'
            put(case,'constant/reactiveProperties',f'''closure HEM; thermoConfiguration "{local_config}"; initialization conserved;
chemistry false; dynamicViscosity 0; thermalConductivity 0; molecularDiffusivity 0;
physics {{ chemistry false; phaseChange true; viscosity false; heatConduction false; speciesDiffusion false; turbulence {turbulence}; surfaceTension false; }}
waleCw .325; transportBackend cuda; closureBackend cuda; closureCpuFallback false;
closureScalarBackend cpu; closureJacobian analytic; thermoExactReuse true;
thermoWorkers 1; thermoBatchCells 4096; maxThermoBatchMemoryMB 128;
maxDeviceMemoryGB 4; maxHostMemoryGB 8; waveSpeedFactor 1.1;
checkpointFirstStep true; checkpointOnFailure true;
boundaryConditions {{
 inletX {{ {inlet_dict} }}
 inletY {{ {inlet_dict} }}
 inletPlates {{ type slipWall; }}
 ambient {{ {ambient_dict} }}
}}
''')
            put(case,'0/reactiveStateIdentity',f'identitySchema 2; fingerprint "{b.fingerprint}"; speciesCount {b.ns}; closure HEM; physicalModelHash "{b.physical_hash}"; numericalPolicyHash "{b.policy_hash}";\n')
            (case/'impinging_n2o.foam').touch()
            definition={'schema':1,'name':name,'geometry':geometry,'species':b.names,'n2oSpeciesIndex':b.names.index('N2O'),'n2oLiquidIndex':0,
               'n2oSolidIndex':solid[0]['slot'] if solid else None,'condensedPhases':b.condensed,
               'temperatureK':temperature,'supplyGaugePa':55e5,'gaugeReferencePa':ATM,'supplyAbsolutePa':SUPPLY,
               'ambientAbsolutePa':back,'pressureDifferencePa':SUPPLY-back,
               'inletPhase':'equilibrium pure liquid N2O','ambientMoleFractions':{'N2':.79,'O2':.21},
               'boundaryModel':'stationary fixedState reservoir ghost with HLL pressure-driven flux; not a total-pressure/nozzle solution',
               'closure':'HEM','phaseChange':True,'chemistry':False,'turbulence':turbulence,'cpuFallback':False,
               'surfaceTension':False,'geometricInterface':False,'capillaryFlowCoupling':False,'primaryBreakupValidated':False,
               'maxDeltaT':3e-8,'requestedEndTime':1e-3,'physicalModelHash':b.physical_hash,'numericalPolicyHash':b.policy_hash,
               'inletState':recovered.as_dict(),'ambientState':state.as_dict()}
            definition['inputHashes']={str(p.relative_to(case)):common.sha256(p) for d in ('0','constant','system') for p in sorted((case/d).rglob('*')) if p.is_file()}
            common.atomic_json(case/'benchmark-definition.json',definition);definitions[name]=definition
    matrix={'schema':1,'purpose':'90-degree pressure-driven liquid N2O impingement and equilibrium flashing',
      'cases':list(definitions),'geometry':geometry,'supplyAbsolutePa':SUPPLY,'temperatureK':temperature,
      'sourceConfiguration':str(source),'sourceConfigurationSha256':common.sha256(source),'sourceMechanismSha256':common.sha256(mechanism),
      'localThermoHashes':{str(p.relative_to(out)):common.sha256(p) for p in thermo.iterdir()},
      'generatorSha256':common.sha256(Path(__file__)),'meshCheck':'mesh/checkMesh.log','meshCheckCommand':check_mesh,
      'limits':['fixedState reservoir is not a resolved nozzle or characteristic total-pressure inlet','HEM flash is thermodynamic equilibrium, not finite-rate nucleation','no geometric interface/surface-tension breakup prediction','30 ns is an upper bound; solver may reduce it for CFL or recovery']}
    if cells_per_axis is not None:
        matrix['meshGeneratorSha256']=common.sha256(Path(__file__).with_name('impinging_cartesian_mesh.py'))
    common.atomic_json(out/'benchmark-matrix.json',matrix)
    (out/'README.ko.md').write_text(f'''# 90도 액체 N₂O 충돌 벤치마크\n\n두 공급구 모두 {temperature-273.15:g}°C, 55 bar(g)=56.01325 bar(abs). 외부는 각각 1.01325 bar(abs), 40 bar(abs).\n\n메시: {geometry['cells']:,}개 동일 정육면체 셀, 변 길이 {spacing_mm:g} mm. 영역 20 × 20 × 10 mm, 각 정사각 분사구 2 × 2 mm. 공칭 축 교점 (6,6,5) mm.\n\n초기 공간은 정지한 79% N₂/21% O₂ 공기(몰분율)이며 입구 플럭스는 공급구와 공간의 압력 차로 발생한다. `fixedState U=0` 저장조 ghost 경계이며 노즐 출구 속도나 실제 초킹 유량을 지정/검증한 조건은 아니다. 외부 열린 면도 배압 공기 저장조 ghost, 공급 판은 slipWall이다.\n\n상평형·상변화와 CUDA 수송/열역학을 켰고 CPU fallback은 껐다. 기본 WALE는 운동량 응력 범위이며 열/종 SGS는 없다. 계면을 해상하는 VOF·표면장력·액적 분열 해석으로 해석하면 안 된다.\n\n`maxDeltaT=30 ns`, 예비 종료시간 1 ms, 저장 간격 10 μs. 종료시간까지 실행한 결과는 아직 아니다. CFL/복원에 따라 dt가 작아질 수 있다.\n\n환경을 준비한 뒤 `ReactiveFoam -case <케이스 경로>`로 실행한다. `.foam` 파일로 ParaView에서 연다. 전체 실행 전 `tools/validate_impinging_n2o.py --benchmark <이 폴더> --output <새 검증 폴더>`로 각 케이스의 1스텝을 확인할 수 있다.\n''')
    if solid:
        data=settings['condensables'][solid[0]['slot']]['solid']
        with (out/'README.ko.md').open('a') as stream:
            stream.write(f"\n고체 N₂O 프로필: 액체 N₂O와 고체 N₂O가 응축상 슬롯을 공유하며 IPA 액상은 포함하지 않는다. "
                f"고체 적용 범위는 {data['temperature-table'][0]:g}–{data['temperature-table'][-1]:g} K, "
                f"절대압 {data['maximum-pressure']/1e5:g} bar 이하이고 기체 공존이 필요하다. "
                "고체 물성은 잠정 모델이며 독립 승화압 정확도 검증을 별도로 확인해야 한다. "
                "현재 고체 프로필은 GPU 내 유한차분 Jacobian을 사용하므로 계산 비용이 크게 늘 수 있다. "
                "저장소 docs/benchmarks/impinging-n2o-solid-recovery.ko.md에 검증 결과와 제한을 기록했다.\n")
    if cells_per_axis is not None:
        path=out/'README.ko.md';text=path.read_text()
        old=f"메시: {geometry['cells']:,}개 동일 정육면체 셀, 변 길이 {spacing_mm:g} mm. 영역 20 × 20 × 10 mm, 각 정사각 분사구 2 × 2 mm."
        new=(f"메시: {cells_per_axis}³ = {geometry['cells']:,}셀, 영역 {domain_mm:g} × {domain_mm:g} × {domain_mm:g} mm. "
             f"셀 간격 {np.array(geometry['cellSpacingM'])*1000} mm. 각 원형 분사구 지름 {nozzle_diameter_mm:g} mm. "
             f"원 내부에 중심이 있는 경계 면을 선택했으며 면적 오차는 {geometry['nozzleAreaRelativeError']:.4%}. "
             "셀 절단·변형 없이 격자 경계에서 원을 근사한다.")
        path.write_text(text.replace(old,new).replace('교점 (6,6,5) mm',f'교점 (6,6,{inlet_z_mm:g}) mm'))
    return matrix

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);ap.add_argument('--configuration',required=True,type=Path)
    ap.add_argument('--cell-mm',type=float,default=.25);ap.add_argument('--temperature',type=float,default=293.15);ap.add_argument('--turbulence',choices=('none','WALE'),default='WALE')
    ap.add_argument('--cells-per-axis',type=int,choices=(40,80,160));ap.add_argument('--nozzle-diameter-mm',type=float,default=5.)
    ap.add_argument('--domain-mm',type=float,default=80.)
    ap.add_argument('--inlet-z-mm',type=float,default=40.)
    a=ap.parse_args()
    with common.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);report=prepare(a.output,a.configuration,a.cell_mm,a.temperature,a.turbulence,
            cells_per_axis=a.cells_per_axis,nozzle_diameter_mm=a.nozzle_diameter_mm,domain_mm=a.domain_mm,inlet_z_mm=a.inlet_z_mm)
    print(json.dumps(report))
if __name__=='__main__':main()
