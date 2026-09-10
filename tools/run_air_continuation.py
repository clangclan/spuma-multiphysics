#!/usr/bin/env python3
"""MAIN: isolated pure-air continuation of the supplied rhoPimpleFoam case."""
import argparse
from decimal import Decimal
import fcntl
import json
from pathlib import Path
import re
import subprocess
import numpy as np
import benchmark as b
from prepare_minicase import header


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',default='0.00049');p.add_argument('--duration',default='1e-7')
    p.add_argument('--linear-tolerance',type=float,default=1e-13);p.add_argument('--steps',type=int,default=4);p.add_argument('--outer',type=int,default=4)
    p.add_argument('--wave-bound',type=float,default=1.5,help='Acceptance ceiling for measured wave Co with fixed dt; distinct from adaptive maxWaveCo')
    a=p.parse_args();source=a.source.resolve();case=a.output.resolve();start=Decimal(a.start)
    duration=Decimal(a.duration);end=start+duration;dt=duration/a.steps
    if a.steps<1 or duration<=0 or not np.isfinite(a.wave_bound) or a.wave_bound<=0 or case.exists() or b.is_relative_to(case,source):raise ValueError('New isolated output and positive duration/steps/wave bound required')
    env=b.sourced_environment();initial=source/a.start
    with b.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        protected=[f for folder in [source/'constant',source/'system'] for f in folder.rglob('*') if f.is_file()]
        protected += [initial/name for name in ['p','T','U','rho','phi']]
        before={str(f):b.sha256(f) for f in protected}
        case.mkdir(parents=True)
        for folder in ['constant','system']:b.copy_case(source/folder,case/folder)
        target=case/a.start;target.mkdir()
        for name in ['p','T','U','rho']:
            subprocess.run(['cp','-a','--reflink=auto','--',str(initial/name),str(target/name)],check=True)
        pressure=(target/'p').read_bytes()
        match=re.search(rb'FoamFile\s*\{(.*?)\}',pressure,re.S)
        head,count=re.subn(rb'(\bobject\s+)p(\s*;)',rb'\1p_rgh\2',match[0])
        if count!=1:raise ValueError('Cannot rename pressure header')
        (target/'p_rgh').write_bytes(pressure[:match.start()]+head+pressure[match.end():])
        del pressure
        flux=(initial/'phi').read_bytes()
        match=re.search(rb'FoamFile\s*\{(.*?)\}',flux,re.S)
        head,count=re.subn(rb'(\bobject\s+)phi(\s*;)',rb'\1pintleInitialMassFlux\2',match[0])
        if count!=1:raise ValueError('Cannot rename mass-flux header')
        (target/'pintleInitialMassFlux').write_bytes(flux[:match.start()]+head+flux[match.end():])
        del flux
        boundaries=(case/'constant/polyMesh/boundary').read_text()
        patches=re.findall(r'\b(\w+)\s*\{\s*type\s+(\w+)\s*;',boundaries)
        if len(patches)!=9:raise ValueError('Expected the nine supplied physical boundary patches')
        for phase in b.PHASES:
            value=int(phase=='air')
            bc=' '.join(f'{name} {{ type zeroGradient; }}' for name,_ in patches)
            (target/('alpha.'+phase)).write_text(header('alpha.'+phase,'volScalarField')+f'dimensions [0 0 0 0 0 0 0]; internalField uniform {value}; boundaryField {{ {bc} }}\n')
        (case/'constant/g').write_text(header('g','uniformDimensionedVectorField')+'dimensions [0 1 -2 0 0 0 0]; value (0 0 0);\n')
        (case/'constant/thermophysicalProperties').write_text(header('thermophysicalProperties')+'phases (ipa n2o air); sigmas ((ipa n2o) 0 (ipa air) 0 (n2o air) 0); pMin 1;\n')
        for phase in b.PHASES:
            name='thermophysicalProperties.'+phase
            (case/'constant'/name).write_text(header(name)+'''thermoType { type pintleDeviceHeRhoThermo; device false; mixture pureMixture; transport const; thermo hConst;
equationOfState perfectGas; specie specie; energy sensibleInternalEnergy; }
mixture { specie { molWeight 28.96; } thermodynamics { Cp 1005; Hf 0; } transport { mu 1.8e-5; Pr .7; } } updateT false;
''')
        (case/'system/controlDict').write_text(header('controlDict')+f'''application spumaPintleColdFoam;
startFrom startTime; startTime {start}; stopAt endTime; endTime {end}; deltaT {dt}; adjustTimeStep no;
maxCo .25; maxAlphaCo .15; maxWaveCo .5;
writeControl runTime; writeInterval {duration}; writeFormat binary; writeCompression off; writePrecision 17;
timePrecision 12; runTimeModifiable false; functions {{}} libs ("libpintleMultiphaseThermo.so");
pintleGasDynamics true; pintleGasPhase air; pintleGasResidualTolerance 1e-4;
pintleStrictState true; pintleMassTolerance 1e-6; pintleLimiter gpu; pintleSurfaceCache true;
''')
        (case/'system/fvSchemes').write_text(header('fvSchemes')+'''ddtSchemes { default Euler; } gradSchemes { default Gauss linear; }
divSchemes { default Gauss linear; div(rhoPhi,U) Gauss upwind; div(rhoPhi,T) Gauss upwind; div(rhoPhi,K) Gauss upwind;
div(rhoPhi,e) Gauss upwind; div(phi,p) Gauss upwind; div(phi,rho) Gauss upwind; div(phid,p_rgh) Gauss upwind; }
laplacianSchemes { default Gauss linear corrected; } interpolationSchemes { default linear; }
snGradSchemes { default corrected; } fluxRequired { default no; p_rgh; }
''')
        (case/'system/fvSolution').write_text(header('fvSolution')+f'''solvers {{
p_rgh {{ solver PBiCGStab; preconditioner diagonal; tolerance {a.linear_tolerance:.17g}; relTol 0; minIter 1; maxIter 1000; }} p_rghFinal {{ $p_rgh; }}
U {{ solver PBiCGStab; preconditioner diagonal; tolerance {a.linear_tolerance:.17g}; relTol 0; minIter 1; maxIter 1000; }} UFinal {{ $U; }}
T {{ solver PBiCGStab; preconditioner diagonal; tolerance {a.linear_tolerance:.17g}; relTol 0; minIter 1; maxIter 1000; }} TFinal {{ $T; }}
"alpha.*" {{ nAlphaSubCycles 1; cAlpha 0; }} }}
PIMPLE {{ momentumPredictor yes; nOuterCorrectors {a.outer}; nCorrectors 2; nNonOrthogonalCorrectors 0; }}
relaxationFactors {{ equations {{ ".*" 1; }} }}
''')
        definition=dict(source=str(source),start_time=str(start),end_time=str(end),delta_t=str(dt),steps=a.steps,
            outer_correctors=a.outer,linear_tolerance=a.linear_tolerance,fixed_step_wave_acceptance_bound=a.wave_bound,
            harness_sha256=b.sha256(Path(__file__)),executable_sha256=b.sha256(b.DEFAULT_EXECUTABLE),
            thermo_library_sha256=b.sha256(b.PROJECT_ROOT/'lib/libpintleMultiphaseThermo.so'),protected_before=before,
            input_sha256={str(f.relative_to(case)):b.sha256(f) for folder in ['system','constant',a.start] for f in (case/folder).rglob('*') if f.is_file()},
            scope='Pure-air branch of an existing solution; fixed timestep is explicitly recorded; this is not multiphase shock validation.')
        b.atomic_json(case/'run-definition.json',definition)
        run=b.run_solver(case,b.DEFAULT_EXECUTABLE,definition,env,max(600,a.steps*30))
        log=(case/'stdout.log').read_text();diagnostics={}
        records={}
        for key in ['waveCo','maxMach','massResidualMax','energyResidualMax','energyBalanceRelative']:
            values=[float(v) for v in re.findall(r'\b'+key+r'=(\S+)',log)]
            records[key]=values
            if values:diagnostics[key+'_maximum_abs']=max(abs(v) for v in values) if np.isfinite(values).all() else None
        result=dict(run=run,diagnostics=diagnostics,source_unchanged=all(b.sha256(Path(f))==h for f,h in before.items()))
        imports=[float(v) for v in re.findall(r'PINTLE_GAS_RESTART importedMassFlux=1 relativeError=(\S+)',log)]
        complete_records=all(len(v)==a.steps*(2 if key=='maxMach' else 1) and np.isfinite(v).all() for key,v in records.items())
        checks=dict(completed=run['completed_ok'],source_unchanged=result['source_unchanged'],
            gas_mode='PINTLE_GAS_MODE active=1 phase=air' in log,
            restart_import=len(imports)==1 and bool(np.isfinite(imports).all()) and abs(imports[0])<=1e-12,
            complete_finite_records=bool(complete_records))
        for key,bound in [('waveCo',a.wave_bound),('massResidualMax',1e-4),('energyResidualMax',1e-4),('energyBalanceRelative',1e-6)]:
            checks[key]=bool(complete_records and max(abs(v) for v in records[key])<=bound)
        if run['completed_ok']:
            final=b.expected_final_directory(case,end)
            result['field_health']=b.field_health_report(final)
            health=result['field_health'];fields=health['fields']
            checks['fields']=bool(not health['field_errors'] and not health['missing_core_fields'] and not health['missing_phase_fields']
                and all(v['finite'] for v in fields.values()) and all(fields[name]['min']>0 for name in ['p','T','rho'])
                and health['alpha_sum']['max_abs_error_from_one']<=1e-12)
            reference=source/format(end,'f').rstrip('0')
            if reference.is_dir():
                result['comparison_to_original']={name:b.compare_fields(reference/name,final/name) for name in ['p','T','rho','U']}
            result['evidence_sha256']={str(f):b.sha256(f) for f in b.logical_field_files(final).values()}
        result['acceptance']=dict(passed=all(checks.values()),checks=checks,
            wave_scope='Measured fixed-dt wave Co ceiling; maxWaveCo in controlDict only controls adaptive timestepping.')
        b.atomic_json(case/'result.json',result)
        print(json.dumps({'completed':run['completed_ok'],'elapsed_s':run['elapsed_wall_s'],'steps':run['completed_steps'],'diagnostics':diagnostics},indent=2))
        if not result['acceptance']['passed']:raise RuntimeError('Continuation failed; see '+str(case/'stdout.log'))


if __name__=='__main__':main()
