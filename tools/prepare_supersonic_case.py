#!/usr/bin/env python3
"""MAIN: independent three-phase pure-gas benchmarks; all inputs are generated."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import benchmark as b
from prepare_minicase import header
from supersonic_reference import AIR_R, GAMMA, normal_shock, area_mach, acoustic_state


def prepare(case,kind='sod',cells=128,steps=160,outer=2,gas='air',mean_mach=0.,new_physics=False):
    if not np.isfinite(mean_mach) or mean_mach<=-1:raise ValueError('mean Mach must be finite and > -1')
    if kind=='weak-shock' and cells%4:raise ValueError('Weak-shock front requires cells divisible by four')
    if kind=='nozzle' and steps<8*cells:raise ValueError('Nozzle needs at least 8*cells steps for the axial acoustic CFL')
    if cells<16 or cells%2 or steps<4:raise ValueError('Even cells>=16 and steps>=4 required')
    if gas not in ['air','n2o'] or (gas=='n2o' and kind!='acoustic'):raise ValueError('PR benchmark is acoustic only')
    case=Path(case).resolve();case.mkdir(parents=True,exist_ok=False)
    def put(name,text):
        p=case/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
    x=(np.arange(cells)+.5)/cells
    rho=np.ones(cells);velocity=np.zeros((cells,3));pressure=np.full(cells,1e5)
    temperature=np.full(cells,300.);a0=np.sqrt(GAMMA*AIR_R*300)
    if kind=='sod':
        rho=np.where(x<.5,1.,.125);pressure=np.where(x<.5,1e5,1e4)
        temperature=pressure/(rho*AIR_R);end=5e-4
    elif kind=='weak-shock':
        ratios=normal_shock(1.1);rho0=1e5/(AIR_R*300)
        rho=np.where(x<.25,rho0*ratios['density_ratio'],rho0)
        pressure=np.where(x<.25,1e5*ratios['pressure_ratio'],1e5)
        temperature=pressure/(rho*AIR_R)
        velocity[:,0]=np.where(x<.25,1.1*a0*(1-1/ratios['density_ratio']),0);end=5e-4
    elif kind=='normal-shock':
        ratios=normal_shock(1.1);rho0=1e5/(AIR_R*300);u1=1.1*a0
        rho=np.where(x<.5,rho0,rho0*ratios['density_ratio'])
        pressure=np.where(x<.5,1e5,1e5*ratios['pressure_ratio'])
        temperature=pressure/(rho*AIR_R)
        velocity[:,0]=np.where(x<.5,u1,u1/ratios['density_ratio']);end=5e-4
    elif kind=='acoustic':
        initial,base=acoustic_state(x,mean_mach=mean_mach,gas=gas)
        pressure=initial['p'];rho=initial['rho'];temperature=initial['T'];velocity[:,0]=initial['U']
        end=1/(4*base['a']*(1+mean_mach))
    elif kind=='nozzle':
        area=.001+.002*np.abs(x-.5)
        mach=np.array([area_mach(s/.001,t>.5) for s,t in zip(area,x)])
        temperature=300/(1+(GAMMA-1)*mach*mach/2)
        pressure=2e5*(temperature/300)**(GAMMA/(GAMMA-1))
        rho=pressure/(AIR_R*temperature);velocity[:,0]=mach*np.sqrt(GAMMA*AIR_R*temperature)
        end=.002
    else:raise ValueError(kind)
    dt=end/steps
    if kind=='nozzle':
        vertices='((0 -.001 0) (.5 -.0005 0) (.5 .0005 0) (0 .001 0) (0 -.001 .001) (.5 -.0005 .001) (.5 .0005 .001) (0 .001 .001) (1 -.001 0) (1 .001 0) (1 -.001 .001) (1 .001 .001))'
        blocks=f'(hex (0 1 2 3 4 5 6 7) ({cells//2} 1 1) simpleGrading (1 1 1) hex (1 8 9 2 5 10 11 6) ({cells//2} 1 1) simpleGrading (1 1 1))'
        leftfaces='((0 3 7 4))';rightfaces='((8 10 11 9))'
        sidefaces='((0 4 5 1) (3 2 6 7) (1 5 10 8) (2 9 11 6))'
        emptyfaces='((0 1 2 3) (4 7 6 5) (1 8 9 2) (5 6 11 10))'
    else:
        vertices='((0 0 0) (1 0 0) (1 .001 0) (0 .001 0) (0 0 .001) (1 0 .001) (1 .001 .001) (0 .001 .001))'
        blocks=f'(hex (0 1 2 3 4 5 6 7) ({cells} 1 1) simpleGrading (1 1 1))'
        leftfaces='((0 3 7 4))';rightfaces='((1 5 6 2))';sidefaces='((0 4 5 1) (3 2 6 7))';emptyfaces='((0 1 2 3) (4 7 6 5))'
    cyclic=kind=='acoustic';lefttype='type cyclic; neighbourPatch right;' if cyclic else 'type patch;'
    righttype='type cyclic; neighbourPatch left;' if cyclic else 'type patch;'
    put('system/blockMeshDict',header('blockMeshDict')+f'''scale 1;
vertices {vertices}; blocks {blocks}; edges ();
boundary (left {{ {lefttype} faces {leftfaces}; }} right {{ {righttype} faces {rightfaces}; }}
walls {{ type wall; faces {sidefaces}; }} frontAndBack {{ type empty; faces {emptyfaces}; }});
mergePatchPairs ();
''')
    put('system/controlDict',header('controlDict')+f'''application spumaPintleColdFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {end:.17g}; deltaT {dt:.17g};
adjustTimeStep no; maxCo .25; maxAlphaCo .15;
writeControl timeStep; writeInterval {steps}; writeFormat binary; writePrecision 17; writeCompression off;
timeFormat general; timePrecision 12; runTimeModifiable false;
functions {{}} libs ("libpintleMultiphaseThermo.so");
pintleLimiter gpu; pintleSurfaceCache true; pintleFuseAlphaSource false;
pintleStrictState true; pintleMassTolerance 1e-3;
pintleGasPhase {gas};
pintleGasDynamics {'true' if new_physics else 'false'};
''')
    put('system/fvSchemes',header('fvSchemes')+'''
ddtSchemes { default Euler; } gradSchemes { default Gauss linear; }
divSchemes { default Gauss linear; div(rhoPhi,U) Gauss upwind;
div(phi,alpha) Gauss vanLeer; div(phirb,alpha) Gauss linear; div(phi,rho) Gauss upwind;
div(rhoPhi,T) Gauss upwind; div(rhoPhi,K) Gauss upwind; div(phi,p) Gauss upwind;
div(phid,p_rgh) Gauss upwind; div(rhoPhi,e) Gauss upwind; }
laplacianSchemes { default Gauss linear uncorrected; }
interpolationSchemes { default linear; } snGradSchemes { default uncorrected; }
''')
    put('system/fvSolution',header('fvSolution')+f'''
solvers {{ "alpha.*" {{ nAlphaSubCycles 1; cAlpha 0; }} "rho.*" {{ solver diagonal; }}
p_rgh {{ solver GAMG; smoother multicolorGaussSeidel; tolerance 1e-11; relTol 0; minIter 1; maxIter 1000; }}
p_rghFinal {{ $p_rgh; }}
U {{ solver smoothSolver; smoother multicolorGaussSeidel; tolerance 1e-10; relTol 0; minIter 1; }} UFinal {{ $U; }}
T {{ solver pintleTemperatureIncrement; smoother multicolorGaussSeidel; tolerance 1e-11; relTol 0; minIter 1; }} TFinal {{ $T; }} }}
PIMPLE {{ momentumPredictor yes; nOuterCorrectors {outer}; nCorrectors 2; nNonOrthogonalCorrectors 0; }}
relaxationFactors {{ equations {{ ".*" 1; }} }}
''')
    put('constant/g',header('g','uniformDimensionedVectorField')+'dimensions [0 1 -2 0 0 0 0]; value (0 0 0);\n')
    put('constant/turbulenceProperties',header('turbulenceProperties')+'simulationType laminar;\n')
    put('constant/thermophysicalProperties',header('thermophysicalProperties')+'phases (ipa n2o air); sigmas ((ipa n2o) 0 (ipa air) 0 (n2o air) 0); pMin 1;\n')
    put('constant/thermophysicalProperties.ipa',header('thermophysicalProperties.ipa')+'''thermoType { type pintleDeviceHeRhoThermo; device false; mixture pureMixture; properties pintleIpa; energy sensibleInternalEnergy; }
mixture { iC3H8O; } updateT false;
''')
    for phase,mw,cp,eos,extra in [('air',28.965,AIR_R*3.5,'perfectGas',''),
        ('n2o',44.0128,881.80380464488,'pintlePengRobinsonGas','equationOfState { Tc 309.52067823146; Pc 7244816.7016072; Vc .097173197657602; omega .1613; }')]:
        put(f'constant/thermophysicalProperties.{phase}',header(f'thermophysicalProperties.{phase}')+f'''thermoType {{ type pintleDeviceHeRhoThermo; device false; mixture pureMixture; transport const; thermo hConst;
equationOfState {eos}; specie specie; energy sensibleInternalEnergy; }}
mixture {{ specie {{ molWeight {mw}; }} {extra} thermodynamics {{ Cp {cp:.17g}; Hf 0; }} transport {{ mu 1e-12; Pr .71; }} }} updateT false;
''')
    # The dormant phases use the same gas. This is an intentional pure-gas
    # embedding, avoiding unrelated absent-liquid model limits in shock tests.
    active=(case/('constant/thermophysicalProperties.'+gas)).read_text()
    for phase in ['ipa','n2o','air']:
        if phase!=gas:put('constant/thermophysicalProperties.'+phase,active.replace('object thermophysicalProperties.'+gas,'object thermophysicalProperties.'+phase))
    def scalar(v):return f'{v:.17g}'
    def boundary(name,at_left):
        if cyclic:return 'type cyclic;'
        if kind=='normal-shock':
            index=0 if at_left else -1
            if name=='p':return f'type calculated; value uniform {pressure[index]:.17g};'
            if name=='p_rgh':return f'type fixedValue; value uniform {pressure[index]:.17g};'
            if at_left and name=='T':return 'type fixedValue; value uniform 300;'
            if at_left and name=='U':return f'type fixedValue; value uniform ({velocity[0,0]:.17g} 0 0);'
            return 'type zeroGradient;'
        if kind in ['sod','weak-shock']:return 'type zeroGradient;'
        index=0 if at_left else -1
        value={'p':pressure[index],'p_rgh':pressure[index],'T':temperature[index]}.get(name)
        if kind=='nozzle':
            if not at_left:return 'type zeroGradient;'
            if name=='p':return f'type calculated; value uniform {value:.17g};'
            if name=='p_rgh':return f'type totalPressure; p0 uniform 200000; gamma {GAMMA}; rho rho; psi thermo:psi; value uniform {value:.17g};'
            if name=='T':return f'type totalTemperature; T0 uniform 300; psi thermo:psi; gamma {GAMMA}; value uniform {value:.17g};'
            if name=='U':return f'type pressureInletOutletVelocity; value uniform ({velocity[0,0]:.17g} 0 0);'
        return 'type zeroGradient;'
    for name,dim,values in [('p','1 -1 -2 0 0 0 0',pressure),('p_rgh','1 -1 -2 0 0 0 0',pressure),
                          ('T','0 0 0 1 0 0 0',temperature),('U','0 1 -1 0 0 0 0',velocity)]:
        vector=name=='U';entries=['('+ ' '.join(map(scalar,v))+')' if vector else scalar(v) for v in values]
        wall='type slip;' if vector else ('type fixedFluxPressure; value uniform 100000;' if name=='p_rgh' else 'type zeroGradient;')
        put('0/'+name,header(name,'volVectorField' if vector else 'volScalarField')+f'dimensions [{dim}];\ninternalField nonuniform List<{"vector" if vector else "scalar"}> {cells}\n(\n'+'\n'.join(entries)+'\n);\nboundaryField {\n'+
            f'left {{ {boundary(name,True)} }} right {{ {boundary(name,False)} }} walls {{ {wall} }} frontAndBack {{ type empty; }}\n}}\n')
    for phase in b.PHASES:
        val=int(phase==gas);ends='type cyclic;' if cyclic else 'type zeroGradient;'
        put('0/alpha.'+phase,header('alpha.'+phase,'volScalarField')+f'dimensions [0 0 0 0 0 0 0]; internalField uniform {val}; boundaryField {{ left {{ {ends} }} right {{ {ends} }} walls {{ type zeroGradient; }} frontAndBack {{ type empty; }} }}\n')
    env=b.sourced_environment()
    with (case/'blockMesh.log').open('w') as log:subprocess.run(['blockMesh','-case',str(case)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    definition={'kind':kind,'gas':gas,'cells':cells,'steps':steps,'outer_correctors':outer,'mean_mach':mean_mach,
                'start_time':'0','end_time':f'{end:.17g}','delta_t':f'{dt:.17g}','gamma':GAMMA,'R':AIR_R,
                'new_physics':new_physics,'initial_x':x.tolist(),'initial_rho':rho.tolist(),'initial_p':pressure.tolist(),'initial_T':temperature.tolist(),'initial_U':velocity[:,0].tolist(),
                'generator_sha256':b.sha256(Path(__file__)),'scope':'Laminar gas-only limit of the three-phase solver, negligible viscosity; not a three-material shock validation.'}
    b.atomic_json(case/'run-definition.json',definition)
    return definition


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    p.add_argument('--kind',choices=['sod','weak-shock','normal-shock','acoustic','nozzle'],default='sod');p.add_argument('--cells',type=int,default=128)
    p.add_argument('--steps',type=int,default=160);p.add_argument('--outer',type=int,default=2);p.add_argument('--mean-mach',type=float,default=0)
    p.add_argument('--new-physics',action='store_true');a=p.parse_args()
    print(json.dumps(prepare(a.output,a.kind,a.cells,a.steps,a.outer,mean_mach=a.mean_mach,new_physics=a.new_physics),indent=2))
