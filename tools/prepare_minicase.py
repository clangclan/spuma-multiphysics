#!/usr/bin/env python3
"""MAIN: generate a three-phase contact-angle case without a Pintle restart."""
import argparse
from pathlib import Path
import subprocess
import benchmark as b


def header(name, cls='dictionary'):
    return f'FoamFile {{ format ascii; class {cls}; object {name}; }}\n'


def prepare(case: Path):
    case.mkdir(parents=True, exist_ok=False)
    def put(path, content):
        p=case/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
    put('system/blockMeshDict',header('blockMeshDict')+'''
scale 1;
vertices ((0 0 0) (.012 0 0) (.012 .006 0) (0 .006 0)
          (0 0 .0002) (.012 0 .0002) (.012 .006 .0002) (0 .006 .0002));
blocks (hex (0 1 2 3 4 5 6 7) (48 24 1) simpleGrading (1 1 1));
edges ();
boundary
(
 bottomWall { type wall; faces ((0 1 5 4)); }
 rightWall { type wall; faces ((1 2 6 5)); }
 topWall { type wall; faces ((2 3 7 6)); }
 leftWall { type wall; faces ((3 0 4 7)); }
 frontAndBack { type empty; faces ((0 3 2 1) (4 5 6 7)); }
);
mergePatchPairs ();
''')
    put('system/controlDict',header('controlDict')+'''
application spumaPintleColdFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime 1.6e-6;
deltaT 2e-7; adjustTimeStep no; maxCo .25; maxAlphaCo .15;
writeControl runTime; writeInterval 1.6e-6;
writeFormat binary; writePrecision 17; writeCompression off;
timeFormat general; timePrecision 12; runTimeModifiable false;
functions {}; libs ("libpintleMultiphaseThermo.so"); pintleLimiter gpu; pintleVerifyLimiter true;
pintleSurfaceCache true; pintleFuseAlphaSource false;
pintleStrictState true;
''')
    put('system/fvSchemes',header('fvSchemes')+'''
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes
{
 default Gauss linear;
 div(rhoPhi,U) Gauss linearUpwind grad(U);
 div(phi,alpha) Gauss vanLeer;
 div(phirb,alpha) Gauss linear;
 div(phi,rho) Gauss upwind;
 div(rhoPhi,T) Gauss upwind;
 div(rhoPhi,K) Gauss upwind;
 div(phi,p) Gauss upwind;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
''')
    put('system/fvSolution',header('fvSolution')+'''
solvers
{
 "alpha.*" { nAlphaSubCycles 2; cAlpha 1; }
 "rho.*" { solver diagonal; }
 p_rgh { solver GAMG; smoother multicolorGaussSeidel; tolerance 1e-12; relTol 0; minIter 1; }
 p_rghFinal { $p_rgh; }
 U { solver smoothSolver; smoother multicolorGaussSeidel; tolerance 1e-10; relTol 0; minIter 1; }
 UFinal { $U; }
 T { solver pintleTemperatureIncrement; smoother multicolorGaussSeidel; tolerance 1e-12; relTol 0; minIter 1; }
 TFinal { $T; }
}
PIMPLE { momentumPredictor yes; nOuterCorrectors 2; nCorrectors 2; nNonOrthogonalCorrectors 1; }
relaxationFactors { equations { ".*" 1; } }
''')
    put('constant/g',header('g','uniformDimensionedVectorField')+'dimensions [0 1 -2 0 0 0 0]; value (0 0 0);\n')
    put('constant/turbulenceProperties',header('turbulenceProperties')+'simulationType laminar;\n')
    put('constant/thermophysicalProperties',header('thermophysicalProperties')+'''
phases (ipa n2o air);
sigmas ((ipa n2o) .0206230480452 (ipa air) .0206230480452 (n2o air) 0);
pMin 10000;
''')
    put('constant/thermophysicalProperties.ipa',header('thermophysicalProperties.ipa')+'''
thermoType { type pintleDeviceHeRhoThermo; device false; mixture pureMixture; properties pintleIpa; energy sensibleInternalEnergy; }
mixture { iC3H8O; }
updateT false;
''')
    for phase,mw,cp,mu,pr,eos,extra in [
        ('air',28.965,1006,1.86e-5,.71,'perfectGas',''),
        ('n2o',44.0128,881.80380464488,1.49e-5,.75,'pintlePengRobinsonGas',
         'equationOfState { Tc 309.52067823146; Pc 7244816.7016072; Vc .097173197657602; omega .1613; }')]:
        put(f'constant/thermophysicalProperties.{phase}',header(f'thermophysicalProperties.{phase}')+f'''
thermoType {{ type pintleDeviceHeRhoThermo; device false; mixture pureMixture; transport const; thermo hConst;
 equationOfState {eos}; specie specie; energy sensibleInternalEnergy; }}
mixture {{ specie {{ molWeight {mw}; }} {extra}
 thermodynamics {{ Cp {cp}; Hf 0; }} transport {{ mu {mu}; Pr {pr}; }} }}
updateT false;
''')
    def field(name,dimensions,value,wall,vector=False):
        put('0/'+name,header(name,'volVectorField' if vector else 'volScalarField')+
            f'dimensions [{dimensions}];\ninternalField uniform {value};\nboundaryField\n{{\n'+
            ''.join(f'{patch} {{ {wall} }}\n' for patch in ['bottomWall','rightWall','topWall','leftWall'])+
            'frontAndBack { type empty; }\n}\n')
    field('U','0 1 -1 0 0 0 0','(.2 0 0)','type noSlip;',True)
    field('p','1 -1 -2 0 0 0 0','1980000','type zeroGradient;')
    field('p_rgh','1 -1 -2 0 0 0 0','1980000','type fixedFluxPressure; value uniform 1980000;')
    field('T','0 0 0 1 0 0 0','295','type zeroGradient;')
    for phase in b.PHASES:
        value='1' if phase=='air' else '0'
        field('alpha.'+phase,'0 0 0 0 0 0 0',value,'''
type alphaContactAngle;
thetaProperties ((ipa n2o) 90 .05 100 80 (ipa air) 95 .05 105 85 (n2o air) 90 .05 100 80);
value uniform '''+value+';')
    put('system/setFieldsDict',header('setFieldsDict')+'''
defaultFieldValues (volScalarFieldValue alpha.ipa 0 volScalarFieldValue alpha.n2o 0 volScalarFieldValue alpha.air 1);
regions
(
 boxToCell { box (.0015 0 -1) (.00575 .0025 1); fieldValues
 (volScalarFieldValue alpha.ipa 1 volScalarFieldValue alpha.n2o 0 volScalarFieldValue alpha.air 0); }
 boxToCell { box (.00575 0 -1) (.0105 .0025 1); fieldValues
 (volScalarFieldValue alpha.ipa 0 volScalarFieldValue alpha.n2o 1 volScalarFieldValue alpha.air 0); }
 boxToCell { box (0 0 -1) (.006 .006 1); fieldValues
 (volScalarFieldValue p 2020000 volScalarFieldValue p_rgh 2020000); }
);
''')
    env=b.sourced_environment()
    for command in ['blockMesh','pintleRegressionCheck']:
        with (case/(command+'.log')).open('w') as log:
            extra=['-mode','initialize','-pool','fixedSizeMemoryPool','-poolSize','1'] if command=='pintleRegressionCheck' else []
            subprocess.run([command,'-case',str(case),*extra],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    expected={'ipa':170,'n2o':190,'air':792};counts={}
    for phase in b.PHASES:
        field=b.read_internal_field(case/'0'/('alpha.'+phase))
        if field.entries!=1152 or not ((field.values==0)|(field.values==1)).all():
            raise ValueError('Invalid initialized phase '+phase)
        counts[phase]=int((field.values==1).sum())
    if counts!=expected:raise ValueError('Initialized material regions differ from the specified mesh boxes')
    paths=[f for folder in ['system','constant','0'] for f in (case/folder).rglob('*') if f.is_file()]
    b.atomic_json(case/'preparation.json',{'generator':str(Path(__file__).resolve()),'generator_sha256':b.sha256(Path(__file__)),
        'initializer_sha256':b.sha256(b.PROJECT_ROOT/'bin/pintleRegressionCheck'),
        'thermo_library_sha256':b.sha256(b.PROJECT_ROOT/'lib/libpintleMultiphaseThermo.so'),
        'input_sha256':{str(f.relative_to(case)):b.sha256(f) for f in paths},'cells':1152,'initial_phase_cells':counts,
        'external_restart_required':False,'initializer':'pintleRegressionCheck -mode initialize; setFieldsDict is a human-readable region reference only',
        'scope':'laminar numerical regression; synthetic initial state, not a physical validation experiment'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    prepare(parser.parse_args().output.resolve())
