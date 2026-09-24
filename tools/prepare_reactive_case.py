#!/usr/bin/env python3
"""Generate isolated conservative HEM verification cases. Caller owns run lock."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cantera as ct
import numpy as np
from scipy.optimize import root

import benchmark as common
from benchmark import header
from real_fluid_backend import RealFluidBackend as Backend
from write_real_fluid_manifest import model_manifest
from validate_reactive_thermo import saturation


KINDS = ("uniform", "acoustic", "contact", "release", "shock", "reacting-shock", "viscous", "viscous-zero", "conduction", "conduction-zero", "diffusion", "diffusion-zero", "chemistry", "coupled")


def prepare(case, thermo_dir, kind="uniform", cells=32, mach=2., cfl=.25, end=None, dt_scale=1.,
            transport_backend="cpu", chemical_linear_solver="dense", transport_gas_properties="auto",
            thermo_workers=1, thermo_batch_cells=64, transport_bridge_cells=0, optimization_policy=None, physics=None,
            thermo_exact_reuse=False, closure_scalar_backend="cpu"):
    physics=dict(physics or {})
    known={'chemistry','combustion','phaseChange','viscosity','heatConduction','speciesDiffusion'}
    if set(physics)-known or any(type(v) is not bool for v in physics.values()):
        raise ValueError('Unknown physics option or non-boolean switch')
    if 'combustion' in physics and 'chemistry' in physics and physics['combustion'] != physics['chemistry']:
        raise ValueError('combustion and chemistry must agree')
    chemistry=physics.get('combustion',physics.get('chemistry',kind in ('chemistry','coupled','reacting-shock')))
    phase_change=physics.get('phaseChange',True)
    if closure_scalar_backend not in ('cpu','cuda') or ((thermo_exact_reuse or closure_scalar_backend=='cuda') and (chemistry or not phase_change)):
        raise ValueError('Closure acceleration requires nonreacting HEM equilibrium and cpu/cuda scalar backend')
    if transport_backend not in ("cpu", "cuda") or chemical_linear_solver not in ("dense", "sparse", "auto", "matrixFree", "matrixFreeWoodbury"):
        raise ValueError("Unknown reactive execution backend")
    if transport_gas_properties not in ("auto", "host", "deviceNasa"):
        raise ValueError("Unknown transport gas property mode")
    if physics.get("speciesDiffusion",True) and transport_gas_properties == "deviceNasa" and (transport_backend != "cuda" or kind not in ("diffusion", "coupled")):
        raise ValueError("deviceNasa requires CUDA and an active diffusion case")
    if not (1<=thermo_workers<=64 and thermo_workers<=min(cells,thermo_batch_cells) and transport_bridge_cells>=0):
        raise ValueError("Invalid worker/batch/bridge limits")
    policy=Path(optimization_policy or Path(__file__).resolve().parents[1]/"policies/real-fluid-optimization-v2.yaml").resolve()
    case, thermo_dir = Path(case).resolve(), Path(thermo_dir).resolve()
    if (case.exists() or cells < 4 or cells % 2 or kind not in KINDS or not (0<dt_scale<=1)
            or (end is not None and (not np.isfinite(end) or end<=0))
            or not np.isfinite(mach) or not (0<cfl<=.5)):
        raise ValueError("Require a new output path, an even cell count >=4, and a known case kind")
    case.mkdir(parents=True)
    def put(name, text):
        path = case / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
    configuration = thermo_dir / ("reactive-dilute-config.yaml" if kind == "coupled" else
                                   "chemistry-config.yaml" if kind in ("chemistry","reacting-shock","diffusion","diffusion-zero") else "cold-pr-config.yaml")
    with Backend(configuration) as backend:
        backend.check(backend.lib.reactive_rt_load_optimization_policy(backend.handle,str(policy).encode()))
        if chemistry:backend.set_chemical_linear_solver(chemical_linear_solver)
        put("constant/realFluidPolicy.yaml",policy.read_text())
        ns, nv = backend.ns, backend.ns+4
        q = np.zeros((cells, nv)); states = []
        x = (np.arange(cells)+.5)/cells
        mode = np.sin(2*np.pi*x)
        periodic = kind not in ("release", "shock", "reacting-shock")
        # Large, prescribed coefficients isolate the diffusion operator above
        # truncation noise. They are verification inputs, not fluid property fits.
        viscosity = 500. if kind == "viscous" else 0.
        conductivity = 1e7 if kind == "conduction" else 0.
        diffusivity = 500. if kind == "diffusion" else 0.
        if kind == "reacting-shock":
            viscosity,conductivity,diffusivity=3e-5,.1,1e-5
        coefficients=[viscosity,conductivity,diffusivity]
        for i,option in enumerate(('viscosity','heatConduction','speciesDiffusion')):
            if physics.get(option,coefficients[i]>0):
                if coefficients[i]<=0:raise ValueError(f'{option} requires a case with a positive coefficient')
            else:coefficients[i]=0.
        viscosity,conductivity,diffusivity=coefficients
        frozen=not phase_change and backend.nl>0
        identity_closure='HEM-frozen' if frozen else 'HEM'
        backend.bind_case(chemistry,viscosity,conductivity,diffusivity,transport_backend,transport_gas_properties,phase_change=phase_change)
        if thermo_exact_reuse or closure_scalar_backend=='cuda':
            backend.closure_acceleration(thermo_exact_reuse,closure_scalar_backend=='cuda',common.PROJECT_ROOT/'lib/libreactiveTransport.so')
        reference = {"kind": kind, "mean_mach_requested": mach}
        def pack(T, p, Y, liquid=(0, 0), u=None):
            mass, E, state = backend.make_state(T, p, Y, liquid)
            state = backend.recover(mass, E, state,equilibrium=phase_change)
            velocity = np.array(u if u is not None else [mach*state.soundEquilibrium, 0, 0.])
            conserved = np.r_[mass, state.rho*velocity, E+.5*state.rho*velocity.dot(velocity)]
            return conserved, state
        baseY = backend.vector({"N2": .8, "O2": .2})
        baseT=500 if kind.startswith("diffusion") else 300
        base, state = pack(baseT, 1e5 if kind=="coupled" else 2e6, baseY)
        base_velocity = base[ns:ns+3]/state.rho
        duration = .05/(abs(base_velocity[0])+state.soundEquilibrium)
        if kind == "acoustic":
            p = saturation(backend, "N2O", 270.)
            base, state = pack(270, p, {"N2O": 1}, (.7, 0))
            u0 = mach*state.soundEquilibrium
            for i, wave in enumerate(mode):
                drho = state.rho*1e-4*wave
                mass = base[:ns]*(1+1e-4*wave)
                E = (state.rho+drho)*(state.e+state.p/state.rho**2*drho)
                guess=state.copy()
                if frozen:
                    for j in range(backend.nl):guess.liquidMass[j]*=1+1e-4*wave
                s = backend.recover(mass, E, guess,equilibrium=phase_change)
                u = u0+state.soundEquilibrium*drho/state.rho
                q[i] = np.r_[mass, s.rho*u, 0, 0, E+.5*s.rho*u*u]; states.append(s)
            duration = 1/(4*(u0+state.soundEquilibrium))
            reference.update(base=state.as_dict(), epsilon=1e-4, mean_velocity=u0)
        elif kind == "contact":
            a, sa = pack(300, 1e5, {"N2": .4, "N2O": .6})
            b, sb = pack(300, 1e5, {"N2": .3, "IC3H7OH": .7}, (0, .95))
            # Construct both sides at one pressure and temperature in phase
            # equilibrium by assigning the saturated vapour composition.
            # For this numerical contact test, retain the actual recovered
            # common pressure/temperature by solving each total energy/density
            # initialization from the equilibrium phase partition at target p/T.
            def equilibrated_at_pt(Y, guess_liquid):
                y = backend.vector(Y)
                def affinity(f):
                    mass, E, s = backend.make_state(300, 1e5, y, (0, f))
                    gasmass = mass.copy();gasmass[backend.names.index("IC3H7OH")] -= s.liquidMass[1]
                    g = backend.phase(300, 1e5, Y=gasmass/gasmass.sum(), selected="IC3H7OH")
                    l = backend.phase(300, 1e5, phase=1)
                    return l.chemicalPotential-g.chemicalPotential
                from scipy.optimize import brentq
                fraction = brentq(affinity, 0, 1-1e-10)
                return pack(300, 1e5, y, (0, fraction))
            b, sb = equilibrated_at_pt({"N2": .3, "IC3H7OH": .7}, .95)
            commonU = mach*max(sa.soundEquilibrium, sb.soundEquilibrium)
            for i in range(cells):
                row, s = (a.copy(), sa) if x[i] < .5 else (b.copy(), sb)
                internal = row[-1]-.5*np.dot(row[ns:ns+3],row[ns:ns+3])/s.rho
                row[ns:ns+3] = (s.rho*commonU, 0, 0);row[-1] = internal+.5*s.rho*commonU**2
                q[i]=row;states.append(s.copy())
            duration=.1/max(abs(commonU),sa.soundEquilibrium,sb.soundEquilibrium)
            reference.update(p=1e5, T=300, mean_velocity=commonU)
        elif kind == "release":
            a, sa = pack(283.137492, 5e6, {"N2O": 1}, (1, 0), u=[0, 0, 0])
            b, sb = pack(283.137492, 1e6, {"N2O": 1}, u=[0, 0, 0])
            for i in range(cells):
                row, s = (a, sa) if x[i] < .5 else (b, sb)
                q[i]=row;states.append(s.copy())
            duration=1e-4
        elif kind in ("shock","reacting-shock"):
            T1,p1=(1100.,2e4) if kind=="reacting-shock" else (300.,1e5)
            Y=backend.mole_to_mass({"N2O":3,"IC3H7OH":1,"N2":100}) if kind=="reacting-shock" else backend.vector({"N2":1})
            a, sa = pack(T1, p1, Y, u=[0, 0, 0])
            u1=mach*sa.soundEquilibrium;massFlux=sa.rho*u1
            gas=ct.Solution(str(thermo_dir/("chemistry-ideal.yaml" if kind=="reacting-shock" else "cold-pr.yaml")),"gas",transport_model=None)
            gas.TPY=T1,p1,Y;h0=gas.enthalpy_mass+.5*u1*u1
            def residual(logstate):
                T2,rho2=np.exp(logstate);gas.TDY=T2,rho2,Y
                return [(gas.P+massFlux**2/rho2-(sa.p+sa.rho*u1*u1))/1e6,
                        (gas.enthalpy_mass+.5*(massFlux/rho2)**2-h0)/1e6]
            solved=root(residual,np.log([T1*2.,sa.rho*2.7]))
            if not solved.success or np.linalg.norm(residual(solved.x),np.inf)>1e-9:raise RuntimeError("Shock reference solve failed")
            T2,rho2=np.exp(solved.x);gas.TDY=T2,rho2,Y;p2=gas.P
            a,sa=pack(T1,p1,Y,u=[u1,0,0]);b,sb=pack(T2,p2,Y,u=[massFlux/rho2,0,0])
            for i in range(cells):
                row,s=(a,sa) if x[i]<.5 else (b,sb);q[i]=row;states.append(s.copy())
            duration=2e-6 if kind=="reacting-shock" else 1e-4
            reference.update(upstream=sa.as_dict(),downstream=sb.as_dict(),u1=u1,u2=massFlux/rho2,
                                         RH_residual=float(np.linalg.norm(residual(solved.x),np.inf)))
        elif kind in ("chemistry", "coupled"):
            if kind == "chemistry":
                Y=backend.mole_to_mass({"N2O":3,"IC3H7OH":1,"N2":100})
                base,state=pack(2200,1e5,Y);duration=2e-5
            else:
                Y=backend.mole_to_mass({"N2O":10,"IC3H7OH":20,"N2":50,"H":.02,"O":.01,"OH":.02})
                base,state=pack(300,1e5,Y,(0,.95));duration=1e-7
            for i in range(cells):q[i]=base;states.append(state.copy())
            reference.update(base=state.as_dict(),scope="Synthetic advected closed-cell source test; not ignition validation")
        else:
            for i in range(cells):
                if kind in ("viscous", "viscous-zero"):
                    row,s=pack(300,2e6,baseY,u=[base_velocity[0],10*mode[i],0])
                elif kind in ("conduction", "conduction-zero"):
                    row,s=pack(300+.01*mode[i],2e6,baseY,u=[base_velocity[0],0,0])
                elif kind in ("diffusion", "diffusion-zero"):
                    Y={"N2":.8+.001*mode[i],"O2":.2-.001*mode[i]}
                    row,s=pack(baseT,2e6,Y,u=[base_velocity[0],0,0])
                else:row,s=base.copy(),state.copy()
                q[i]=row;states.append(s)
            if kind in ("viscous", "viscous-zero", "conduction", "conduction-zero", "diffusion", "diffusion-zero"):duration=1e-6
            reference.update(base=state.as_dict(),mean_velocity=float(base_velocity[0]))
        duration=float(end if end is not None else duration)
        max_dt=duration/(2 if kind in ("chemistry","coupled","conduction","conduction-zero","viscous","viscous-zero","diffusion","diffusion-zero") else 4)
        max_dt*=dt_scale
        lefttype="type cyclic; neighbourPatch right; transform translational; separationVector (1 0 0);" if periodic else "type patch;"
        righttype="type cyclic; neighbourPatch left; transform translational; separationVector (-1 0 0);" if periodic else "type patch;"
        put("system/blockMeshDict",header("blockMeshDict")+f'''scale 1;
vertices ((0 0 0) (1 0 0) (1 .001 0) (0 .001 0) (0 0 .001) (1 0 .001) (1 .001 .001) (0 .001 .001));
blocks (hex (0 1 2 3 4 5 6 7) ({cells} 1 1) simpleGrading (1 1 1)); edges ();
boundary (left {{ {lefttype} faces ((0 3 7 4)); }} right {{ {righttype} faces ((1 5 6 2)); }}
walls {{ type wall; faces (); }} frontAndBack {{ type empty; faces ((0 1 2 3) (4 7 6 5) (0 4 5 1) (3 2 6 7)); }}); mergePatchPairs ();
''')
        put("system/controlDict",header("controlDict")+f'''application ReactiveFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {duration:.17g}; deltaT {max_dt:.17g};
maxDeltaT {max_dt:.17g}; maxCo {cfl:.17g}; writeControl runTime; writeInterval {duration:.17g};
writeFormat binary; writePrecision 17; writeCompression off; timeFormat general; timePrecision 15;
runTimeModifiable false; functions {{}};
''')
        put("system/fvSchemes",header("fvSchemes")+'''ddtSchemes { default Euler; } gradSchemes { default Gauss linear; }
divSchemes { default none; } laplacianSchemes { default Gauss linear uncorrected; }
interpolationSchemes { default linear; } snGradSchemes { default uncorrected; }
''')
        put("system/fvSolution",header("fvSolution")+"solvers {}\n")
        # Native HLL boundaries are deliberately explicit and separate from
        # passive output-field patch types.
        bc="walls { type slipWall; }\n"
        if not periodic:
            for side,index in [("left",0),("right",-1)]:
                if kind == "release":bc+=f"{side} {{ type extrapolate; }}\n";continue
                s=states[index];u=q[index,ns:ns+3]/s.rho;Y=q[index,:ns]/s.rho
                liquid=[s.liquidMass[j]/q[index,backend.liquid_indices[j]] if q[index,backend.liquid_indices[j]]>0 else 0 for j in range(backend.nl)]
                liquid += [0]*(2-len(liquid))
                bc+=f'{side} {{ type fixedState; p {s.p:.17g}; T {s.T:.17g}; U ('+" ".join(f"{v:.17g}" for v in u)+'); Y ('+" ".join(f"{v:.17g}" for v in Y)+'); liquidFractions ('+" ".join(f"{v:.17g}" for v in liquid)+"); }\n"
        put("constant/reactiveProperties",header("reactiveProperties")+f'''closure HEM;
thermoConfiguration "{configuration}"; initialization conserved;
chemistry {str(chemistry).lower()}; dynamicViscosity {viscosity}; thermalConductivity {conductivity}; molecularDiffusivity {diffusivity};
chemicalRelativeTolerance 1e-8; chemicalAbsoluteTolerance 1e-14;
transportBackend {transport_backend}; chemicalLinearSolver {chemical_linear_solver}; maxDeviceMemoryGB 2;
transportGasProperties {transport_gas_properties};
optimizationPolicy "{case/'constant/realFluidPolicy.yaml'}";
thermoWorkers {thermo_workers}; thermoBatchCells {thermo_batch_cells}; maxThermoBatchMemoryMB 64;
thermoExactReuse {str(thermo_exact_reuse).lower()}; closureScalarBackend {closure_scalar_backend};
transportBridgeCells {transport_bridge_cells};
transportStagingBytes 1048576; maxPinnedTransportBytes 1048576; transportBlockThreads 256; transportDetailedGasCounters false;
waveSpeedFactor 1.1; maxHostMemoryGB 2; boundaryConditions {{ {bc} }}
physics {{ {" ".join(k+" "+str(v).lower()+";" for k,v in physics.items())} }}
''')
        def field(name, values, dimensions):
            vector=values.ndim==2
            entries=["("+" ".join(f"{v:.17g}" for v in row)+")" if vector else f"{row:.17g}" for row in values]
            ends="type cyclic;" if periodic else "type zeroGradient;"
            wall="type slip;" if vector else "type zeroGradient;"
            put("0/"+name,header(name,"volVectorField" if vector else "volScalarField")+f"dimensions [{dimensions}];\ninternalField nonuniform List<{'vector' if vector else 'scalar'}> {cells}\n(\n"+"\n".join(entries)+f"\n);\nboundaryField {{ left {{ {ends} }} right {{ {ends} }} walls {{ {wall} }} frontAndBack {{ type empty; }} }}\n")
        field("p",np.array([s.p for s in states]),"1 -1 -2 0 0 0 0")
        field("T",np.array([s.T for s in states]),"0 0 0 1 0 0 0")
        field("U",q[:,ns:ns+3]/q[:,:ns].sum(axis=1)[:,None],"0 1 -1 0 0 0 0")
        for k in range(ns):field(f"q{k}",q[:,k],"1 -3 0 0 0 0 0")
        field("rhoMomentum",q[:,ns:ns+3],"1 -2 -1 0 0 0 0")
        field("rhoTotalEnergy",q[:,-1],"1 -1 -2 0 0 0 0")
        if frozen:
            liquid_mass=np.array([[s.liquidMass[j] for j in range(backend.nl)] for s in states])
            for j in range(backend.nl):field(f'rhoLiquid{j}',liquid_mass[:,j],'1 -3 0 0 0 0 0')
            q=np.column_stack((q,liquid_mass))
        put("0/reactiveStateIdentity",header("reactiveStateIdentity")+f'identitySchema 2; fingerprint "{backend.fingerprint}"; speciesCount {ns}; closure {identity_closure}; physicalModelHash "{backend.physical_hash}"; numericalPolicyHash "{backend.policy_hash}";\n')
        put("physical-model-manifest.json",json.dumps(model_manifest(configuration,backend),indent=2)+"\n")
        put("capability-matrix.json",json.dumps(backend.capabilities(),indent=2)+"\n")
        np.savez_compressed(case/"initial-conserved.npz",q=q,x=x)
        definition={"kind":kind,"cells":cells,"end_time":duration,"max_delta_t":max_dt,"cfl":cfl,"dt_scale":dt_scale,
                    "configuration":str(configuration),"species":backend.names,"liquid_indices":backend.liquid_indices,
                    "reference":reference,"viscosity":viscosity,"conductivity":conductivity,"diffusivity":diffusivity,
                    "model_fingerprint":backend.fingerprint,"physicalModelHash":backend.physical_hash,"numericalPolicyHash":backend.policy_hash,
                    "thermo_workers":thermo_workers,"thermo_batch_cells":thermo_batch_cells,"transport_bridge_cells":transport_bridge_cells,"policy_sha256":common.sha256(policy),
                    "transport_backend":transport_backend,"chemical_linear_solver":chemical_linear_solver,
                    "thermo_exact_reuse":thermo_exact_reuse,"closure_scalar_backend":closure_scalar_backend,
                    "transport_gas_properties":transport_gas_properties,"physics":physics,"chemistry":chemistry,"phase_change":phase_change,"frozen_liquids":backend.nl if frozen else 0,
                    "initial_states":[s.as_dict() for s in states],"generator_sha256":common.sha256(Path(__file__))}
    env=common.sourced_environment()
    with (case/"blockMesh.log").open("w") as log:
        subprocess.run(["blockMesh","-case",str(case)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    definition["input_sha256"]={str(p.relative_to(case)):common.sha256(p) for d in ("system","constant","0") for p in (case/d).rglob("*") if p.is_file()}
    common.atomic_json(case/"run-definition.json",definition)
    return definition


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path);parser.add_argument("--thermo-dir",type=Path,required=True)
    parser.add_argument("--kind",choices=KINDS,default="uniform");parser.add_argument("--cells",type=int,default=32)
    parser.add_argument("--mach",type=float,default=2.);parser.add_argument("--cfl",type=float,default=.25)
    parser.add_argument("--end",type=float)
    parser.add_argument("--dt-scale",type=float,default=1.)
    parser.add_argument("--transport-backend",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--chemical-linear-solver",choices=("dense","sparse","auto","matrixFree","matrixFreeWoodbury"),default="dense")
    parser.add_argument("--transport-gas-properties",choices=("auto","host","deviceNasa"),default="auto")
    parser.add_argument("--thermo-workers",type=int,default=1)
    parser.add_argument("--thermo-batch-cells",type=int,default=64)
    parser.add_argument("--thermo-exact-reuse",action='store_true')
    parser.add_argument("--closure-scalar-backend",choices=('cpu','cuda'),default='cpu')
    parser.add_argument("--transport-bridge-cells",type=int,default=0)
    parser.add_argument("--optimization-policy",type=Path)
    parser.add_argument('--chemistry',choices=('on','off'))
    parser.add_argument('--phase-change',choices=('equilibrium','frozen'))
    parser.add_argument('--viscosity',choices=('on','off'))
    parser.add_argument('--heat-conduction',choices=('on','off'))
    parser.add_argument('--species-diffusion',choices=('on','off'))
    args=parser.parse_args()
    physics={key:getattr(args,attr)=='on' for attr,key in [('chemistry','chemistry'),('viscosity','viscosity'),
        ('heat_conduction','heatConduction'),('species_diffusion','speciesDiffusion')] if getattr(args,attr) is not None}
    if args.phase_change is not None:physics['phaseChange']=args.phase_change=='equilibrium'
    print(json.dumps(prepare(args.output,args.thermo_dir,args.kind,args.cells,args.mach,args.cfl,args.end,args.dt_scale,
                             args.transport_backend,args.chemical_linear_solver,args.transport_gas_properties,args.thermo_workers,
                             args.thermo_batch_cells,args.transport_bridge_cells,args.optimization_policy,physics,
                             args.thermo_exact_reuse,args.closure_scalar_backend),indent=2))
