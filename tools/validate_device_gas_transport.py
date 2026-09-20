#!/usr/bin/env python3
"""MFC/Pele-inspired gas-property kernels versus Cantera and face-scatter fluxes.

CPU executes the same operators but is not GPU execution evidence. CUDA mode
requires a working device. Test diagnostics download Y/h; normal stages do not.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path

import cantera as ct
import numpy as np
import yaml

from reactive_backend import Backend, GasThermoSpecies, GasThermoRegion, GasPartition
import validate_reactive_transport as tr


class DeviceProfile(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in ("gasPropertyBuilds", "gasPropertyCells", "partitionUploadBytes",
        "thermoTableBytes", "gasStatusChecks", "cflFaceLaunches", "transportFaceLaunches", "faceWorkspaceBytes")]


def require(ok, message):
    if not ok: raise AssertionError(message)


def load(root, library=None):
    lib = tr.load(library or root / "lib/libpintleReactiveTransport.so")
    v, d, s, part = C.c_void_p, C.POINTER(C.c_double), C.POINTER(tr.State), C.POINTER(GasPartition)
    signatures = {
        "device_profile": ([v, C.POINTER(DeviceProfile)], C.c_int),
        "set_gas_thermo": ([v, C.POINTER(GasThermoSpecies), C.c_size_t, C.POINTER(GasThermoRegion), C.c_size_t,
            C.POINTER(C.c_int64), C.c_size_t], C.c_int),
        "gas_properties_resident": ([v, s, part, d, d], C.c_int),
        "rhs_gas": ([v, d, s, part, d, d], C.c_int),
        "stage_resident_gas": ([v, s, part, C.c_double, C.c_int, C.c_uint64, C.c_uint64, d, d], C.c_int),
    }
    for name, (args, ret) in signatures.items():
        f = getattr(lib, "pintle_transport_"+name); f.argtypes, f.restype = args, ret
    return lib


def profile(lib, handle, cls=DeviceProfile):
    out = cls(); name = {DeviceProfile: "device_profile", tr.Profile: "profile", tr.Stats: "stats"}[cls]
    require(getattr(lib, "pintle_transport_"+name)(handle, C.byref(out)) == 0, "Profile failed")
    return {name: getattr(out, name) for name, _ in out._fields_}


def arrays(b, q, states):
    compact = (tr.State*len(q))(*[tr.State(s.p, s.T, s.rho, s.cv, s.soundFrozen, s.gasMass, 0) for s in states])
    partition = (GasPartition*len(q))(*[GasPartition((C.c_double*2)(*s.liquidMass)) for s in states]) if b.nl else None
    y, h = np.zeros((len(q), b.ns)), np.zeros((len(q), b.ns))
    for c, s in enumerate(states):
        if s.gasMass > 0:
            y[c] = q[c, :b.ns]
            for i, k in enumerate(b.liquid_indices): y[c, k] -= s.liquidMass[i]
            y[c] /= s.gasMass; h[c] = b.gas_enthalpies(q[c, :b.ns], s)
    return compact, partition, y, h


def state_rows(b, kind, nc):
    q = np.zeros((nc, b.ns+4)); states = []
    for c in range(nc):
        T, p, X, fraction = {
            "ideal": (1400+3*c, 1e5*(1+.01*c), {"N2O": 3, "IC3H7OH": 1, "N2": 100}, (0, 0)),
            "mixed": (300+.1*c, 1e5*(1+.01*c), {"N2O": 10, "IC3H7OH": 20, "N2": 50}, (0, .95)),
            "liquid": (270+.1*c, 1e5*(1+.01*c), {"IC3H7OH": 1}, (0, 1)),
        }[kind]
        mass, e, s = b.make_state(T, p, b.mole_to_mass(X), fraction); s = b.recover(mass, e, s)
        velocity = np.array([1+c*.1, -.3, .7]); q[c, :b.ns] = mass
        q[c, b.ns:b.ns+3] = s.rho*velocity; q[c, b.ns+3] = e+.5*s.rho*np.dot(velocity, velocity)
        states.append(s)
    if kind == "mixed": require(all(s.activeLiquids and s.gasMass > 0 for s in states), "Missing coexistence")
    if kind == "liquid": require(all(s.gasMass == 0 for s in states), "Missing pure-liquid state")
    return q, states


class Fixture:
    def __init__(self, lib, backend, b, q, states, fixed=True, install=True):
        self.lib, self.b = lib, b; nc, nv = q.shape
        self.volumes = np.linspace(.1, .2, nc)
        normals = np.array([[1, 2, 3], [-2, 1, 1], [1, -1, 2]], dtype=float)
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        self.faces = [tr.Face(c, (c+1)%nc, -1, (C.c_double*3)(*normals[c%3]), .7, .15, .37, 0) for c in range(nc)]
        if fixed:
            for c, kind in enumerate((1, 2, 3)):
                self.faces.append(tr.Face(c%nc, -1, 0 if kind == 3 else -1, (C.c_double*3)(*normals[c]), .2, .1, .5, kind))
        self.cfg = tr.Config(nc, b.ns, nv, len(self.faces), int(fixed), .007, 2, .003, 1.1, 2e8, 0)
        self.fixed = q[:1].copy() if fixed else q[:0].copy()
        if fixed:self.fs, _, self.fy, self.fh = arrays(b, self.fixed, states[:1])
        else:self.fs, self.fy, self.fh = (tr.State*0)(), None, None
        geometry = (tr.Face*len(self.faces))(*self.faces); error = C.create_string_buffer(8192)
        self.handle = lib.pintle_transport_create(backend, C.byref(self.cfg), tr.ptr(self.volumes), geometry,
            tr.ptr(self.fixed), self.fs, tr.ptr(self.fy), tr.ptr(self.fh), error, len(error))
        require(bool(self.handle), error.value.decode())
        try:
            self.records, self.regions = b.export_gas_thermo()
            self.liquids = (C.c_int64*b.nl)(*b.liquid_indices)
            if install:self.check(lib.pintle_transport_set_gas_thermo(self.handle, self.records, b.ns, self.regions, len(self.regions), self.liquids, b.nl))
        except Exception:
            lib.pintle_transport_destroy(self.handle); raise

    def check(self, status):
        require(status == 0, self.lib.pintle_transport_error(self.handle).decode())

    def close(self): self.lib.pintle_transport_destroy(self.handle)

    def reference(self, q, compact, y, h):
        return tr.reference(q, compact, self.faces, self.volumes, self.cfg, self.fixed, self.fs, y, h, self.fy, self.fh)


def properties(lib, backend, directory, configuration=None):
    cfg = configuration or directory / "chemistry-config.yaml"
    with Backend(cfg) as b:
        records,regions = b.export_gas_thermo(); model = yaml.safe_load(cfg.read_text())
        gas = ct.Solution(model["mechanism"], model["gas-phase"], transport_model=None)
        mids = sorted(set(regions[r.regionOffset+i].minimumTemperature for r in records for i in range(1,r.regionCount)))
        temperatures = sorted(set([270., 300., 900., 2200., 3500.] +
            [x for t in mids for x in (np.nextafter(t, 0), t, np.nextafter(t, np.inf))]))
        # Cantera permits NASA polynomial extrapolation; explicitly include 270 K
        # independently of the full reactive model's configured flash bounds.
        temperatures = np.resize(temperatures, max(257,len(temperatures)))
        q = np.zeros((len(temperatures), b.ns+4)); states = []
        rng = np.random.default_rng(9313); Y = rng.uniform(.1, 1, b.ns); Y /= Y.sum()
        from reactive_backend import State
        expected = []
        for c, T in enumerate(temperatures):
            gas.TPY = float(T), 1e5*(1+c/100), Y
            rho, cv, cp = gas.density, gas.cv_mass, gas.cp_mass
            q[c, :b.ns] = rho*Y; q[c, b.ns+3] = rho*gas.int_energy_mass
            s = State(); s.T, s.p, s.rho, s.cv = T, gas.P, rho, cv
            s.soundFrozen = np.sqrt(cp/cv*gas.P/rho); s.gasMass = rho; states.append(s)
            expected.append(gas.partial_molar_enthalpies/gas.molecular_weights)
        f = Fixture(lib, backend, b, q, states, fixed=False)
        try:
            compact=(tr.State*len(q))(*[tr.State(s.p,s.T,s.rho,s.cv,s.soundFrozen,s.gasMass,0) for s in states])
            partition=None;y=np.tile(Y,(len(q),1))
            f.check(lib.pintle_transport_upload_conserved(f.handle, tr.ptr(q), 1))
            gy, gh = np.empty_like(y), np.empty_like(y)
            f.check(lib.pintle_transport_gas_properties_resident(f.handle, compact, partition, tr.ptr(gy), tr.ptr(gh)))
            error = float(np.max(np.abs(gh-expected)/np.maximum(1, np.abs(expected))))
            require(error < 2e-11 and np.max(np.abs(gy-y)) < 2e-15, "Generated NASA7/NASA9/Y differ from Cantera")
            # Export includes coefficient ordering, molecular weights and species ordering.
            for k, r in enumerate(records):
                data=gas.species(k).input_data['thermo']
                require(r.regionCount==len(data['data']),"Region count changed")
                for i,coefficient in enumerate(data['data']):
                    region=regions[r.regionOffset+i]
                    require(np.array_equal(list(region.coefficient)[:r.polynomial],coefficient),"Coefficient order changed")
                    require(region.minimumTemperature==data['temperature-ranges'][i] and region.maximumTemperature==data['temperature-ranges'][i+1],"Temperature region changed")
            return dict(species=b.ns, cells=len(q), regions=len(regions),NASA7_species=sum(r.polynomial==7 for r in records),
                NASA9_species=sum(r.polynomial==9 for r in records), unique_boundaries=mids, midpoint_sides="nextafter below / exact / above",
                h_scaled_error=error, Y_Linf=float(np.max(np.abs(gy-y))), extrapolated_temperature=270,
                profile=profile(lib,f.handle), gpu_execution=bool(backend))
        finally: f.close()


def nasa9_terms(lib,backend,directory,scratch):
    # The four NASA9 species in the detailed mechanism may have zero inverse
    # powers; use Cantera's independent air data to exercise those terms too.
    source=ct.Solution("airNASA9.yaml")
    gas=ct.Solution(name="gas",thermo="ideal-gas",kinetics=None,species=[source.species("N2"),source.species("O2")])
    mechanism=scratch/"air-nasa9.yaml";gas.write_yaml(mechanism)
    cfg=scratch/"air-nasa9-config.yaml";template=yaml.safe_load((directory/"chemistry-config.yaml").read_text())
    template["mechanism"]=str(mechanism);template["gas-phase"]="gas";cfg.write_text(yaml.safe_dump(template))
    require(any(a[0]!=0 and a[1]!=0 for s in gas.species() for a in s.input_data['thermo']['data']),"Missing inverse/log terms")
    return properties(lib,backend,directory,cfg)


def operators(lib, backend, directory, kind):
    cfg = directory / ("chemistry-config.yaml" if kind == "ideal" else "reactive-dilute-config.yaml")
    with Backend(cfg) as b:
        q, states = state_rows(b, kind, 3); f = Fixture(lib, backend, b, q, states)
        try:
            compact, partition, y, h = arrays(b,q,states)
            expected, boundary, _ = f.reference(q,compact,y,h)
            rhs, br = np.empty_like(q), np.empty(q.shape[1])
            f.check(lib.pintle_transport_rhs_gas(f.handle,tr.ptr(q),compact,partition,tr.ptr(rhs),tr.ptr(br)))
            error = float(np.max(np.abs(rhs-expected)/np.maximum(1,np.max(np.abs(expected),axis=0))))
            budget = np.maximum(1,(np.abs(expected)*f.volumes[:,None]).sum(axis=0))
            boundary_error = float(np.max(np.abs(br-boundary)/budget))
            require(error<3e-12 and boundary_error<3e-12,"Generated-property face operator mismatch")
            require(np.max(np.abs((rhs*f.volumes[:,None]).sum(axis=0)+br)/budget)<2e-11,"Conservation failed")
            return dict(kind=kind,species=b.ns,scaled_error=error,boundary_scaled_error=boundary_error,
                gas_mass=[s.gasMass for s in states],liquid_mass=[list(s.liquidMass) for s in states],
                gas_upload_bytes=profile(lib,f.handle,tr.Profile)["gasUploadBytes"])
        finally: f.close()


def resident(lib, backend, directory):
    rows=[]
    for kind in ("ideal", "mixed"):
        cfg=directory/("chemistry-config.yaml" if kind=="ideal" else "reactive-dilute-config.yaml")
        with Backend(cfg) as b:
            q,states=state_rows(b,kind,7); initial=q.copy(); f=Fixture(lib,backend,b,q,states)
            try:
                compact,partition,y,h=arrays(b,q,states); before=profile(lib,f.handle,tr.Profile); device=profile(lib,f.handle)
                step=C.c_double();dt=1e-9
                for _ in range(2): f.check(lib.pintle_transport_stable_step_primitives(f.handle,tr.primitives(q,compact,b.ns),compact,.23,.1,C.byref(step)))
                f.check(lib.pintle_transport_upload_conserved(f.handle,tr.ptr(q),1))
                errors=[]
                for stage in (0,1):
                    rhs,boundary,_=f.reference(q,compact,y,h)
                    expected=q+dt*rhs if stage==0 else .5*initial+.5*(q+dt*rhs)
                    br=np.empty(q.shape[1])
                    f.check(lib.pintle_transport_stage_resident_gas(f.handle,compact,partition,dt,stage,1+stage,2+stage,tr.ptr(q),tr.ptr(br)))
                    error=float(np.max(np.abs(q-expected)/np.maximum(1,np.abs(expected))))
                    require(error<3e-12,"Resident RK state differs from independent operator")
                    errors.append(error)
                    if stage==0:
                        states=[b.recover(q[c,:b.ns],q[c,b.ns+3]-.5*np.sum(q[c,b.ns:b.ns+3]**2)/q[c,:b.ns].sum(),s) for c,s in enumerate(states)]
                        compact,partition,y,h=arrays(b,q,states)
                        f.check(lib.pintle_transport_stable_step_primitives(f.handle,tr.primitives(q,compact,b.ns),compact,.23,.1,C.byref(step)))
                after=profile(lib,f.handle,tr.Profile); dev=profile(lib,f.handle)
                delta={k:after[k]-before[k] for k in after}; dd={k:dev[k]-device[k] for k in dev}
                require(delta["conservedUploads"]==1 and delta["conservedUploadBytes"]==initial.nbytes
                    and delta["conservedDownloadBytes"]==2*initial.nbytes and delta["gasUploadBytes"]==0,"Large field transfer contract failed")
                require(dd["gasPropertyBuilds"]==2 and dd["partitionUploadBytes"]==(2*len(q)*C.sizeof(GasPartition) if b.nl else 0),"Property/partition count mismatch")
                require(dd["cflFaceLaunches"]==3 and dd["transportFaceLaunches"]==2,"CFL used full flux work")
                # Invalid versions or phase data must leave caller outputs intact;
                # a fresh upload then starts a valid stage 0 after rollback.
                saved=q.copy();br.fill(123)
                require(lib.pintle_transport_stage_resident_gas(f.handle,compact,partition,dt,1,2,4,tr.ptr(q),tr.ptr(br))!=0,"Stale version accepted")
                require(np.array_equal(q,saved) and np.all(br==123),"Rejected version changed outputs")
                q=initial.copy();compact,partition,_,_=arrays(b,q,state_rows(b,kind,len(q))[1])
                f.check(lib.pintle_transport_upload_conserved(f.handle,tr.ptr(q),4))
                f.check(lib.pintle_transport_stage_resident_gas(f.handle,compact,partition,dt,0,4,5,tr.ptr(q),tr.ptr(br)))
                rows.append(dict(kind=kind,stage_scaled_errors=errors,profile_delta=delta,device_delta=dd,
                    stale_version_rejected=True,rollback_fresh_upload=True,face_workspace_bytes=dev["faceWorkspaceBytes"],faces=len(f.faces)))
            finally:f.close()
    return dict(runs=rows)


def rejection(lib, backend, directory, scratch):
    results=[]
    with Backend(directory/"chemistry-config.yaml") as b:
        q,states=state_rows(b,"ideal",3)
        for case in ("bad-region-offset","unknown-polynomial","noncontiguous-regions","nan-coefficient"):
            f=Fixture(lib,backend,b,q,states,install=False)
            try:
                records=(GasThermoSpecies*b.ns).from_buffer_copy(f.records)
                regions=(GasThermoRegion*len(f.regions)).from_buffer_copy(f.regions)
                if case=="bad-region-offset":records[0].regionOffset=2**64-1
                elif case=="unknown-polynomial":records[0].polynomial=8
                elif case=="noncontiguous-regions":regions[1].minimumTemperature+=1
                else:regions[0].coefficient[0]=np.nan
                before=profile(lib,f.handle,tr.Stats)["allocatedBytes"]
                require(lib.pintle_transport_set_gas_thermo(f.handle,records,b.ns,regions,len(regions),f.liquids,b.nl)!=0,"Invalid table accepted")
                message=lib.pintle_transport_error(f.handle).decode()
                require(profile(lib,f.handle,tr.Stats)["allocatedBytes"]==before,"Invalid table allocated memory")
                f.check(lib.pintle_transport_set_gas_thermo(f.handle,f.records,b.ns,f.regions,len(f.regions),f.liquids,b.nl))
                results.append(dict(case=case,error=message))
            finally:f.close()
    with Backend(directory/"reactive-dilute-config.yaml") as b:
        initial,states=state_rows(b,"mixed",3);f=Fixture(lib,backend,b,initial,states)
        try:
            for index,case in enumerate(("missing-partition","excess-liquid","negative-species","gas-mass-mismatch","enthalpy-overflow")):
                q=initial.copy();compact,part,_,_=arrays(b,q,states)
                if case=="missing-partition":part=None
                elif case=="excess-liquid":part[0].liquidMass[1]=q[0,b.liquid_indices[1]]+1
                elif case=="negative-species":q[0,0]=-1
                elif case=="gas-mass-mismatch":compact[0].gasMass*=1.01
                else:compact[0].T=1e200
                version=1+2*index;f.check(lib.pintle_transport_upload_conserved(f.handle,tr.ptr(q),version))
                output=np.full_like(q,17);br=np.full(q.shape[1],19.)
                status=lib.pintle_transport_stage_resident_gas(f.handle,compact,part,1e-9,0,version,version+1,tr.ptr(output),tr.ptr(br))
                require(status!=0 and np.all(output==17) and np.all(br==19),"Invalid gas input escaped or changed outputs")
                message=lib.pintle_transport_error(f.handle).decode();results.append(dict(case=case,error=message))
            compact,part,_,_=arrays(b,initial,states)
            f.check(lib.pintle_transport_upload_conserved(f.handle,tr.ptr(initial),20))
            f.check(lib.pintle_transport_stage_resident_gas(f.handle,compact,part,1e-9,0,20,21,tr.ptr(output),tr.ptr(br)))
        finally:f.close()
    # Unsupported thermodynamic representations leave the export buffer intact.
    template=yaml.safe_load((directory/"chemistry-config.yaml").read_text())
    n2=ct.Species("N2",{"N":2});n2.thermo=ct.ConstantCp(200,5000,ct.one_atm,[298.15,0,0,3.5*ct.gas_constant])
    gas=ct.Solution(name="gas",thermo="ideal-gas",kinetics=None,species=[n2]);mechanism=scratch/"constant-cp.yaml";gas.write_yaml(mechanism)
    template["mechanism"]=str(mechanism);template["gas-phase"]="gas"
    cfg=scratch/"constant-cp-config.yaml";cfg.write_text(yaml.safe_dump(template))
    for path in (directory/"cold-pr-config.yaml",cfg):
        with Backend(path) as b:
            records=(GasThermoSpecies*b.ns)();regions=(GasThermoRegion*4)();count=C.c_size_t(77)
            C.memset(C.addressof(records),0x5A,C.sizeof(records));before=bytes(records)
            status=b.lib.pintle_rt_export_gas_thermo(b.handle,records,b.ns,regions,len(regions),C.byref(count))
            require(status!=0 and bytes(records)==before and count.value==77,"Unsupported export was not atomic")
            results.append(dict(case=path.name,error=b.lib.pintle_rt_error(b.handle).decode()))
    return dict(rejections=results,next_source_recovered=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thermo-dir",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--backend",choices=("cpu","cuda"),default="cpu");parser.add_argument("--checks",nargs="+")
    parser.add_argument("--transport-library",type=Path,help="Alternate CPU/CUDA build to test")
    a=parser.parse_args();root=Path(__file__).resolve().parents[1];directory=a.thermo_dir.resolve()
    if a.output.exists():raise SystemExit("Refusing to overwrite evidence")
    scratch=directory/"device-gas-inputs"/a.output.stem;scratch.mkdir(parents=True,exist_ok=False)
    library=(a.transport_library or root/"lib/libpintleReactiveTransport.so").resolve()
    lib=load(root,library);backend=int(a.backend=="cuda")
    checks=[("NASA7-NASA9-Cantera",lambda:properties(lib,backend,directory))]
    checks += [("NASA9-inverse-log-terms",lambda:nasa9_terms(lib,backend,directory,scratch))]
    checks += [("operator-"+kind,lambda kind=kind:operators(lib,backend,directory,kind)) for kind in ("ideal","mixed","liquid")]
    checks += [("resident-RK-transfer",lambda:resident(lib,backend,directory)),("rejection-recovery",lambda:rejection(lib,backend,directory,scratch))]
    if a.checks:
        unknown=set(a.checks)-{name for name,_ in checks}
        if unknown:parser.error("Unknown checks: "+str(unknown))
        checks=[(name,fn) for name,fn in checks if name in a.checks]
    report=dict(baseline="69bd3f0e4a81f89c6ca083f9ea69acd6849111fd",backend=a.backend,cantera=ct.__version__,tests=[],
        selected_checks=[name for name,_ in checks],validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        libraries={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/"lib/libpintleReactiveBackend.so",library)})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for name,fn in checks:
        try:row=dict(name=name,passed=True,**fn())
        except Exception as ex:row=dict(name=name,passed=False,error=str(ex))
        report["tests"].append(row);report["passed"]=all(t["passed"] for t in report["tests"])
        a.output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(row),flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__=="__main__":main()
