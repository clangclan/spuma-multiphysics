#!/usr/bin/env python3
"""Probe the fixed 55 barg, 293.15 K N2O benchmark state with PR and CoolProp."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path

import CoolProp
from CoolProp.CoolProp import PropsSI
import numpy as np
from scipy.optimize import brentq
import yaml

from reactive_backend import Backend

ROOT=Path(__file__).resolve().parents[1]
SUPPLY_T=293.15
AMBIENT_P=101325.0
SUPPLY_P=5_500_000.0+AMBIENT_P
OUTLET_PRESSURES=(AMBIENT_P,4_000_000.0)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def saturation_pressure(backend,T):
    liquid=backend.liquid_indices.index(backend.names.index('N2O'))
    def affinity(p):
        vapour=backend.phase(T,p,Y={'N2O':1},selected='N2O')
        condensed=backend.phase(T,p,phase=liquid)
        return condensed.chemicalPotential-vapour.chemicalPotential
    previous=None
    for p in np.geomspace(1001.0,7.0e6,400):
        try:value=affinity(float(p))
        except RuntimeError:
            previous=None
            continue
        if previous is not None and value*previous[1]<0:
            return brentq(affinity,previous[0],p,xtol=1e-7,rtol=1e-13)
        previous=(p,value)
    raise RuntimeError(f'No N2O saturation bracket at T={T} K')

def saturation_temperature(backend,p):
    reference=PropsSI('T','P',p,'Q',0,'NitrousOxide')
    lower=max(182.4,reference-8.0);upper=min(309.4,reference+8.0)
    return brentq(lambda T:saturation_pressure(backend,T)-p,
                  lower,upper,xtol=1e-10,rtol=1e-13)

def quality(value,liquid,vapour):
    return (value-liquid)/(vapour-liquid)

def volume_fraction(vapour_quality,rho_liquid,rho_vapour):
    vapour_volume=vapour_quality/rho_vapour
    liquid_volume=(1-vapour_quality)/rho_liquid
    return vapour_volume/(vapour_volume+liquid_volume)

def require(condition,message):
    if not condition:raise RuntimeError(message)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration',type=Path,required=True)
    parser.add_argument('--output',type=Path,
                        default=ROOT/'results/benchmarks/impinging-n2o-thermodynamics.json')
    args=parser.parse_args()
    configuration=args.configuration.resolve();output=args.output.resolve()
    if output.exists():raise SystemExit(f'Refusing to overwrite evidence: {output}')
    config_data=yaml.safe_load(configuration.read_text())
    mechanism=Path(config_data['mechanism']).resolve()

    with Backend(configuration) as backend:
        liquid=backend.liquid_indices.index(backend.names.index('N2O'))
        pr_psat=saturation_pressure(backend,SUPPLY_T)
        cp_psat=PropsSI('P','T',SUPPLY_T,'Q',0,'NitrousOxide')
        mass,energy,initial=backend.make_state(SUPPLY_T,SUPPLY_P,{'N2O':1},(1,0))
        recovered=backend.recover(mass,energy,initial)
        supply=backend.phase(SUPPLY_T,SUPPLY_P,phase=liquid)
        require(pr_psat<SUPPLY_P and cp_psat<SUPPLY_P,'Supply is not compressed liquid')
        require(abs(recovered.p/SUPPLY_P-1)<1e-12 and abs(recovered.T-SUPPLY_T)<1e-10,
                'Supply fixedState does not preserve p/T')
        require(abs(recovered.liquidMass[0]/recovered.rho-1)<1e-12
                and recovered.alphaGas==0,'Supply fixedState is not pure liquid')
        reference_supply={
            'rho_kg_m3':PropsSI('D','T',SUPPLY_T,'P',SUPPLY_P,'NitrousOxide'),
            'h_J_kg':PropsSI('H','T',SUPPLY_T,'P',SUPPLY_P,'NitrousOxide'),
            's_J_kgK':PropsSI('S','T',SUPPLY_T,'P',SUPPLY_P,'NitrousOxide')}
        destinations=[]
        for pressure in OUTLET_PRESSURES:
            pr_temperature=saturation_temperature(backend,pressure)
            pr_liquid=backend.phase(pr_temperature,pressure,phase=liquid)
            pr_vapour=backend.phase(pr_temperature,pressure,Y={'N2O':1},selected='N2O')
            xh=quality(supply.h,pr_liquid.h,pr_vapour.h)
            xs=quality(supply.s,pr_liquid.s,pr_vapour.s)
            q,e,state=backend.make_state(pr_temperature,pressure,{'N2O':1},(1-xh,0))
            roundtrip=backend.recover(q,e,state)

            cp_temperature=PropsSI('T','P',pressure,'Q',0,'NitrousOxide')
            cp_hl=PropsSI('H','P',pressure,'Q',0,'NitrousOxide')
            cp_hv=PropsSI('H','P',pressure,'Q',1,'NitrousOxide')
            cp_sl=PropsSI('S','P',pressure,'Q',0,'NitrousOxide')
            cp_sv=PropsSI('S','P',pressure,'Q',1,'NitrousOxide')
            cp_rhol=PropsSI('D','P',pressure,'Q',0,'NitrousOxide')
            cp_rhov=PropsSI('D','P',pressure,'Q',1,'NitrousOxide')
            cp_xh=quality(reference_supply['h_J_kg'],cp_hl,cp_hv)
            cp_xs=quality(reference_supply['s_J_kgK'],cp_sl,cp_sv)
            require(all(0<value<1 for value in (xh,xs,cp_xh,cp_xs)),
                    f'Outlet at {pressure} Pa is not a two-phase equilibrium result')
            require(abs(roundtrip.p/pressure-1)<1e-9
                    and abs(roundtrip.T-pr_temperature)<1e-7,
                    f'HEM outlet round trip failed at {pressure} Pa')
            require(max(roundtrip.volumeResidual,roundtrip.energyResidual,
                        roundtrip.chemicalResidual)<1e-8,
                    f'HEM outlet residual failed at {pressure} Pa')
            destinations.append({
                'p_abs_Pa':pressure,'pressure_drop_Pa':SUPPLY_P-pressure,
                'repository_PR':{
                    'Tsat_K':pr_temperature,'latent_h_J_kg':pr_vapour.h-pr_liquid.h,
                    'quality_isenthalpic':xh,
                    'gas_volume_fraction_isenthalpic':volume_fraction(xh,pr_liquid.rho,pr_vapour.rho),
                    'quality_isentropic':xs,
                    'gas_volume_fraction_isentropic':volume_fraction(xs,pr_liquid.rho,pr_vapour.rho),
                    'saturated_liquid':pr_liquid.as_dict(),'saturated_vapour':pr_vapour.as_dict(),
                    'HEM_isenthalpic_roundtrip':{
                        'p_relative_error':roundtrip.p/pressure-1,
                        'T_error_K':roundtrip.T-pr_temperature,
                        'liquid_mass_fraction':roundtrip.liquidMass[0]/roundtrip.rho,
                        'gas_volume_fraction':roundtrip.alphaGas,
                        'volume_residual':roundtrip.volumeResidual,
                        'energy_residual':roundtrip.energyResidual,
                        'chemical_residual':roundtrip.chemicalResidual}},
                'CoolProp_Lemmon_Span':{
                    'Tsat_K':cp_temperature,'latent_h_J_kg':cp_hv-cp_hl,
                    'rho_liquid_kg_m3':cp_rhol,'rho_vapour_kg_m3':cp_rhov,
                    'quality_isenthalpic':cp_xh,
                    'gas_volume_fraction_isenthalpic':volume_fraction(cp_xh,cp_rhol,cp_rhov),
                    'quality_isentropic':cp_xs,
                    'gas_volume_fraction_isentropic':volume_fraction(cp_xs,cp_rhol,cp_rhov)}})

        report={
            'schema':1,
            'scope':'Equilibrium thermodynamic probe; not a nozzle, choking, or finite-rate flashing model',
            'conditions':{'species':'N2O','supply_T_K':SUPPLY_T,
                'supply_p_gauge_Pa':5_500_000.0,'gauge_reference_Pa':AMBIENT_P,
                'supply_p_abs_Pa':SUPPLY_P,'outlet_p_abs_Pa':list(OUTLET_PRESSURES)},
            'formulae':{
                'quality_isenthalpic':'(h_in-h_l_sat)/(h_v_sat-h_l_sat)',
                'quality_isentropic':'(s_in-s_l_sat)/(s_v_sat-s_l_sat)',
                'gas_volume_fraction':'(x/rho_v)/(x/rho_v+(1-x)/rho_l)'},
            'model':{'physical_model_fingerprint':backend.fingerprint,
                'configuration':str(configuration),'mechanism':str(mechanism),
                'backend':str(backend.library_path),'python':platform.python_version(),
                'CoolProp':CoolProp.__version__},
            'hashes':{'configuration_sha256':sha256(configuration),
                'mechanism_sha256':sha256(mechanism),'backend_sha256':sha256(backend.library_path),
                'probe_sha256':sha256(Path(__file__).resolve()),
                'reactive_backend_py_sha256':sha256(ROOT/'tools/reactive_backend.py')},
            'supply':{'repository_PR':{'psat_Pa':pr_psat,
                    'subcool_pressure_margin_Pa':SUPPLY_P-pr_psat,
                    'liquid':supply.as_dict(),
                    'fixedState_roundtrip':{
                        'p_relative_error':recovered.p/SUPPLY_P-1,
                        'T_error_K':recovered.T-SUPPLY_T,
                        'liquid_mass_fraction':recovered.liquidMass[0]/recovered.rho,
                        'gas_volume_fraction':recovered.alphaGas}},
                'CoolProp_Lemmon_Span':{'psat_Pa':cp_psat,
                    'subcool_pressure_margin_Pa':SUPPLY_P-cp_psat,**reference_supply}},
            'destinations':destinations}

    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(output.suffix+'.tmp')
    temporary.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    temporary.replace(output)
    print(output)

if __name__=='__main__':main()
