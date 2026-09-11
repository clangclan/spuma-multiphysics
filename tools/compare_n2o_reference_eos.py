#!/usr/bin/env python3
"""PR/CoolProp N2O property comparison; successful calculation is not an accuracy certificate."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import cantera as ct
import CoolProp
from CoolProp.CoolProp import PropsSI
from reactive_backend import Backend
from validate_reactive_thermo import saturation
import benchmark as b


def run(config,output):
    report={'scope':'Independent reference-EOS comparison; no experimental fit or claimed validated operating envelope',
        'reference':'CoolProp NitrousOxide multiparameter reference EOS','CoolProp_version':CoolProp.__version__,
        'Cantera_version':ct.__version__,'validated_operating_range':None,'saturation':[],'fixed_states':[]}
    with Backend(config) as m:
        report['model_fingerprint']=m.fingerprint
        for T in (240.,270.,283.137492,300.,305.):
            pr=saturation(m,'N2O',T);pref=PropsSI('P','T',T,'Q',0,'NitrousOxide')
            liq=m.phase(T,pr,phase=0);gas=m.phase(T,pr,Y={'N2O':1},selected='N2O')
            pairs={'pressure':(pr,pref),'latent_heat':(gas.h-liq.h,PropsSI('Hmass','T',T,'Q',1,'NitrousOxide')-PropsSI('Hmass','T',T,'Q',0,'NitrousOxide'))}
            for label,s,quality in [('liquid',liq,0),('gas',gas,1)]:
                for prop,key in [('rho','Dmass'),('cp','Cpmass'),('cv','Cvmass'),('sound','A')]:
                    pairs[label+'_'+prop]=(getattr(s,prop),PropsSI(key,'T',T,'Q',quality,'NitrousOxide'))
            report['saturation'].append(dict(T=T,comparison_path='Each EOS at its own saturation pressure at the same T',
                properties={name:dict(PR=got,reference=ref,relative_error=got/ref-1) for name,(got,ref) in pairs.items()}))
        for T,p,phase in [(283.137492,5e6,0),(283.137492,1e7,0),(300.,1e5,-1),(300.,2e6,-1)]:
            s=m.phase(T,p,phase=phase,Y={'N2O':1},selected='N2O')
            report['fixed_states'].append(dict(T=T,p=p,phase='liquid' if phase==0 else 'gas',properties={
                name:dict(PR=getattr(s,name),reference=PropsSI(key,'T',T,'P',p,'NitrousOxide'),
                    relative_error=getattr(s,name)/PropsSI(key,'T',T,'P',p,'NitrousOxide')-1)
                for name,key in [('rho','Dmass'),('cp','Cpmass'),('cv','Cvmass'),('sound','A')]}))
    report['calculation_completed']=True;b.atomic_json(output,report)
    print(json.dumps(report),flush=True);return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise SystemExit('Refusing to overwrite evidence')
    run(a.config.resolve(),a.output.resolve())
