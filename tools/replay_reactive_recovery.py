#!/usr/bin/env python3
"""Replay exact JSONL UV inputs against an explicitly selected same-EOS library.

Accepts ReactiveFoam schema-1 failure records or the original R04 tracer records.
Summary-only records are counted but never fabricated into complete inputs.
"""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from real_fluid_backend import RealFluidBackend, ptr
from reactive_backend import State


def bind(backend):
    for name,args,result in [
        ('set_recovery_v1',[C.c_void_p,C.c_int,C.c_int],C.c_int),
        ('recovery_diagnostic_v1',[C.c_void_p],C.c_char_p),
        ('runtime_manifest_v1',[C.c_void_p],C.c_char_p)]:
        function=getattr(backend.lib,'pintle_rt_'+name);function.argtypes=args;function.restype=result


def guess_of(row):
    values=row.get('guess',row);state=State()
    for name,kind in state._fields_:
        if name in values:
            if issubclass(kind,C.Array):getattr(state,name)[:]=values[name]
            else:setattr(state,name,values[name])
    return state


def replay(backend,row,mode):
    if row.get('operation','recover')!='recover':raise ValueError('UV replay cannot replay chemical source records')
    backend.check(backend.lib.pintle_rt_set_recovery_v1(backend.handle,int(mode=='boundaryFallback'),1))
    q=np.asarray(row['q'],dtype=np.float64).copy();original=q.tobytes();state=guess_of(row);before=bytes(state)
    energy=float(row['energy']);start=time.perf_counter()
    status=backend.lib.pintle_rt_recover(backend.handle,ptr(q),energy,int(row.get('equilibrium',True)),C.byref(state))
    result={'success':status==0,'seconds':time.perf_counter()-start,'inputUnchanged':q.tobytes()==original,
            'failureSeedUnchanged':not status or bytes(state)==before}
    if status:
        result['error']=backend.lib.pintle_rt_error(backend.handle).decode()
        detail=backend.lib.pintle_rt_recovery_diagnostic_v1(backend.handle)
        result['search']=json.loads(detail) if detail else None
    else:
        result['state']=state.as_dict()
        result['stateBitPatternSha256']=hashlib.sha256(bytes(state)).hexdigest()
        result['residualsPass']=state.volumeResidual<=1e-9 and state.energyResidual<=1e-9 and state.chemicalResidual<=1e-7
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',type=Path,nargs='+');p.add_argument('--config',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('reference','boundaryFallback'),default='reference')
    a=p.parse_args();report={'schema':1,'mode':a.mode,'inputs':[],'records':[],'summariesSkipped':0}
    with RealFluidBackend(a.config,a.library) as backend:
        bind(backend)
        for path in a.inputs:
            report['inputs'].append({'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            for line,text in enumerate(path.read_text().splitlines(),1):
                if not text.strip():continue
                row=json.loads(text)
                if 'q' not in row:report['summariesSkipped']+=1;continue
                result=replay(backend,row,a.mode);result.update(file=str(path),line=line,globalCell=row.get('globalCell'))
                report['records'].append(result)
        manifest=backend.lib.pintle_rt_runtime_manifest_v1(backend.handle)
        if not manifest:raise RuntimeError(backend.lib.pintle_rt_error(backend.handle).decode())
        report['runtime']=json.loads(manifest)
    report['passed']=bool(report['records']) and all(r['success'] and r['residualsPass'] and r['inputUnchanged'] for r in report['records'])
    a.output.parent.mkdir(parents=True,exist_ok=True);temporary=a.output.with_suffix(a.output.suffix+'.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temporary.replace(a.output)
    print(json.dumps({'records':len(report['records']),'passed':report['passed'],'output':str(a.output)}))
    raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()
