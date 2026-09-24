#!/usr/bin/env python3
"""Build and run the standalone resolved-interface CPU and CUDA primitives."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]

def run(command):
    subprocess.run(command,check=True,cwd=ROOT)

def output(command):
    return subprocess.run(command,check=True,cwd=ROOT,text=True,
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.strip()

def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nvcc',type=Path,default=Path('/home/jsw/nvidia/hpc_sdk/Linux_x86_64/26.5/compilers/bin/nvcc'))
    parser.add_argument('--skip-cuda',action='store_true')
    parser.add_argument('--gpu-lock',type=Path,default=Path('/home/jsw/cae-benchmark/run.lock'))
    parser.add_argument('--evidence',type=Path,
                        default=ROOT/'results/spray-physics/interface-primitives.json')
    args=parser.parse_args()
    report={'schema':1,'generated_utc':datetime.now(timezone.utc).isoformat(),
            'architecture':'sm_120','cpu':{'passed':False},
            'cuda':{'passed':False,'skipped':bool(args.skip_cuda)},'hashes':{}}
    with tempfile.TemporaryDirectory(prefix='reactive-interface-') as temporary:
        build=Path(temporary)
        cpu=build/'interface_primitives_cpu'
        run(['g++','-std=c++17','-O2','-Wall','-Wextra','-pedantic',
             str(ROOT/'tests/interface_primitives.cpp'),'-o',str(cpu)])
        run([str(cpu)])
        report['cpu'].update(passed=True,compiler=output(['g++','--version']).splitlines()[0])
        print('interface CPU primitives passed')
        if not args.skip_cuda:
            gpu=build/'interface_primitives_cuda'
            run([str(args.nvcc),'-std=c++17','-O2','-arch=sm_120',
                 str(ROOT/'tests/interface_primitives_cuda.cu'),'-o',str(gpu)])
            args.gpu_lock.parent.mkdir(parents=True,exist_ok=True)
            with args.gpu_lock.open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                run([str(gpu)])
            report['cuda'].update(passed=True,skipped=False,
                                  compiler=output([str(args.nvcc),'--version']).splitlines()[-1],
                                  gpu_lock=str(args.gpu_lock))
    sources=[ROOT/'src/reactiveInterface/reactiveResolvedInterface.h',
             ROOT/'tests/interface_primitives.cpp',ROOT/'tests/interface_primitives_cuda.cu',
             ROOT/'tools/validate_interface_primitives.py',
             ROOT/'docs/spray-physics/resolved-interface-design.md']
    report['hashes']={str(path.relative_to(ROOT)):sha256(path) for path in sources}
    evidence=args.evidence.resolve();evidence.parent.mkdir(parents=True,exist_ok=True)
    temporary=evidence.with_suffix(evidence.suffix+'.tmp')
    temporary.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    temporary.replace(evidence)
    print(f'interface evidence: {evidence}')

if __name__=='__main__':main()
