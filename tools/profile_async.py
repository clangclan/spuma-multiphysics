#!/usr/bin/env python3
"""MAIN-authored isolated async-smoother profiling or one-step memcheck."""
import argparse
from decimal import Decimal
import fcntl
from pathlib import Path
import re
import subprocess
import benchmark as b


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['baseline','async'],default='async')
    parser.add_argument('--memcheck',action='store_true')
    args=parser.parse_args();output=args.output.resolve();source=b.PROJECT_ROOT/'cases/pintle45us'
    if output.exists() or b.is_relative_to(output,source) or b.is_relative_to(source,output):parser.error('new output outside source required')
    env=b.sourced_environment();b.verify_prepared_case(source,Decimal(b.get_dictionary_entry(env,source/'system/controlDict','startTime')))
    output.mkdir(parents=True);case=output/'case';b.copy_case(source,case)
    config=b.configure_case(case,'gpu',1 if args.memcheck else 12,env);b.configure_thermo(case,'device',env)
    control=case/'system/controlDict';solution=case/'system/fvSolution'
    if args.mode=='async':
        libs=b.get_dictionary_entry(env,control,'libs').strip()
        if not libs.startswith('(') or not libs.endswith(')'):raise b.BenchmarkError('libs syntax')
        b.set_dictionary_entry(env,control,'libs',libs[:-1]+' "libpintleAsyncSmoother.so")')
        s,n=re.subn(r'(\bsmoother\s+)multicolorGaussSeidel(\s*;)',r'\g<1>pintleAsyncGaussSeidel\2',solution.read_text())
        if n<3:raise b.BenchmarkError('missing smoother entries')
        solution.write_text(s)
    binaries=[b.DEFAULT_EXECUTABLE,b.PROJECT_ROOT/'lib/libpintleMultiphaseThermo.so',b.PROJECT_ROOT/'lib/libpintleAsyncSmoother.so']
    config.update(effective_source_sha256={str(p):b.sha256(p) for p in sorted((b.PROJECT_ROOT/'src').rglob('*')) if p.is_file() and p.suffix in ('.C','.H','.cu','.cuh','.h') and 'Make' not in p.parts and 'lnInclude' not in p.parts},mode=args.mode,smoother_selection='pintleAsyncGaussSeidel' if args.mode=='async' else 'multicolorGaussSeidel',control_dict_sha256=b.sha256(control),fv_solution_sha256=b.sha256(solution),binary_sha256={str(p):b.sha256(p) for p in binaries})
    if args.memcheck:
        prefix=['compute-sanitizer','--tool','memcheck','--error-exitcode','97','--report-api-errors','explicit','--kernel-name','kns=pintleAsync','--print-limit','0']
    else:
        prefix=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--delay=20','--duration=10','--kill=none','--stats=false','--force-overwrite=false','--output',str(output/'trace')]
    with b.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        report=b.run_solver(case,b.DEFAULT_EXECUTABLE,config,env,900,command_prefix=prefix)
    if not report['completed_ok']:raise b.BenchmarkError(f'Run failed: {case}')
    if args.memcheck and 'ERROR SUMMARY: 0 errors' not in (case/'stdout.log').read_text():raise b.BenchmarkError('No zero-error sanitizer summary')
    if config['binary_sha256']!={str(p):b.sha256(p) for p in binaries}:raise b.BenchmarkError('Binary changed')
    if not args.memcheck:
        subprocess.run(['nsys','export','--type','sqlite','--output',str(output/'trace.sqlite'),str(output/'trace.nsys-rep')],env=env,check=True,stdout=subprocess.DEVNULL)
    b.atomic_json(output/'result.json',report)
    print(f'PASS {args.mode} {output}',flush=True)

if __name__=='__main__':main()
