#!/usr/bin/env python3
"""MAIN: post-run input equality against the retained original case manifest."""
import argparse
import json
from pathlib import Path
import benchmark as b


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',type=Path,default=Path('benchmarks/async-final12/benchmark.json'))
    parser.add_argument('--reference',type=Path,default=Path('reports/precision-input-integrity.json'))
    parser.add_argument('--output',type=Path,default=Path('reports/async-input-integrity.json'))
    args=parser.parse_args();full=json.loads(args.benchmark.read_text());reference=json.loads(args.reference.read_text())
    source=Path(reference['source_case']);manifest=reference['source_manifest']
    source_mismatch=[p for p,h in manifest.items() if b.sha256(source/p)!=h]
    runs=[]
    for run in full['runs']:
        case=Path(run['case']);mismatch=[p for p,h in manifest.items() if b.sha256(case/p)!=h]
        thermo={p:b.sha256(Path(v['dictionary']))==v['sha256'] for p,v in run['thermo_configuration']['phases'].items()}
        runs.append({'mode':run['configuration'],'repeat':run['repeat'],'input_mismatches':mismatch,'thermo_matches_run_definition':thermo})
    result={'coverage':'Post-run original initial-time, mesh and unchanged constant files; thermo changes checked against each run definition. Runtime/control/fvSolution before/after hashes are in the benchmark.','source_case':str(source),'reference_manifest_sha256':b.sha256(args.reference),'source_manifest':manifest,'source_mismatches':source_mismatch,'runs':runs,'all_match':not source_mismatch and all(not r['input_mismatches'] and all(r['thermo_matches_run_definition'].values()) for r in runs)}
    b.atomic_json(args.output,result)
    if not result['all_match']:raise ValueError('Input mismatch')
    print('PASS',len(manifest),'files per case,',len(runs),'copied cases')

if __name__=='__main__':main()
