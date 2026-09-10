#!/usr/bin/env python3
"""MAIN: consolidate canonical comparisons and verify their saved evidence."""
import fcntl
import json
from pathlib import Path
import benchmark as b


def main():
    root=b.PROJECT_ROOT;bench=root/'benchmarks'
    def read(path):return json.loads(path.read_text())
    def compact_comparison(c):
        return dict(reference=c['reference'],candidate=c['candidate'],
            fields={name:{key:value.get(key) for key in ['max_abs','max_scaled','rel_l2']} for name,value in c['fields'].items()})
    with b.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        oldwaves=read(bench/'gas-final-waves/campaign.json')
        current=read(bench/'gas-final-release-waves/campaign.json')
        shocks=read(bench/'gas-final-shocks/campaign.json');nozzle=read(bench/'gas-final-nozzle/campaign.json')
        for group in [current,shocks,nozzle]:
            if not group['unchanged'] or not all(b.sha256(Path(f))==h for f,h in {**group['protected'],**group['source_sha256']}.items()):
                raise RuntimeError('Final source/executable differs from a current campaign')
        canonical=[r for r in oldwaves['runs'] if r['name'].endswith('-base')]+current['runs']+shocks['runs']+nozzle['runs']
        pairs={};records=[]
        for row in canonical:
            case=Path(row['case']);d=read(case/'run-definition.json');result=row['result'];analysis=result['analysis']
            if not result['input_unchanged'] or not result['run']['completed_ok']:raise RuntimeError('Incomplete or mutated canonical run')
            evidence={**analysis['evidence_sha256'],str(case/'stdout.log'):result['stdout_sha256']}
            if not all(b.sha256(Path(f))==h for f,h in evidence.items()):raise RuntimeError('Saved field/log evidence changed')
            mode='new' if d['new_physics'] else 'base';key=row['name'].rsplit('-',1)[0]
            pairs.setdefault(key,{})[mode]=d
            records.append(dict(name=row['name'],case=str(case.relative_to(root)),executable_sha256=d['executable_sha256'],
                library_sha256=d['thermo_library_sha256'],steps=d['steps'],outer=d['outer_correctors'],delta_t=d['delta_t'],
                analysis=analysis,acceptance=result['acceptance']))
        physical_keys=['kind','gas','cells','steps','outer_correctors','mean_mach','end_time','delta_t','initial_x','initial_p','initial_T','initial_rho','initial_U']
        for key,pair in pairs.items():
            if set(pair)!={'base','new'} or any(pair['base'][k]!=pair['new'][k] for k in physical_keys):raise RuntimeError('Unmatched physical inputs: '+key)
            for name,h in pair['base']['input_sha256'].items():
                if name!='system/controlDict' and pair['new']['input_sha256'].get(name)!=h:raise RuntimeError('Unmatched schemes/mesh/initial files: '+key+'/'+name)
        replay=[]
        previous={r['name']:r for r in oldwaves['runs']}
        for row in current['runs']:
            first=Path(previous[row['name']]['result']['analysis']['final_time_directory'])
            final=Path(row['result']['analysis']['final_time_directory'])
            comparisons={name:b.compare_fields(first/name,final/name) for name in [*b.CORE_FIELDS,*b.PHASE_FIELDS]}
            replay.append(dict(name=row['name'],max_abs_by_field={name:c['max_abs'] for name,c in comparisons.items()}))
        legacy=read(bench/'gas-legacy-ab/comparison.json');regressions=read(bench/'gas-legacy-regressions/checks.json')
        legacy_sources=read(bench/'gas-legacy-source-audit/checks.json')
        guards=[read(bench/name/'checks.json') for name in ['gas-final-guards','gas-final-seed-guard']]
        if not legacy_sources['passed'] or not regressions['pass'] or not all(g['passed'] for g in guards):raise RuntimeError('A required numerical/source regression or guard failed')
        air=read(bench/'gas-air-490-500/result.json');air_comparison=read(bench/'gas-air-490-500-comparison.json')
        if not air['acceptance']['passed'] or not air_comparison['passed']:raise RuntimeError('Real-case check failed')
        eos=[json.loads(line) for line in (root/'logs/build-gas-solver-release.log').read_text().splitlines() if line.startswith('{"states"')][-1]
        if not eos['passed']:raise RuntimeError('EOS check failed')
        compact_legacy=dict(passed=legacy['passed'],evidence='benchmarks/gas-legacy-ab/comparison.json',
            evidence_sha256=b.sha256(bench/'gas-legacy-ab/comparison.json'),comparisons=[compact_comparison(c) for c in legacy['comparisons']])
        compact_sources=dict(passed=legacy_sources['passed'],evidence='benchmarks/gas-legacy-source-audit/checks.json',
            evidence_sha256=b.sha256(bench/'gas-legacy-source-audit/checks.json'),cases=[
                dict(name=c['name'],passed=c['passed'],acceptance=c['acceptance'],replay=c['replay'],
                    same_binary_repeats=[dict(mode=r['mode'],exact=r['exact'],comparison=compact_comparison(r['comparison'])) for r in c['same_binary_repeats']])
                for c in legacy_sources['cases']])
        result=dict(scope='Canonical paired numerical validation; reported failed physics criteria are retained.',
            current_source_sha256=current['source_sha256'],current_binary_sha256=current['protected'],paired_input_checks=len(pairs),
            eos=eos,canonical_runs=records,final_wave_replay=replay,
            new_passed=sum(r['acceptance']['passed'] for r in records if r['name'].endswith('-new')),
            new_failed=[r['name'] for r in records if r['name'].endswith('-new') and not r['acceptance']['passed']],
            legacy_exact=compact_legacy,legacy_numerical_and_sources=compact_sources,
            regression_checks=[dict(name=c['name'],passed=c['pass']) for c in regressions['checks']],
            gas_guards=[dict(name=c['name'],passed=c['passed']) for g in guards for c in g['checks']],
            real_case=dict(acceptance=air['acceptance'],diagnostics=air['diagnostics'],source_unchanged=air['source_unchanged'],
                comparison=air_comparison),collection_tool_sha256=b.sha256(Path(__file__)))
        b.atomic_json(root/'reports/supersonic-validation-results.json',result)
        print(json.dumps(dict(pairs=len(pairs),new_passed=result['new_passed'],new_failed=result['new_failed'],
            legacy_exact=legacy['passed'],legacy_numerical_and_sources=legacy_sources['passed'],
            regressions=len(regressions['checks']),guards=len(result['gas_guards']),
            wave_replay_exact=all(v==0 for row in replay for v in row['max_abs_by_field'].values())),indent=2))


if __name__=='__main__':main()
