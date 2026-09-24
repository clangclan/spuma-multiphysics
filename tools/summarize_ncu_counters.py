#!/usr/bin/env python3
"""Preserve Nsight Compute wide CSV metrics and build a concise kernel table."""
import argparse
import csv
import io
from pathlib import Path
import benchmark as common
from analyze_gpu_trace import short,write_csv


def parse(path):
    text=path.read_text();start=text.find('"ID","Process ID"')
    if start<0:raise ValueError('No Nsight Compute raw CSV header')
    reader=csv.DictReader(io.StringIO(text[start:]));units=next(reader);rows=[]
    for raw in reader:
        if not raw.get('ID','').isdigit():continue
        values={}
        for key,value in raw.items():
            if value is None:continue
            try:values[key]=float(value.replace(',',''))
            except ValueError:values[key]=value
        rows.append(values)
    if not rows:raise ValueError('No profiled launches')
    return units,rows


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('csv',type=Path)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    units,rows=parse(a.csv);out=[]
    common.atomic_json(a.output/'all-counters.json',dict(units=units,launches=rows,
        scope='Selected kernel invocations under replay, uncontrolled clocks and caches. Not a weighted whole-solver average.'))
    for r in rows:
        def metric(*names):
            for name in names:
                if isinstance(r.get(name),(int,float)):return r[name]
            return None
        d=dict(id=r['ID'],kernel=short(r['Kernel Name']),grid=r['Grid Size'],block=r['Block Size'])
        mapping={
            'durationNs':('gpu__time_duration.sum',),
            'registersPerThread':('launch__registers_per_thread',),
            'theoreticalOccupancyPercent':('sm__maximum_warps_per_active_cycle_pct',),
            'achievedOccupancyPercent':('sm__warps_active.avg.pct_of_peak_sustained_active',),
            'smThroughputPercent':('sm__throughput.avg.pct_of_peak_sustained_elapsed',),
            'fp64PipePercent':('sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed',),
            'fmaPipePercent':('sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed',),
            'aluPipePercent':('sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed',),
            'tensorPipePercent':('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',),
            'l1HitPercent':('l1tex__t_sector_hit_rate.pct',),
            'l2HitPercent':('lts__t_sector_hit_rate.pct',),
            'dramBytesPerSecond':('dram__bytes.sum.per_second',),
            'dramThroughputPercent':('gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed','dram__throughput.avg.pct_of_peak_sustained_elapsed'),
            'l1ThroughputPercent':('l1tex__throughput.avg.pct_of_peak_sustained_active',),
            'l2ThroughputPercent':('lts__throughput.avg.pct_of_peak_sustained_elapsed',),
            'l1RequestedSectors':('l1tex__t_sectors.sum',),
            'l2SectorsPerSecond':('lts__t_sectors.sum.per_second',),
            'dramReadBytesPerSecond':('dram__bytes_op_read.sum.per_second',),
            'dramWriteBytesPerSecond':('dram__bytes_op_write.sum.per_second',),
            'issueActivePercent':('smsp__issue_active.avg.pct_of_peak_sustained_active',),
            'eligibleWarpsPerScheduler':('smsp__warps_eligible.avg.per_cycle_active',),
            'activeWarpsPerScheduler':('smsp__warps_active.avg.per_cycle_active',),
            'branchUniformPercent':('smsp__sass_average_branch_targets_threads_uniform.pct',),
            'localLoadSectors':('l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum',),
            'localStoreSectors':('l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',),
            'smActiveCyclesMean':('sm__cycles_active.avg',),
            'smActiveCyclesMin':('sm__cycles_active.min',),
            'smActiveCyclesMax':('sm__cycles_active.max',),
            'smElapsedCyclesMean':('sm__cycles_elapsed.avg',),
            'activeThreadsPerWarpInstruction':('smsp__thread_inst_executed_pred_on_per_inst_executed.ratio',),
            'waitCyclesPerIssuedInstruction':('smsp__average_warps_issue_stalled_wait_per_issue_active.ratio',),
            'shortScoreboardCyclesPerIssuedInstruction':('smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio',),
            'longScoreboardCyclesPerIssuedInstruction':('smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio',),
        }
        for key,names in mapping.items():d[key]=metric(*names)
        for precision,ops in [('fp64',('dadd','dmul','dfma')),('fp32',('fadd','fmul','ffma'))]:
            counts=[metric('smsp__sass_thread_inst_executed_op_'+op+'_pred_on.sum') for op in ops]
            d[precision+'Flops']=sum(x*(2 if i==2 else 1) for i,x in enumerate(counts)) if all(x is not None for x in counts) else None
            d[precision+'GflopsPerSecond']=d[precision+'Flops']/d['durationNs'] if d[precision+'Flops'] is not None and d['durationNs'] else None
        d['smActiveElapsedPercentDerived']=100*d['smActiveCyclesMean']/d['smElapsedCyclesMean'] if d['smElapsedCyclesMean'] and d['smActiveCyclesMean'] is not None else None
        d['l1SectorEquivalentGBperSecond']=32*d['l1RequestedSectors']/d['durationNs'] if d['l1RequestedSectors'] is not None and d['durationNs'] else None
        d['l2SectorEquivalentGBperSecond']=32*d['l2SectorsPerSecond']/1e9 if d['l2SectorsPerSecond'] is not None else None
        d['percentOutsidePhysicalRange']=','.join(k for k,v in d.items() if k in ('l1HitPercent','l2HitPercent','achievedOccupancyPercent') and v is not None and not 0<=v<=100)
        out.append(d)
    write_csv(a.output/'kernel-summary.csv',out)
    common.atomic_json(a.output/'kernel-summary.json',out)
    print('Summarized',len(rows),'profiled launches')


if __name__=='__main__':main()
