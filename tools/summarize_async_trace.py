#!/usr/bin/env python3
"""MAIN: read-only Nsight SQLite summaries; host wait overlaps GPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

ACTIVITIES='''SELECT k.start,k.end,'kernel' kind,s.value name,
 CASE WHEN s.value LIKE '%GaussSeidelSmoother::smooth_%' THEN 1 ELSE 0 END smoother
 FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
 UNION ALL SELECT start,end,'memcpy',NULL,0 FROM CUPTI_ACTIVITY_KIND_MEMCPY
 UNION ALL SELECT start,end,'memset',NULL,0 FROM CUPTI_ACTIVITY_KIND_MEMSET'''


def summarize(path):
    db=sqlite3.connect(f'file:{path}?mode=ro',uri=True);db.row_factory=sqlite3.Row
    def q(sql):return [dict(r) for r in db.execute(sql)]
    duty=q('WITH activities AS ('+ACTIVITIES+'''), ordered AS (
 SELECT *,max(end) OVER (ORDER BY start,end ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prev_end FROM activities
 ), marked AS (SELECT *,CASE WHEN prev_end IS NULL OR start>prev_end THEN 1 ELSE 0 END new_group FROM ordered),
 grouped AS (SELECT *,sum(new_group) OVER(ORDER BY start,end ROWS UNBOUNDED PRECEDING) gid FROM marked),
 islands AS (SELECT gid,min(start) start,max(end) end FROM grouped GROUP BY gid)
 SELECT sum(end-start)/1e6 busy_ms,(max(end)-min(start))/1e6 span_ms,
 ((max(end)-min(start))-sum(end-start))/1e6 idle_ms,
 100.0*sum(end-start)/(max(end)-min(start)) duty_percent FROM islands''')[0]
    smoother=q('WITH activities AS ('+ACTIVITIES+'''), ordered AS (
 SELECT *,lead(start) OVER(ORDER BY start,end) next_start FROM activities)
 SELECT count(*) calls,sum(end-start)/1e6 kernel_ms,avg(end-start)/1e3 average_kernel_us,
 sum(max(next_start-end,0))/1e6 post_gap_ms,avg(max(next_start-end,0))/1e3 average_post_gap_us
 FROM ordered WHERE smoother=1''')[0]
    api=q('''SELECT CASE WHEN n.value LIKE 'cudaLaunchKernel%' THEN 'cudaLaunchKernel'
 WHEN n.value LIKE 'cudaDeviceSynchronize%' THEN 'cudaDeviceSynchronize'
 WHEN n.value LIKE 'cudaMemcpyAsync%' THEN 'cudaMemcpyAsync'
 WHEN n.value LIKE 'cudaMemcpy%' THEN 'cudaMemcpy'
 WHEN n.value LIKE 'cudaMemset%' THEN 'cudaMemset' END api,
 count(*) calls,sum(r.end-r.start)/1e6 host_ms
 FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds n ON n.id=r.nameId
 WHERE n.value LIKE 'cudaLaunchKernel%' OR n.value LIKE 'cudaDeviceSynchronize%'
 OR n.value LIKE 'cudaMemcpy%' OR n.value LIKE 'cudaMemset%' GROUP BY api''')
    kernels=q('''SELECT count(*) calls,sum(end-start)/1e6 milliseconds,
 sum(end-start<=10000) under_10us_calls FROM CUPTI_ACTIVITY_KIND_KERNEL''')[0]
    copies=q('''SELECT e.label,count(*) calls,sum(m.bytes) bytes,sum(m.end-m.start)/1e6 milliseconds
 FROM CUPTI_ACTIVITY_KIND_MEMCPY m JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id=m.copyKind GROUP BY e.label''')
    db.close()
    return {'sqlite':str(path),'gpu_timeline':duty,'smoother':smoother,'runtime_api':api,'kernels':kernels,'copies':copies}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('sqlite',nargs='+',type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();result={'scope':'Profiled 10-second windows. GPU activity union includes kernels/memcpy/memset. Post-gap attribution is temporal, not complete causal proof; host API wait overlaps GPU work. Profiles are excluded from wall benchmarks.','traces':[summarize(p.resolve()) for p in args.sqlite]}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
