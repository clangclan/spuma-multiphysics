#!/usr/bin/env python3
"""MAIN: plot saved pressure fields against independent wave/Riemann references."""
import fcntl
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import benchmark as b
from supersonic_reference import acoustic_state,riemann


def main():
    root=b.PROJECT_ROOT;bench=root/'benchmarks';out=root/'reports/assets';out.mkdir(exist_ok=True)
    with b.RUN_LOCK.open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');evidence={}
        for kind,ax in zip(['wave','sod'],axes):
            for mode,color in [('base','#C26319'),('new','#2368A0')]:
                case=bench/('gas-final-waves' if mode=='base' else 'gas-final-release-waves')/('acoustic-n2o-m0-n160-'+mode) if kind=='wave' else bench/'gas-final-shocks'/('sod-air-m0-n160-'+mode)
                definition=json.loads((case/'run-definition.json').read_text());result=json.loads((case/'result.json').read_text())
                x=np.array(definition['initial_x']);pfile=Path(result['analysis']['final_time_directory'])/'p'
                pressure=b.read_internal_field(pfile).values[:,0];t=float(definition['end_time'])
                evidence[str(pfile.relative_to(root))]=b.sha256(pfile)
                if kind=='wave':
                    exact,_=acoustic_state(x,t,0,'n2o');values=pressure-exact['p']
                else:values=pressure/1000
                ax.plot(x,values,color=color,label='Previous' if mode=='base' else 'Gas mode',lw=1.8)
            if kind=='wave':
                ax.axhline(0,color='#333333',lw=1,ls='--',label='Linear acoustic reference')
                ax.set(title='PR gas: wave-pressure error (160 cells)',ylabel='Pressure error (Pa)')
            else:
                xx=np.linspace(0,1,4000);exact=riemann(xx,t)
                ax.plot(xx,exact['p']/1000,color='#333333',lw=1,ls='--',label='Exact Euler reference')
                ax.set(title='Sod shock tube (160 cells)',ylabel='Pressure (kPa)')
            ax.set(xlabel='Axial position (m)',xlim=(0,1));ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
        svg=out/'supersonic-validation.svg';fig.savefig(svg)
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        fig.savefig(bench/'supersonic-validation.png',dpi=150)
        plt.close(fig)
        b.atomic_json(out/'supersonic-plot-evidence.json',dict(input_sha256=evidence,plot_tool_sha256=b.sha256(Path(__file__)),
            scope='PR wave error uses the linear small-amplitude solution; Sod plots pointwise Euler reference against finite-volume cell values.'))


if __name__=='__main__':main()
