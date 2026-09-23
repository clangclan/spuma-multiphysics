#!/usr/bin/env python3
"""Draw the prescribed geometry, not simulated jet trajectories."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('benchmark',type=Path);a=ap.parse_args()
    j=json.loads((a.benchmark/'benchmark-matrix.json').read_text());g=j['geometry']
    fig=plt.figure(figsize=(11,7.5),facecolor='#f5f7fa');ax=fig.add_subplot(111,projection='3d',facecolor='#f5f7fa')
    corners=np.array(list(itertools.product((0,20),(0,20),(0,10))))
    for p,q in itertools.combinations(corners,2):
        if np.count_nonzero(p!=q)==1:ax.plot(*np.stack((p,q)).T,color='#8091a5',lw=1)
    colors=('#1177cc','#e77b25')
    apertures=[[[0,5,4],[0,7,4],[0,7,6],[0,5,6]],[[5,0,4],[7,0,4],[7,0,6],[5,0,6]]]
    for verts,color in zip(apertures,colors):
        ax.add_collection3d(Poly3DCollection([verts],facecolor=color,edgecolor=color,alpha=.85))
    for c,d,color in zip(np.array(g['inletCentersM'])*1000,g['nominalDirections'],colors):
        ax.quiver(*c,*(np.array(d)*6),color=color,lw=2.8,arrow_length_ratio=.15)
    ax.scatter([6],[6],[5],s=42,c='#19283c');ax.text(6.2,6.5,5.2,'90°\n(6, 6, 5) mm',fontsize=10,color='#19283c')
    ax.text(0,8,7,'Liquid N₂O\n2 × 2 mm',color=colors[0],fontsize=10)
    ax.text(8,0,7,'Liquid N₂O\n2 × 2 mm',color=colors[1],fontsize=10)
    ax.set(xlim=(0,20),ylim=(0,20),zlim=(0,10),xlabel='x [mm]',ylabel='y [mm]',zlabel='z [mm]')
    ax.set_box_aspect((2,2,1));ax.view_init(elev=25,azim=225)
    ax.set_xticks([0,5,10,15,20]);ax.set_yticks([0,5,10,15,20]);ax.set_zticks([0,5,10])
    fig.suptitle('90° impinging liquid-N₂O benchmark',fontsize=18,fontweight='bold',x=.5,y=.96)
    fig.text(.5,.905,'Supply: 55 bar(g) = 56.01325 bar(abs), 20°C  |  Ambient: 1.01325 or 40 bar(abs)',ha='center',fontsize=11)
    fig.text(.5,.07,f"Uniform cubic mesh: {g['shape'][0]} × {g['shape'][1]} × {g['shape'][2]} = {g['cells']:,} cells  |  Δ = {g['cellEdgeM']*1000:g} mm",ha='center',fontsize=11)
    fig.text(.5,.035,'Arrows show prescribed inlet axes; no computed jet or breakup result is shown.',ha='center',fontsize=9,color='#526376')
    fig.subplots_adjust(top=.86,bottom=.11,left=.02,right=.98)
    for ext in ('png','svg'):fig.savefig(a.benchmark/f'geometry-preview.{ext}',dpi=160)
    plt.close(fig)
if __name__=='__main__':main()
