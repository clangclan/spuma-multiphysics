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
    domain=np.array(g['domainM'])*1000
    corners=np.array(list(itertools.product(*[(0,float(v)) for v in domain])))
    for p,q in itertools.combinations(corners,2):
        if np.count_nonzero(p!=q)==1:ax.plot(*np.stack((p,q)).T,color='#8091a5',lw=1)
    colors=('#1177cc','#e77b25')
    apertures=[[[0,5,4],[0,7,4],[0,7,6],[0,5,6]],[[5,0,4],[7,0,4],[7,0,6],[5,0,6]]]
    if g['nozzleShape']=='circle':
        angle=np.linspace(0,2*np.pi,129)
        r=g['nozzleDiameterM']*500
        apertures=[]
        for axis,center in enumerate(np.array(g['inletCentersM'])*1000):
            vertices=np.tile(center,(len(angle),1))
            vertices[:,1-axis]+=r*np.cos(angle)
            vertices[:,2]+=r*np.sin(angle)
            apertures.append(vertices)
    for verts,color in zip(apertures,colors):
        ax.add_collection3d(Poly3DCollection([verts],facecolor=color,edgecolor=color,alpha=.85))
    for c,d,color in zip(np.array(g['inletCentersM'])*1000,g['nominalDirections'],colors):
        ax.quiver(*c,*(np.array(d)*6),color=color,lw=2.8,arrow_length_ratio=.15)
    intersection=np.array(g['nominalIntersectionM'])*1000
    ax.scatter(*intersection,s=42,c='#19283c')
    port=f"Ø {g['nozzleDiameterM']*1000:g} mm" if g['nozzleShape']=='circle' else '2 × 2 mm'
    position=', '.join(f'{v:g}' for v in intersection)
    ax.text2D(.02,.92,'Two liquid N₂O ports: '+port+f'\nAxes intersect at ({position}) mm, 90°',transform=ax.transAxes,fontsize=10)
    ax.set(xlim=(0,domain[0]),ylim=(0,domain[1]),zlim=(0,domain[2]),xlabel='x [mm]',ylabel='y [mm]',zlabel='z [mm]')
    ax.set_box_aspect(domain.copy());ax.view_init(elev=25,azim=225)
    ax.set_xticks(np.linspace(0,domain[0],5));ax.set_yticks(np.linspace(0,domain[1],5));ax.set_zticks(np.linspace(0,domain[2],5))
    fig.suptitle('90° impinging liquid-N₂O benchmark',fontsize=18,fontweight='bold',x=.5,y=.96)
    fig.text(.5,.905,'Supply: 55 bar(g) = 56.01325 bar(abs), 20°C  |  Ambient: 1.01325 or 40 bar(abs)',ha='center',fontsize=11)
    spacing=np.array(g.get('cellSpacingM',[g['cellEdgeM']]*3))*1000
    delta=' × '.join(f'{v:g}' for v in spacing)
    fig.text(.5,.07,f"Uniform Cartesian mesh: {g['shape'][0]} × {g['shape'][1]} × {g['shape'][2]} = {g['cells']:,} cells  |  Δ = {delta} mm",ha='center',fontsize=11)
    fig.text(.5,.035,'Arrows show prescribed inlet axes; no computed jet or breakup result is shown.',ha='center',fontsize=9,color='#526376')
    fig.subplots_adjust(top=.86,bottom=.11,left=.02,right=.98)
    for ext in ('png','svg'):fig.savefig(a.benchmark/f'geometry-preview.{ext}',dpi=160)
    plt.close(fig)
if __name__=='__main__':main()
