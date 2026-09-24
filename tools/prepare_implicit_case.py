#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Actual-solver static-drop fixture with independent sharp sphere volumes.

Sphere parameters are used only by the fixture generator. ReactiveFoam sees
the native mesh, conservative phase inventories and normal physical fields.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import yaml
from prepare_capillary_case import CartesianGrid, prepare
from vof_implicit_reconstruction import load_test_vof


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--configuration',type=Path,required=True)
    p.add_argument('--geometry',choices=['diffuse','cartesianImplicit'],default='cartesianImplicit')
    p.add_argument('--end-time',type=float,default=6e-8)
    p.add_argument('--volume-tolerance',type=float,default=1e-9)
    args=p.parse_args()
    root=Path(__file__).resolve().parent.parent
    configuration=args.configuration.resolve()
    model=yaml.safe_load(configuration.read_text())
    mechanism=Path(model['mechanism'])
    if not mechanism.is_absolute():mechanism=configuration.parent/mechanism
    if not mechanism.is_file():raise FileNotFoundError(mechanism)
    model['mechanism']=str(mechanism.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    resolved=args.output.parent/(args.output.name+'-thermo.yaml')
    resolved.write_text(yaml.safe_dump(model,sort_keys=False))
    with tempfile.TemporaryDirectory(prefix='sharp-drop-fixture-') as directory:
        exporter=Path(directory)/'oracle'
        subprocess.run(['g++','-std=c++17','-O2',str(root/'tools/geometric_oracle_export.cpp'),'-o',str(exporter)],check=True)
        color,h,radius=load_test_vof(exporter,args.n)
    grid=CartesianGrid((args.n,)*3,(.004,)*3)
    result=prepare(args.output,grid,'sphere','interfaceColor',radius=radius,
                   run_block_mesh=True,configuration=resolved,
                   end_time=args.end_time,color_override=color.ravel(),curved_primitive=args.geometry=='cartesianImplicit')
    case=args.output.resolve();properties=case/'constant/reactiveProperties'
    with properties.open('a') as stream:
        if args.geometry=='cartesianImplicit':stream.write('\ncapillaryInitialColorField interfaceColor;\n')
        stream.write(f'\ncapillaryGeometry {args.geometry};\n'
                     f'capillaryVolumeTolerance {args.volume_tolerance:.17g};\n'
                     'capillaryQuadratureTolerance 1e-9;\ncapillaryFitIterations 40;\n')
    result.update(centerM=[.002071,.001947,.002037],geometry=args.geometry,
                  volumeTolerance=args.volume_tolerance,independentCellWidthM=h)
    (case/'implicit-fixture.json').write_text(json.dumps(result,indent=2)+'\n')
    (case/'geometry-definition.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
