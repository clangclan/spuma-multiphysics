#!/usr/bin/env python3
"""Reconstruct a smooth spherical-cap graph from VOF cell volumes, then test face balance.

The sphere supplies input cell volumes and an independent reference only. The
reconstruction sees no sphere center, radius, curvature or analytic face data.
This probes a single-valued graph patch, not a closed-drop production method.
"""
import argparse
import json
import numpy as np
from scipy.interpolate import RectBivariateSpline


def quadrature(order):
    points, weights = np.polynomial.legendre.leggauss(order)
    return .5 * (points + 1), .5 * weights


def split_quadrature(order):
    # Each voxel crosses a spline knot at its center; integrate each smooth
    # half separately to avoid a Gauss rule spanning a third-derivative jump.
    u,w=quadrature(order)
    return np.r_[.5*u,.5+.5*u],np.r_[.5*w,.5*w]


def sample_analytic_cap(n, radius, order):
    width = .9*radius
    dx = width/n
    dz = 1.05*radius/n
    centers = -width/2+(np.arange(n)+.5)*dx
    u,w = quadrature(order)
    x = centers[:,None,None,None]+(u[None,:,None,None]-.5)*dx
    y = centers[None,None,:,None]+(u[None,None,None,:]-.5)*dx
    height = np.sqrt(radius*radius-x*x-y*y)
    voxel = np.empty((n,n,n))
    for k in range(n):
        fraction = np.clip((height-k*dz)/dz,0,1)
        voxel[:,:,k] = np.einsum('a,b,iajb->ij',w,w,fraction)
    return centers,dx,dz,voxel


def deconvolved_height(voxel, centers, dx, dz):
    # The column sum is a cell-average height. Undo its O(h^2) box-filter bias
    # with only neighboring column volumes; no analytic shape data enters.
    averaged = voxel.sum(axis=2)*dz
    hxx = np.gradient(np.gradient(averaged,dx,axis=0,edge_order=2),dx,axis=0,edge_order=2)
    hyy = np.gradient(np.gradient(averaged,dx,axis=1,edge_order=2),dx,axis=1,edge_order=2)
    point_height = averaged-dx*dx*(hxx+hyy)/24
    return RectBivariateSpline(centers,centers,point_height,kx=3,ky=3,s=0),point_height


def local_quadratic(values, centers, i, j):
    x0,y0=centers[i],centers[j]
    matrix=[];target=[]
    for a in range(i-1,i+2):
        for b in range(j-1,j+2):
            x,y=centers[a]-x0,centers[b]-y0
            matrix.append((1,x,y,x*x,x*y,y*y))
            target.append(values[a,b])
    coefficients=np.linalg.lstsq(matrix,target,rcond=None)[0]
    return lambda x,y: np.dot(coefficients,(1,x-x0,y-y0,(x-x0)**2,
                                          (x-x0)*(y-y0),(y-y0)**2))


def cell_forces(height, i, j, k, centers, dx, dz, sigma, pressure_jump, order):
    u,w=split_quadrature(order)
    xmin=centers[i]-dx/2; ymin=centers[j]-dx/2; zmin=k*dz
    normals=((1.,0.,0.),(-1.,0.,0.),(0.,1.,0.),(0.,-1.,0.))
    pressure=np.array([0.,0.,-pressure_jump*dx*dx])
    traction=np.zeros(3)
    def edge(axis,side):
        coord=(xmin if side<0 else xmin+dx) if axis==0 else (ymin if side<0 else ymin+dx)
        varying=(ymin+dx*u) if axis==0 else (xmin+dx*u)
        x=np.full_like(varying,coord) if axis==0 else varying
        y=varying if axis==0 else np.full_like(varying,coord)
        h,hx,hy=height(x,y)[:3]
        n=np.column_stack((-hx,-hy,np.ones_like(h)))
        n/=np.linalg.norm(n,axis=1)[:,None]
        face=np.array(normals[2*axis+(0 if side>0 else 1)])
        projected=face[None,:]-np.sum(n*face[None,:],axis=1)[:,None]*n
        m=projected/np.linalg.norm(projected,axis=1)[:,None]
        ds=np.sqrt(1+(hy if axis==0 else hx)**2)
        area=dx*np.dot(w,h-zmin)
        force=sigma*dx*np.einsum('a,a,ad->d',w,ds,m)
        return pressure_jump*area*face,force
    for axis in (0,1):
        for side in (-1,1):
            p,t=edge(axis,side);pressure+=p;traction+=t
    # Independent 2D surface quadrature measures both geometric identities.
    x=xmin+dx*u[:,None]
    y=ymin+dx*u[None,:]
    h,hx,hy,hxx,hxy,hyy=height(x,y)
    w2=np.outer(w,w)*dx*dx
    vector_area=np.array([np.sum(w2*-hx),np.sum(w2*-hy),np.sum(w2)])
    surface_area=np.sum(w2*np.sqrt(1+hx*hx+hy*hy))
    den=(1+hx*hx+hy*hy)**1.5
    curvature=-((1+hy*hy)*hxx-2*hx*hy*hxy+(1+hx*hx)*hyy)/den
    curved_area=np.array([np.sum(w2*curvature*-hx),
                          np.sum(w2*curvature*-hy),np.sum(w2*curvature)])
    return pressure,traction,vector_area,curved_area,surface_area


def evaluate_grid(n, radius, sigma, volume_order, face_order):
    centers,dx,dz,voxel=sample_analytic_cap(n,radius,volume_order)
    spline,point_height=deconvolved_height(voxel,centers,dx,dz)
    def rebuilt(x,y):
        return (spline.ev(x,y),spline.ev(x,y,dx=1,dy=0),
                spline.ev(x,y,dx=0,dy=1),spline.ev(x,y,dx=2,dy=0),
                spline.ev(x,y,dx=1,dy=1),spline.ev(x,y,dx=0,dy=2))
    def analytic(x,y):
        h=np.sqrt(radius*radius-x*x-y*y)
        return h,-x/h,-y/h,-1/h-x*x/h**3,-x*y/h**3,-1/h-y*y/h**3
    pressure_jump=2*sigma/radius
    u,w=split_quadrature(face_order)
    residual=[]; reference=[];wrong_jump=[];volume_error=[];trace_error=[];scale=[]
    aperture_closure=[];traction_closure=[]
    area_error=[]
    for i in range(4,n-4):
        for j in range(4,n-4):
            xedges=centers[i]+np.array([-.5,.5])*dx
            yedges=centers[j]+np.array([-.5,.5])*dx
            truth=[float(analytic(x,y)[0]) for x in xedges for y in yedges]
            reconstructed=[np.asarray(rebuilt(x,y)[0]).item() for x in xedges for y in yedges]
            lo=min(min(truth),min(reconstructed));hi=max(max(truth),max(reconstructed))
            k=int(np.floor(lo/dz))
            # A graph wholly inside this voxel has only vertical contact
            # curves. Horizontal faces have full/empty liquid apertures.
            if k<0 or k>=n or hi>=(k+1)*dz:
                continue
            p,t,N,K,A=cell_forces(rebuilt,i,j,k,centers,dx,dz,sigma,pressure_jump,face_order)
            pt,tt,_,_,A_truth=cell_forces(analytic,i,j,k,centers,dx,dz,sigma,pressure_jump,face_order)
            residual.append(np.linalg.norm(p-t));reference.append(np.linalg.norm(pt-tt))
            wrong_jump.append(np.linalg.norm(1.01*p-t))
            scale.append(max(np.linalg.norm(p),np.linalg.norm(t),1e-30))
            aperture_closure.append(np.linalg.norm(p/pressure_jump+N)/(dx*dx))
            traction_closure.append(np.linalg.norm(t+sigma*K)
                                    /max(np.linalg.norm(t),1e-30))
            area_error.append(abs(A/A_truth-1))
            x=centers[i]+(u[:,None]-.5)*dx
            y=centers[j]+(u[None,:]-.5)*dx
            h=rebuilt(x,y)[0]
            fraction_geom=np.einsum('a,b,ab->',w,w,(h-k*dz)/dz)
            volume_error.append(abs(fraction_geom-voxel[i,j,k]))
            if i<n-5:
                left=local_quadratic(point_height,centers,i,j)
                right=local_quadratic(point_height,centers,i+1,j)
                trace_error.append(abs(left(centers[i]+dx/2,centers[j])
                                       -right(centers[i]+dx/2,centers[j]))/dz)
    assert residual and trace_error
    residual=np.asarray(residual);scale=np.asarray(scale)
    candidates=(n-8)**2
    return {'n':n,'candidateColumns':candidates,'eligibleCells':len(residual),
            'excludedHorizontalCrossingColumns':candidates-len(residual),
            'eligibleFraction':len(residual)/candidates,'cellWidthM':dx,
            'maxScaledForceResidual':float(np.max(residual/scale)),
            'maxScaledWrongJumpResidual':float(np.max(np.asarray(wrong_jump)/scale)),
            'l2ScaledForceResidual':float(np.sqrt(np.mean((residual/scale)**2))),
            'p95ScaledForceResidual':float(np.quantile(residual/scale,.95)),
            'maxAnalyticForceResidualN':float(np.max(reference)),
            'maxGeometricVolumeFractionError':float(np.max(volume_error)),
            'maxInterfaceAreaRelativeError':float(np.max(area_error)),
            'maxApertureVectorClosure':float(np.max(aperture_closure)),
            'maxTractionCurvatureClosure':float(np.max(traction_closure)),
            'maxIndependentQuadraticTraceJumpOverDz':float(np.max(trace_error)),
            'sharedSplineTraceJump':0.0}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    grids=[evaluate_grid(n,1e-3,.01,8,12) for n in (16,24,32,48)]
    result={'scope':'single-valued 3D cap patch reconstructed from cell VOF volume only',
            'notProduction':True,
            'eligibility':'Only interior columns whose analytic and reconstructed graph lie wholly inside one z cell; horizontal face intersections and chart seams are excluded.',
            'grids':grids}
    with open(args.output,'w') as output:json.dump(result,output,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
