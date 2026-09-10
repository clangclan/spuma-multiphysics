// SPDX-License-Identifier: GPL-3.0-or-later
// Main-agent implementation. Algorithm reference: SPUMA v2512 MULESTemplates.C.
#include "pintleLimiter.H"
#include "MULES.H"
#include "localEulerDdtScheme.H"
#include "upwind.H"
#include "syncTools.H"
#include "wedgeFvPatch.H"
#include <cmath>
#include <limits>
#include <type_traits>

namespace Foam
{
namespace pintleKernel
{
#ifdef have_cuda
template<class F> __global__ void loop(F f, label n)
{
    for (label i=blockIdx.x*blockDim.x+threadIdx.x;
         i<n; i+=blockDim.x*gridDim.x) f(i);
}
#endif
// Same default stream as SPUMA. All referenced storage survives until finish().
// No per-kernel device synchronization inside the limiter iteration.
template<class F> void launch(F f, label n)
{
    if (!n) return;
#ifdef have_cuda
    loop<<<min(label(65535),(n+255)/256),256>>>(f,n);
    const auto err=cudaGetLastError();
    if (err!=cudaSuccess)
        FatalErrorInFunction << cudaGetErrorString(err) << abort(FatalError);
#else
    for (label i=0;i<n;++i) f(i);
#endif
}
void finish()
{
#ifdef have_cuda
    const auto err=cudaDeviceSynchronize();
    if (err!=cudaSuccess)
        FatalErrorInFunction << cudaGetErrorString(err) << abort(FatalError);
#endif
}
template<class Ratio> FOAM_DEVICE Ratio boundedRatio(scalar numerator, scalar denominator)
{
    const scalar value=max(scalar(0),min(scalar(1),numerator/(denominator+ROOTVSMALL)));
    Ratio result=static_cast<Ratio>(value);
    if constexpr (std::is_same<Ratio,float>::value)
    {
        // Directed rounding: float storage must never relax the FP64 bound.
        if (scalar(result)>value) result=::nextafterf(result,0.0f);
    }
    return result;
}
}

pintleLimiter::pintleLimiter(const fvMesh& mesh)
:
    mesh_(mesh), boundaryStart_(mesh.nCells()+1,Zero,poolSwitch(true)),
    boundaryFaces_(mesh.nFaces()-mesh.nInternalFaces(),poolSwitch(true)),
    boundaryOwner_(boundaryFaces_.size(),label(-1),poolSwitch(true)), boundaryKind_(boundaryFaces_.size(),label(-1),poolSwitch(true)),
    boundaryPsi_(boundaryFaces_.size()), boundaryBD_(boundaryFaces_.size()),
    boundaryCorr_(boundaryFaces_.size()),
    upperBudget_(mesh.nCells()), lowerBudget_(mesh.nCells()),
    positive_(mesh.nCells()), negative_(mesh.nCells()),
    ratioIn_(mesh.nCells()), ratioOut_(mesh.nCells()), lambda_(mesh.nFaces()),
    ratioIn32_(mesh.nCells()), ratioOut32_(mesh.nCells()), coupled_(false)
{
    // One-time topology setup. Cell gather also handles several patch faces
    // belonging to the same cell, without atomic writes or patch races.
    forAll(mesh.boundary(),p)
    {
        const auto& patch=mesh.boundary()[p];
        coupled_=coupled_ || patch.coupled();
        forAll(patch.faceCells(),j)
        {
            const label b=patch.start()+j-mesh.nInternalFaces();
            const label c=patch.faceCells()[j];
            boundaryOwner_[b]=c;
            ++boundaryStart_[c+1];
            boundaryKind_[b]=isA<wedgeFvPatch>(patch) ? 2 : patch.coupled() ? 1 : 0;
        }
    }
    for(label c=1;c<boundaryStart_.size();++c) boundaryStart_[c]+=boundaryStart_[c-1];
    labelList next(boundaryStart_);
    forAll(boundaryOwner_,b)
        if(boundaryOwner_[b]>=0) boundaryFaces_[next[boundaryOwner_[b]]++]=b;
    // Trigger SPUMA's lazy CSR construction before any asynchronous kernels.
    mesh.lduAddr().ownerStartAddr();
    mesh.lduAddr().losortStartAddr();
    mesh.lduAddr().losortAddr();
}

template<class Ratio> void pintleLimiter::iterate
(
    const volScalarField& psi, const surfaceScalarField& phiBD,
    const surfaceScalarField& corr, const scalar rdt, const label iterations,
    const scalar extrema, const scalar boundaryExtra, const scalar smooth,
    Field<Ratio>& rin, Field<Ratio>& rout
)
{
    const auto& addr=mesh_.lduAddr();
    const auto os=addr.ownerStartAddr().cbegin(), ns=addr.losortStartAddr().cbegin();
    const auto nf=addr.losortAddr().cbegin(), own=mesh_.owner().cbegin();
    const auto nei=mesh_.neighbour().cbegin();
    const auto bs=boundaryStart_.cbegin(), bf=boundaryFaces_.cbegin();
    const auto bo=boundaryOwner_.cbegin(), bk=boundaryKind_.cbegin();
    const auto a=psi.cbegin(), a0=psi.oldTime().cbegin();
    const auto V=mesh_.V().cbegin(), bd=phiBD.cbegin(), fc=corr.cbegin();
    const auto bp=boundaryPsi_.cbegin(), bbd=boundaryBD_.cbegin(), bc=boundaryCorr_.cbegin();
    auto up=upperBudget_.begin(), low=lowerBudget_.begin();
    auto pos=positive_.begin(), neg=negative_.begin(), lam=lambda_.begin();
    auto ri=rin.begin(), ro=rout.begin();
    const label nc=mesh_.nCells(), ni=mesh_.nInternalFaces(), nt=mesh_.nFaces();
    pintleKernel::launch([=](label c)
    {
        scalar hi=0, lo=1, base=0, plus=0, minus=0;
        // Incoming faces precede outgoing faces in the upstream face ordering.
        for(label j=ns[c];j<ns[c+1];++j)
        {
            const label f=nf[j]; const scalar q=-fc[f];
            hi=max(hi,a[own[f]]); lo=min(lo,a[own[f]]); base-=bd[f];
            plus+=max(q,scalar(0)); minus+=max(-q,scalar(0));
        }
        for(label f=os[c];f<os[c+1];++f)
        {
            const scalar q=fc[f];
            hi=max(hi,a[nei[f]]); lo=min(lo,a[nei[f]]); base+=bd[f];
            plus+=max(q,scalar(0)); minus+=max(-q,scalar(0));
        }
        for(label j=bs[c];j<bs[c+1];++j)
        {
            const label b=bf[j]; const scalar q=bc[b];
            // NaN sentinel marks a non-fixed uncoupled boundary.
            if (bp[b]==bp[b]) {hi=max(hi,bp[b]);lo=min(lo,bp[b]);}
            else {hi+=boundaryExtra;lo-=boundaryExtra;}
            base+=bbd[b]; plus+=max(q,scalar(0)); minus+=max(-q,scalar(0));
        }
        hi=min(hi+extrema,scalar(1)); lo=max(lo-extrema,scalar(0));
        if (smooth>SMALL)
        {hi=min(smooth*a[c]+(1-smooth)*hi,scalar(1));lo=max(smooth*a[c]+(1-smooth)*lo,scalar(0));}
        up[c]=V[c]*(rdt*hi-rdt*a0[c])+base;
        low[c]=V[c]*(-rdt*lo+rdt*a0[c])-base;
        pos[c]=plus;neg[c]=minus;
    },nc);
    pintleKernel::launch([=](label f){lam[f]=1;},nt);
    for(label iteration=0;iteration<iterations;++iteration)
    {
        // Fuse gather with both cell ratio calculations: no sum scratch arrays.
        pintleKernel::launch([=](label c)
        {
            scalar plus=0,minus=0;
            for(label j=ns[c];j<ns[c+1];++j)
            {const label f=nf[j];const scalar q=-lam[f]*fc[f];plus+=max(q,scalar(0));minus+=max(-q,scalar(0));}
            for(label f=os[c];f<os[c+1];++f)
            {const scalar q=lam[f]*fc[f];plus+=max(q,scalar(0));minus+=max(-q,scalar(0));}
            for(label j=bs[c];j<bs[c+1];++j)
            {const label b=bf[j];const scalar q=lam[ni+b]*bc[b];plus+=max(q,scalar(0));minus+=max(-q,scalar(0));}
            ri[c]=pintleKernel::boundedRatio<Ratio>(plus+up[c],neg[c]);
            ro[c]=pintleKernel::boundedRatio<Ratio>(minus+low[c],pos[c]);
        },nc);
        pintleKernel::launch([=](label f)
        {
            if(f<ni)
            {
                const scalar r=fc[f]>0 ? min(scalar(ro[own[f]]),scalar(ri[nei[f]]))
                    : min(scalar(ri[own[f]]),scalar(ro[nei[f]]));
                lam[f]=min(lam[f],r);
            }
            else
            {
                const label b=f-ni;
                if(bk[b]==2) lam[f]=0;
                else if(bk[b]==1)
                    lam[f]=min(lam[f],bc[b]>0 ? scalar(ro[bo[b]]) : scalar(ri[bo[b]]));
            }
        },nt);
        if(coupled_)
        {
            pintleKernel::finish();
            syncTools::syncFaceList(mesh_,lambda_,minEqOp<scalar>());
        }
    }
    pintleKernel::finish();
}

void pintleLimiter::limit
(
    const volScalarField& psi, const surfaceScalarField& phi,
    surfaceScalarField& phiPsi, bool mixed, bool returnCorr
)
{
    if(mesh_.moving() || mesh_.topoChanging() || fv::localEulerDdt::enabled(mesh_))
        FatalErrorInFunction << "pintleLimiter requires a fixed mesh and global Euler time step" << abort(FatalError);
    const auto& controls=mesh_.solverDict(psi.name());
    const label it=controls.getOrDefault<label>("nLimiterIter",3);
    const scalar ex=controls.getOrDefault<scalar>("extremaCoeff",0);
    const scalar be=controls.getOrDefault<scalar>("boundaryExtremaCoeff",ex);
    const scalar sm=controls.getOrDefault<scalar>("smoothLimiter",0);
    if(it<1) FatalErrorInFunction << "nLimiterIter must be positive" << abort(FatalError);
    surfaceScalarField bd(upwind<scalar>(mesh_,phi).flux(psi));
    forAll(bd.boundaryField(),p)
        if(!bd.boundaryField()[p].coupled()) bd.boundaryFieldRef()[p]=phiPsi.boundaryField()[p];
    phiPsi-=bd;
    auto bp=boundaryPsi_.begin(), bbd=boundaryBD_.begin(), bc=boundaryCorr_.begin();
    foamExecutor exec;
    forAll(mesh_.boundary(),p)
    {
        const auto& pp=psi.boundaryField()[p];
        const label start=mesh_.boundary()[p].start()-mesh_.nInternalFaces();
        const auto bdptr=bd.boundaryField()[p].cbegin(), cptr=phiPsi.boundaryField()[p].cbegin();
        tmp<scalarField> neighbour;
        const scalar* ap=pp.cbegin();
        if(pp.coupled()) {neighbour=pp.patchNeighbourField();ap=neighbour().cbegin();}
        const bool useValue=pp.coupled() || pp.fixesValue();
        const scalar marker=std::numeric_limits<scalar>::quiet_NaN();
        auto pack=[=](label j){bp[start+j]=useValue ? ap[j] : marker;bbd[start+j]=bdptr[j];bc[start+j]=cptr[j];};
        exec.parallelFor(pack,pp.size()); // Neighbour temporary must survive packing.
    }
    if(mixed) iterate(psi,bd,phiPsi,1/mesh_.time().deltaTValue(),it,ex,max(be-ex,scalar(0)),sm,ratioIn32_,ratioOut32_);
    else iterate(psi,bd,phiPsi,1/mesh_.time().deltaTValue(),it,ex,max(be-ex,scalar(0)),sm,ratioIn_,ratioOut_);
    slicedSurfaceScalarField lambda
    (
        IOobject("pintleLambda",mesh_.time().timeName(),mesh_,IOobject::NO_READ,IOobject::NO_WRITE,IOobject::NO_REGISTER),
        mesh_,dimless,lambda_,false
    );
    phiPsi*=lambda;
    if(!returnCorr) phiPsi+=bd;
}

void pintleLimiter::explicitSolve(volScalarField& psi,const surfaceScalarField& flux,
    const volScalarField::Internal& Sp,const volScalarField::Internal& Su)
{
    const auto& addr=mesh_.lduAddr();
    const auto os=addr.ownerStartAddr().cbegin(),ns=addr.losortStartAddr().cbegin();
    const auto nf=addr.losortAddr().cbegin(),bs=boundaryStart_.cbegin(),bf=boundaryFaces_.cbegin();
    const auto a0=psi.oldTime().cbegin(),sp=Sp.cbegin(),su=Su.cbegin(),V=mesh_.V().cbegin();
    const auto q=flux.cbegin();auto bq=boundaryCorr_.begin(),a=psi.begin();
    const scalar rdt=1/mesh_.time().deltaTValue();
    forAll(flux.boundaryField(),p)
    {
        const auto qp=flux.boundaryField()[p].cbegin();
        const label start=mesh_.boundary()[p].start()-mesh_.nInternalFaces();
        pintleKernel::launch([=](label j){bq[start+j]=qp[j];},flux.boundaryField()[p].size());
    }
    pintleKernel::launch([=](label c)
    {
        scalar net=0;
        for(label j=ns[c];j<ns[c+1];++j) net-=q[nf[j]];
        for(label f=os[c];f<os[c+1];++f) net+=q[f];
        for(label j=bs[c];j<bs[c+1];++j) net+=bq[bf[j]];
        a[c]=(a0[c]*rdt+su[c]-net/V[c])/(rdt-sp[c]);
    },mesh_.nCells());
    pintleKernel::finish();
    psi.correctBoundaryConditions();
    Info<< "Pintle MULES: Solving for " << psi.name() << endl;
}

void pintleLimiter::limitSum(PtrList<surfaceScalarField>& fields)
{
    if(fields.size()!=3) {MULES::limitSum(fields);return;}
    auto apply=[](scalar* a,scalar* b,scalar* c,label size)
    {
        pintleKernel::launch([=](label f)
        {
            const scalar x=a[f],y=b[f],z=c[f];
            const scalar pos=max(x,scalar(0))+max(y,scalar(0))+max(z,scalar(0));
            const scalar neg=min(x,scalar(0))+min(y,scalar(0))+min(z,scalar(0));
            const scalar sum=pos+neg;
            if(sum>0 && pos>VSMALL)
            {const scalar l=-neg/pos;a[f]=x>0 ? x*l:x;b[f]=y>0 ? y*l:y;c[f]=z>0 ? z*l:z;}
            else if(sum<0 && neg<-VSMALL)
            {const scalar l=-pos/neg;a[f]=x<0 ? x*l:x;b[f]=y<0 ? y*l:y;c[f]=z<0 ? z*l:z;}
        },size);
    };
    apply(fields[0].begin(),fields[1].begin(),fields[2].begin(),fields[0].size());
    forAll(fields[0].boundaryField(),p)
    {
        // Match stock MULES: only coupled boundary correction fluxes are limited.
        if(fields[0].boundaryField()[p].coupled())
            apply(fields[0].boundaryFieldRef()[p].begin(),fields[1].boundaryFieldRef()[p].begin(),
                fields[2].boundaryFieldRef()[p].begin(),fields[0].boundaryField()[p].size());
    }
    pintleKernel::finish();
}
}
