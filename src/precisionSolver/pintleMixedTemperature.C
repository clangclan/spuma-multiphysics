// SPDX-License-Identifier: GPL-3.0-or-later
// Main-agent mixed-precision refinement of the unchanged temperature increment
// system. Only inner Jacobi arithmetic/storage changes. True residuals and the
// accumulated increment are FP64. A failed inner path falls back to FP64.
#include "smoothSolver.H"
#include "Pstream.H"
#include "pintleIrKernels.h"
#include <cmath>

namespace Foam
{
class pintleMixedTemperature : public lduMatrix::solver
{
public:
    TypeName("pintleMixedTemperature");
    pintleMixedTemperature(const word& name,const lduMatrix& matrix,
        const FieldField<Field,scalar>& bou,const FieldField<Field,scalar>& internal,
        const lduInterfaceFieldPtrsList& interfaces,const dictionary& controls)
    :lduMatrix::solver(name,matrix,bou,internal,interfaces,controls){}

    solverPerformance solve(scalarField& psi,const scalarField& source,
                            const direction cmpt=0) const override
    {
        static_assert(sizeof(scalar)==8&&sizeof(label)==4,"FP64/Int32 ABI required");
        if(Pstream::parRun())
            FatalErrorInFunction << "Experimental refinement requires one rank" << abort(FatalError);
        forAll(interfaces_,i)if(interfaces_.set(i))
            FatalErrorInFunction << "Experimental refinement excludes coupled interfaces" << abort(FatalError);
        const word precision(controlDict_.getOrDefault<word>("innerPrecision","fp32"));
        const int mode=precision=="fp64"?0:precision=="fp32"?1:precision=="ds"?2:precision=="ts"?3:-1;
        if(mode<0)FatalErrorInFunction << "Unknown innerPrecision " << precision << abort(FatalError);
        const label sweeps=controlDict_.getOrDefault<label>("innerSweeps",6);
        const label maxOuter=controlDict_.getOrDefault<label>("maxRefinements",8);
        if(sweeps<1||maxOuter<1)FatalErrorInFunction << "Refinement counts must be positive" << abort(FatalError);

        const scalarField rhs(matrix_.residual(psi,source,interfaceBouCoeffs_,interfaces_,cmpt));
        scalarField delta(psi.size(),Zero),defect(rhs);
        const scalar rhsNorm=gSumMag(rhs,matrix_.mesh().comm());
        if(!std::isfinite(rhsNorm))FatalErrorInFunction << "Nonfinite original RHS" << abort(FatalError);
        // For delta=0, the stock DEFAULT/L1 normalization reduces exactly to
        // sum(abs(rhs))+small. NO_NORM keeps the unscaled residual norm.
        const scalar norm=normType_==lduMatrix::normTypes::NO_NORM ? 1:rhsNorm+solverPerformance::small_;
        solverPerformance perf(typeName,fieldName_,rhsNorm/norm,rhsNorm/norm);
        bool fallback=matrix_.diagonal();label outer=0,totalInner=0;int invalid=0;void* workspace=nullptr;
        if(!fallback)
        {
            const auto& addr=matrix_.lduAddr();
            const PintleIrMatrix data
            {
                label(psi.size()),label(addr.lowerAddr().size()),
                addr.ownerStartAddr().cbegin(),addr.losortStartAddr().cbegin(),addr.losortAddr().cbegin(),
                addr.lowerAddr().cbegin(),addr.upperAddr().cbegin(),
                matrix_.diag().cbegin(),matrix_.lower().cbegin(),matrix_.upper().cbegin()
            };
            const int status=pintleIrPrepare(mode,&data,&workspace,&invalid);
            if(status)FatalErrorInFunction << "CUDA prepare: " << cudaGetErrorString(cudaError_t(status)) << abort(FatalError);
            fallback=invalid!=0;
        }
        while(!fallback && outer<maxOuter && perf.nIterations()<maxIter_
            && (perf.nIterations()<minIter_ || !perf.checkConvergence(tolerance_,relTol_)))
        {
            const label count=min(sweeps,maxIter_-perf.nIterations());
            const int status=pintleIrSweep(workspace,defect.cbegin(),delta.begin(),count);
            if(status)FatalErrorInFunction << "CUDA inner sweep: " << cudaGetErrorString(cudaError_t(status)) << abort(FatalError);
            // Ordered on the same default stream after the FP64 delta update.
            defect=matrix_.residual(delta,rhs,interfaceBouCoeffs_,interfaces_,cmpt);
            perf.finalResidual()=gSumMag(defect,matrix_.mesh().comm())/norm;
            ++outer;totalInner+=count;perf.nIterations()+=count;
            if(!std::isfinite(perf.finalResidual()))fallback=true;
        }
        fallback=fallback || perf.nIterations()<minIter_ || !perf.checkConvergence(tolerance_,relTol_);
        if(fallback)
        {
            delta=Zero;
            smoothSolver exact(fieldName_,matrix_,interfaceBouCoeffs_,interfaceIntCoeffs_,interfaces_,controlDict_);
            perf=exact.solve(delta,rhs,cmpt);
            // Fixed-count smoothSolver (nSweeps<0) leaves its residual fields
            // at zero. Independently verify the actual fallback correction.
            defect=matrix_.residual(delta,rhs,interfaceBouCoeffs_,interfaces_,cmpt);
            perf.initialResidual()=rhsNorm/norm;
            perf.finalResidual()=gSumMag(defect,matrix_.mesh().comm())/norm;
            if(!std::isfinite(perf.finalResidual()) || !perf.checkConvergence(tolerance_,relTol_))
                FatalErrorInFunction << "FP64 fallback failed: " << perf << abort(FatalError);
        }
        psi+=delta;perf.solverName()=typeName;
        Info<< "PINTLE_IR precision=" << precision << " outer=" << outer
            << " innerSweeps=" << totalInner << " finalResidual=" << perf.finalResidual()
            << " fallback=" << fallback << " invalidPacked=" << invalid << endl;
        if(controlDict_.getOrDefault<bool>("diagnostics",false))
        {
            const scalarField residual(matrix_.residual(psi,source,interfaceBouCoeffs_,interfaces_,cmpt));
            scalarField Apsi(psi.size());matrix_.Amul(Apsi,psi,interfaceBouCoeffs_,interfaces_,cmpt);
            Info<< "PINTLE_T_AUDIT backwardError="
                << gSumMag(residual)/(gSum(mag(source)+mag(Apsi))+VSMALL)
                << " maxResidualOverDiagK=" << gMax(mag(residual)/mag(matrix_.diag()))
                << " maxIncrementK=" << gMax(mag(delta)) << endl;
        }
        return perf;
    }
};
defineTypeNameAndDebug(pintleMixedTemperature,0);
lduMatrix::solver::addsymMatrixConstructorToTable<pintleMixedTemperature> pintleIrSym;
lduMatrix::solver::addasymMatrixConstructorToTable<pintleMixedTemperature> pintleIrAsym;
}
