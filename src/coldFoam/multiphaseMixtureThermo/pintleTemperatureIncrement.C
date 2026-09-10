// SPDX-License-Identifier: GPL-3.0-or-later
// Same linear system as the absolute-T equation, solved for its increment.
#include "smoothSolver.H"
namespace Foam
{
class pintleTemperatureIncrement : public lduMatrix::solver
{
public:
    TypeName("pintleTemperatureIncrement");
    pintleTemperatureIncrement(const word& name,const lduMatrix& matrix,
        const FieldField<Field,scalar>& bou,const FieldField<Field,scalar>& internal,
        const lduInterfaceFieldPtrsList& interfaces,const dictionary& controls)
    :lduMatrix::solver(name,matrix,bou,internal,interfaces,controls){}
    solverPerformance solve(scalarField& psi,const scalarField& source,
                            const direction cmpt=0) const override
    {
        const scalarField rhs(matrix_.residual(psi,source,interfaceBouCoeffs_,interfaces_,cmpt));
        scalarField delta(psi.size(),Zero);
        smoothSolver inner(fieldName_,matrix_,interfaceBouCoeffs_,interfaceIntCoeffs_,interfaces_,controlDict_);
        solverPerformance perf=inner.solve(delta,rhs,cmpt);
        if(!perf.checkConvergence(tolerance_,relTol_))
            FatalErrorInFunction << "Temperature increment did not converge: " << perf << abort(FatalError);
        psi+=delta;
        perf.solverName()="pintleTemperatureIncrement";
        if(controlDict_.getOrDefault<bool>("diagnostics",false))
        {
            const scalarField residual(matrix_.residual(psi,source,interfaceBouCoeffs_,interfaces_,cmpt));
            scalarField Apsi(psi.size());
            matrix_.Amul(Apsi,psi,interfaceBouCoeffs_,interfaces_,cmpt);
            Info<< "PINTLE_T_AUDIT backwardError="
                << gSumMag(residual)/(gSum(mag(source)+mag(Apsi))+VSMALL)
                << " maxResidualOverDiagK=" << gMax(mag(residual)/mag(matrix_.diag()))
                << " maxIncrementK=" << gMax(mag(delta)) << endl;
        }
        return perf;
    }
};
defineTypeNameAndDebug(pintleTemperatureIncrement,0);
lduMatrix::solver::addsymMatrixConstructorToTable<pintleTemperatureIncrement> pintleTSym;
lduMatrix::solver::addasymMatrixConstructorToTable<pintleTemperatureIncrement> pintleTAsym;
}
