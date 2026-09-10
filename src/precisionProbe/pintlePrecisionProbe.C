// SPDX-License-Identifier: GPL-3.0-or-later
// Main-agent diagnostic: delegate the unchanged solve, then export its first
// scalar system for standalone precision experiments. Never used in timings.
#include "lduMatrix.H"
#include "Pstream.H"
#include "fileName.H"
#include <cstdint>
#include <fstream>
#include <vector>
#include <limits>

namespace Foam
{
class pintlePrecisionProbe : public lduMatrix::solver
{
public:
    TypeName("pintlePrecisionProbe");
    pintlePrecisionProbe(const word& name,const lduMatrix& matrix,
        const FieldField<Field,scalar>& bou,const FieldField<Field,scalar>& internal,
        const lduInterfaceFieldPtrsList& interfaces,const dictionary& controls)
    :lduMatrix::solver(name,matrix,bou,internal,interfaces,controls){}

    solverPerformance solve(scalarField& psi,const scalarField& source,
                            const direction cmpt=0) const override
    {
        static bool written=false;
        if(Pstream::parRun())
            FatalErrorInFunction << "Snapshot requires one rank" << abort(FatalError);
        forAll(interfaces_,i)
            if(interfaces_.set(i))
                FatalErrorInFunction << "Snapshot excludes coupled interfaces" << abort(FatalError);
        dictionary delegate(controlDict_);
        const word name(controlDict_.getOrDefault<word>("delegate","GAMG"));
        if(name==typeName)
            FatalErrorInFunction << "Recursive probe delegate" << abort(FatalError);
        delegate.set("solver",name);
        auto solver=lduMatrix::solver::New(fieldName_,matrix_,interfaceBouCoeffs_,
                                          interfaceIntCoeffs_,interfaces_,delegate);
        const solverPerformance result=solver->solve(psi,source,cmpt);
        if(written) return result;

        const auto& addr=matrix_.lduAddr();
        const auto& own=addr.lowerAddr(); const auto& nei=addr.upperAddr();
        const auto& os=addr.ownerStartAddr(); const auto& ns=addr.losortStartAddr();
        const auto& nf=addr.losortAddr();
        const auto& diag=matrix_.diag(); const auto& low=matrix_.lower();
        const auto& up=matrix_.upper();
        const uint64_t n=diag.size(), nz=n+2*uint64_t(own.size());
        if(nz>uint64_t(std::numeric_limits<int32_t>::max()) || sizeof(scalar)!=8)
            FatalErrorInFunction << "Snapshot requires FP64 and 32-bit CSR offsets" << abort(FatalError);
        scalarField amul(n);
        matrix_.Amul(amul,psi,interfaceBouCoeffs_,interfaces_,cmpt);
        const scalarField residual(matrix_.residual(psi,source,interfaceBouCoeffs_,interfaces_,cmpt));
        std::vector<int32_t> rows(n+1),cols(nz);
        std::vector<double> values(nz);
        int32_t k=0;
        for(label c=0;c<label(n);++c)
        {
            rows[c]=k; cols[k]=c; values[k++]=diag[c];
            for(label j=ns[c];j<ns[c+1];++j)
            {const label f=nf[j];cols[k]=own[f];values[k++]=low[f];}
            for(label f=os[c];f<os[c+1];++f)
            {cols[k]=nei[f];values[k++]=up[f];}
        }
        rows[n]=k;
        if(uint64_t(k)!=nz)
            FatalErrorInFunction << "CSR size mismatch" << abort(FatalError);
        const fileName path(controlDict_.get<fileName>("snapshotFile"));
        std::ifstream existing(path.c_str(),std::ios::binary);
        if(existing.good())
            FatalErrorInFunction << "Refusing to overwrite " << path << abort(FatalError);
        std::ofstream out(path.c_str(),std::ios::binary);
        auto write=[&](const void* p,size_t count)
        {out.write(static_cast<const char*>(p),count);};
        const char magic[8]={'P','I','N','T','A','0','0','1'};
        write(magic,8);write(&n,8);write(&nz,8);
        write(rows.data(),rows.size()*sizeof(int32_t));
        write(cols.data(),cols.size()*sizeof(int32_t));
        write(values.data(),values.size()*sizeof(double));
        write(psi.cbegin(),n*8);write(source.cbegin(),n*8);
        write(amul.cbegin(),n*8);write(residual.cbegin(),n*8);
        out.close();
        if(!out)
            FatalErrorInFunction << "Could not write " << path << abort(FatalError);
        written=true;
        Info<< "PINTLE_PRECISION_SNAPSHOT file=" << path << " rows=" << n
            << " nonzeros=" << nz << " field=" << fieldName_ << endl;
        return result;
    }
};
defineTypeNameAndDebug(pintlePrecisionProbe,0);
lduMatrix::solver::addsymMatrixConstructorToTable<pintlePrecisionProbe> pintleProbeSym;
lduMatrix::solver::addasymMatrixConstructorToTable<pintlePrecisionProbe> pintleProbeAsym;
}
