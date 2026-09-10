/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2015 OpenFOAM Foundation
    Copyright (C) 2017-2019 OpenCFD Ltd.
    Copyright (C) 2026 Cineca
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "pintleAsyncGaussSeidelSmoother.H"
#include "PrecisionAdaptor.H"

// MAIN changes: queue colors on the same default stream. Partition metadata
// and all fields remain live until the explicit end-of-sweep fence.
namespace Foam { namespace pintleAsync {
#ifdef have_cuda
template<class F> __global__ void colorLoop(F f, label n)
{ for(label i=blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=blockDim.x*gridDim.x) f(i); }
#endif
template<class F> void launch(F f, label n)
{
    if(!n) return;
#ifdef have_cuda
    const label block=cudaDeviceInit::getNumberOfThreadsPerBlock();
    colorLoop<<<device::setNumBlocks(n,block),block>>>(f,n);
    const auto err=cudaGetLastError();
    if(err!=cudaSuccess) FatalErrorInFunction << cudaGetErrorString(err) << abort(FatalError);
#else
    for(label i=0;i<n;++i) f(i);
#endif
}
void finish()
{
#ifdef have_cuda
    const auto err=cudaDeviceSynchronize();
    if(err!=cudaSuccess) FatalErrorInFunction << cudaGetErrorString(err) << abort(FatalError);
#endif
}
} }

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(pintleAsyncGaussSeidelSmoother, 0);

    lduMatrix::smoother::addsymMatrixConstructorToTable<pintleAsyncGaussSeidelSmoother>
        addpintleAsyncGaussSeidelSmootherSymMatrixConstructorToTable_;

    lduMatrix::smoother::addasymMatrixConstructorToTable<pintleAsyncGaussSeidelSmoother>
        addpintleAsyncGaussSeidelSmootherAsymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::pintleAsyncGaussSeidelSmoother::pintleAsyncGaussSeidelSmoother
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::smoother
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    ),
    rD_(matrix.diag().size())
{
    readControls();

    const label nCells = matrix.diag().size();

    const scalar* const __restrict__ diagPtr =
        matrix.diag().cbegin();
    solveScalar* __restrict__ rDPtr = rD_.begin();

    foamExecutor exec;

    // -- Calculate the inverse of the diagonal matrix (D^-1)
    auto LambdarD = [=](label celli)
    {
        rDPtr[celli] = 1. / diagPtr[celli];
    };
    exec.parallelFor(LambdarD, nCells);
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::pintleAsyncGaussSeidelSmoother::readControls()
{
}


void Foam::pintleAsyncGaussSeidelSmoother::smooth_
(
    const word& fieldName,
    solveScalarField& psi,
    const lduMatrix& matrix,
    const solveScalarField& source,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const direction cmpt,
    const label nSweeps
) const
{
    solveScalar* __restrict__ psiPtr = psi.begin();
    const solveScalar* const __restrict__ bPtr = source.cbegin();

    const label nCells = psi.size();
    const label nInternalFaces = matrix.upper().size();
 
    solveScalarField& bPrime = matrix.work(nCells);
    bool haveInterface=false;
    forAll(interfaces,patchi) haveInterface=haveInterface || interfaces.set(patchi);
    // In a serial matrix without interfaces source is invariant through all
    // sweeps. Pooled storage is device-accessible. Preserve the copy path for
    // host-only source arrays, aliasing, MPI, and active interfaces.
    const bool directSource=!Pstream::parRun() && !haveInterface && source.usePool()
        && source.cbegin()!=psi.cbegin()
        && controlDict_.getOrDefault<bool>("pintleDirectSource",true);
    const solveScalar* __restrict__ bPrimePtr=directSource ? bPtr:bPrime.cbegin();

    const solveScalar* const __restrict__ rDPtr = rD_.cbegin();

    const scalar* const __restrict__ diagPtr = 
        matrix_.diag().cbegin();
    const scalar* const __restrict__ upperPtr =
        matrix_.upper().begin();
    const scalar* const __restrict__ lowerPtr =
        matrix_.lower().begin();
    const scalar* const __restrict__ lowercsrPtr = 
        matrix_.lowerCSR().begin();

    const lduAddressing& addr = matrix.lduAddr();

    const label* const __restrict__ uPtr =
        addr.upperAddr().begin();
    const label* const __restrict__ lPtr =
        addr.lowerAddr().begin();

    const label* const __restrict__ oStartPtr =
        addr.ownerStartAddr().begin();
    const label* const __restrict__ loStartPtr =
        addr.losortStartAddr().begin();
    const label* const __restrict__ lcsrPtr =
        addr.lowerCSRAddr().begin();

    dictionary smootherDict;
    if (controlDict_.found("multicolorGaussSeidel"))
    {
        smootherDict = controlDict_.subDict("multicolorGaussSeidel");
    }

    const List<DynamicList<label>>& partitions = addr.partitions(smootherDict);

    foamExecutor exec;

    // Parallel boundary initialisation.  The parallel boundary is treated
    // as an effective jacobi interface in the boundary.
    // Note: there is a change of sign in the coupled
    // interface update.  The reason for this is that the
    // internal coefficients are all located at the l.h.s. of
    // the matrix whereas the "implicit" coefficients on the
    // coupled boundaries are all created as if the
    // coefficient contribution is of a source-kind (i.e. they
    // have a sign as if they are on the r.h.s. of the matrix.
    // To compensate for this, it is necessary to turn the
    // sign of the contribution.

    for (label sweep=0; sweep<nSweeps; sweep++)
    {
        if(!directSource)
        {
        bPrime = source;

        const label startRequest = UPstream::nRequests();

        matrix.initMatrixInterfaces
        (
            false,
            interfaceBouCoeffs,
            interfaces,
            psi,
            bPrime,
            cmpt
        );

        matrix.updateMatrixInterfaces
        (
            false,
            interfaceBouCoeffs,
            interfaces,
            psi,
            bPrime,
            cmpt,
            startRequest
        );
        }

        for (const DynamicList<label>& partition : partitions)
        {
            const label* const __restrict__ partitionPtr = partition.cbegin();

            auto LambdaColor = [=](label i)
            {
                const label celli = partitionPtr[i];

                solveScalar psii;

                // Get the accumulated neighbour side
                psii = bPrimePtr[celli];

                // Add lower contributions
                {
                    const label start = loStartPtr[celli];
                    const label end = loStartPtr[celli + 1];

                    for (label i=start; i<end; ++i)
                    {
                        const label nbrCell = lcsrPtr[i];
                        psii -= lowercsrPtr[i] * psiPtr[nbrCell];
                    }
                }

                // Add upper contributions
                {
                    const label start = oStartPtr[celli]; 
                    const label end = oStartPtr[celli + 1];

                    for (label i=start; i<end; ++i)
                    {
                        const label nbrCell = uPtr[i];
                        psii -= upperPtr[i] * psiPtr[nbrCell];
                    }
                }

                // Finish psi for this cell
                psii *= rDPtr[celli];

                psiPtr[celli] = psii;
            };
            pintleAsync::launch(LambdaColor, partition.size());
        }
        // Interface work and bPrime assignment in the next sweep may read
        // managed storage on the host; complete this sweep before returning.
        if(!directSource) pintleAsync::finish();
    }
    if(directSource) pintleAsync::finish();
}


void Foam::pintleAsyncGaussSeidelSmoother::smooth
(
    solveScalarField& psi,
    const scalarField& source,
    const direction cmpt,
    const label nSweeps
) const
{
    smooth_
    (
        fieldName_,
        psi,
        matrix_,
        ConstPrecisionAdaptor<solveScalar, scalar>(source),
        interfaceBouCoeffs_,
        interfaces_,
        cmpt,
        nSweeps
    );
}


void Foam::pintleAsyncGaussSeidelSmoother::scalarSmooth
(
    solveScalarField& psi,
    const solveScalarField& source,
    const direction cmpt,
    const label nSweeps
) const
{
    smooth_
    (
        fieldName_,
        psi,
        matrix_,
        source,
        interfaceBouCoeffs_,
        interfaces_,
        cmpt,
        nSweeps
    );
}


// ************************************************************************* //
