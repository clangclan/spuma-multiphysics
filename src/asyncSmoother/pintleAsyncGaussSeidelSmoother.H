/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011 OpenFOAM Foundation
    Copyright (C) 2019 OpenCFD Ltd.
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

Class
    Foam::pintleAsyncGaussSeidelSmoother

Group
    grpLduMatrixSmoothers

Description
    A race-condition-free implementation (based on color graph partitioning)
    of Gauss-Seidel designed for multithreading.

Usage
    Example usage:
    \verbatim
    smoother pintleAsyncGaussSeidel;
    multicolorGaussSeidel
    {
        graphColoring      randomized;
        deterministic      true;
        shrinkingFactor    1.0;
    }
    \endverbatim

SourceFiles
    pintleAsyncGaussSeidelSmoother.H
    pintleAsyncGaussSeidelSmoother.C

\*---------------------------------------------------------------------------*/

#ifndef pintleAsyncGaussSeidelSmoother_H
#define pintleAsyncGaussSeidelSmoother_H

#include "lduMatrix.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                           Class multicolorGaussSeidel Smoother Declaration
\*---------------------------------------------------------------------------*/

class pintleAsyncGaussSeidelSmoother
:
    public lduMatrix::smoother
{
private:

    // Private data

        //- The inverse (reciprocal for scalars) diagonal
        Field<solveScalar> rD_;


    // Member functions

        //- Read the control parameters from the controlDict_
        void readControls();


public:

    //- Runtime type information
    TypeName("pintleAsyncGaussSeidel");


    // Constructors

        //- Construct from components
        pintleAsyncGaussSeidelSmoother
        (
            const word& fieldName,
            const lduMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );


    // Member Functions

        //- Smooth for the given number of sweeps
        void smooth_
        (
            const word& fieldName,
            solveScalarField& psi,
            const lduMatrix& matrix,
            const solveScalarField& source,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const direction cmpt,
            const label nSweeps
        ) const;


        //- Smooth the solution for a given number of sweeps
        virtual void smooth
        (
            solveScalarField& psi,
            const scalarField& Source,
            const direction cmpt,
            const label nSweeps
        ) const;

        //- Smooth the solution for a given number of sweeps
        virtual void scalarSmooth
        (
            solveScalarField& psi,
            const solveScalarField& Source,
            const direction cmpt,
            const label nSweeps
        ) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
