/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2014-2017 OpenFOAM Foundation
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
    Foam::pintlePengRobinsonGas

Group
    grpSpecieEquationOfState

Description
    pintlePengRobinsonGas gas equation of state.

SourceFiles
    pintlePengRobinsonGasI.H
    pintlePengRobinsonGas.C

\*---------------------------------------------------------------------------*/

#ifndef pintlePengRobinsonGas_H
#define pintlePengRobinsonGas_H
#include "specie.H"

#include "autoPtr.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward declaration of friend functions and operators

template<class Specie> class pintlePengRobinsonGas;

template<class Specie>
inline pintlePengRobinsonGas<Specie> operator+
(
    const pintlePengRobinsonGas<Specie>&,
    const pintlePengRobinsonGas<Specie>&
);

template<class Specie>
inline pintlePengRobinsonGas<Specie> operator*
(
    const scalar,
    const pintlePengRobinsonGas<Specie>&
);

template<class Specie>
inline pintlePengRobinsonGas<Specie> operator==
(
    const pintlePengRobinsonGas<Specie>&,
    const pintlePengRobinsonGas<Specie>&
);

template<class Specie>
Ostream& operator<<
(
    Ostream&,
    const pintlePengRobinsonGas<Specie>&
);



/*---------------------------------------------------------------------------*\
                           Class pintlePengRobinsonGas Declaration
\*---------------------------------------------------------------------------*/

template<class Specie>
class pintlePengRobinsonGas
:
    public Specie
{
    scalar pintleRR_ = RR;
    scalar pintlePstd_ = Pstd;

    // Private data

        //- Critical Temperature [K]
        scalar Tc_;

        //- Critical volume [m^3/kmol]
        scalar Vc_;

        //- Critical compression factor [-]
        scalar Zc_;

        //- Critical Pressure [Pa]
        scalar Pc_;

        //- Acentric factor [-]
        scalar omega_;


public:

    // Constructors

        //- Construct from components
        inline pintlePengRobinsonGas
        (
            const Specie& sp,
            const scalar& Tc,
            const scalar& Vc,
            const scalar& Zc,
            const scalar& Pc,
            const scalar& omega
        );

        //- Construct from dictionary
        pintlePengRobinsonGas(const dictionary& dict);

        //- Construct as named copy
        inline pintlePengRobinsonGas(const word& name, const pintlePengRobinsonGas&);

        //- Construct and return a clone
        inline autoPtr<pintlePengRobinsonGas> clone() const;

        // Selector from dictionary
        inline static autoPtr<pintlePengRobinsonGas> New
        (
            const dictionary& dict
        );


    // Member functions

        //- Return the instantiated type name
        static word typeName()
        {
            return "pintlePengRobinsonGas<" + word(Specie::typeName_()) + '>';
        }

        // Fundamental properties


            //- Is the equation of state is incompressible i.e. rho != f(p)
            static const bool incompressible = false;

            //- Is the equation of state is isochoric i.e. rho = const
            static const bool isochoric = false;

            //- Return density [kg/m^3]
            FOAM_DEVICE inline scalar rho(scalar p, scalar T) const;

            //- Return enthalpy departure [J/kg]
            FOAM_DEVICE inline scalar H(const scalar p, const scalar T) const;

            //- Return Cp departure [J/(kg K]
            FOAM_DEVICE inline scalar Cp(scalar p, scalar T) const;

            //- Return internal energy departure [J/kg]
            FOAM_DEVICE inline scalar E(const scalar p, const scalar T) const;

            //- Return Cv departure [J/(kg K]
            FOAM_DEVICE inline scalar Cv(scalar p, scalar T) const;

            //- Return entropy [J/(kg K)]
            FOAM_DEVICE inline scalar S(const scalar p, const scalar T) const;

            //- Return compressibility rho/p [s^2/m^2]
            FOAM_DEVICE inline scalar psi(scalar p, scalar T) const;

            //- Return compression factor []
            FOAM_DEVICE inline scalar Z(scalar p, scalar T) const;

            //- Return (Cp - Cv) [J/(kg K]
            FOAM_DEVICE inline scalar CpMCv(scalar p, scalar T) const;


    // IO

            //- Write to Ostream
            void write(Ostream& os) const;

    // Member operators

        inline void operator+=(const pintlePengRobinsonGas&);
        inline void operator*=(const scalar);


    // Friend operators

        friend pintlePengRobinsonGas operator+ <Specie>
        (
            const pintlePengRobinsonGas&,
            const pintlePengRobinsonGas&
        );

        friend pintlePengRobinsonGas operator* <Specie>
        (
            const scalar s,
            const pintlePengRobinsonGas&
        );

        friend pintlePengRobinsonGas operator== <Specie>
        (
            const pintlePengRobinsonGas&,
            const pintlePengRobinsonGas&
        );


    // Ostream Operator

        friend Ostream& operator<< <Specie>
        (
            Ostream&,
            const pintlePengRobinsonGas&
        );
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "pintlePengRobinsonGasI.H"

#ifdef NoRepository
    #include "pintlePengRobinsonGas.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
