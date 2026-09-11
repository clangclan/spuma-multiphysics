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

\*---------------------------------------------------------------------------*/

#include "pintlePengRobinsonGas.H"
#include "mathematicalConstants.H"

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

template<class Specie>
inline Foam::pintlePengRobinsonGas<Specie>::pintlePengRobinsonGas
(
    const Specie& sp,
    const scalar& Tc,
    const scalar& Vc,
    const scalar& Zc,
    const scalar& Pc,
    const scalar& omega
)
:
    Specie(sp),
    Tc_(Tc),
    Vc_(Vc),
    Zc_(Zc),
    Pc_(Pc),
    omega_(omega)
{}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class Specie>
inline Foam::pintlePengRobinsonGas<Specie>::pintlePengRobinsonGas
(
    const word& name,
    const pintlePengRobinsonGas& pg
)
:
    Specie(name, pg),
    Tc_(pg.Tc_),
    Vc_(pg.Vc_),
    Zc_(pg.Zc_),
    Pc_(pg.Pc_),
    omega_(pg.omega_)
{}


template<class Specie>
inline Foam::autoPtr<Foam::pintlePengRobinsonGas <Specie>>
Foam::pintlePengRobinsonGas<Specie>::clone() const
{
    return autoPtr<pintlePengRobinsonGas<Specie>>::New(*this);
}


template<class Specie>
inline Foam::autoPtr<Foam::pintlePengRobinsonGas<Specie>>
Foam::pintlePengRobinsonGas<Specie>::New
(
    const dictionary& dict
)
{
    return autoPtr<pintlePengRobinsonGas<Specie>>::New(dict);
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class Specie>
FOAM_DEVICE inline Foam::scalar Foam::pintlePengRobinsonGas<Specie>::rho
(
    scalar p,
    scalar T
) const
{
    const scalar Z = this->Z(p, T);
    return p/(Z*this->R()*T);
}


template<class Specie>
FOAM_DEVICE inline Foam::scalar Foam::pintlePengRobinsonGas<Specie>::H(scalar p, scalar T) const
{
    const scalar Tr = T/Tc_;
    const scalar a = 0.45724*sqr(pintleRR_*Tc_)/Pc_;
    const scalar b = 0.07780*pintleRR_*Tc_/Pc_;
    const scalar B = b*p/(pintleRR_*T);
    const scalar kappa = 0.37464 + 1.54226*omega_ - 0.26992*sqr(omega_);
    const scalar q = 1 + kappa*(1 - sqrt(Tr));
    const scalar Z = this->Z(p, T);
    const scalar root2 = sqrt(2.0);
    const scalar L = log((Z + (1 + root2)*B)/(Z + (1 - root2)*B));
    // T*a'(T)-a(T) = -a0*(1+kappa)*q. Keep signed q; sqrt(q*q)
    // changes the derivative above its zero. H is the enthalpy departure.
    return this->R()*T*(Z - 1) - a*(1 + kappa)*q*L/(2*root2*b*this->W());
}


template<class Specie>
FOAM_DEVICE inline Foam::scalar Foam::pintlePengRobinsonGas<Specie>::Cp(scalar p, scalar T) const
{
    const scalar Tr = T/Tc_;
    const scalar a = 0.45724*sqr(pintleRR_*Tc_)/Pc_;
    const scalar b = 0.07780*pintleRR_*Tc_/Pc_;
    const scalar kappa = 0.37464 + 1.54226*omega_ - 0.26992*sqr(omega_);
    const scalar alpha = sqr(1 + kappa*(1 - sqrt(Tr)));

    const scalar A = a*alpha*p/sqr(pintleRR_*T);
    const scalar B = b*p/(pintleRR_*T);

    const scalar Z = this->Z(p, T);

    const scalar ap = kappa*a*(kappa/Tc_ - (1 + kappa)/sqrt(T*Tc_));
    const scalar app = kappa*a*(1 + kappa)/(2*sqrt(pow3(T)*Tc_));

    const scalar M = (sqr(Z) + 2*B*Z - sqr(B))/(Z - B);
    const scalar N = ap*B/(b*pintleRR_);

    const scalar root2 = sqrt(2.0);

    return
    (
        app*(T/(2*root2*b))*log((Z + (root2 + 1)*B)/(Z - (root2 - 1)*B))
      + pintleRR_*sqr(M - N)/(sqr(M) - 2*A*(Z + B))
      - pintleRR_
    )/this->W();
}


template<class Specie>
FOAM_DEVICE inline Foam::scalar Foam::pintlePengRobinsonGas<Specie>::E(scalar p, scalar T) const
{
    const scalar Tr = T/Tc_;
    const scalar a = 0.45724*sqr(pintleRR_*Tc_)/Pc_;
    const scalar b = 0.07780*pintleRR_*Tc_/Pc_;
    const scalar B = b*p/(pintleRR_*T);
    const scalar kappa = 0.37464 + 1.54226*omega_ - 0.26992*sqr(omega_);
    const scalar q = 1 + kappa*(1 - sqrt(Tr));
    const scalar Z = this->Z(p, T);
    const scalar root2 = sqrt(2.0);
    const scalar L = log((Z + (1 + root2)*B)/(Z + (1 - root2)*B));
    // Internal-energy departure only: an ideal-gas -R*T is not added here.
    return -a*(1 + kappa)*q*L/(2*root2*b*this->W());
}


template<class Specie>
FOAM_DEVICE inline Foam::scalar Foam::pintlePengRobinsonGas<Specie>::Cv(scalar p, scalar T) const
{
    const scalar a = 0.45724*sqr(pintleRR_*Tc_)/Pc_;
    const scalar b = 0.07780*pintleRR_*Tc_/Pc_;
    const scalar kappa = 0.37464 + 1.54226*omega_ - 0.26992*sqr(omega_);

    const scalar B = b*p/(pintleRR_*T);

    const scalar Z = this->Z(p, T);

    const scalar app = kappa*a*(1 + kappa)/(2*sqrt(pow3(T)*Tc_));

    const scalar root2 = sqrt(2.0);

    return
    (
        app*(T/(2*root2*b))*log((Z + (root2 + 1)*B)/(Z - (root2 - 1)*B))
    )/this->W();
}


template<class Specie>
FOAM_DEVICE inline Foam::scalar Foam::pintlePengRobinsonGas<Specie>::S
(
    scalar p,
    scalar T
) const
{
    const scalar Tr = T/Tc_;
    const scalar a = 0.45724*sqr(pintleRR_*Tc_)/Pc_;
    const scalar b = 0.07780*pintleRR_*Tc_/Pc_;
    const scalar B = b*p/(pintleRR_*T);
    const scalar kappa = 0.37464 + 1.54226*omega_ - 0.26992*sqr(omega_);
    const scalar q = 1 + kappa*(1 - sqrt(Tr));
    const scalar ap = -a*kappa*q/sqrt(T*Tc_);
    const scalar Z = this->Z(p, T);
    const scalar root2 = sqrt(2.0);
    const scalar L = log((Z + (1 + root2)*B)/(Z + (1 - root2)*B));
    // EOS entropy contains its ideal-gas pressure contribution as required by
    // hConstThermo; the temperature contribution is supplied by that wrapper.
    return this->R()*(-log(p/pintlePstd_) + log(Z - B)) + ap*L/(2*root2*b*this->W());
}


template<class Specie>
FOAM_DEVICE inline Foam::scalar Foam::pintlePengRobinsonGas<Specie>::psi
(
    scalar p,
    scalar T
) const
{
    const scalar Z = this->Z(p, T);

    return 1.0/(Z*this->R()*T);
}


template<class Specie>
FOAM_DEVICE inline Foam::scalar Foam::pintlePengRobinsonGas<Specie>::Z
(
    scalar p,
    scalar T
) const
{
    const scalar Tr = T/Tc_;
    const scalar a = 0.45724*sqr(pintleRR_*Tc_)/Pc_;
    const scalar b = 0.07780*pintleRR_*Tc_/Pc_;
    const scalar kappa = 0.37464 + 1.54226*omega_ - 0.26992*sqr(omega_);
    const scalar alpha = sqr(1 + kappa*(1 - sqrt(Tr)));

    const scalar A = a*alpha*p/sqr(pintleRR_*T);
    const scalar B = b*p/(pintleRR_*T);

    const scalar a2 = B - 1;
    const scalar a1 = A - 2*B - 3*sqr(B);
    const scalar a0 = -A*B + sqr(B) + pow3(B);

    const scalar Q = (3*a1 - a2*a2)/9.0;
    const scalar Rl = (9*a2*a1 - 27*a0 - 2*a2*a2*a2)/54.0;

    const scalar Q3 = Q*Q*Q;
    const scalar D = Q3 + Rl*Rl;

    scalar root = -1;

    if (D <= 0)
    {
        const scalar th = ::acos(Rl/sqrt(-Q3));
        const scalar qm = 2*sqrt(-Q);
        const scalar r1 = qm*cos(th/3.0) - a2/3.0;
        const scalar r2 =
            qm*cos((th + 2*constant::mathematical::pi)/3.0) - a2/3.0;
        const scalar r3 =
            qm*cos((th + 4*constant::mathematical::pi)/3.0) - a2/3.0;

        root = max(r1, max(r2, r3));
    }
    else
    {
        // One root is real
        const scalar D05 = sqrt(D);
        const scalar S = cbrt(Rl + D05);
        scalar Tl = 0;
        if (D05 > Rl)
        {
            Tl = -cbrt(mag(Rl - D05));
        }
        else
        {
            Tl = cbrt(Rl - D05);
        }

        root = S + Tl - a2/3.0;
    }

    return root;
}


template<class Specie>
FOAM_DEVICE inline Foam::scalar Foam::pintlePengRobinsonGas<Specie>::CpMCv
(
    scalar p,
    scalar T
) const
{
    const scalar Tr = T/Tc_;
    const scalar a = 0.45724*sqr(pintleRR_*Tc_)/Pc_;
    const scalar b = 0.07780*pintleRR_*Tc_/Pc_;
    const scalar kappa = 0.37464 + 1.54226*omega_ - 0.26992*sqr(omega_);
    const scalar alpha = sqr(1 + kappa*(1 - sqrt(Tr)));

    const scalar A = alpha*a*p/sqr(pintleRR_*T);
    const scalar B = b*p/(pintleRR_*T);

    const scalar Z = this->Z(p, T);

    const scalar ap = kappa*a*(kappa/Tc_ - (1 + kappa)/sqrt(T*Tc_));
    const scalar M = (sqr(Z) + 2*B*Z - sqr(B))/(Z - B);
    const scalar N = ap*B/(b*pintleRR_);

    return this->R()*sqr(M - N)/(sqr(M) - 2*A*(Z + B));
}


// * * * * * * * * * * * * * * * Member Operators  * * * * * * * * * * * * * //

template<class Specie>
inline void Foam::pintlePengRobinsonGas<Specie>::operator+=
(
    const pintlePengRobinsonGas<Specie>& pg
)
{
    scalar Y1 = this->Y();
    Specie::operator+=(pg);

    if (mag(this->Y()) > SMALL)
    {
        Y1 /= this->Y();
        const scalar Y2 = pg.Y()/this->Y();

        Tc_ = Y1*Tc_ + Y2*pg.Tc_;
        Vc_ = Y1*Vc_ + Y2*pg.Vc_;
        Zc_ = Y1*Zc_ + Y2*pg.Zc_;
        Pc_ = pintleRR_*Zc_*Tc_/Vc_;
        omega_ = Y1*omega_ + Y2*pg.omega_;
    }
}


template<class Specie>
inline void Foam::pintlePengRobinsonGas<Specie>::operator*=(const scalar s)
{
     Specie::operator*=(s);
}


// * * * * * * * * * * * * * * * Friend Operators  * * * * * * * * * * * * * //


template<class Specie>
Foam::pintlePengRobinsonGas<Specie> Foam::operator+
(
    const pintlePengRobinsonGas<Specie>& pg1,
    const pintlePengRobinsonGas<Specie>& pg2
)
{
    Specie sp
    (
        static_cast<const Specie&>(pg1)
      + static_cast<const Specie&>(pg2)
    );

    if (mag(sp.Y()) < SMALL)
    {
        return pintlePengRobinsonGas<Specie>
        (
            sp,
            pg1.Tc_,
            pg1.Vc_,
            pg1.Zc_,
            pg1.Pc_,
            pg1.omega_
        );
    }
    else
    {
        const scalar Y1 = pg1.Y()/sp.Y();
        const scalar Y2 = pg2.Y()/sp.Y();

        const scalar Tc = Y1*pg1.Tc_ + Y2*pg2.Tc_;
        const scalar Vc = Y1*pg1.Vc_ + Y2*pg2.Vc_;
        const scalar Zc = Y1*pg1.Zc_ + Y2*pg2.Zc_;

        return pintlePengRobinsonGas<Specie>
        (
            sp,
            Tc,
            Vc,
            Zc,
            RR*Zc*Tc/Vc,
            Y1*pg1.omega_ + Y2*pg2.omega_
        );
    }
}


template<class Specie>
Foam::pintlePengRobinsonGas<Specie> Foam::operator*
(
    const scalar s,
    const pintlePengRobinsonGas<Specie>& pg
)
{
    return pintlePengRobinsonGas<Specie>
    (
        s*static_cast<const Specie&>(pg),
        pg.Tc_,
        pg.Vc_,
        pg.Zc_,
        pg.Pc_,
        pg.omega_
    );
}


template<class Specie>
Foam::pintlePengRobinsonGas<Specie> Foam::operator==
(
    const pintlePengRobinsonGas<Specie>& pg1,
    const pintlePengRobinsonGas<Specie>& pg2
)
{
    Specie sp
    (
        static_cast<const Specie&>(pg1)
     == static_cast<const Specie&>(pg2)
    );

    const scalar Y1 = pg1.Y()/sp.Y();
    const scalar Y2 = pg2.Y()/sp.Y();

    const scalar Tc = Y2*pg2.Tc_ - Y1*pg1.Tc_;
    const scalar Vc = Y2*pg2.Vc_ - Y1*pg1.Vc_;
    const scalar Zc = Y2*pg2.Zc_ - Y1*pg1.Zc_;

    return pintlePengRobinsonGas<Specie>
    (
        sp,
        Tc,
        Vc,
        Zc,
        RR*Zc*Tc/Vc,
        Y2*pg2.omega_ - Y1*pg1.omega_
    );
}


// ************************************************************************* //
