/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
    Copyright (C) 2015-2022 OpenCFD Ltd.
    Copyright (C) 2025 Cineca
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

#include "pintleDeviceHeRhoThermo.H"

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

template<class BasicPsiThermo, class MixtureType>
void Foam::pintleDeviceHeRhoThermo<BasicPsiThermo, MixtureType>::calculate
(
    const volScalarField& p,
    volScalarField& T,
    volScalarField& he,
    volScalarField& psi,
    volScalarField& rho,
    volScalarField& mu,
    volScalarField& alpha,
    const bool doOldTimes
)
{
    // Note: update oldTimes before current time so that if T.oldTime() is
    // created from T, it starts from the unconverted T
    if (doOldTimes && (p.nOldTimes() || T.nOldTimes()))
    {
        calculate
        (
            p.oldTime(),
            T.oldTime(),
            he.oldTime(),
            psi.oldTime(),
            rho.oldTime(),
            mu.oldTime(),
            alpha.oldTime(),
            true
        );
    }

    const scalarField& hCells = he.primitiveField();
    const scalarField& pCells = p.primitiveField();

    scalarField& TCells = T.primitiveFieldRef();
    scalarField& psiCells = psi.primitiveFieldRef();
    scalarField& rhoCells = rho.primitiveFieldRef();
    scalarField& muCells = mu.primitiveFieldRef();
    scalarField& alphaCells = alpha.primitiveFieldRef();

    const bool updateT = this->updateT();

    foamExecutor exec;
    auto TCellsPtr = TCells.begin();
    auto psiCellsPtr = psiCells.begin();
    auto rhoCellsPtr = rhoCells.begin();
    auto muCellsPtr = muCells.begin();
    auto alphaCellsPtr = alphaCells.begin();
    const auto hCellsPtr = hCells.cbegin();
    const auto pCellsPtr = pCells.cbegin();
    const MixtureType mixture(*this);

    auto Lambda = [=](label celli)
    {
        const typename MixtureType::thermoType& cellMixture =
            mixture.cellMixture(celli);

        if (updateT)
        {
            TCellsPtr[celli] = cellMixture.THE
            (
                hCellsPtr[celli],
                pCellsPtr[celli],
                TCellsPtr[celli]
            );
        }

        psiCellsPtr[celli] = cellMixture.psi(pCellsPtr[celli], TCellsPtr[celli]);
        rhoCellsPtr[celli] = cellMixture.rho(pCellsPtr[celli], TCellsPtr[celli]);
        muCellsPtr[celli] = cellMixture.mu(pCellsPtr[celli], TCellsPtr[celli]);
        alphaCellsPtr[celli] = cellMixture.alphah(pCellsPtr[celli], TCellsPtr[celli]);
    };
    exec.parallelFor(Lambda, TCells.size());

    const volScalarField::Boundary& pBf = p.boundaryField();
    volScalarField::Boundary& TBf = T.boundaryFieldRef();
    volScalarField::Boundary& psiBf = psi.boundaryFieldRef();
    volScalarField::Boundary& rhoBf = rho.boundaryFieldRef();
    volScalarField::Boundary& heBf = he.boundaryFieldRef();
    volScalarField::Boundary& muBf = mu.boundaryFieldRef();
    volScalarField::Boundary& alphaBf = alpha.boundaryFieldRef();

    forAll(pBf, patchi)
    {
        const fvPatchScalarField& pp = pBf[patchi];
        fvPatchScalarField& pT = TBf[patchi];
        fvPatchScalarField& ppsi = psiBf[patchi];
        fvPatchScalarField& prho = rhoBf[patchi];
        fvPatchScalarField& phe = heBf[patchi];
        fvPatchScalarField& pmu = muBf[patchi];
        fvPatchScalarField& palpha = alphaBf[patchi];

        const auto ppPtr = pp.cbegin();
        auto pTPtr = pT.begin();
        auto phePtr = phe.begin();
        auto ppsiPtr = ppsi.begin();
        auto prhoPtr = prho.begin();
        auto pmuPtr = pmu.begin();
        auto palphaPtr = palpha.begin();

        if (pT.fixesValue())
        {
            auto LambdaPatch = [=](label facei)
            {
                const typename MixtureType::thermoType& pFaceMixture =
                    mixture.patchFaceMixture(patchi, facei);

                phePtr[facei] = pFaceMixture.HE(ppPtr[facei], pTPtr[facei]);
                ppsiPtr[facei] = pFaceMixture.psi(ppPtr[facei], pTPtr[facei]);
                prhoPtr[facei] = pFaceMixture.rho(ppPtr[facei], pTPtr[facei]);
                pmuPtr[facei] = pFaceMixture.mu(ppPtr[facei], pTPtr[facei]);
                palphaPtr[facei] = pFaceMixture.alphah(ppPtr[facei], pTPtr[facei]);
            };
            exec.parallelFor(LambdaPatch, pT.size());
        }
        else
        {
            auto LambdaPatch = [=](label facei)
            {
                const typename MixtureType::thermoType& pFaceMixture =
                    mixture.patchFaceMixture(patchi, facei);

                if (updateT)
                {
                    pTPtr[facei] = pFaceMixture.THE(phePtr[facei], ppPtr[facei], pTPtr[facei]);
                }

                ppsiPtr[facei] = pFaceMixture.psi(ppPtr[facei], pTPtr[facei]);
                prhoPtr[facei] = pFaceMixture.rho(ppPtr[facei], pTPtr[facei]);
                pmuPtr[facei] = pFaceMixture.mu(ppPtr[facei], pTPtr[facei]);
                palphaPtr[facei] = pFaceMixture.alphah(ppPtr[facei], pTPtr[facei]);
            };
            exec.parallelFor(LambdaPatch, pT.size());
        }
    }
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicPsiThermo, class MixtureType>
Foam::pintleDeviceHeRhoThermo<BasicPsiThermo, MixtureType>::pintleDeviceHeRhoThermo
(
    const fvMesh& mesh,
    const word& phaseName
)
:
    deviceHeThermo<BasicPsiThermo, MixtureType>(mesh, phaseName)
{
    calculate
    (
        this->p_,
        this->T_,
        this->he_,
        this->psi_,
        this->rho_,
        this->mu_,
        this->alpha_,
        true                    // Create old time fields
    );
}


template<class BasicPsiThermo, class MixtureType>
Foam::pintleDeviceHeRhoThermo<BasicPsiThermo, MixtureType>::pintleDeviceHeRhoThermo
(
    const fvMesh& mesh,
    const word& phaseName,
    const word& dictName
)
:
    deviceHeThermo<BasicPsiThermo, MixtureType>(mesh, phaseName, dictName)
{
    calculate
    (
        this->p_,
        this->T_,
        this->he_,
        this->psi_,
        this->rho_,
        this->mu_,
        this->alpha_,
        true                    // Create old time fields
    );
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class BasicPsiThermo, class MixtureType>
Foam::pintleDeviceHeRhoThermo<BasicPsiThermo, MixtureType>::~pintleDeviceHeRhoThermo()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicPsiThermo, class MixtureType>
void Foam::pintleDeviceHeRhoThermo<BasicPsiThermo, MixtureType>::correct()
{
    DebugInFunction << endl;

    calculate
    (
        this->p_,
        this->T_,
        this->he_,
        this->psi_,
        this->rho_,
        this->mu_,
        this->alpha_,
        false           // No need to update old times
    );

    DebugInFunction << "Finished" << endl;
}

// ************************************************************************* //
