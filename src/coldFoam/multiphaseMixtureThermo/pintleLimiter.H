// SPDX-License-Identifier: GPL-3.0-or-later
// Fixed-mesh VOF specialization of SPUMA v2512 MULES (rho=1, Sp=Su=0).
#ifndef pintleLimiter_H
#define pintleLimiter_H
#include "fvCFD.H"
#include "slicedSurfaceFields.H"

namespace Foam
{
class pintleLimiter
{
    const fvMesh& mesh_;
    labelList boundaryStart_, boundaryFaces_, boundaryOwner_, boundaryKind_;
    scalarField boundaryPsi_, boundaryBD_, boundaryCorr_;
    scalarField upperBudget_, lowerBudget_, positive_, negative_;
    scalarField ratioIn_, ratioOut_, lambda_;
    Field<float> ratioIn32_, ratioOut32_;
    bool coupled_;

    template<class Ratio> void iterate
    (
        const volScalarField&, const surfaceScalarField&,
        const surfaceScalarField&, const scalar, const label,
        const scalar, const scalar, const scalar,
        Field<Ratio>&, Field<Ratio>&
    );
public:
    explicit pintleLimiter(const fvMesh&);
    // No mesh mutation while this cache exists. Coupled min sync is retained.
    void limit(const volScalarField&, const surfaceScalarField&,
               surfaceScalarField&, bool mixed, bool returnCorr=true);
    void explicitSolve(volScalarField&,const surfaceScalarField&,
                       const volScalarField::Internal&,const volScalarField::Internal&);
    static void limitSum(PtrList<surfaceScalarField>&);
};
}
#endif
