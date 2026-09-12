// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef PINTLE_SPARSE_JACOBIAN_H
#define PINTLE_SPARSE_JACOBIAN_H
#include <cstddef>
#include <vector>

// J = S + uT*vT' + uP*vP'. The dense thermodynamic couplings must
// never be dropped merely because the kinetic concentration block is sparse.
// CSR contains all entries supplied by Cantera, including third bodies/falloff.
struct PintleSparseJacobian {
    std::vector<std::size_t> row, column;
    std::vector<double> value, uT, vT, uP, vP;
    void apply(const double* x, double* y) const {
        double t=0,p=0;
        for(std::size_t j=0;j<vT.size();++j) {t+=vT[j]*x[j];p+=vP[j]*x[j];}
        for(std::size_t i=0;i<uT.size();++i) {
            double sum=0;
            for(std::size_t j=row[i];j<row[i+1];++j) sum+=value[j]*x[column[j]];
            y[i]=sum+uT[i]*t+uP[i]*p;
        }
    }
};
#endif
