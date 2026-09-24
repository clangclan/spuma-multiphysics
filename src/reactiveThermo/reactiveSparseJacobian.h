// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef REACTIVE_SPARSE_JACOBIAN_H
#define REACTIVE_SPARSE_JACOBIAN_H
#include <cstddef>
#include <vector>
#include <algorithm>

// J = S + uT*vT' + uP*vP'. The dense thermodynamic couplings must
// never be dropped merely because the kinetic concentration block is sparse.
// CSR contains all entries supplied by Cantera, including third bodies/falloff.
struct ReactiveSparseJacobian {
    std::vector<std::size_t> row, column;
    std::vector<double> value, uT, vT, uP, vP;
    std::vector<int> cscOuter, cscInner;
    std::vector<std::size_t> cscToCsr;
    // Rebuild structural indices only when the actual stored pattern changes.
    // Numerical zero entries remain present; a zero crossing is not pruning.
    bool assignCsc(std::size_t n,const int* outer,const int* inner,const double* values) {
        const std::size_t nnz=outer[n];
        const bool changed=cscOuter.size()!=n+1||cscInner.size()!=nnz
            ||!std::equal(cscOuter.begin(),cscOuter.end(),outer)
            ||!std::equal(cscInner.begin(),cscInner.end(),inner);
        if(changed) {
            cscOuter.assign(outer,outer+n+1);cscInner.assign(inner,inner+nnz);
            row.assign(n+1,0);column.resize(nnz);value.resize(nnz);cscToCsr.resize(nnz);
            for(std::size_t j=0;j<nnz;++j) ++row[inner[j]+1];
            for(std::size_t j=0;j<n;++j) row[j+1]+=row[j];
            auto next=row;
            for(std::size_t col=0;col<n;++col) for(int j=outer[col];j<outer[col+1];++j) {
                const auto target=next[inner[j]]++;column[target]=col;cscToCsr[j]=target;
            }
        }
        for(std::size_t j=0;j<nnz;++j) value[cscToCsr[j]]=values[j];
        return changed;
    }
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
