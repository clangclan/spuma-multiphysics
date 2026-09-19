// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef PINTLE_SMALL_SYSTEM_H
#define PINTLE_SMALL_SYSTEM_H
#include <Eigen/Dense>
#include <stdexcept>
// Equilibrated pivoted solve, never an explicit inverse. Residual is measured
// in the original equations; rcond describes the equilibrated system.
inline Eigen::VectorXd pintleSmallSolve(const Eigen::MatrixXd& D,const Eigen::VectorXd& b,
                                      double* rcond=nullptr,double* residual=nullptr)
{
    if(D.rows()!=D.cols()||D.rows()!=b.size()||!D.allFinite()||!b.allFinite())
        throw std::runtime_error("Invalid small tangent system");
    Eigen::VectorXd rows(D.rows()),cols(D.cols());Eigen::MatrixXd A=D;
    for(Eigen::Index i=0;i<A.rows();++i) {rows[i]=A.row(i).cwiseAbs().maxCoeff();
        if(!(rows[i]>0))throw std::runtime_error("Zero tangent equation");
        A.row(i)/=rows[i];}
    for(Eigen::Index j=0;j<A.cols();++j) {cols[j]=A.col(j).cwiseAbs().maxCoeff();
        if(!(cols[j]>0))throw std::runtime_error("Zero tangent variable");
        A.col(j)/=cols[j];}
    const auto lu=A.fullPivLu();
    if(!lu.isInvertible()||lu.rcond()<1e-10)throw std::runtime_error("Ill-conditioned scaled tangent");
    const Eigen::VectorXd x=lu.solve((b.array()/rows.array()).matrix()).array()/cols.array();
    const double error=(D*x-b).lpNorm<Eigen::Infinity>()/
        std::max(1.,D.lpNorm<Eigen::Infinity>()*x.lpNorm<Eigen::Infinity>()+b.lpNorm<Eigen::Infinity>());
    if(!x.allFinite()||!std::isfinite(error)||error>1e-10)throw std::runtime_error("Small tangent residual failed");
    if(rcond)*rcond=lu.rcond();
    if(residual)*residual=error;
    return x;
}
// Flash retains its previous FP64 pivoted reference solve if equilibration is
// rejected. A new tangent screening threshold must not silently remove an old
// entropy candidate. The flash still applies its original nonlinear acceptance.
inline Eigen::VectorXd pintleFlashSolve(const Eigen::MatrixXd& D,const Eigen::VectorXd& b)
{
    try {return pintleSmallSolve(D,b);}
    catch(const std::exception&) {
        const auto lu=D.fullPivLu();
        if(!lu.isInvertible())throw std::runtime_error("Singular reference flash system");
        const Eigen::VectorXd x=lu.solve(b);
        if(!x.allFinite())throw std::runtime_error("Nonfinite reference flash correction");
        return x;
    }
}
#endif
