// SPDX-License-Identifier: GPL-3.0-or-later
// Host orchestration only: geometry, sparse operators, Krylov vectors and
// reductions execute through Runtime (CPU reference or CUDA). The 10x10
// coarse solve and scalar convergence decisions are explicit host work.
#ifndef REACTIVE_IMPLICIT_FIT_H
#define REACTIVE_IMPLICIT_FIT_H
#include "reactiveImplicitFitKernels.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace ReactiveImplicitFit {
struct Settings {
    double quadratureTolerance=1e-8,volumeTolerance=1e-7,smoothness=.01;
    unsigned nonlinearIterations=24,linearIterations=1600;
    double linearTolerance=1e-9;
    void (*observer)(unsigned,double,double,double)=nullptr;
    void (*linearObserver)(unsigned,double)=nullptr;
};
struct Progress {
    unsigned nonlinearIterations=0,linearIterations=0,rejectedTrials=0;
    unsigned coarseIterations=0;
    double rmsVolumeError=0,regularizerNorm=0;
    double lastLinearRelativeResidual=0;
    bool converged=false,coarseConverged=false;
};
inline void check(bool condition,const char* text){if(!condition)throw std::runtime_error(text);}
inline Grid validated(Grid grid) {
    for(int d=0;d<3;++d)check(grid.cells[d]>=4&&grid.cells[d]<=1024,"Invalid implicit reconstruction grid");
    return grid;
}
inline void cholesky(double matrix[100]) {
    double maximum=0;for(int i=0;i<10;++i)maximum=std::max(maximum,matrix[10*i+i]);
    check(std::isfinite(maximum)&&maximum>0,"Invalid reconstruction coarse operator");
    for(int i=0;i<10;++i)matrix[10*i+i]+=1e-12*maximum;
    for(int i=0;i<10;++i)for(int j=0;j<=i;++j) {
        double value=matrix[10*i+j];
        for(int k=0;k<j;++k)value-=matrix[10*i+k]*matrix[10*j+k];
        if(i==j){check(value>0&&std::isfinite(value),"Reconstruction coarse factorization failed");matrix[10*i+j]=std::sqrt(value);}
        else matrix[10*i+j]=value/matrix[10*j+j];
    }
}
inline void solveCoarse(const double lower[100],double rhs[10]) {
    for(int i=0;i<10;++i){for(int j=0;j<i;++j)rhs[i]-=lower[10*i+j]*rhs[j];rhs[i]/=lower[10*i+i];}
    for(int i=9;i>=0;--i){for(int j=i+1;j<10;++j)rhs[i]-=lower[10*j+i]*rhs[j];rhs[i]/=lower[10*i+i];}
}
template<class Runtime> class Workspace {
    Runtime& runtime;Grid grid;size_t cells,controls;
    double *color,*a,*trial,*gauge,*normal,*volume,*trialVolume,*jacobian;
    double *cellWork,*thirdWork,*residual,*rhs,*diagonal,*update,*r,*z,*p,*ap;
    double *coarseSamples,*coarseWeights;
    double lastLinearRelativeResidual=0;
    static constexpr double ridge=1e-12;
    double dot(size_t n,const double* x,const double* y){return runtime.sum(n,Dot{x,y});}
    void regularize(const double* x){runtime.launch(10*controls,ThirdForward{grid,x,thirdWork});}
    bool geometry(const double* x,double* output,double* derivatives,double tolerance) {
        runtime.launch(cells,Geometry{grid,x,tolerance,output,derivatives});
        return runtime.sum(cells,BadGeometry{output})==0;
    }
    void apply(const double* x,double lambda,double* output) {
        runtime.launch(cells,JacobianForward{grid,jacobian,x,cellWork});regularize(x);
        const double gx=dot(controls,gauge,x);
        runtime.launch(controls,Transpose{grid,jacobian,cellWork,thirdWork,gauge,lambda,gx,output,false,x,ridge});
    }
    void precondition(const double lower[100]) {
        double weights[10];
        for(int m=0;m<10;++m)weights[m]=runtime.sum(controls,CoarseDot{grid,r,m});
        solveCoarse(lower,weights);runtime.upload(coarseWeights,weights,10);
        runtime.launch(controls,Precondition{grid,r,diagonal,coarseWeights,z});
    }
    unsigned linearSolve(double lambda,const Settings& settings) {
        // RHS is J^T(c-V) - lambda D3^T D3 a. Gauge RHS is zero because
        // accepted iterates are rescaled to the same positive linear gauge.
        regularize(a);
        runtime.launch(controls,Transpose{grid,jacobian,residual,thirdWork,nullptr,-lambda,0,rhs});
        runtime.launch(controls,Transpose{grid,jacobian,nullptr,nullptr,gauge,lambda,0,diagonal,true,nullptr,ridge});
        runtime.launch(10*cells,CoarseSamples{grid,jacobian,coarseSamples});
        double gp[10],lower[100];
        for(int i=0;i<10;++i)gp[i]=runtime.sum(controls,CoarseDot{grid,gauge,i});
        for(int i=0;i<10;++i)for(int j=0;j<=i;++j) {
            const double value=dot(cells,coarseSamples+i*cells,coarseSamples+j*cells)+100*gp[i]*gp[j];
            lower[10*i+j]=value;lower[10*j+i]=value;
        }
        cholesky(lower);
        runtime.launch(controls,LinearCombination{0,a,0,nullptr,update});
        runtime.launch(controls,LinearCombination{1,rhs,0,nullptr,r});
        precondition(lower);
        runtime.launch(controls,LinearCombination{1,z,0,nullptr,p});
        double rz=dot(controls,r,z);
        const double before=dot(controls,r,r);
        if(before<1e-30){lastLinearRelativeResidual=0;return 0;}
        auto finish=[&](unsigned iterations) {
            // Recompute the true residual; a capped Krylov solve supplies an
            // inexact descent candidate, never a claimed converged solve.
            apply(update,lambda,ap);runtime.launch(controls,Difference{rhs,ap,r});
            lastLinearRelativeResidual=std::sqrt(dot(controls,r,r)/before);
            if(settings.linearObserver)settings.linearObserver(iterations,lastLinearRelativeResidual);
            return iterations;
        };
        check(rz>0&&std::isfinite(rz),"Invalid reconstruction initial Krylov residual");
        for(unsigned iteration=0;iteration<settings.linearIterations;++iteration) {
            apply(p,lambda,ap);const double pap=dot(controls,p,ap);
            check(pap>0&&std::isfinite(pap),"Nonpositive reconstruction Krylov operator");
            const double alpha=rz/pap;
            runtime.launch(controls,LinearCombination{1,update,alpha,p,update});
            runtime.launch(controls,LinearCombination{1,r,-alpha,ap,r});
            const double norm=dot(controls,r,r);
            if(norm<=before*settings.linearTolerance*settings.linearTolerance)return finish(iteration+1);
            precondition(lower);const double next=dot(controls,r,z);
            check(next>0&&std::isfinite(next),"Invalid reconstruction Krylov residual");
            runtime.launch(controls,LinearCombination{1,z,next/rz,p,p});rz=next;
        }
        return finish(settings.linearIterations);
    }
    bool coarseFit(const Settings& settings,Progress& progress) {
        // Solve first in the nullspace of the D3 regularizer. If an interface
        // satisfies every volume constraint here, it has the minimum possible
        // bending cost. Otherwise retain all local spline degrees of freedom.
        // This receives no shape parameters and is identical for all targets.
        const double* candidate=a;bool feasible=false;double bestCost=0;
        auto finish=[&]() {
            if(!feasible)return false;
            if(!geometry(z,volume,nullptr,settings.quadratureTolerance))return false;
            runtime.launch(cells,Difference{color,volume,residual});
            if(runtime.sum(cells,Exceeds{residual,settings.volumeTolerance})!=0)return false;
            bestCost=dot(cells,residual,residual);
            runtime.launch(controls,LinearCombination{1,z,0,nullptr,a});
            progress.rmsVolumeError=std::sqrt(bestCost/cells);regularize(a);
            progress.regularizerNorm=std::sqrt(dot(10*controls,thirdWork,thirdWork));
            progress.converged=progress.coarseConverged=true;return true;
        };
        for(unsigned iteration=0;iteration<=10;++iteration){
            if(!geometry(candidate,volume,iteration==10?nullptr:jacobian,settings.quadratureTolerance))return finish();
            runtime.launch(cells,Difference{color,volume,residual});
            const double dataCost=dot(cells,residual,residual);
            if(candidate!=a&&runtime.sum(cells,Exceeds{residual,settings.volumeTolerance})==0){
                // Feasibility alone does not choose the best coarse surface.
                // Continue minimizing the overdetermined volume residual to
                // avoid an arbitrary curvature inside the allowed interval.
                runtime.launch(controls,LinearCombination{1,candidate,0,nullptr,z});
                bestCost=dataCost;feasible=true;
            }
            if(iteration==10)return finish();
            runtime.launch(10*cells,CoarseSamples{grid,jacobian,coarseSamples});
            double gp[10],matrix[100],weights[10],coarseRhs[10];
            for(int m=0;m<10;++m)gp[m]=runtime.sum(controls,CoarseDot{grid,gauge,m});
            // J*a=0 because multiplying an implicit field by a positive
            // constant leaves every sharp volume unchanged. Thus the linear
            // absolute-coefficient equation is J*P*w = c-V, with g*P*w=1.
            for(int i=0;i<10;++i){
                coarseRhs[i]=weights[i]=dot(cells,coarseSamples+i*cells,residual)+(candidate==a?100*gp[i]:0);
                for(int j=0;j<=i;++j){const double value=dot(cells,coarseSamples+i*cells,coarseSamples+j*cells)+100*gp[i]*gp[j];
                    matrix[10*i+j]=matrix[10*j+i]=value;}
            }
            // Plane-like zero sets may leave polynomial directions
            // unidentified. The existing diagonal regularizer gives a
            // bounded candidate; failure falls back to the full space.
            try {cholesky(matrix);}catch(const std::runtime_error&){return finish();}
            solveCoarse(matrix,weights);
            if(candidate!=a){
                double predicted=0;for(int m=0;m<10;++m)predicted+=weights[m]*coarseRhs[m];
                if(predicted<=settings.linearTolerance*dataCost)return finish();
            }
            runtime.upload(coarseWeights,weights,10);
            runtime.launch(controls,CoarseExpand{grid,coarseWeights,update});
            // Once in the nullspace solve for a correction, so the rank
            // regularizer damps an update rather than biasing the solution.
            if(candidate!=a)runtime.launch(controls,LinearCombination{1,candidate,1,update,update});
            const double gu=dot(controls,gauge,update);
            if(!(gu>0&&std::isfinite(gu)))return finish();
            runtime.launch(controls,LinearCombination{1/gu,update,0,nullptr,update});
            bool accepted=false;double step=1;
            // The first candidate enters the polynomial subspace directly.
            // Later damped steps stay in that same subspace.
            for(unsigned attempt=0;attempt<(candidate==a?1u:10u);++attempt){
                runtime.launch(controls,LinearCombination{1-step,candidate,step,update,ap});
                if(geometry(ap,trialVolume,nullptr,settings.quadratureTolerance)){
                    runtime.launch(cells,Difference{color,trialVolume,cellWork});
                    if(dot(cells,cellWork,cellWork)<dataCost){accepted=true;break;}
                }
                step*=.5;
            }
            if(!accepted)return finish();
            runtime.launch(controls,LinearCombination{1,ap,0,nullptr,trial});candidate=trial;
            progress.coarseIterations=iteration+1;
        }
        return finish();
    }
public:
    Workspace(Runtime& execution,Grid dimensions):runtime(execution),grid(validated(dimensions)),
        cells(size_t(grid.count())),controls(size_t(grid.controls())) {
        auto alloc=[&](size_t n){return runtime.template allocate<double>(n);};
        color=alloc(cells);a=alloc(controls);trial=alloc(controls);gauge=alloc(controls);
        normal=alloc(3*cells);volume=alloc(cells);trialVolume=alloc(cells);jacobian=alloc(64*cells);
        cellWork=alloc(cells);thirdWork=alloc(10*controls);residual=alloc(cells);
        rhs=alloc(controls);diagonal=alloc(controls);update=alloc(controls);r=alloc(controls);
        z=alloc(controls);p=alloc(controls);ap=alloc(controls);
        coarseSamples=alloc(10*cells);coarseWeights=alloc(10);
    }
    // Inputs/outputs live on the Runtime's backend. CPU and CUDA both execute
    // this exact algorithm. No coefficients from an analytic shape are needed.
    const double* coefficients()const{return a;}
    double* target(){return color;}
    double* initialCoefficients(){return a;}
    size_t coefficientCount()const{return controls;}
    // Caller must have completed a successful fit with this coefficient
    // field. Its volume quadrature is then a certificate for a new target;
    // accepting it is the same hard constraint as fit()'s first iteration.
    bool targetMatches(double tolerance) {
        runtime.launch(cells,Difference{color,volume,residual});
        return runtime.sum(cells,Exceeds{residual,tolerance})==0;
    }
    bool certifyRestored(const Settings& settings) {
        if(!geometry(a,volume,nullptr,settings.quadratureTolerance))return false;
        return targetMatches(settings.volumeTolerance);
    }
    Progress fit(const Settings& settings,bool reuse=false) {
        check(settings.quadratureTolerance>0&&settings.quadratureTolerance<.01
            &&settings.volumeTolerance>0&&settings.smoothness>0
            &&std::isfinite(settings.volumeTolerance)&&std::isfinite(settings.smoothness)
            &&settings.nonlinearIterations>0&&settings.linearIterations>0
            &&settings.linearTolerance>0&&settings.linearTolerance<1,"Invalid reconstruction settings");
        const double mixed=runtime.sum(cells,MixedCount{color});
        if(!reuse)runtime.launch(controls,Seed{grid,color,a});
        if(mixed==0) {
            check(geometry(a,volume,nullptr,settings.quadratureTolerance),"Unresolved binary interface seed");
            runtime.launch(cells,Difference{color,volume,residual});
            Progress uniform{};uniform.converged=runtime.sum(cells,Exceeds{residual,settings.volumeTolerance})==0;
            if(reuse&&!uniform.converged){
                runtime.launch(controls,Seed{grid,color,a});
                check(geometry(a,volume,nullptr,settings.quadratureTolerance),"Unresolved changed binary interface seed");
                runtime.launch(cells,Difference{color,volume,residual});
                uniform.converged=runtime.sum(cells,Exceeds{residual,settings.volumeTolerance})==0;
            }
            uniform.rmsVolumeError=std::sqrt(dot(cells,residual,residual)/cells);
            return uniform;
        }
        runtime.launch(cells,SeedNormal{grid,color,a,normal});
        runtime.launch(controls,Gauge{grid,normal,1/mixed,gauge});
        const double ga=dot(controls,gauge,a);check(ga>0&&std::isfinite(ga),"Unresolved reconstruction seed normals");
        runtime.launch(controls,LinearCombination{1/ga,a,0,nullptr,a});
        Progress result{};
        if(!reuse&&coarseFit(settings,result))return result;
        double lambda=settings.smoothness;
        for(unsigned iteration=0;iteration<settings.nonlinearIterations;++iteration) {
            check(geometry(a,volume,jacobian,settings.quadratureTolerance),"Unresolved implicit reconstruction quadrature");
            runtime.launch(cells,Difference{color,volume,residual});
            const double dataCost=dot(cells,residual,residual);regularize(a);
            const double regularCost=dot(10*controls,thirdWork,thirdWork);
            result.rmsVolumeError=std::sqrt(dataCost/cells);result.regularizerNorm=std::sqrt(regularCost);
            result.nonlinearIterations=iteration;
            if(settings.observer)settings.observer(iteration,result.rmsVolumeError,result.regularizerNorm,lambda);
            if(runtime.sum(cells,Exceeds{residual,settings.volumeTolerance})==0) {
                result.converged=true;return result;
            }
            const bool feasibilityPhase=runtime.sum(cells,Exceeds{residual,100*settings.volumeTolerance})==0;
            if(feasibilityPhase&&regularCost>0) {
                // Near the requested hard volume constraint, smoothing the
                // far-field coefficients must not buy a larger volume error.
                // Keep D3 as a secondary regularizer and require data descent.
                lambda=std::min(lambda,std::max(1e-14,.1*dataCost/regularCost));
            }
            result.linearIterations+=linearSolve(lambda,settings);
            result.lastLinearRelativeResidual=lastLinearRelativeResidual;
            result.nonlinearIterations=iteration+1;
            const double before=dataCost+lambda*regularCost;double step=1;bool accepted=false;
            double acceptedData=dataCost;
            for(unsigned attempt=0;attempt<12;++attempt) {
                runtime.launch(controls,LinearCombination{1,a,step,update,trial});
                const double gt=dot(controls,gauge,trial);
                if(gt>0&&std::isfinite(gt)) {
                    runtime.launch(controls,LinearCombination{1/gt,trial,0,nullptr,trial});
                    if(geometry(trial,trialVolume,nullptr,settings.quadratureTolerance)) {
                        runtime.launch(cells,Difference{color,trialVolume,cellWork});regularize(trial);
                        const double trialData=dot(cells,cellWork,cellWork);
                        const double objective=trialData+lambda*dot(10*controls,thirdWork,thirdWork);
                        if(std::isfinite(objective)&&objective<before&&(!feasibilityPhase||trialData<dataCost)) {
                            std::swap(a,trial);accepted=true;acceptedData=trialData;break;
                        }
                    }
                }
                step*=.5;++result.rejectedTrials;
            }
            if(!accepted)lambda*=.1;
            else if(acceptedData>.81*dataCost||(iteration&&iteration%6==0))lambda*=.1;
            // A stationary penalized solution is not a solution of the hard
            // volume constraints. Continue the penalty as soon as the data
            // residual stalls, instead of spending six nearly identical
            // Krylov solves at the same regularization strength.
            // A successful feasibility step may legitimately need a tiny
            // regularizer. Stop only after a failed line search at its floor.
            if(!accepted&&lambda<1e-14)break;
        }
        check(geometry(a,volume,nullptr,settings.quadratureTolerance),"Unresolved final implicit geometry");
        runtime.launch(cells,Difference{color,volume,residual});
        result.rmsVolumeError=std::sqrt(dot(cells,residual,residual)/cells);
        regularize(a);result.regularizerNorm=std::sqrt(dot(10*controls,thirdWork,thirdWork));
        result.converged=runtime.sum(cells,Exceeds{residual,settings.volumeTolerance})==0;
        return result;
    }
};
} // namespace ReactiveImplicitFit
#endif
