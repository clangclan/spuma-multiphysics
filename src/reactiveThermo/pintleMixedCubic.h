// SPDX-License-Identifier: GPL-3.0-or-later
// Included inside PintleDevicePR::detail. Only initial transcendentals use
// FP32; cubic coefficients, topology, refinement and acceptance remain FP64.
PINTLE_HD inline bool mixedCubicRoots(double center,double delta,double delta2,
    double q,double h,double disc,double bn,double cn,double dn,double b,double* raw,int& count) {
#if defined(__CUDA_ARCH__) && PINTLE_HEM_FP32_SEEDS
    if(!finite(disc)||abs(disc)<=128.*FLT_EPSILON*(q*q+h*h))return false;
    double seeds[3]{};int n=0;
    if(disc>1e-14||!(delta2>0)){
        const double sd=.5*::sqrt(max(0.,disc)),u=-.5*q+sd,v=-.5*q-sd;
        if(!finite(u)||!finite(v)||abs(u)>FLT_MAX||abs(v)>FLT_MAX
           ||(u!=0&&abs(u)<FLT_MIN)||(v!=0&&abs(v)<FLT_MIN))return false;
        const double a=double(::cbrtf(float(u))),b=double(::cbrtf(float(v)));
        // Cancellation in the two cube roots makes a float seed unreliable.
        if(abs(a+b)<1e-4*(abs(a)+abs(b)))return false;
        seeds[0]=center+a+b;n=1;
    }else if(disc<-1e-14){
        const double arg=-.5*q/(delta*delta*delta);
        if(!finite(arg)||abs(arg)>1.-1e-4)return false;
        const float theta=::acosf(float(arg))/3.f;
        seeds[0]=center+2.*delta*double(::cosf(theta));
        seeds[1]=center+2.*delta*double(::cosf(theta+2.f*float(Pi)/3.f));
        seeds[2]=center+2.*delta*double(::cosf(theta+4.f*float(Pi)/3.f));n=3;
    }else return false;
    for(int i=0;i<n;++i){
        double x=seeds[i];bool converged=false;
        for(int iteration=0;iteration<12;++iteration){
            if(!finite(x))return false;
            // Keep each seed in its monotone cubic interval throughout Newton.
            if(n==3&&((i==0&&x<=center+delta)||(i==1&&x>=center-delta)
                ||(i==2&&(x<=center-delta||x>=center+delta))))return false;
            const double f=cubicResidual(x,bn,cn,dn),df=(3.*x+2.*bn)*x+cn;
            if(!finite(df)||abs(df)<=1e-300)return false;
            const double step=f/df;x-=step;
            if(abs(step)<=4e-15*max(1.,abs(x))){converged=true;break;}
        }
        if(!converged||!finite(x)||abs(x-b)<=1e-7*max(1.,max(abs(x),abs(b)))
           ||(n==1&&abs(x-center)<=1e-10*max(1.,abs(x))))return false;
        const double f=cubicResidual(x,bn,cn,dn),df=(3.*x+2.*bn)*x+cn;
        const double scale=abs(x*x*x)+abs(bn*x*x)+abs(cn*x)+abs(dn);
        if(!finite(f)||abs(f)>64.*DBL_EPSILON*scale||abs(df)<=1e-300
            ||abs(f/df)>8e-15*max(1.,abs(x)))return false;
        if(n==3&&((i==0&&x<=center+delta)||(i==1&&x>=center-delta)
            ||(i==2&&(x<=center-delta||x>=center+delta))))return false;
        seeds[i]=x;
    }
    for(int i=0;i<n;++i)raw[i]=seeds[i];count=n;return true;
#else
    (void)center;(void)delta;(void)delta2;(void)q;(void)h;(void)disc;
    (void)b;(void)bn;(void)cn;(void)dn;(void)raw;(void)count;return false;
#endif
}
