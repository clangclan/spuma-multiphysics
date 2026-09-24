// SPDX-License-Identifier: GPL-3.0-or-later
// Experimental device NASA polynomial arithmetic. Tables, T, log(T), 1/T,
// EOS roots, conserved fields and nonlinear acceptance retain the FP64 ABI.
// Included inside ReactiveDevicePR::detail after NasaRegion is declared.
struct DoubleSingle {
    float hi,lo;
    REACTIVE_HD DoubleSingle(double x=0):hi(float(x)),lo(float(x-double(hi))){}
    REACTIVE_HD static DoubleSingle parts(float a,float b){DoubleSingle r;r.hi=a;r.lo=b;return r;}
    REACTIVE_HD static DoubleSingle renormalize(float a,float b){const float s=a+b;return parts(s,b-(s-a));}
    REACTIVE_HD explicit operator double()const{return double(hi)+double(lo);}
    REACTIVE_HD friend DoubleSingle operator+(DoubleSingle a,DoubleSingle b){
        const float s=a.hi+b.hi,v=s-a.hi;
        const float e=(a.hi-(s-v))+(b.hi-v)+a.lo+b.lo;
        return renormalize(s,e);
    }
    REACTIVE_HD friend DoubleSingle operator-(DoubleSingle a){return parts(-a.hi,-a.lo);}
    REACTIVE_HD friend DoubleSingle operator-(DoubleSingle a,DoubleSingle b){return a+(-b);}
    REACTIVE_HD friend DoubleSingle operator*(DoubleSingle a,DoubleSingle b){
        const float p=a.hi*b.hi;
#ifdef __CUDA_ARCH__
        const float error=__fmaf_rn(a.hi,b.hi,-p);
#else
        const float error=::fmaf(a.hi,b.hi,-p);
#endif
        const float e=error+a.hi*b.lo+a.lo*b.hi+a.lo*b.lo;
        return renormalize(p,e);
    }
};
template<class Number>
REACTIVE_HD inline bool mixedNasa(const NasaRegion& region,double temperature,
    double& cp_R,double& h_RT,double& s_R){
    // Bounded experiment domain avoids float range loss. Other states retain
    // the canonical FP64 implementation; this is not a silent clamp.
    if(temperature<100.||temperature>10000.)return false;
    Number a[9];for(int i=0;i<region.polynomial;++i){
        const double c=region.coefficient[i];if(!finite(c)||abs(c)>1e20||(c!=0&&abs(c)<1e-30))return false;
        a[i]=Number(c);
    }
    const Number T(temperature),T2=T*T,T3=T2*T,T4=T3*T;
    const Number invT(1./temperature),logT(::log(temperature));
    Number cp,h,s;
    if(region.polynomial==7){
        cp=a[0]+a[1]*T+a[2]*T2+a[3]*T3+a[4]*T4;
        h=a[0]+Number(.5)*a[1]*T+Number(1./3.)*a[2]*T2
            +Number(.25)*a[3]*T3+Number(.2)*a[4]*T4+a[5]*invT;
        s=a[0]*logT+a[1]*T+Number(.5)*a[2]*T2+Number(1./3.)*a[3]*T3
            +Number(.25)*a[4]*T4+a[6];
    }else{
        const Number c0=a[0]*invT*invT,c1=a[1]*invT,c2=a[2],c3=a[3]*T,c4=a[4]*T2,c5=a[5]*T3,c6=a[6]*T4;
        cp=c0+c1+c2+c3+c4+c5+c6;
        h=-c0+logT*c1+c2+Number(.5)*c3+Number(1./3.)*c4+Number(.25)*c5+Number(.2)*c6+a[7]*invT;
        s=-Number(.5)*c0-c1+logT*c2+c3+Number(.5)*c4+Number(1./3.)*c5+Number(.25)*c6+a[8];
    }
    const double dc=double(cp),dh=double(h),ds=double(s);
    if(!finite(dc)||!finite(dh)||!finite(ds))return false;
    cp_R=dc;h_RT=dh;s_R=ds;return true;
}
