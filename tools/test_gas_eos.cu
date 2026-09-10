// MAIN: independent finite-difference identities and actual CPU/GPU execution.
#include <cuda_runtime.h>
#define FOAM_DEVICE __host__ __device__
#include "../src/coldFoam/pintleGasEOS.H"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>
struct Sample {pintle::GasEOS eos; double rho,T;};
__global__ void evaluate(const Sample* in,pintle::GasState* out,int n)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)out[i]=in[i].eos.evaluate(in[i].rho,in[i].T);
}
void cudaCheck(cudaError_t status) {if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
double density(const pintle::GasEOS& eos,double p,double t,double r)
{
    // The root finder deliberately uses a numerical pressure derivative.
    for(int j=0;j<15;++j)
    {
        double h=r*1e-5;
        double dpdr=(eos.evaluate(r+h,t).p-eos.evaluate(r-h,t).p)/(2*h);
        double delta=(eos.evaluate(r,t).p-p)/dpdr;r-=delta;
        if(std::abs(delta)<1e-13*r)break;
    }
    return r;
}
int main()
{
    std::vector<Sample> in;
    const double RR=6.022141793e23*1.38065e-23*1000;
    for(bool pr:{false,true})for(double t:{270.,300.,340.,450.})for(double r:{1.,10.,40.,80.})
    {
        pintle::GasEOS eos;eos.R=RR/(pr?44.0128:28.965);eos.cp0=pr?881.80380464488:3.5*eos.R;
        eos.pengRobinson=pr;eos.Tc=309.52067823146;eos.Pc=7244816.7016072;eos.omega=.1613;
        in.push_back({eos,r,t});
    }
    double worst=0,gpuWorst=0;int checks=0;
    auto check=[&](double got,double wanted,double scale)
    {
        double err=std::abs(got-wanted)/std::max(std::abs(wanted),scale);
        if(!std::isfinite(err))throw std::runtime_error("nonfinite identity");
        worst=std::max(worst,err);++checks;
    };
    for(const auto& v:in)
    {
        const auto& eos=v.eos;double r=v.rho,t=v.T,hr=r*1e-4,ht=t*1e-4;
        auto s=eos.evaluate(r,t);auto rp=eos.evaluate(r+hr,t),rm=eos.evaluate(r-hr,t);
        auto tp=eos.evaluate(r,t+ht),tm=eos.evaluate(r,t-ht);
        double pr=(rp.p-rm.p)/(2*hr),pt=(tp.p-tm.p)/(2*ht);
        check(1/s.drhodpT,pr,1);check(s.drhodTp,-pt/pr,1e-6);
        check(s.cv,(tp.e-tm.e)/(2*ht),1);
        check((rp.e-rm.e)/(2*hr),(s.p-t*pt)/(r*r),1);
        const double rtplus=density(eos,s.p,t+ht,r),rtminus=density(eos,s.p,t-ht,r);
        const auto ep=eos.evaluate(rtplus,t+ht),em=eos.evaluate(rtminus,t-ht);
        check(s.dedTp,(ep.e-em.e)/(2*ht),1);
        check(s.cp,((ep.e+s.p/rtplus)-(em.e+s.p/rtminus))/(2*ht),1);
        check(s.soundSpeedSqr,s.cp/s.cv/s.drhodpT,1);
        if(!eos.pengRobinson)check(s.soundSpeedSqr,1.4*eos.R*t,1);
    }
    Sample* devIn=nullptr;pintle::GasState* devOut=nullptr;
    std::vector<pintle::GasState> out(in.size());
    cudaCheck(cudaMalloc(&devIn,in.size()*sizeof(Sample)));
    cudaCheck(cudaMalloc(&devOut,in.size()*sizeof(pintle::GasState)));
    cudaCheck(cudaMemcpy(devIn,in.data(),in.size()*sizeof(Sample),cudaMemcpyHostToDevice));
    evaluate<<<1,128>>>(devIn,devOut,in.size());cudaCheck(cudaGetLastError());cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(out.data(),devOut,out.size()*sizeof(pintle::GasState),cudaMemcpyDeviceToHost));
    for(size_t i=0;i<in.size();++i)
    {
        auto s=in[i].eos.evaluate(in[i].rho,in[i].T);
        const double host[]={s.p,s.e,s.cv,s.cp,s.drhodpT,s.drhodTp,s.dedTp,s.soundSpeedSqr};
        const auto& d=out[i];const double gpu[]={d.p,d.e,d.cv,d.cp,d.drhodpT,d.drhodTp,d.dedTp,d.soundSpeedSqr};
        for(int j=0;j<8;++j)
        {
            if(!std::isfinite(host[j]) || !std::isfinite(gpu[j]))throw std::runtime_error("nonfinite CPU/GPU property");
            gpuWorst=std::max(gpuWorst,std::abs(host[j]-gpu[j])/std::max(1e-12,std::abs(host[j])));
        }
    }
    cudaCheck(cudaFree(devIn));cudaCheck(cudaFree(devOut));
    const bool pass=worst<3e-6 && gpuWorst<2e-12;
    std::cout<<std::setprecision(17)<<"{\"states\":"<<in.size()<<",\"identity_checks\":"<<checks
        <<",\"finite_difference_max_scaled_error\":"<<worst<<",\"gpu_max_relative_error\":"<<gpuWorst
        <<",\"passed\":"<<(pass?"true":"false")<<"}\n";
    return pass?0:1;
}
