// A bounded capability/counter-access probe, not a performance benchmark.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
static void check(cudaError_t e) {
    if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);}
}
__global__ void profileProbe(double* x) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    double v=double(i)+1.;
    for(int j=0;j<1024;++j)v=v*1.00000001+0.00000001;
    x[i]=v;
}
int main(){
    cudaDeviceProp p{};check(cudaGetDeviceProperties(&p,0));
    int ratio=0;check(cudaDeviceGetAttribute(&ratio,cudaDevAttrSingleToDoublePrecisionPerfRatio,0));
    std::printf("{\"name\":\"%s\",\"computeMajor\":%d,\"computeMinor\":%d,"
        "\"smCount\":%d,\"warpSize\":%d,\"maxThreadsPerSM\":%d,\"maxBlocksPerSM\":%d,"
        "\"registersPerSM\":%d,\"sharedMemoryPerSM\":%zu,\"globalMemoryBytes\":%zu,"
        "\"l2Bytes\":%d,\"memoryBusBits\":%d,\"singleToDoublePerfRatioAttribute\":%d}\n",
        p.name,p.major,p.minor,p.multiProcessorCount,p.warpSize,p.maxThreadsPerMultiProcessor,
        p.maxBlocksPerMultiProcessor,p.regsPerMultiprocessor,p.sharedMemPerMultiprocessor,
        p.totalGlobalMem,p.l2CacheSize,p.memoryBusWidth,ratio);
    double* x=nullptr;check(cudaMalloc(&x,65536*sizeof(double)));
    profileProbe<<<256,256>>>(x);check(cudaGetLastError());check(cudaDeviceSynchronize());
    check(cudaFree(x));
}
