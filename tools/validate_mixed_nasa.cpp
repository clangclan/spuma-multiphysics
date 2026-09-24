// SPDX-License-Identifier: GPL-3.0-or-later
#include "../src/reactiveThermo/pintleDeviceFlash.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cassert>
int main(int argc,char** argv){
    assert(argc==2);std::ifstream f(argv[1],std::ios::binary);uint64_t header[8]{};
    f.read((char*)header,sizeof(header));assert(header[0]==0x48454d4341503031ULL&&header[2]==sizeof(PintleDeviceFlash::Model));
    PintleDeviceFlash::Model model{};f.read((char*)&model,sizeof(model));assert(f.good());
    double singleError[3]{},dsError[3]{};size_t samples=0;
    for(int phase=0;phase<=model.nl;++phase){const auto& table=model.phase[phase];
        for(int k=0;k<table.ns;++k)for(int r=0;r<table.species[k].regionCount;++r){
            const auto& region=table.species[k].regions[r];
            const double low=std::max(100.,region.minimumTemperature),high=std::min(10000.,region.maximumTemperature);
            if(!(high>low))continue;
            for(int i=0;i<=2000;++i){const double T=low+(high-low)*i/2000.;double ref[3]{},single[3]{},ds[3]{};
                PintleDevicePR::detail::nasa(region,T,ref[0],ref[1],ref[2]);
                assert(PintleDevicePR::detail::mixedNasa<float>(region,T,single[0],single[1],single[2]));
                assert(PintleDevicePR::detail::mixedNasa<PintleDevicePR::detail::DoubleSingle>(region,T,ds[0],ds[1],ds[2]));
                for(int j=0;j<3;++j){singleError[j]=std::max(singleError[j],std::fabs(single[j]-ref[j])/std::max(1.,std::fabs(ref[j])));
                    dsError[j]=std::max(dsError[j],std::fabs(ds[j]-ref[j])/std::max(1.,std::fabs(ref[j])));}
                ++samples;
            }
        }
    }
    for(double e:dsError)assert(e<1e-10);
    std::cout.precision(17);std::cout<<"{\"passed\":true,\"samples\":"<<samples<<",\"fp32MaxScaled\":["<<singleError[0]<<","<<singleError[1]<<","<<singleError[2]<<"],\"dsMaxScaled\":["<<dsError[0]<<","<<dsError[1]<<","<<dsError[2]<<"]}"<<std::endl;
}
