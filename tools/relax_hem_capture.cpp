// SPDX-License-Identifier: GPL-3.0-or-later
// Explicit tolerance experiment on an immutable solver-input capture copy.
// This is not a checkpoint and never rewrites solver restart identities.
#include "../src/reactiveThermo/pintleDeviceFlash.h"
#include <fstream>
#include <iterator>
#include <vector>
#include <cstring>
#include <iostream>
#include <cassert>
int main(int argc,char** argv){
    assert(argc==4);std::ifstream in(argv[1],std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)),{});assert(bytes.size()>64);
    uint64_t header[8];std::memcpy(header,bytes.data(),64);
    assert(header[0]==0x48454d4341503031ULL&&header[2]==sizeof(PintleDeviceFlash::Model));
    PintleDeviceFlash::Model model;std::memcpy(&model,bytes.data()+64,sizeof(model));
    const double tolerance=std::stod(argv[3]);assert(tolerance>=1e-7&&tolerance<=1e-3);
    std::cout<<"old tolerances "<<model.vtol<<" "<<model.etol<<" "<<model.mutol<<"; new "<<tolerance<<std::endl;
    model.vtol=model.etol=model.mutol=tolerance;std::memcpy(bytes.data()+64,&model,sizeof(model));
    std::ofstream out(argv[2],std::ios::binary);out.write(bytes.data(),bytes.size());assert(out.good());
}
