// SPDX-License-Identifier: GPL-3.0-or-later
class PersistentIOError : public std::runtime_error {public:using std::runtime_error::runtime_error;};
std::string jsonQuote(const std::string& value) {
    std::ostringstream out;out<<'"';
    for(unsigned char c:value){if(c=='"'||c=='\\')out<<'\\'<<char(c);
        else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;
        else out<<char(c);}
    out<<'"';return out.str();
}
void jsonNumber(std::ostream& out,double value) {
    if(std::isfinite(value))out<<value;
    else out<<jsonQuote(std::isnan(value)?"NaN":(value<0?"-Infinity":"Infinity"));
}
void jsonState(std::ostream& out,const ReactiveThermoState& s,const std::vector<int>& condensedKinds,
               const std::vector<std::string>& condensedNames) {
    out<<'{';bool comma=false;
#define STATE_VALUE(n) if(comma)out<<',';comma=true;out<<"\"" #n "\":";jsonNumber(out,s.n);
    STATE_VALUE(p) STATE_VALUE(T) STATE_VALUE(rho) STATE_VALUE(e) STATE_VALUE(entropy)
    STATE_VALUE(soundFrozen) STATE_VALUE(soundEquilibrium) STATE_VALUE(volumeResidual)
    STATE_VALUE(energyResidual) STATE_VALUE(chemicalResidual) STATE_VALUE(alphaGas)
    STATE_VALUE(rhoGas) STATE_VALUE(gasMass) STATE_VALUE(cp) STATE_VALUE(cv)
#undef STATE_VALUE
    out<<",\"activeLiquids\":"<<s.activeLiquids<<",\"iterations\":"<<s.iterations;
    for(int k=0;k<3;++k){const char* name=k==0?"liquidMass":k==1?"rhoLiquid":"alphaLiquid";
        const double* values=k==0?s.liquidMass:k==1?s.rhoLiquid:s.alphaLiquid;
        out<<','<<jsonQuote(name)<<":[";jsonNumber(out,values[0]);out<<',';jsonNumber(out,values[1]);out<<']';}
    out<<",\"condensedPhases\":[";
    for(size_t i=0;i<condensedNames.size();++i){if(i)out<<',';out<<"{\"slot\":"<<i
        <<",\"name\":"<<jsonQuote(condensedNames[i])<<",\"kind\":"<<jsonQuote(condensedKinds[i]?"solid":"liquid")
        <<",\"mass\":";jsonNumber(out,s.liquidMass[i]);out<<",\"density\":";jsonNumber(out,s.rhoLiquid[i]);
        out<<",\"alpha\":";jsonNumber(out,s.alphaLiquid[i]);out<<'}';}
    out<<']';
    out<<'}';
}
