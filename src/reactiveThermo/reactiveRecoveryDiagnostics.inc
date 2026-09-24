// SPDX-License-Identifier: GPL-3.0-or-later
// Fixed-capacity, per-model trace; no allocations in successful EOS probes.
std::string recoveryQuote(const std::string& text) {
    std::ostringstream out;out<<'"';
    for(unsigned char c:text) {if(c=='"'||c=='\\')out<<'\\'<<char(c);
        else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;
        else out<<char(c);}
    out<<'"';return out.str();
}
const char* recoveryCategory(const std::string& e) {
    if(e.find("No stable UV flash candidate")!=std::string::npos)return "stability_or_search_exhausted";
    if(e.find("non-finite")!=std::string::npos||e.find("Non-finite")!=std::string::npos)return "nonfinite_input_or_state";
    if(e.find("Negative")!=std::string::npos||e.find("CFL")!=std::string::npos)return "positivity_or_cfl";
    if(e.find("domain")!=std::string::npos||e.find("solid-state")!=std::string::npos)return "eos_domain";
    if(e.find("root")!=std::string::npos||e.find("branch")!=std::string::npos||e.find("critical")!=std::string::npos)return "candidate_branch_unavailable";
    if(e.find("Jacobian")!=std::string::npos||e.find("derivative")!=std::string::npos)return "derivative_or_jacobian";
    if(e.find("line search")!=std::string::npos)return "line_search";
    if(e.find("iteration limit")!=std::string::npos)return "iteration_limit";
    if(e.find("stable")!=std::string::npos||e.find("stability")!=std::string::npos)return "stability_or_search_exhausted";
    return "other";
}
struct RecoveryTrace {
    struct Candidate {
        int mask=0,dimension=0,iteration=0,phase=-99,branch=-99;
        bool adaptive=false,accepted=false;
        double seed=0,p=0,T=0,density=0;
        std::array<double,4> initialX{},x{},residual{};
        std::array<char,512> reason{};
    };
    bool enabled=false;size_t count=0,omitted=0;
    std::array<Candidate,64> candidates{};
    Candidate* current=nullptr;
    void begin(int mask,double seed,bool adaptive) {
        if(!enabled)return;
        if(count==candidates.size()){current=nullptr;++omitted;return;}
        current=&candidates[count++];*current={};current->mask=mask;current->seed=seed;current->adaptive=adaptive;
    }
    void initial(const Eigen::VectorXd& x) {
        if(!current)return;
        current->dimension=x.size();for(int j=0;j<x.size();++j){current->initialX[j]=x[j];current->x[j]=x[j];current->residual[j]=std::numeric_limits<double>::quiet_NaN();}
    }
    void iterate(const Eigen::VectorXd& x,const Eigen::VectorXd& f,int iteration) {
        if(!current)return;
        current->dimension=x.size();current->iteration=iteration;
        for(int j=0;j<x.size();++j){current->x[j]=x[j];current->residual[j]=f[j];}
    }
    void finish(bool accepted,const std::string& reason) {
        if(!current)return;
        current->accepted=accepted;
        std::snprintf(current->reason.data(),current->reason.size(),"%s",reason.c_str());current=nullptr;
    }
    static void number(std::ostream& out,double v) {if(std::isfinite(v))out<<v;else out<<"null";}
    std::string json() const {
        std::ostringstream out;out<<std::setprecision(17)<<"{\"schema\":1,\"candidatesOmitted\":"<<omitted<<",\"candidates\":[";
        for(size_t i=0;i<count;++i) {
            const auto& c=candidates[i];if(i)out<<',';
            out<<"{\"activeMask\":"<<c.mask<<",\"seed\":";number(out,c.seed);
            out<<",\"coordinates\":"<<recoveryQuote(c.adaptive?"logP_logT_logVaporFraction":"logP_logT_liquidFraction")
               <<",\"adaptive\":"<<(c.adaptive?"true":"false")<<",\"iteration\":"<<c.iteration
               <<",\"accepted\":"<<(c.accepted?"true":"false")<<",\"category\":"<<recoveryQuote(recoveryCategory(c.reason.data()))
               <<",\"reason\":"<<recoveryQuote(c.reason.data())<<",\"initialX\":[";
            for(int j=0;j<c.dimension;++j){if(j)out<<',';number(out,c.initialX[j]);}
            out<<"],\"x\":[";
            for(int j=0;j<c.dimension;++j){if(j)out<<',';number(out,c.x[j]);}
            out<<"],\"residual\":[";for(int j=0;j<c.dimension;++j){if(j)out<<',';number(out,c.residual[j]);}
            out<<"],\"lastEosProbe\":{\"phase\":"<<c.phase<<",\"branch\":"<<c.branch<<",\"p\":";number(out,c.p);
            out<<",\"T\":";number(out,c.T);out<<",\"density\":";number(out,c.density);out<<"}}";
        }
        out<<"]}";return out.str();
    }
};
