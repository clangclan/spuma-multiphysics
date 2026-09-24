// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef REACTIVE_CALORIC_FAST_H
#define REACTIVE_CALORIC_FAST_H
// Exact reorganization of Cantera 3.2 NASA reference calorics and PR mixing.
// No mechanism coefficients, pair corrections or residual energy are omitted.
struct ReactiveCaloricData {
    struct Species {int polynomial=0;double gasConstant=0;std::vector<ReactiveGasThermoRegion> regions;};
    struct Pair {size_t i,j;double delta;};
    std::vector<Species> nasa;Vector boundaries,a,b,A,B;std::vector<Pair> pairs;
    bool supported=false,pr=false;
    explicit ReactiveCaloricData(Cantera::ThermoPhase& gas) {
        try {
            require(gas.type()=="ideal-gas"||gas.type()=="Peng-Robinson","Unsupported fast caloric EOS");
            pr=gas.type()=="Peng-Robinson";require(!pr||Cantera::version()=="3.2.0","Fast PR coefficients audited against Cantera 3.2.0 only");
            const size_t n=gas.nSpecies();nasa.resize(n);
            for(size_t k=0;k<n;++k) {
                const auto thermo=gas.species(k)->thermo;Vector c(thermo->nCoeffs());
                double lo,hi,p;size_t index;int type;thermo->reportParameters(index,type,lo,hi,p,c.data());
                for(double x:c)require(std::isfinite(x),"Invalid fast NASA data");
                auto& rec=nasa[k];rec.gasConstant=Cantera::GasConstant/gas.molecularWeight(k);
                auto append=[&](double l,double h,const double* coeff,size_t count){ReactiveGasThermoRegion r{};
                    r.minimumTemperature=l;r.maximumTemperature=h;std::copy(coeff,coeff+count,r.coefficient);rec.regions.push_back(r);};
                if(type==NASA1){require(c.size()==7,"NASA1 layout");rec.polynomial=7;append(lo,hi,c.data(),7);}
                else if(type==NASA2){require(c.size()==15,"NASA2 layout");rec.polynomial=7;append(lo,c[0],c.data()+8,7);append(c[0],hi,c.data()+1,7);}
                else if(type==NASA9||type==NASA9MULTITEMP){require(c.size()>=12&&(c.size()-1)%11==0&&c[0]==double((c.size()-1)/11),"NASA9 layout");
                    rec.polynomial=9;for(size_t j=1;j<c.size();j+=11)append(c[j],c[j+1],c.data()+j+2,9);}
                else throw std::runtime_error("Unsupported reference caloric representation");
                for(size_t j=1;j<rec.regions.size();++j)boundaries.push_back(rec.regions[j].minimumTemperature);
            }
            std::sort(boundaries.begin(),boundaries.end());boundaries.erase(std::unique(boundaries.begin(),boundaries.end()),boundaries.end());
            if(pr) {
                a.resize(n);b.resize(n);A.resize(n);B.resize(n);
                std::vector<Cantera::AnyMap> data(n);
                for(size_t k=0;k<n;++k) {
                    // Serialize the INSTALLED coefficients, including binary-a.
                    // No interpreting binary-a as dimensionless kij.
                    gas.getSpeciesParameters(gas.speciesName(k),data[k]);data[k].applyUnits();
                    auto& eos=data[k]["equation-of-state"].getMapWhere("model","Peng-Robinson");
                    a[k]=eos.convert("a","Pa*m^6/kmol^2");b[k]=eos.convert("b","m^3/kmol");
                    const double w=eos["acentric-factor"].asDouble();
                    require(a[k]>0&&b[k]>0&&std::isfinite(a[k])&&std::isfinite(b[k])&&std::isfinite(w),"Unsupported PR coefficient");
                    const double kap=w<=.491?.37464+1.54226*w-.26992*w*w:.374642+1.487503*w-.164423*w*w+.016666*w*w*w;
                    const double Tc=a[k]*7.77960739038885E-02/(b[k]*4.5723552892138218E-01*Cantera::GasConstant);
                    A[k]=1+kap;B[k]=kap/std::sqrt(Tc);
                }
                for(size_t i=0;i<n;++i){auto& eos=data[i]["equation-of-state"].getMapWhere("model","Peng-Robinson");
                    if(eos.hasKey("binary-a")){auto& bin=eos["binary-a"].as<Cantera::AnyMap>();
                        for(const auto& item:bin){const size_t j=gas.speciesIndex(item.first);require(j<n,"PR pair species missing");
                            if(j>i)pairs.push_back({i,j,bin.convert(item.first,"Pa*m^6/kmol^2")-std::sqrt(a[i]*a[j])});}}
                }
            }
            supported=true;
        }catch(const std::exception&){supported=false;} // same-EOS public fallback
    }
    // Polynomial in sqrt(T), exact while alpha square-root signs are fixed.
    std::array<double,3> mixingPolynomial(const Vector& x,double T) const {
        const double t=std::sqrt(T);double s0=0,s1=0;Vector u(A.size()),v(A.size());
        for(size_t k=0;k<A.size();++k){const double f=A[k]-B[k]*t;
            require(std::abs(f)>1e-10,"PR alpha cusp requires reference evaluation");const double sign=f>0?1:-1;
            u[k]=sign*A[k];v[k]=-sign*B[k];s0+=x[k]*std::sqrt(a[k])*u[k];s1+=x[k]*std::sqrt(a[k])*v[k];}
        std::array<double,3> c{s0*s0,2*s0*s1,s1*s1};
        // Exact correction tiles: dense corrections retain every supplied pair.
        for(size_t begin=0;begin<pairs.size();begin+=1024)for(size_t p=begin;p<std::min(pairs.size(),begin+1024);++p){
            const auto& z=pairs[p];const double w=2*x[z.i]*x[z.j]*z.delta;
            c[0]+=w*u[z.i]*u[z.j];c[1]+=w*(u[z.i]*v[z.j]+v[z.i]*u[z.j]);c[2]+=w*v[z.i]*v[z.j];}
        return c;
    }
};
struct ReactiveFixedCaloric {
    const ReactiveCaloricData& d;ReactiveCostProfileV21& cost;Vector Y,x;
    std::vector<size_t> active;Vector boundaries;
    double W=0,b=0,r=0;std::array<double,8> coeff{};std::array<double,3> mix{};
    size_t regionKey=SIZE_MAX;double alphaLo=0,alphaHi=HUGE_VAL;
    ReactiveFixedCaloric(const ReactiveCaloricData& data,const Vector& y,const Vector& weights,ReactiveCostProfileV21& stats)
        :d(data),cost(stats),Y(y),x(y.size()) {
        require(d.supported,"Unsupported fast caloric model");double inv=0;
        for(size_t k=0;k<y.size();++k)inv+=y[k]/weights[k];
        W=1/inv;r=Cantera::GasConstant/W;
        for(size_t k=0;k<y.size();++k){x[k]=y[k]*W/weights[k];if(d.pr)b+=x[k]*d.b[k];
            if(y[k]!=0){active.push_back(k);for(size_t i=1;i<d.nasa[k].regions.size();++i)boundaries.push_back(d.nasa[k].regions[i].minimumTemperature);}}
        std::sort(boundaries.begin(),boundaries.end());boundaries.erase(std::unique(boundaries.begin(),boundaries.end()),boundaries.end());
    }
    void prepare(double T) {
        const auto it=std::lower_bound(boundaries.begin(),boundaries.end(),T);
        const size_t key=2*size_t(it-boundaries.begin())+(it!=boundaries.end()&&*it==T);
        if(key!=regionKey){coeff.fill(0);for(size_t k:active){
            const auto& species=d.nasa[k];size_t region=0;
            while(region+1<species.regions.size()&&(species.polynomial==7?T>species.regions[region+1].minimumTemperature:T>=species.regions[region+1].minimumTemperature))++region;
            const double* a=species.regions[region].coefficient;const double w=Y[k]*species.gasConstant;
            if(species.polynomial==7){for(int j=0;j<5;++j)coeff[j+2]+=w*a[j];coeff[7]+=w*a[5];}
            else {for(int j=0;j<8;++j)coeff[j]+=w*a[j];}
            ++cost.nasaCoefficientVisits;
        }regionKey=key;++cost.nasaAggregateBuilds;}
        if(d.pr&&(T<=alphaLo||T>=alphaHi||alphaLo==0)) {
            mix=d.mixingPolynomial(x,T);alphaLo=0;alphaHi=HUGE_VAL;
            for(size_t k=0;k<d.A.size();++k)if(d.B[k]!=0&&d.A[k]/d.B[k]>0){const double cross=std::pow(d.A[k]/d.B[k],2);
                if(cross<T)alphaLo=std::max(alphaLo,cross);else alphaHi=std::min(alphaHi,cross);}
            // A negative lower sentinel denotes a valid open interval from 0.
            if(alphaLo==0)alphaLo=-1;
        }
    }
    struct Value {double energy,cv,p,dpdv,attraction,b,volume;bool uniqueRoot;};
    Value evaluate(double T,double rho) {
        prepare(T);++cost.energyCvEvaluations;++cost.nasaAggregateEvaluations;
        const double inv=1/T,T2=T*T,T3=T2*T,T4=T3*T;
        const double h0=-coeff[0]*inv+coeff[1]*std::log(T)+coeff[2]*T+coeff[3]*T2/2+coeff[4]*T3/3+coeff[5]*T4/4+coeff[6]*T4*T/5+coeff[7];
        const double cv0=coeff[0]*inv*inv+coeff[1]*inv+coeff[2]+coeff[3]*T+coeff[4]*T2+coeff[5]*T3+coeff[6]*T4-r;
        Value v{h0-r*T,cv0,rho*r*T,-rho*rho*r*T/W,0,b,W/rho,true};
        if(d.pr){++cost.exactMixingEvaluations;const double t=std::sqrt(T);
            const double a=mix[0]+mix[1]*t+mix[2]*T,at=mix[1]/(2*t)+mix[2],att=-mix[1]/(4*T*t);
            const double V=v.volume,den=V*V+2*V*b-b*b;
            require(V>b&&den>0,"Invalid PR covolume");const double root2=std::sqrt(2.);
            const double L=std::log((V+(1+root2)*b)/(V+(1-root2)*b))/(2*root2*b);
            v.energy+=(T*at-a)*L/W;v.cv+=T*att*L/W;
            v.p=Cantera::GasConstant*T/(V-b)-a/den;
            v.dpdv=-Cantera::GasConstant*T/((V-b)*(V-b))+2*a*(V+b)/(den*den);v.attraction=a;
            // Exact cubic discriminant certificate at this (EOS,Y,rho,T).
            // Negative discriminant with a relative margin means one real root.
            // Ambiguous/near-multiple roots retain Cantera densityCalc.
            const long double rt=Cantera::GasConstant*T/v.p,ap=a/v.p;
            const long double bn=(b-rt)/V,cn=-(2*rt*b-ap+3*b*b)/(V*V);
            const long double dn=(b*b*rt+b*b*b-ap*b)/(V*V*V);
            const long double terms[]={bn*bn*cn*cn,-4*cn*cn*cn,-4*bn*bn*bn*dn,-27*dn*dn,18*bn*cn*dn};
            long double disc=0,scale=0;for(auto z:terms){disc+=z;scale+=std::abs(z);}
            v.uniqueRoot=std::isfinite(disc)&&disc < -1e-10L*std::max(1e-30L,scale);
            // When there are three real roots, the other two may both lie
            // outside the admissible V>b domain. Deflate the known input root
            // only if its polynomial residual is small, with a strict margin.
            const long double residual=1+bn+cn+dn;
            const long double qb=bn+1,qd=qb*qb+4*dn;
            if(!v.uniqueRoot&&std::abs(residual)<=1e-13L*std::max(1.L,1+std::abs(bn)+std::abs(cn)+std::abs(dn))
                &&qd>=0&&std::isfinite(qd)) {
                const long double otherMax=(-qb+std::sqrt(qd))/2;
                v.uniqueRoot=otherMax < (long double)b/V-1e-8L*std::max(1.L,std::abs((long double)b/V));
            }
            if(v.uniqueRoot)++cost.branchCertificates;
        }
        require(std::isfinite(v.energy)&&std::isfinite(v.cv)&&v.cv>0&&std::isfinite(v.p)&&v.p>0&&v.dpdv<0,"Invalid minimal caloric state");return v;
    }
};
#endif
