// SPDX-License-Identifier: GPL-3.0-or-later
// Resolve optional operators before allocating conserved fields or workers.
struct ReactivePhysics {
    bool chemistry,phaseChange,frozen,mechanical,wale,surfaceTension;
    double waleCw,turbulentPrandtl,turbulentSchmidt;
    double viscosity,conductivity,diffusivity;
    double sigma=0,capillaryCfl=.25,capillaryGeometryTolerance=1e-10;

    ReactivePhysics(const Foam::dictionary& dict,size_t liquids)
    {
        const Foam::dictionary empty;
        const auto& options=dict.found("physics")?dict.subDict("physics"):empty;
        for(const auto& key:options.toc())
            demand(key=="chemistry"||key=="combustion"||key=="phaseChange"||key=="viscosity"
                ||key=="heatConduction"||key=="speciesDiffusion"||key=="turbulence"
                ||key=="surfaceTension"||key=="turbulentHeatFlux"||key=="turbulentSpeciesMixing",
                "Unknown physics option: "+std::string(key.c_str()));
        surfaceTension=options.getOrDefault<Foam::Switch>("surfaceTension",false);
        chemistry=options.getOrDefault<Foam::Switch>("chemistry",dict.getOrDefault<Foam::Switch>("chemistry",false));
        if(options.found("combustion")) {
            const bool combustion=options.get<Foam::Switch>("combustion");
            demand(!options.found("chemistry")||combustion==chemistry,
                "combustion and chemistry must agree: reaction heat is part of conserved total energy");
            chemistry=combustion;
        }
        phaseChange=options.getOrDefault<Foam::Switch>("phaseChange",true)&&liquids>0;
        frozen=!phaseChange&&liquids>0;
        mechanical=dict.get<Foam::word>("closure")=="mechanicalEquilibrium";
        if(surfaceTension) {
            demand(liquids==1&&!chemistry&&!mechanical,
                "The diffuse capillary model requires one liquid and nonreacting HEM closure");
            demand(dict.getOrDefault<Foam::word>("transportBackend","cpu")=="cuda"
                &&dict.getOrDefault<Foam::word>("closureBackend","cpu")=="cuda"
                &&!dict.getOrDefault<bool>("closureCpuFallback",true),
                "Capillary coupling requires CUDA transport and closure with closureCpuFallback false");
            sigma=dict.get<Foam::scalar>("surfaceTensionCoefficient");
            capillaryCfl=dict.getOrDefault<Foam::scalar>("capillaryCfl",.25);
            capillaryGeometryTolerance=dict.getOrDefault<Foam::scalar>("capillaryGeometryTolerance",1e-10);
            demand(std::isfinite(sigma)&&sigma>0&&std::isfinite(capillaryCfl)&&capillaryCfl>0&&capillaryCfl<=.5
                &&std::isfinite(capillaryGeometryTolerance)&&capillaryGeometryTolerance>0,
                "Require positive constant surfaceTensionCoefficient, 0<capillaryCfl<=0.5 and positive geometry tolerance");
        }
        const Foam::word turbulence=options.getOrDefault<Foam::word>("turbulence","none");
        demand(turbulence=="none"||turbulence=="WALE","Unknown turbulence model (none or WALE)");
        wale=turbulence=="WALE";waleCw=dict.getOrDefault<Foam::scalar>("waleCw",0.325);
        demand(std::isfinite(waleCw)&&waleCw>=0,"Invalid WALE coefficient");
        demand(!wale||(!chemistry&&!mechanical),
            "Experimental WALE stress supports nonreacting HEM/frozen flow; SGS chemistry/mechanical coupling is not implemented");
        demand(!wale||dict.getOrDefault<Foam::word>("transportBackend","cpu")=="cuda",
            "ReactiveFoam WALE requires CUDA transport; portable CPU operator checks are available separately");
        auto scalarClosure=[&](const char* option,const char* coefficient,double fallback) {
            if(!options.getOrDefault<Foam::Switch>(option,false))return 0.0;
            demand(wale,std::string(option)+" requires turbulence WALE");
            const double value=dict.getOrDefault<Foam::scalar>(coefficient,fallback);
            demand(std::isfinite(value)&&value>0,std::string(option)+" requires a finite positive "+coefficient);
            return value;
        };
        turbulentPrandtl=scalarClosure("turbulentHeatFlux","turbulentPrandtl",0.85);
        turbulentSchmidt=scalarClosure("turbulentSpeciesMixing","turbulentSchmidt",0.7);
        demand(!frozen||turbulentSchmidt==0,
            "turbulentSpeciesMixing with frozen condensed phases requires matching condensed-mass fluxes; use phaseChange true or disable turbulentSpeciesMixing");
        demand(!(mechanical&&frozen),
            "phaseChange off with liquid mechanical environments is not implemented; use closure HEM for frozen-phase transport");
        auto coefficient=[&](const char* option,const char* legacy) {
            const double value=dict.getOrDefault<Foam::scalar>(legacy,0);
            const bool enabled=options.getOrDefault<Foam::Switch>(option,value!=0);
            if(!enabled)return 0.0;
            demand(std::isfinite(value)&&value>0,std::string(option)+" requires a finite positive "+legacy);
            return value;
        };
        viscosity=coefficient("viscosity","dynamicViscosity");
        conductivity=coefficient("heatConduction","thermalConductivity");
        diffusivity=coefficient("speciesDiffusion","molecularDiffusivity");
    }
};
