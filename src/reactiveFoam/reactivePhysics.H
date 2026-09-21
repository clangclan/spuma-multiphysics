// SPDX-License-Identifier: GPL-3.0-or-later
// Resolve optional operators before allocating conserved fields or workers.
struct ReactivePhysics {
    bool chemistry,phaseChange,frozen,mechanical;
    double viscosity,conductivity,diffusivity;

    ReactivePhysics(const Foam::dictionary& dict,size_t liquids)
    {
        const Foam::dictionary empty;
        const auto& options=dict.found("physics")?dict.subDict("physics"):empty;
        for(const auto& key:options.toc())
            demand(key=="chemistry"||key=="combustion"||key=="phaseChange"||key=="viscosity"
                ||key=="heatConduction"||key=="speciesDiffusion",
                "Unknown physics option: "+std::string(key.c_str()));
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
