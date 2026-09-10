// SPDX-License-Identifier: GPL-3.0-or-later
#include "rhoThermo.H"
#include "makeThermo.H"
#include "specie.H"
#include "PengRobinsonGas.H"
#include "hConstThermo.H"
#include "sensibleInternalEnergy.H"
#include "thermo.H"
#include "constTransport.H"
#include "heRhoThermo.H"
#include "pureMixture.H"
namespace Foam {
makeThermos(rhoThermo, heRhoThermo, pureMixture, constTransport,
    sensibleInternalEnergy, hConstThermo, PengRobinsonGas, specie);
}

#include "pintleDeviceHeRhoThermo.H"
#include "pintlePengRobinsonGas.H"
#include "perfectGas.H"
#include "pintleIpa.H"
#include "addToRunTimeSelectionTable.H"
namespace Foam {
makeThermos(rhoThermo,pintleDeviceHeRhoThermo,pureMixture,constTransport,
    sensibleInternalEnergy,hConstThermo,pintlePengRobinsonGas,specie);
makeThermos(rhoThermo,pintleDeviceHeRhoThermo,pureMixture,constTransport,
    sensibleInternalEnergy,hConstThermo,perfectGas,specie);
typedef pintleDeviceHeRhoThermo<rhoThermo,
    pureMixture<species::thermo<pintleIpa,sensibleInternalEnergy>>> pintleIpaRhoThermo;
defineTemplateTypeNameAndDebugWithName(pintleIpaRhoThermo,
    "pintleDeviceHeRhoThermo<pureMixture<pintleIpa,sensibleInternalEnergy>>",0);
addToRunTimeSelectionTable(basicThermo,pintleIpaRhoThermo,fvMesh);
addToRunTimeSelectionTable(fluidThermo,pintleIpaRhoThermo,fvMesh);
addToRunTimeSelectionTable(rhoThermo,pintleIpaRhoThermo,fvMesh);
}
