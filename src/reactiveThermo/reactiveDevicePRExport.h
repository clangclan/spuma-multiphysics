// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef REACTIVE_DEVICE_PR_EXPORT_H
#define REACTIVE_DEVICE_PR_EXPORT_H

// Host-only extraction of the installed Cantera 3.2 phase into the immutable,
// pointer-free table consumed by reactiveDevicePR.h.

#include "reactiveDevicePR.h"

#include "cantera/base/AnyMap.h"
#include "cantera/base/global.h"
#include "cantera/thermo/Species.h"
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/thermo/speciesThermoTypes.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace ReactiveDevicePR {

inline bool exportPhase(Cantera::ThermoPhase& phase, Table<>& output,
                        std::string& error)
{
    try {
        if (Cantera::version() != "3.2.0")
            throw std::runtime_error("Device PR table is audited for Cantera 3.2.0");
        if (phase.type() != "Peng-Robinson")
            throw std::runtime_error("Device PR export requires a Peng-Robinson phase");
        if (!phase.nSpecies() || phase.nSpecies() > MaxSpecies)
            throw std::runtime_error("Device PR export supports 1..16 species");

        Table<> table{};
        table.ns = static_cast<int>(phase.nSpecies());
        table.refPressure = phase.refPressure();
        std::vector<Cantera::AnyMap> data(table.ns);

        auto append = [](Species<>& species, double low, double high,
                         int polynomial, const double* coefficient) {
            if (species.regionCount >= DefaultMaxRegions)
                throw std::runtime_error("NASA region count exceeds device table capacity");
            auto& region = species.regions[species.regionCount++];
            region.minimumTemperature = low;
            region.maximumTemperature = high;
            region.polynomial = polynomial;
            std::copy(coefficient, coefficient + polynomial, region.coefficient);
        };

        for (int k = 0; k < table.ns; ++k) {
            auto& record = table.species[k];
            record.molecularWeight = phase.molecularWeight(k);
            const auto thermo = phase.species(k)->thermo;
            if (std::abs(thermo->refPressure() - table.refPressure)
                > 1e-12*std::max(1.0, table.refPressure)) {
                throw std::runtime_error("Species reference pressures are not uniform");
            }
            std::vector<double> c(thermo->nCoeffs());
            size_t index = 0;
            int type = 0;
            double low = 0, high = 0, reference = 0;
            thermo->reportParameters(index, type, low, high, reference, c.data());
            if (type == NASA1) {
                if (c.size() != 7) throw std::runtime_error("Invalid NASA7 layout");
                append(record, low, high, 7, c.data());
            } else if (type == NASA2) {
                if (c.size() != 15) throw std::runtime_error("Invalid NASA7 two-region layout");
                append(record, low, c[0], 7, c.data()+8);
                append(record, c[0], high, 7, c.data()+1);
            } else if (type == NASA9 || type == NASA9MULTITEMP) {
                if (c.size() < 12 || (c.size()-1)%11
                    || c[0] != static_cast<double>((c.size()-1)/11)) {
                    throw std::runtime_error("Invalid NASA9 multi-region layout");
                }
                for (size_t j = 1; j < c.size(); j += 11)
                    append(record, c[j], c[j+1], 9, c.data()+j+2);
            } else {
                throw std::runtime_error("Unsupported device reference caloric model");
            }

            phase.getSpeciesParameters(phase.speciesName(k), data[k]);
            data[k].applyUnits();
            auto& eos = data[k]["equation-of-state"].getMapWhere(
                "model", "Peng-Robinson");
            record.a = eos.convert("a", "Pa*m^6/kmol^2");
            record.b = eos.convert("b", "m^3/kmol");
            const double omega = eos["acentric-factor"].asDouble();
            record.kappa = omega <= .491
                ? .37464 + 1.54226*omega - .26992*omega*omega
                : .374642 + 1.487503*omega - .164423*omega*omega
                    + .016666*omega*omega*omega;
        }

        // Default geometric mixing. Explicit binary-a contains the actual
        // Cantera a_ij coefficient, not a dimensionless k_ij correction.
        for (int i = 0; i < table.ns; ++i)
            for (int j = 0; j < table.ns; ++j)
                table.aBinary[i][j] = std::sqrt(table.species[i].a*table.species[j].a);
        for (int i = 0; i < table.ns; ++i) {
            auto& eos = data[i]["equation-of-state"].getMapWhere(
                "model", "Peng-Robinson");
            if (!eos.hasKey("binary-a")) continue;
            auto& binary = eos["binary-a"].as<Cantera::AnyMap>();
            for (const auto& item : binary) {
                const size_t j = phase.speciesIndex(item.first);
                if (j >= static_cast<size_t>(table.ns))
                    throw std::runtime_error("binary-a references a species outside the phase");
                const double value = binary.convert(item.first, "Pa*m^6/kmol^2");
                table.aBinary[i][j] = table.aBinary[j][i] = value;
            }
        }
        const Status status = detail::validate(table);
        if (status != Status::Success)
            throw std::runtime_error("Exported device PR table failed validation");
        output = table;
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

} // namespace ReactiveDevicePR

#endif
