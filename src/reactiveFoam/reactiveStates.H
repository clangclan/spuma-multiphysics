// SPDX-License-Identifier: GPL-3.0-or-later
// Active state storage. CellState remains the schema-3 mechanical wire layout.
#ifndef REACTIVE_ACTIVE_STATES_H
#define REACTIVE_ACTIVE_STATES_H
#include "pintleReactiveThermo.h"
#include <vector>
#include <stdexcept>
struct CellState : PintleThermoState {PintleMechanicalState mechanical{};};
class States {
    bool mechanical_;
    std::vector<PintleThermoState> thermo_;
    std::vector<CellState> full_;
public:
    explicit States(size_t count=0,bool mechanical=false):mechanical_(mechanical),
        thermo_(mechanical?0:count),full_(mechanical?count:0){}
    size_t size()const{return mechanical_?full_.size():thermo_.size();}
    bool isMechanical()const{return mechanical_;}
    PintleThermoState& operator[](size_t c){
        return mechanical_?static_cast<PintleThermoState&>(full_[c]):thermo_[c];
    }
    const PintleThermoState& operator[](size_t c)const{
        return mechanical_?static_cast<const PintleThermoState&>(full_[c]):thermo_[c];
    }
    CellState& full(size_t c){
        if(!mechanical_)throw std::logic_error("Mechanical storage is not active");
        return full_[c];
    }
    const CellState& full(size_t c)const{
        if(!mechanical_)throw std::logic_error("Mechanical storage is not active");
        return full_[c];
    }
    PintleMechanicalState& mechanical(size_t c){return full(c).mechanical;}
    const PintleMechanicalState& mechanical(size_t c)const{return full(c).mechanical;}
};
#endif
