// SPDX-License-Identifier: GPL-3.0-or-later
// Active state storage. CellState remains the schema-3 mechanical wire layout.
#ifndef REACTIVE_ACTIVE_STATES_H
#define REACTIVE_ACTIVE_STATES_H
#include "reactiveThermo.h"
#include <vector>
#include <stdexcept>
struct CellState : ReactiveThermoState {ReactiveMechanicalState mechanical{};};
class States {
    bool mechanical_;
    std::vector<ReactiveThermoState> thermo_;
    std::vector<CellState> full_;
public:
    explicit States(size_t count=0,bool mechanical=false):mechanical_(mechanical),
        thermo_(mechanical?0:count),full_(mechanical?count:0){}
    size_t size()const{return mechanical_?full_.size():thermo_.size();}
    bool isMechanical()const{return mechanical_;}
    ReactiveThermoState& operator[](size_t c){
        return mechanical_?static_cast<ReactiveThermoState&>(full_[c]):thermo_[c];
    }
    const ReactiveThermoState& operator[](size_t c)const{
        return mechanical_?static_cast<const ReactiveThermoState&>(full_[c]):thermo_[c];
    }
    CellState& full(size_t c){
        if(!mechanical_)throw std::logic_error("Mechanical storage is not active");
        return full_[c];
    }
    const CellState& full(size_t c)const{
        if(!mechanical_)throw std::logic_error("Mechanical storage is not active");
        return full_[c];
    }
    ReactiveMechanicalState& mechanical(size_t c){return full(c).mechanical;}
    const ReactiveMechanicalState& mechanical(size_t c)const{return full(c).mechanical;}
};
#endif
