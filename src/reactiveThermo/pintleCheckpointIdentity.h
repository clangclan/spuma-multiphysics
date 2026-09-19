// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef PINTLE_CHECKPOINT_IDENTITY_H
#define PINTLE_CHECKPOINT_IDENTITY_H
#include <string>
#include <stdexcept>
#include <algorithm>
struct PintleCheckpointIdentity {
    int schema=1;long long speciesCount=-1;
    std::string fingerprint,closure,physicalModelHash,numericalPolicyHash;
};
inline bool pintleIdentityHash(const std::string& s) {
    return s.size()==64&&std::all_of(s.begin(),s.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');});
}
// Current numerical settings have already passed the solver's strict parsers.
// Identity-only old checkpoints cannot describe an old policy's individual
// fields: report both hashes, never fabricate a detailed policy delta.
inline bool pintleValidateIdentity(const PintleCheckpointIdentity& old,const PintleCheckpointIdentity& now) {
    if(old.schema!=1&&old.schema!=2)throw std::runtime_error("Unsupported checkpoint identity schema");
    if(old.schema==2&&(!pintleIdentityHash(old.fingerprint)||!pintleIdentityHash(old.physicalModelHash)
        ||!pintleIdentityHash(old.numericalPolicyHash)||old.speciesCount<=0||old.closure.empty()))
        throw std::runtime_error("Schema 2 requires valid fingerprint/speciesCount/closure/physicalModelHash/numericalPolicyHash");
    const auto closure=old.schema==1&&old.closure.empty()?"HEM":old.closure;
    if(old.fingerprint!=now.fingerprint||old.speciesCount!=now.speciesCount||closure!=now.closure)
        throw std::runtime_error("Legacy fingerprint/species/closure mismatch; explicit migration required");
    if((old.schema==2||!old.physicalModelHash.empty())&&old.physicalModelHash!=now.physicalModelHash)
        throw std::runtime_error("Restart physical model hash differs");
    return old.schema==2&&old.numericalPolicyHash!=now.numericalPolicyHash;
}
#endif
