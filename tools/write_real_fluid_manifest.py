#!/usr/bin/env python3
"""Record implemented physical models and explicit data/capability gaps."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import cantera as ct
import yaml
from real_fluid_backend import RealFluidBackend

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def plain(value):
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if hasattr(value,'item'):return value.item()
    return value
def model_manifest(config,backend):
    raw=yaml.safe_load(Path(config).read_text());mechanism=Path(raw['mechanism'])
    gas=ct.Solution(str(mechanism),raw['gas-phase'],transport_model=None)
    # Cantera uses YAML 1.2; PyYAML 1.1 would parse the species name NO as False.
    byname={s.name:plain(s.input_data) for s in gas.species()}
    eos_missing=[];species=[];pair_overrides=[]
    for name,weight in zip(gas.species_names,gas.molecular_weights):
        s=byname[name];eos=s.get('equation-of-state')
        if not isinstance(eos,dict) or not all(k in eos for k in ('a','b','acentric-factor')):eos_missing.append(name)
        if isinstance(eos,dict) and 'binary-a' in eos:pair_overrides.append({'species':name,'binary-a':eos['binary-a']})
        species.append({'name':name,'molecular_weight_kg_per_kmol':weight,'composition':s['composition'],
                        'standard_state':s.get('thermo'),'eos':eos,'transport':s.get('transport')})
    liquids=[];inputs={str(mechanism):sha(mechanism)}
    for entry in raw['condensables']:
        file=Path(entry.get('mechanism',mechanism));inputs[str(file)]=sha(file)
        liquid=ct.Solution(str(file),entry['phase'],transport_model=None);selected=plain(liquid.species(0).input_data)
        liquids.append({'configuration':entry,'species_data':selected,'mixing':'immiscible pure liquids'})
    reaction_types=Counter(r.reaction_type for r in gas.reactions())
    result={'configuration':str(Path(config).resolve()),'configuration_sha256':sha(config),
        'physicalModelHash':backend.physical_hash,'numericalPolicyHash':backend.policy_hash,
        'legacyCheckpointFingerprint':backend.fingerprint,'identity_scope':'thermodynamic model; a bound case also includes closure/chemistry/transport settings',
        'cantera':ct.__version__,'eos':gas.thermo_model,'gas_phase':raw['gas-phase'],'species':species,
        'mechanism_files':inputs,'explicit_binary_overrides':pair_overrides,
        'mixing_rule':'Cantera Peng-Robinson including every explicit binary-a override' if gas.thermo_model=='Peng-Robinson' else 'ideal gas',
        'default_pair_semantics':'Unspecified PR binary pairs use the selected Cantera geometric mixing rule; missing measured pair data is not a claim of experimental accuracy',
        'liquid_phases':liquids,'energy_reference':'unchanged species NASA formation+sensible internal energy; no added reaction/latent source',
        'entropy_reference':'unchanged NASA reference entropy and reference pressure, full EOS/mixing contribution',
        'reference_pressure_Pa':gas.reference_pressure,'reactions':gas.n_reactions,'reaction_types':dict(reaction_types),
        'reaction_definitions':'Complete self-contained input files identified by SHA256; no reactions removed',
        'reverse_rate_policy':'Selected Cantera mechanism semantics (reversible/irreversible and equilibrium constants unchanged)',
        'domain':{k:raw[k] for k in ('temperature-min','temperature-max','pressure-min','pressure-max')},
        'transport_scope':'Runtime-prescribed constant viscosity/conductivity; common-D Fick only for ideal gas; nonideal diffusion unsupported',
        'capabilities':backend.capabilities(),
        'high_pressure_detailed_combustion_gaps':{'species_without_explicit_PR_data':eos_missing,
            'missing_models':['validated high-pressure mixture transport/correlations and validity domain','nonideal multicomponent diffusion mobility/pressure/thermal terms'],
            'binary_pair_data':'Only listed overrides are explicit inputs; unprovided measured interactions are not inferred or invented',
            'production_support':False}}
    return plain(result)
def write_manifests(directory,library,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);models=[]
    for name in ('cold-pr-config.yaml','chemistry-config.yaml','reactive-dilute-config.yaml'):
        config=Path(directory)/name
        with RealFluidBackend(config,library) as b:models.append(model_manifest(config,b))
    for name,value in [('physical-model-manifest.json',{'schema':1,'models':models}),
                       ('capability-matrix.json',{'schema':1,'cases':[{'configuration':m['configuration'],'physicalModelHash':m['physicalModelHash'],**m['capabilities']} for m in models]})]:
        path=output/name
        if path.exists():raise FileExistsError('Refusing to overwrite '+str(path))
        path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    return models
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--thermo-dir',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();models=write_manifests(a.thermo_dir.resolve(),a.library.resolve(),a.output.resolve())
    print(json.dumps([{'model':m['configuration'],'physicalModelHash':m['physicalModelHash'],'species':len(m['species']),
                       'missing_PR_species':len(m['high_pressure_detailed_combustion_gaps']['species_without_explicit_PR_data'])} for m in models],indent=2))
