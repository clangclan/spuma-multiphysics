# WALE tensor helper: research basis and capability boundary

## Implemented model

`src/reactiveTransport/pintleWale.h` implements the local tensor part of the
Wall-Adapting Local Eddy-viscosity (WALE) model. For the row-major resolved
velocity gradient \(g_{ij}=\partial \widetilde u_i/\partial x_j\), it forms

\[
S=\operatorname{symm}(g),\qquad
S^d=\operatorname{dev}(\operatorname{symm}(g g))
\]

and evaluates

\[
\nu_t=(C_w\Delta)^2
\frac{(S^d:S^d)^{3/2}}
{(S:S)^{5/2}+(S^d:S^d)^{5/4}}.
\]

The product is `g*g`, not `g*transpose(g)`. The helper is usable from ordinary
C++ and CUDA device code. It takes \(\Delta\) explicitly. A separate opt-in
helper computes \(\Delta=\sqrt[3]{V}\), so the mesh-width choice cannot be
mistaken for an intrinsic part of the tensor model.

The implementation first divides the gradient by its maximum absolute entry.
It constructs both invariants at that safe scale and uses the degree-one
homogeneity of \(\nu_t\) to restore the gradient magnitude. The final product
is assembled with binary mantissas and exponents. This prevents `g*g`,
`delta*delta`, and `Cw*Cw` from overflowing or underflowing when their combined
result is representable. The exact zero-gradient and zero-\(S^d\) limits return
zero. There is no denominator epsilon, viscosity floor, wall damping, or
maximum-viscosity clip. Nonfinite inputs, negative model parameters, and an
unrepresentable result fail explicitly and return NaN.

## Primary and official sources

- F. Nicoud and F. Ducros, “Subgrid-Scale Stress Modelling Based on the Square
  of the Velocity Gradient Tensor,” *Flow, Turbulence and Combustion* 62,
  183–200 (1999), [DOI 10.1023/A:1009995426001](https://doi.org/10.1023/A:1009995426001),
  [accessible PDF copy](https://gibbs.science/les/handouts/nicoud_ducros_1999.pdf).
  Equation 10 defines \(S^d=\operatorname{dev}(\operatorname{symm}(g^2))\),
  equation 13 gives the viscosity above, and the paper derives the
  \(O(y^3)\) near-wall behavior and exact pure-shear zero. It also states
  coordinate translation/rotation invariance. The paper calibrated
  \(C_w\) around 0.5 for its schemes and cases; that is evidence for those
  cases, not a universal coefficient.
- The OpenFOAM Foundation’s official version-13 implementation forms
  `dev(symm(gradU & gradU))` and recovers the same viscosity through its
  modeled SGS kinetic-energy expression. Its shipped coefficients are
  `Cw=0.325` and `Ck=0.094`:
  [OpenFOAM-13 WALE.C at the researched revision](https://github.com/OpenFOAM/OpenFOAM-13/blob/18870c24d21c6b982e2cdec27b2f59738cca5f90/src/MomentumTransportModels/momentumTransportModels/LES/WALE/WALE.C).
  OpenFOAM adds a dimensional `small` term inside its derived-`k` denominator;
  this standalone helper instead handles the mathematical zero separately and
  does not copy that numerical floor.
- WALE in the Pele suite is implemented in PeleC rather than PelePhysics.
  PeleC’s GPU-capable source explicitly builds `g2=g*g`, `S`, and the
  traceless symmetric `D`, uses cube-root Cartesian cell volume for the filter
  width, adds the deviatoric stress work to total energy, and supplies a
  turbulent heat-flux model:
  [PeleC LES.H at the researched revision](https://github.com/Pele-Suite/PeleC/blob/c3f925cc9f54b761b7d5a1efe9e5409c98ed3a3c/Source/LES.H#L423-L576).
  PeleC’s own source comments warn that its \(c_p\nabla T\) heat-flux
  approximation is unsuitable for real-gas and supercritical regimes:
  [PeleC LES.cpp](https://github.com/Pele-Suite/PeleC/blob/c3f925cc9f54b761b7d5a1efe9e5409c98ed3a3c/Source/LES.cpp#L1-L53).

These sources agree on the defining tensor and powers. Their coefficient,
filter, regularization, scalar-flux, and discretization choices are not
interchangeable validation data.

## Verification

`tools/wale_test.cpp` independently checks the direct unscaled formula at
moderate values, zero flow, exact laminar simple-shear zero, response to
rotation rate, coordinate-rotation invariance, degree-one gradient
homogeneity, \(O(y^3)\) wall scaling, cube-root-volume behavior, invalid
inputs, and representable results combining extreme gradients and filter
widths.

`tools/wale_cuda_test.cu` runs a bounded 34-case CUDA kernel and compares its
status and viscosity with the host result. It includes zero, simple shear,
near-wall, extreme-scale, deterministic general-tensor, and invalid cases.
The verified commands were:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror -fno-fast-math \
  tools/wale_test.cpp -o /tmp/wale_test
/tmp/wale_test

flock /home/jsw/cae-benchmark/run.lock \
  /home/jsw/nvidia/hpc_sdk/Linux_x86_64/26.5/compilers/bin/nvcc \
  -std=c++17 -O2 --fmad=false --ftz=false --prec-div=true \
  --prec-sqrt=true -gencode=arch=compute_120,code=sm_120 \
  -Xcompiler=-Wall,-Wextra,-Werror \
  tools/wale_cuda_test.cu -o /tmp/wale_cuda_test
flock /home/jsw/cae-benchmark/run.lock /tmp/wale_cuda_test
```

The CPU suite passed, and device parity passed on an NVIDIA GeForce RTX 5080.
This is tensor-kernel verification. It is not a turbulent-flow validation.

## Compressible Favre closure and energy requirements

For a compressible solver, `gradient` should be the gradient of resolved Favre
velocity \(\widetilde{\boldsymbol u}=\overline{\rho\boldsymbol u}/\bar\rho\).
The kinematic viscosity from this helper becomes dynamic SGS viscosity
\(\mu_t=\bar\rho\nu_t\). A viscous-stress convention can then use
\(2\mu_t\operatorname{dev}(S)\). The momentum flux and total-energy stress-work
flux must use the same face stress and sign; otherwise resolved kinetic-energy
dissipation is not conservatively converted within total energy.

The deviatoric WALE viscosity does not determine the isotropic SGS stress.
Omitting modeled SGS kinetic energy must remain explicit (`sgsK=none`). It also
does not close filtered heat/enthalpy or species fluxes. A full reacting LES
needs independently specified turbulent Prandtl/Schmidt closures, a
sum-zero species diffusion construction, consistent diffusive enthalpy work,
and real-fluid thermodynamic treatment. In particular, a universal
\(c_p\nabla T\) substitution is not justified across nonlinear real-fluid and
two-phase states. Filtered chemistry additionally needs a TCI model or direct
resolution argument; a finite-rate chemistry kernel alone does not supply it.

## Current capability boundary and known limits

- The current live scope is an experimental Favre/single-mixture deviatoric
  WALE stress with conservative total-energy stress work. It must be labeled
  `sgsScalarClosure=none;sgsK=none` and must not be described as full reacting
  LES.
- Applying \(\mu_t=\rho_{mix}\nu_t\) to cold, nonreacting HEM is a defensible
  single-velocity mixture closure. It does not model subgrid phase slip,
  interfacial turbulence, phase change, or density/interface-filter
  correlations. No liquid-volume-fraction damping is introduced.
- Cube-root cell volume is only one filter-width definition. Strongly
  anisotropic cells and mesh/filter changes require sensitivity studies.
- The present integrated boundary set has slip, extrapolated, and fixed-state
  boundaries, with owner-extrapolated \(\nu_t\); it has no SGS inflow model and
  no no-slip wall boundary. The helper’s analytic \(y^3\) test does not validate
  a solver wall treatment.
- The transport face-gradient correction is shared with the molecular
  viscous path. The live solver therefore restricts WALE cases to orthogonal,
  unskewed face geometry; general skew/nonorthogonal support needs a corrected
  gradient implementation and separate verification.
- WALE supplies dissipation for resolved velocity gradients. It does not add
  interface capturing, physical surface tension, sheet/ligament topology
  change, an Eulerian-liquid-to-droplet conversion rule, or droplet state.
  Therefore it cannot by itself claim liquid-jet/sheet breakup or droplet
  creation, which remain the primary spray-physics objective.
- Required flow-level evidence still includes laminar and transition cases,
  homogeneous turbulence decay/spectra, wall or jet mixing where supported,
  scalar-variance/SGS-dissipation budgets once scalar closures exist, and mesh
  and filter sensitivity with statistical uncertainty.
