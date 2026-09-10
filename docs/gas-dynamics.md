# Pure-gas compressible verification path

`spumaPintleColdFoam` has an opt-in pressure/energy coupling for a uniform pure gas embedded in its three-phase data structure. The default three-phase VOF equations are unchanged. This path does **not** implement compressible liquid IPA, phase change, a mixture sound speed, or shock–material-interface physics.

Supported tests use a fixed mesh, zero gravity, laminar flow, global Euler time integration, and `hConst`/`const` transport with either `perfectGas` or `pintlePengRobinsonGas`. One selected phase must be exactly one; the other two must be zero in cells and at boundaries. The benchmark generators give all three phase thermos the selected gas properties because dormant phase models are still evaluated by the mixture class. Use this deliberate pure-gas embedding; an absent liquid's out-of-range thermo is not a gas verification model.

Example `controlDict` entries:

```foam
pintleGasDynamics true;
pintleGasPhase n2o;
pintleGasResidualTolerance 1e-4;
pintleStrictState true;
adjustTimeStep yes;
maxWaveCo 0.5;
```

Keep the existing three phase names. Use `nOuterCorrectors 4` as a starting point for smooth waves and weak shocks; the tested Sod discontinuity required 8 at the chosen timestep. The physical residual check is the acceptance criterion. A low linear-solver residual alone does not establish convergence.

The pressure matrix includes a nonsymmetric density-transport correction. Use an applicable solver such as GAMG or PBiCGStab. The case needs these convection schemes (the supplied generators use upwind):

```foam
div(phi,rho)      Gauss upwind;
div(phid,p_rgh)   Gauss upwind;
div(rhoPhi,U)     Gauss upwind;
div(rhoPhi,T)     Gauss upwind;
div(rhoPhi,e)     Gauss upwind;
div(rhoPhi,K)     Gauss upwind;
div(phi,p)        Gauss upwind;
```

The energy equation solves a defect correction in temperature whose converged residual is

\[
\partial_t(\rho e)+\nabla\!\cdot(\rho\mathbf U e)
+\partial_t(\rho K)+\nabla\!\cdot(\rho\mathbf U K)
+\nabla\!\cdot(p\mathbf U)-\nabla\!\cdot(k\nabla T)
-\nabla\!\cdot(\tau\cdot\mathbf U)=0.
\]

Here `e=(Cp0-R)*T+eDeparture`, with the exact PR departure coefficients. This energy reference differs from the legacy thermo's sensible-energy reference. Pressure, momentum, energy, and final mass/energy diagnostics share the refreshed `rhoPhi`. The `phi` field is a volume flux; `rhoPhi` is a mass flux. A `rhoPimpleFoam` mass-flux `phi` must never be copied directly into this solver's `phi`. For a matched restart using upwind density transport, supply it instead as `pintleInitialMassFlux` (with the object header renamed). The gas path converts it to volumetric flux using donor-cell density. The continuation helper performs this mapping and keeps the original files unchanged. Both `phi` and `rhoPhi` are written for inspection in gas mode.

`pintleGasEOS.H` evaluates the isothermal density derivatives, heat capacities, the constant-pressure temperature derivative of internal energy, and the isentropic sound speed. It keeps legacy `thermo.psi()` and its `rho/p` meaning intact. Positive mechanical compressibility and heat capacity are necessary checks; they do not establish vapor–liquid equilibrium or validity near a saturation boundary.

For postprocessing, compute Mach number as `mag(U)/gasSoundSpeed`, including for PR gas.

`PINTLE_GAS_BALANCE` reports the maximum local mass and energy residuals, scaled by one physical timestep and positive density/thermal energy scales. The default local limit is `1e-4`. If pressure would hit `pMin`, this path aborts before clipping. `PINTLE_GAS` reports EOS Mach number and the wave Courant number. With `adjustTimeStep yes`, the global timestep also obeys `maxWaveCo`. With fixed timesteps, it is a diagnostic. The all-face wave definition includes nonempty wall-normal directions; the one-dimensional benchmark's axial characteristic CFL is separately reported.

Boundary conditions must reflect the normal characteristic directions. The standard tests use periodic acoustic boundaries, prescribed supersonic inflow plus prescribed subsonic outlet pressure for a stationary shock, and ideal-gas stagnation boundaries for an initialized nozzle profile. PR gas rejects stock `totalPressure`, `totalTemperature`, and `waveTransmissive` in this path because their ideal-gas wave/stagnation assumptions have not been replaced. No generic real-gas nonreflecting boundary has been implemented.

To run a benchmark:

```bash
python3 tools/run_supersonic_case.py --output benchmarks/my-pr-wave \
  --kind acoustic --gas n2o --cells 160 --steps 200 --outer 4 --new-physics
```

Each output directory must be new. Runs acquire the shared benchmark lock and save inputs, executable/library hashes, analyzer/reference hashes, logs, final fields, numerical metrics, and explicit pass/fail checks. A completed solver run can still fail physics acceptance. Coarse-grid damping failures and insufficient-iteration failures are retained as evidence.

The `nozzle` test preserves an initialized quasi-one-dimensional transonic solution. It does not establish choking selection under a back-pressure sweep or validate a resolved multidimensional nozzle. Results and the tested limits are recorded in the supersonic validation report.
