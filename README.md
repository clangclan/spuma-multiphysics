# ReactiveFoam

**ReactiveFoam** is a general-purpose compressible multiphysics research solver with selectable reaction, phase-change and transport models.

See the [Korean project guide](README.ko.md) for setup, the default `./Allwmake` build and execution, and the [solver documentation](README.reactive-phase.ko.md) for physics switches and limitations.

ReactiveFoam supports selectable chemistry, phase change, molecular viscosity, heat conduction and species diffusion with CPU/CUDA transport. This development branch adds CUDA WALE stress and heat/species mixing, and an experimental diffuse-interface capillary/flashing integration with conserved bulk-plus-surface energy. The single-liquid capillary path runs geometry, transport, curved UV recovery and WALE PR species-enthalpy preparation on CUDA; static-drop spurious-current convergence still fails, so it is not a validated primary-breakup solver. TCI and MPI domain decomposition remain unimplemented. An opt-in [Cartesian implicit runtime model](reports/implicit-static-drop-20260923.ko.md) now reconstructs a shared C² interface from liquid volumes on CUDA and uses its apertures and integrated tractions in actual RK transport. It remains experimental; the default diffuse model is unchanged. See the [capillary model and limits](docs/spray-physics/capillary-flashing.ko.md) and [static-drop review](docs/spray-physics/capillary-static-balance-review.md).

Older reports and results retain their original validation scope; ColdFoam results do not describe the current solver. Historical source remains available in Git history.

Current development includes the [40³/80³/160³ N₂O benchmark](reports/impinging-cube80-z40-adaptive-1us-20260923.ko.md), [HEM batching and exact-reuse optimization](reports/hem-utilization-optimization-20260924.ko.md), and [current-solver precision experiments](reports/current-solver-precision-20260924.ko.md). FP64 remains the default. Source files, public APIs and build variables now use generic `reactive` names; see the [naming migration guide](docs/generic-naming.ko.md) before rebuilding external libraries or reusing old commands.
