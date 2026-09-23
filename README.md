# ReactiveFoam

The default solver is **ReactiveFoam**, based on `main` with the validated physics-selection change (`96a781c`, PR #2). ColdFoam and its dedicated build, case-generation and benchmark tools have been removed.

See the [Korean project guide](README.ko.md) for setup, the default `./Allwmake` build and execution, and the [solver documentation](README.reactive-phase.ko.md) for physics switches and limitations.

ReactiveFoam supports selectable chemistry, phase change, molecular viscosity, heat conduction and species diffusion with CPU/CUDA transport. This development branch adds CUDA WALE stress and heat/species mixing, and an experimental diffuse-interface capillary/flashing integration with conserved bulk-plus-surface energy. The single-liquid capillary path runs geometry, transport, curved UV recovery and WALE PR species-enthalpy preparation on CUDA; static-drop spurious-current convergence still fails, so it is not a validated primary-breakup solver. TCI and MPI domain decomposition remain unimplemented. A [GPU geometric face diagnostic](reports/geometric-balance-review-20260923.ko.md) now tests shared liquid apertures and integrated tractions; it does not enable a reconstructed geometric runtime model. See the [capillary model and limits](docs/spray-physics/capillary-flashing.ko.md) and [static-drop review](docs/spray-physics/capillary-static-balance-review.md).

Older reports and results retain their original validation scope; ColdFoam results do not describe the current solver. Historical source remains available in Git history.
