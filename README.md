# ReactiveFoam

The default solver is **ReactiveFoam**, based on `main` with the validated physics-selection change (`96a781c`, PR #2). ColdFoam and its dedicated build, case-generation and benchmark tools have been removed.

See the [Korean project guide](README.ko.md) for setup, the default `./Allwmake` build and execution, and the [solver documentation](README.reactive-phase.ko.md) for physics switches and limitations.

ReactiveFoam supports selectable chemistry, phase change, molecular viscosity, heat conduction and species diffusion with CPU/CUDA transport. Surface tension, droplet breakup, turbulence closures and MPI domain decomposition are not implemented.

Older reports and results retain their original validation scope; ColdFoam results do not describe the current solver. Historical source remains available in Git history.
