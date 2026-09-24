# Static-drop balance review (2026-09-23)

**Status: unresolved.** The integrated GPU capillary path conserves periodic total momentum and energy, but its stationary three-dimensional drop develops a maximum velocity that increases under grid refinement. It must not be described as a validated resolved-interface surface-tension solver on this evidence.

## Current discretization and observed failure

The independently transported liquid inventory supplies a cell color $c$. The face graph builds $g=\nabla_h c$, $n=-g/|g|$, curvature $\kappa=\nabla_h\cdot n$, surface-energy density $e_s=\sigma|g|$, and capillary stress $C=\sigma|g|(I-n\otimes n)$. The conserved energy is $E_{\mathrm{total}}=E_{\mathrm{bulk}}+e_s$. A contact-preserving Riemann calculation uses $\pi=p-\sigma\kappa_f c$, then adds $\sigma\kappa_f c_f I-C_f$ to its shared momentum face flux. Thus the physical face traction remains $pI-C_f$; every interior face updates its two cells with opposite momentum. The corresponding total-energy face flux represents $(E_{\mathrm{total}}+p)u-Cu$. There is no uncompensated cell body force.

The [reproducible solver validation](../../results/capillary-integration-20260923/solver-validation-v5.json) initializes a radius-1 mm liquid sphere in a 4 mm periodic cube, with $\sigma=0.01$ N/m and $p=p_0+(2\sigma/R)c$. Color is sampled with 8 points per cell axis. At the same physical time, 60 ns:

| Mesh | Cell width (mm) | Maximum velocity (m/s) | Final pressure-jump error (Pa) |
| --- | ---: | ---: | ---: |
| 16³ | 0.250 | 5.02×10⁻⁵ | 0.218 |
| 24³ | 0.167 | 8.34×10⁻⁵ | <10⁻⁹ |
| 32³ | 0.125 | 1.16×10⁻⁴ | <10⁻⁹ |

Global conservation and a nearly exact *mean* Laplace jump do not establish local static balance. The pressure profile varies across the one-cell-scale material interface, while the divergence of the reconstructed face stress does not discretely equal the gradient of that pressure profile. The resulting localized acceleration grows as the interface narrows with $h$; the maximum velocity follows that trend over this short run. The present first-order diffuse color transport and Gauss normal/curvature reconstruction do not supply a convergent static-drop force balance.

## Independent stencil diagnosis

An offline, periodic Cartesian reconstruction used the same saved initial color fractions and a centered approximation to the initial pressure/stress face divergence. It is a diagnostic of the spatial mismatch, **not** a replacement for the integrated GPU result. The current stress construction produced maximum initial acceleration of approximately 864, 1374, and 1894 m/s² on the 16³, 24³, and 32³ grids. Averaging cell stresses to faces gave approximately 794, 1305, and 2199 m/s². Reconstructing a face gradient with $(c_R-c_L)/d$ in the face-normal direction and averaged tangential gradients, then forming $C_f=\sigma(|g_f|I-g_f\otimes g_f/|g_f|)$, gave approximately 1211, 2837, and 5290 m/s². These simple stress interpolations do not reverse the trend.

Smoothing only the geometry worsened or left the mismatch because the test pressure still used the original sharp $c$. A fixed physical regularization width applied to *both* geometry and equilibrium pressure reduced the offline residual and gave a decreasing trend in some width choices (for example, about 106, 97, and 80 m/s² with a 0.30 mm binomial-filter standard deviation). That is evidence about the mismatch, not a validated fix: the actual EOS, flashing closure, material transport, surface-energy convention, and initial state would all have to use a defined and consistent regularization. Changing the fixture pressure alone would merely fit a test.

## Required resolution and gates

A credible next implementation needs a better representation of the moving material interface and curvature, such as geometric volume-fraction transport with height-function curvature on supported meshes, and a pressure/capillary-force discretization that balances a static Laplace jump on the *same* faces. The resulting capillary momentum update must remain conservative across each shared face. The energy update must still account for surface-area work, and flashing must move or consume the resolved liquid interface consistently. Balanced-force and height-function methods are established for incompressible volume tracking; adapting them to this compressible HEM/EOS and total-energy formulation requires its own derivation and tests [François et al. (2006)](https://doi.org/10.1016/j.jcp.2005.08.004), [Popinet (2009)](https://doi.org/10.1016/j.jcp.2009.04.042). The present diffuse-stress model follows the continuum-surface-force idea of [Brackbill, Kothe, and Zemach (1992)](https://doi.org/10.1016/0021-9991(92)90240-Y); that model alone does not guarantee discrete balance.

The static-drop gate should require decreasing parasitic velocity and converging Laplace pressure error over at least these three meshes, in addition to periodic momentum, species, and total-energy conservation. Capillary-wave and oscillating-drop comparisons need enough simulated time and samples to measure a period; the current 60 ns traces are too short. Flashing cases require a converged coupled surface-energy/phase closure and bounded liquid inventory. Until those gates pass, the current branch is reviewable as an integrated experimental implementation, with static-drop convergence explicitly failing.

## Review follow-up: production face operator isolation

The external 2026-09-23 review was a CPU prototype/design package, not a tested CUDA patch.
Its [provenance](../../results/capillary-review-20260923/review-provenance.json) is recorded.
The accepted consistency fixes use one owner-weighted face color for geometry/pressure restoration,
and one face velocity (normal component equal to the Riemann contact speed) for capillary traction work.
They preserve the shared-face momentum structure. They do **not** repair the uniform-mesh static balance.

The new [production face-flux diagnostic](../../tools/capillary_face_decomposition.cpp) calls the
actual graph geometry and `faceFlux`, separating Riemann pressure, capillary traction, mass flux,
and final cell momentum RHS before time integration. Density, sound speed and total energy are
synthetic fixtures: this isolates the spatial operator and is not another real-fluid solver run.
With analytic constant curvature, mass flux and the Riemann-pressure RHS are zero, while
maximum residual acceleration is 864, 1374, 1894, 3929 and 6788 m/s² on 16³, 24³, 32³, 48³ and 64³.
The nonzero residual therefore survives even without curvature error or time-stepping error.
Global momentum RHS remains approximately 10⁻²¹ N.

Increasing initial sphere quadrature from 8³ to 32³ samples per intersected cell gives
863, 1371 and 1879 m/s² on 16³/24³/32³, which does not reverse the trend.
Zero normals outside the active band remain a separate geometry issue: 624/1392/2400 faces
have only one active normal at those grids. The [full diagnostic data](../../results/capillary-review-20260923/face-decomposition.json)
also retain numerical-curvature errors and RHS decomposition roundoff.
Neither fixed quadrature nor curvature alone explains away the pressure/stress mismatch.

`validate_capillary_solver.py --require-static-convergence` now returns failure for growing
maximum or volume-weighted L2 velocity, missing refinements, or unequal physical conditions/times.
It reports interface-area-weighted curvature errors without hiding cells with poor curvature.
A passing velocity trend would still be only one gate; it does not certify dynamic area/work or breakup.


## Geometric pressure/traction follow-up (2026-09-23)

The [geometric proposal implementation report](../../reports/geometric-balance-review-20260923.ko.md)
adds a separate host/device `ReactiveGeometricCapillary::faceFlux` and the one-shot
`reactive_transport_geometric_diagnostic_v1` API. The latter uses the production
`Faces` and `Rhs` kernels with caller-supplied liquid face apertures and integrated
surface tractions. It reports cell volume mismatch, vector-area closure, traction
closure and geometric surface energy. It cannot install geometry into the stepper.

The new momentum restoration is `(DeltaP*A_liquid*n - T)/A_face`. HLLC uses
`E_total - sigma*A_interface/V`; the gather removes the corresponding advected
surface contribution before adding the separately supplied surface-energy flux.
Both restoration terms are replaced together. The old diffuse stress is not added.

Exact sphere geometry supplied to CPU and CUDA tests substantially reduces the
static operator residual. This is an analytic-geometry test, not a reconstruction
or time-integrated static-drop convergence result. Finite background-pressure
roundoff remains in the full finite-volume gather. A deliberately wrong pressure
jump leaves a nonzero force, and a separate inconsistent cell-pressure test leaves
a nonzero Riemann mass flux.

The VOF-column spline probe demonstrates compatible aperture/traction identities
on a restricted smooth graph patch, but excludes horizontal face crossings and
chart transitions and still has nonzero cell-volume defects. Current transport
mesh records do not contain the face polygons needed by a general geometric
reconstruction. The runtime remains on v1 until all-face, volume-constrained
reconstruction and consistent swept liquid/surface-energy transport exist.
Phase-change-induced interface motion needs its own area/energy contribution.
No analytic sphere parameters enter the production solver; no checkpoint or
physical-model identity is silently converted, and the static convergence gate
remains unresolved.
