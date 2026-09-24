#!/usr/bin/env python3
"""Write reproducible, small 3D Cartesian interface geometry for ReactiveFoam.

When given a real-fluid configuration, the CLI writes a runnable primitive
HEM case whose liquid inventory has the requested color as its volume fraction.
The solver derives interfaceColor from that inventory; the sampled target is
retained independently in initial-color.npz for audit.

The suite source must already define only liquid N2O, for example
examples/impinging-n2o-supercooled/cold-pr-148K-config.yaml. Run this script
with the project's reactive-env Python so the PR EOS and its libraries load.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import yaml

import benchmark as common
from real_fluid_backend import RealFluidBackend


@dataclass(frozen=True)
class CartesianGrid:
    shape: tuple[int, int, int]
    lengths: tuple[float, float, float]

    def __post_init__(self):
        if len(self.shape) != 3 or any(type(n) is not int or n < 4 for n in self.shape):
            raise ValueError("each Cartesian direction needs at least four cells")
        if len(self.lengths) != 3 or any(not math.isfinite(x) or x <= 0 for x in self.lengths):
            raise ValueError("domain lengths must be finite and positive")

    @property
    def cells(self) -> int:
        return math.prod(self.shape)

    @property
    def spacing(self) -> np.ndarray:
        return np.asarray(self.lengths) / np.asarray(self.shape)

    @property
    def centers(self) -> np.ndarray:
        # OpenFOAM's single structured hex block uses x-fastest cell ordering.
        i, j, k = np.meshgrid(*(np.arange(n) for n in self.shape), indexing="ij")
        return np.column_stack([a.ravel(order="F") for a in (i, j, k)]) * self.spacing + .5 * self.spacing


PATCHES = (
    ("x0", "x1", (0, 3, 7, 4), 0, -1),
    ("x1", "x0", (1, 5, 6, 2), 0, 1),
    ("y0", "y1", (0, 4, 5, 1), 1, -1),
    ("y1", "y0", (3, 2, 6, 7), 1, 1),
    ("z0", "z1", (0, 1, 2, 3), 2, -1),
    ("z1", "z0", (4, 7, 6, 5), 2, 1),
)


def put(case: Path, relative: str, body: str, klass: str = "dictionary") -> None:
    path = case / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(common.header(path.name, klass) + body)


def write_mesh(case: Path, grid: CartesianGrid) -> None:
    lx, ly, lz = grid.lengths
    vertices = ((0, 0, 0), (lx, 0, 0), (lx, ly, 0), (0, ly, 0),
                (0, 0, lz), (lx, 0, lz), (lx, ly, lz), (0, ly, lz))
    coords = " ".join("(" + " ".join(f"{x:.17g}" for x in v) + ")" for v in vertices)
    patches = []
    for name, neighbour, face, axis, sign in PATCHES:
        displacement = [0., 0., 0.]
        displacement[axis] = sign * grid.lengths[axis]
        patches.append(f"{name} {{ type cyclic; neighbourPatch {neighbour}; transform translational; "
                       f"separationVector ({' '.join(f'{x:.17g}' for x in displacement)}); "
                       f"faces (({' '.join(map(str, face))})); }}")
    put(case, "system/blockMeshDict", f"scale 1; vertices ({coords});\n"
        f"blocks (hex (0 1 2 3 4 5 6 7) ({' '.join(map(str, grid.shape))}) "
        f"simpleGrading (1 1 1)); edges (); boundary ({' '.join(patches)}); mergePatchPairs ();\n")
    put(case, "system/controlDict", "application ReactiveFoam; startFrom startTime; startTime 0; "
        "stopAt endTime; endTime 0; deltaT 1e-8; writeControl timeStep; writeInterval 1; "
        "runTimeModifiable false;\n")
    put(case, "system/fvSchemes", "ddtSchemes { default Euler; } gradSchemes { default Gauss linear; } "
        "divSchemes { default none; } laplacianSchemes { default Gauss linear uncorrected; } "
        "interpolationSchemes { default linear; } snGradSchemes { default uncorrected; }\n")
    put(case, "system/fvSolution", "solvers {}\n")


def write_field(case: Path, name: str, values: np.ndarray, dimensions: str) -> None:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        kind, klass = "scalar", "volScalarField"
    elif array.ndim == 2 and array.shape[1] == 3:
        kind, klass = "vector", "volVectorField"
    else:
        raise ValueError("field must have shape (cells,) or (cells,3)")
    if not np.isfinite(array).all():
        raise ValueError("nonfinite field")
    entries = ("(" + " ".join(f"{v:.17g}" for v in row) + ")" for row in array) if kind == "vector" else (f"{v:.17g}" for v in array)
    boundary = " ".join(f"{patch[0]} {{ type cyclic; }}" for patch in PATCHES)
    put(case, "0/" + name, f"dimensions [{dimensions}];\ninternalField nonuniform List<{kind}> "
        f"{len(array)}\n(\n" + "\n".join(entries) + f"\n);\nboundaryField {{ {boundary} }}\n", klass)


def _inside(points: np.ndarray, grid: CartesianGrid, kind: str, radius: float, amplitude: float) -> np.ndarray:
    lx, ly, lz = grid.lengths
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    if kind == "plane":
        return (z >= lz / 4) & (z < 3 * lz / 4)
    if kind == "sphere":
        return (x - lx / 2)**2 + (y - ly / 2)**2 + (z - lz / 2)**2 < radius**2
    if kind == "drop":
        dx, dy, dz = x - lx / 2, y - ly / 2, z - lz / 2
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        p2 = .5 * (3 * np.divide(dz * dz, r * r, out=np.zeros_like(r), where=r > 0) - 1)
        return r < radius + amplitude * p2
    if kind == "wave":
        displacement = amplitude * np.cos(2 * np.pi * x / lx)
        return (z >= lz / 4 + displacement) & (z < 3 * lz / 4 + displacement)
    raise ValueError("unknown interface geometry")


def geometric_color(grid: CartesianGrid, kind: str, radius: float | None = None,
                    amplitude: float | None = None, samples_per_axis: int = 8) -> np.ndarray:
    """Deterministic midpoint subcell integration; returns bounded cell volume fractions."""
    if samples_per_axis < 1 or samples_per_axis > 32:
        raise ValueError("samples_per_axis must be in [1, 32]")
    side = min(grid.lengths)
    radius = .25 * side if radius is None else radius
    amplitude = (.025 * radius if kind == "drop" else .01 * grid.lengths[0]) if amplitude is None else amplitude
    if not math.isfinite(radius) or radius <= 0 or 2 * radius >= side:
        raise ValueError("sphere radius must be positive and fit within the domain")
    if not math.isfinite(amplitude) or amplitude < 0:
        raise ValueError("amplitude must be finite and nonnegative")
    if kind == "wave" and amplitude >= grid.lengths[2] / 4:
        raise ValueError("wave amplitude must be less than slab margin")
    if kind == "drop" and 2 * (radius + amplitude) >= side:
        raise ValueError("deformed drop must fit within the domain")
    centers, spacing = grid.centers, grid.spacing
    color = np.zeros(grid.cells)
    offsets = (np.arange(samples_per_axis) + .5) / samples_per_axis - .5
    for ox in offsets:
        for oy in offsets:
            for oz in offsets:
                points = centers + spacing * np.array([ox, oy, oz])
                color += _inside(points, grid, kind, radius, amplitude)
    return color / samples_per_axis**3


def prepare(output: Path, grid: CartesianGrid, kind: str, interface_field: str,
            radius: float | None = None, amplitude: float | None = None,
            samples_per_axis: int = 8, run_block_mesh: bool = False,
            configuration: Path | None = None, sigma: float = 0.01,
            surface_tension: str = "true", phase_change: bool = False,
            temperature: float = 270., pressure: float = 3e6,
            end_time: float = 6e-8, delta_t: float = 3e-8,
            color_override: np.ndarray | None = None, curved_primitive: bool = False) -> dict:
    case = output.resolve()
    if case.exists():
        raise ValueError("output must be a new directory")
    if not interface_field.isidentifier():
        raise ValueError("interface field must be a valid OpenFOAM identifier")
    if surface_tension not in ("true", "false", "omitted"):
        raise ValueError("surface_tension must be true, false or omitted")
    if any(not math.isfinite(x) or x <= 0 for x in (temperature, pressure, end_time, delta_t)):
        raise ValueError("temperature, pressure and times must be finite and positive")
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("surface tension coefficient must be finite and nonnegative")
    case.mkdir(parents=True)
    write_mesh(case, grid)
    color = geometric_color(grid, kind, radius, amplitude, samples_per_axis) if color_override is None \
        else np.asarray(color_override, dtype=np.float64).reshape(-1).copy()
    if color.shape != (grid.cells,) or not np.isfinite(color).all() or np.any((color < 0) | (color > 1)):
        raise ValueError("independent cell volume fractions must be finite, bounded and match the mesh")
    np.savez_compressed(case / "initial-color.npz", color=color)
    if configuration is None:
        write_field(case, interface_field, color, "0 0 0 0 0 0 0")
    else:
        config = configuration.resolve()
        with RealFluidBackend(config) as backend:
            if backend.names != ["N2", "O2", "N2O", "IC3H7OH"] or not backend.condensed \
                    or backend.condensed[0]["kind"] != "liquid" or backend.condensed[0]["species"] != "N2O":
                raise ValueError("fixture requires the versioned N2/O2/N2O/IPA model with liquid N2O in slot 0")
            if surface_tension == "true" and backend.nl != 1:
                raise ValueError("capillary solver requires exactly one condensable liquid slot")
            gas_y = backend.mole_to_mass({"N2": .79, "O2": .21})
            ql, _, liquid = backend.make_state(temperature, pressure, {"N2O": 1}, (1, 0))
            qg, _, gas = backend.make_state(temperature, pressure, gas_y, (0, 0))
            if abs(liquid.alphaLiquid[0] - 1) > 1e-12 or gas.alphaLiquid[0] != 0:
                raise ValueError("reference pure phases are not liquid N2O and gas air")
            pressure_field = np.full(grid.cells, pressure)
            if kind in ("sphere", "drop") and surface_tension == "true" and sigma > 0:
                # Resolved equilibrium pressure rises by 2 sigma/R in liquid.
                # The mixed cells receive the corresponding color-weighted p.
                pressure_field += (2 * sigma / (.25 * min(grid.lengths) if radius is None else radius)) * color
            y = np.empty((grid.cells, backend.ns))
            color_construction_error = 0.
            for value in np.unique(color):
                indices = np.flatnonzero(color == value)
                index = int(indices[0])
                p_local = float(pressure_field[index])
                jump = 2 * sigma / (.25 * min(grid.lengths) if radius is None else radius) \
                    if kind in ("sphere", "drop") and surface_tension == "true" else 0.
                pg = p_local - float(value) * jump if curved_primitive else p_local
                pl = p_local + (1 - float(value)) * jump if curved_primitive else p_local
                liquid_q, _, liquid_local = backend.make_state(temperature, pl, {"N2O": 1}, (1, 0))
                gas_q, _, gas_local = backend.make_state(temperature, pg, gas_y, (0, 0))
                rho = float(value) * liquid_local.rho + (1 - float(value)) * gas_local.rho
                composition = (float(value) * liquid_q + (1 - float(value)) * gas_q) / rho
                y[indices] = composition
                if curved_primitive:
                    liquid_volume = rho * composition[2] / liquid_local.rho
                    gas_volume = rho * (1 - composition[2]) / gas_local.rho
                    recovered_color = liquid_volume / (liquid_volume + gas_volume)
                else:
                    _, _, reconstructed = backend.make_state(temperature, p_local, composition, (1, 0))
                    recovered_color = reconstructed.alphaLiquid[0]
                color_construction_error = max(color_construction_error, abs(recovered_color - float(value)))
            if color_construction_error > 1e-9:
                raise ValueError(f"thermo state does not reproduce geometric color: {color_construction_error}")
            for i in range(backend.ns):
                write_field(case, f"Y{i}", y[:, i], "0 0 0 0 0 0 0")
            if curved_primitive:
                write_field(case, interface_field, color, "0 0 0 0 0 0 0")
            for i in range(backend.nl):
                write_field(case, f"liquidFraction{i}", np.ones(grid.cells) if i == 0 else np.zeros(grid.cells),
                            "0 0 0 0 0 0 0")
            write_field(case, "p", pressure_field, "1 -1 -2 0 0 0 0")
            write_field(case, "T", np.full(grid.cells, temperature), "0 0 0 1 0 0 0")
            write_field(case, "U", np.zeros((grid.cells, 3)), "0 1 -1 0 0 0 0")
            physics_surface = "" if surface_tension == "omitted" else f"surfaceTension {surface_tension};"
            put(case, "constant/reactiveProperties", f'''closure HEM; thermoConfiguration "{config}";
initialization primitive; chemistry false; dynamicViscosity 0; thermalConductivity 0; molecularDiffusivity 0;
physics {{ chemistry false; phaseChange {str(phase_change).lower()}; viscosity false; heatConduction false;
 speciesDiffusion false; turbulence none; {physics_surface} }}
surfaceTensionCoefficient {sigma:.17g}; capillaryCfl .25; capillaryGeometryTolerance 1e-10;
transportBackend cuda; closureBackend cuda; closureCpuFallback false; closureScalarBackend cpu;
closureJacobian finiteDifference; thermoExactReuse false; thermoBatchCells 4096; thermoWorkers 1;
maxDeviceMemoryGB 2; maxHostMemoryGB 4; waveSpeedFactor 1.1; boundaryConditions {{}}
''')
            put(case, "system/controlDict", f'''application ReactiveFoam; startFrom startTime; startTime 0;
stopAt endTime; endTime {end_time:.17g}; deltaT {delta_t:.17g}; maxDeltaT {delta_t:.17g}; maxCo .25;
writeControl runTime; writeInterval {delta_t:.17g}; writeFormat binary; writePrecision 17;
writeCompression off; timeFormat general; timePrecision 15; runTimeModifiable false; functions {{}};
''')
    radius = .25 * min(grid.lengths) if radius is None else radius
    amplitude = (.025 * radius if kind == "drop" else .01 * grid.lengths[0]) if amplitude is None else amplitude
    reference = {"kind": kind, "shape": grid.shape, "lengthsM": grid.lengths,
                 "cellIntegration": "independent supplied volumes" if color_override is not None else "midpoint subcells",
                 "spacingM": grid.spacing.tolist(), "interfaceField": interface_field,
                 "samplesPerAxis": samples_per_axis, "initialColorVolumeM3": float(color.sum() * np.prod(grid.spacing)),
                 "radiusM": radius if kind in ("sphere", "drop") else None,
                 "analyticLaplaceCurvaturePerM": 2 / radius if kind == "sphere" else 0 if kind == "plane" else None,
                 "analyticSphereVolumeM3": 4 * math.pi * radius**3 / 3 if kind in ("sphere", "drop") else None,
                 "wavelengthM": grid.lengths[0] if kind == "wave" else None,
                 "initialAmplitudeM": amplitude if kind in ("wave", "drop") else None,
                 "configuration": str(configuration.resolve()) if configuration else None,
                 "sigmaNPerM": sigma if configuration else None,
                 "capillaryCfl": .25 if configuration else None,
                 "surfaceTension": surface_tension if configuration else None,
                 "phaseChange": phase_change if configuration else None,
                 "temperatureK": temperature if configuration else None,
                 "pressurePa": pressure if configuration else None,
                 "initialPressureJumpPa": (2 * sigma / radius) if configuration and kind in ("sphere", "drop") and surface_tension == "true" else 0 if configuration else None,
                 "pureLiquidDensityKgPerM3": float(liquid.rho) if configuration else None,
                 "pureGasDensityKgPerM3": float(gas.rho) if configuration else None,
                 "colorConstructionMaxAbsoluteError": color_construction_error if configuration else None,
                 "primitivePhasePressure": "curved" if curved_primitive else "common",
                 "endTimeS": end_time if configuration else None}
    if run_block_mesh:
        # A stock OpenFOAM blockMesh writes the same native polyMesh and does
        # not depend on the local CUDA executable's CPU instruction set.
        foam_bashrc = Path("/opt/openfoam14/etc/bashrc")
        if foam_bashrc.is_file():
            command = ["bash", "--noprofile", "--norc", "-c",
                       'source "$1" ParaView_TYPE=none >/dev/null && exec blockMesh -case "$2"',
                       "capillary-blockmesh", str(foam_bashrc), str(case)]
            env = None
        else:
            command = ["blockMesh", "-case", str(case)]
            env = common.sourced_environment()
        with (case / "blockMesh.log").open("w") as stream:
            subprocess.run(command, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, timeout=120, check=True)
        if foam_bashrc.is_file():
            check = ["bash", "--noprofile", "--norc", "-c",
                     'source "$1" ParaView_TYPE=none >/dev/null && exec checkMesh -case "$2" -allGeometry -allTopology',
                     "capillary-checkmesh", str(foam_bashrc), str(case)]
        else:
            check = ["checkMesh", "-case", str(case), "-allGeometry", "-allTopology"]
        with (case / "checkMesh.log").open("w") as stream:
            subprocess.run(check, env=env, stdout=stream, stderr=subprocess.STDOUT,
                           timeout=120, check=True)
        if "Mesh OK." not in (case / "checkMesh.log").read_text():
            raise RuntimeError("checkMesh did not confirm Mesh OK")
    common.atomic_json(case / "geometry-definition.json", reference)
    return reference


def prepare_suite(output: Path, configuration: Path, sigma: float = .01,
                  samples_per_axis: int = 8, run_block_mesh: bool = True) -> dict:
    """Small actual-solver matrix; wave/drop are short dynamics smoke cases."""
    root = output.resolve()
    if root.exists():
        raise ValueError("suite output must be a new directory")
    root.mkdir(parents=True)
    source = configuration.resolve()
    settings = yaml.safe_load(source.read_text())
    mechanism = Path(settings["mechanism"])
    if not mechanism.is_absolute():
        mechanism = source.parent / mechanism
    retained = [entry for entry in settings["condensables"] if entry["species"] == "N2O" and entry["phase"] == "liquid_N2O"]
    if len(retained) != 1 or len(settings["condensables"]) != 1:
        raise ValueError("suite source must already define only liquid N2O; use the single-liquid reference config")
    thermo_dir = root / "thermo"
    thermo_dir.mkdir()
    local_mechanism = thermo_dir / "cold-pr.yaml"
    shutil.copy2(mechanism, local_mechanism)
    settings["mechanism"] = str(local_mechanism)
    local_config = thermo_dir / "capillary-n2o-config.yaml"
    local_config.write_text(yaml.safe_dump(settings, sort_keys=False, allow_unicode=True))
    specs = [
        ("off-omitted", "plane", (8, 8, 8), "omitted", False),
        ("off-explicit", "plane", (8, 8, 8), "false", False),
        ("plane", "plane", (8, 8, 8), "true", False),
        ("sphere-16", "sphere", (16, 16, 16), "true", False),
        ("sphere-24", "sphere", (24, 24, 24), "true", False),
        ("sphere-32", "sphere", (32, 32, 32), "true", False),
        ("wave", "wave", (32, 4, 128), "true", False),
        ("drop", "drop", (32, 32, 32), "true", False),
        ("flash-sphere", "sphere", (16, 16, 16), "true", True),
    ]
    result = {"schema": "capillary-small-suite-v1", "cases": {},
              "configuration": str(configuration.resolve()), "sigmaNPerM": sigma,
              "scope": "two-step operator/integration smoke; wave/drop frequency needs longer runs"}
    for name, kind, shape, switch, flash in specs:
        lengths = (.004, .004, .016) if kind == "wave" else (.004, .004, .004)
        case = root / name
        result["cases"][name] = prepare(case, CartesianGrid(shape, lengths), kind, "interfaceColor",
                                        samples_per_axis=samples_per_axis, run_block_mesh=run_block_mesh,
                                        configuration=local_config, sigma=sigma,
                                        surface_tension=switch, phase_change=flash)
    result["localConfiguration"] = str(local_config)
    result["localThermoSha256"] = {str(p.relative_to(root)): common.sha256(p) for p in thermo_dir.iterdir()}
    result["generatorSha256"] = common.sha256(Path(__file__).resolve())
    result["inputsSha256"] = {str(p.relative_to(root)): common.sha256(p)
                              for case in root.iterdir() if case.is_dir()
                              for directory in ("0", "constant", "system")
                              for p in sorted((case / directory).rglob("*")) if p.is_file()}
    common.atomic_json(root / "suite-definition.json", result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output", type=Path)
    ap.add_argument("--suite", action="store_true", help="write the standard small solver matrix")
    ap.add_argument("--kind", choices=("plane", "sphere", "wave", "drop"))
    ap.add_argument("--shape", type=int, nargs=3, metavar=("NX", "NY", "NZ"))
    ap.add_argument("--lengths", type=float, nargs=3, metavar=("LX", "LY", "LZ"))
    ap.add_argument("--interface-field", default="interfaceColor", help="solver-written geometric color field name")
    ap.add_argument("--configuration", type=Path, help="real-fluid configuration; writes a runnable HEM fixture")
    ap.add_argument("--sigma", type=float, default=.01)
    ap.add_argument("--surface-tension", choices=("true", "false", "omitted"), default="true")
    ap.add_argument("--phase-change", action="store_true")
    ap.add_argument("--temperature", type=float, default=270.)
    ap.add_argument("--pressure", type=float, default=3e6)
    ap.add_argument("--end-time", type=float, default=6e-8)
    ap.add_argument("--delta-t", type=float, default=3e-8)
    ap.add_argument("--radius", type=float)
    ap.add_argument("--amplitude", type=float)
    ap.add_argument("--samples-per-axis", type=int, default=8)
    ap.add_argument("--block-mesh", action="store_true")
    a = ap.parse_args()
    import json
    if a.suite:
        if not a.configuration:
            ap.error("--suite requires --configuration")
        report = prepare_suite(a.output, a.configuration, a.sigma, a.samples_per_axis, True)
        print(json.dumps({"suite": str(a.output.resolve()), "cases": list(report["cases"])}))
    else:
        if a.kind is None or a.shape is None or a.lengths is None:
            ap.error("single case requires --kind, --shape and --lengths")
        print(json.dumps(prepare(a.output, CartesianGrid(tuple(a.shape), tuple(a.lengths)), a.kind,
                                 a.interface_field, a.radius, a.amplitude, a.samples_per_axis, a.block_mesh,
                                 a.configuration, a.sigma, a.surface_tension, a.phase_change,
                                 a.temperature, a.pressure, a.end_time, a.delta_t)))


if __name__ == "__main__":
    main()
