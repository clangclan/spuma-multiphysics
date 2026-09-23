#!/usr/bin/env python3
"""Identify resolved liquid components from an explicit geometric alpha field.

This is a mesh diagnostic, not a breakup model.  In particular, a phase
fraction recovered by an HEM flash is not a geometric interface indicator and
is deliberately rejected by the public API.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


GEOMETRIC_ALPHA = "geometric_volume_fraction"


class ResolvedDropInputError(ValueError):
    """Raised when the mesh or field data cannot define the diagnostic."""


@dataclass(frozen=True)
class ResolvedComponent:
    """Integral measurements for one threshold-connected liquid component."""

    label: int
    cell_count: int
    cells: tuple[int, ...]
    inlet_connected: bool
    liquid_volume: float
    equivalent_diameter: float
    centroid: tuple[float, float, float]
    volume_velocity_moment: tuple[float, float, float]
    mean_velocity: tuple[float, float, float]


@dataclass(frozen=True)
class ResolvedDropResult:
    """All labels and the inlet-connected/detached component partition."""

    alpha_semantics: str
    alpha_threshold: float
    labels: tuple[int, ...]
    components: tuple[ResolvedComponent, ...]
    inlet_connected_labels: tuple[int, ...]
    detached_labels: tuple[int, ...]
    input_liquid_volume: float
    labeled_liquid_volume: float
    unlabeled_liquid_volume: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> None:
        a = self.find(left)
        b = self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResolvedDropInputError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ResolvedDropInputError(f"{name} must be finite")
    return result


def _index(value: Any, name: str, size: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResolvedDropInputError(f"{name} must be an integer")
    if value < 0 or value >= size:
        raise ResolvedDropInputError(f"{name}={value} is outside [0, {size})")
    return value


def _vector3(value: Any, name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise ResolvedDropInputError(f"{name} must contain exactly three numbers")
    return tuple(_finite_number(entry, f"{name}[{axis}]") for axis, entry in enumerate(value))  # type: ignore[return-value]


def _edge_list(values: Iterable[Any], name: str, size: int) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    if isinstance(values, (str, bytes)):
        raise ResolvedDropInputError(f"{name} must be an iterable of cell-index pairs")
    try:
        iterator = iter(values)
    except TypeError as error:
        raise ResolvedDropInputError(f"{name} must be an iterable of cell-index pairs") from error
    for position, value in enumerate(iterator):
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
            raise ResolvedDropInputError(f"{name}[{position}] must contain two cell indices")
        left = _index(value[0], f"{name}[{position}][0]", size)
        right = _index(value[1], f"{name}[{position}][1]", size)
        if left == right:
            raise ResolvedDropInputError(f"{name}[{position}] is a self-edge")
        result.append((left, right))
    return tuple(result)


def identify_resolved_drops(
    *,
    alpha: Sequence[float],
    cell_volumes: Sequence[float],
    cell_centers: Sequence[Sequence[float]],
    edges: Iterable[Sequence[int]],
    velocity: Sequence[Sequence[float]],
    inlet_cells: Iterable[int] = (),
    periodic_edges: Iterable[Sequence[int]] = (),
    alpha_threshold: float = 0.5,
    alpha_semantics: str,
) -> ResolvedDropResult:
    """Label threshold-connected geometric liquid and integrate its measures.

    Ordinary and periodic edges have identical connectivity semantics.  A
    component is an inlet-connected sheet/jet whenever it contains at least
    one listed inlet cell.  Every other component is reported as detached.

    The velocity moment is ``sum(alpha * cell_volume * velocity)`` and the
    mean velocity is that moment divided by liquid volume.  Neither is mass
    momentum because density is intentionally outside this geometric API.
    Periodic edges join labels but do not unwrap coordinates; callers needing
    a seam-independent centroid should supply pre-unwrapped cell centres.
    """

    if alpha_semantics != GEOMETRIC_ALPHA:
        raise ResolvedDropInputError(
            f"alpha_semantics must be {GEOMETRIC_ALPHA!r}; "
            "HEM flash phase fractions are not geometric interface fields"
        )
    if isinstance(alpha, (str, bytes)) or not isinstance(alpha, Sequence):
        raise ResolvedDropInputError("alpha must be a sequence")
    cell_count = len(alpha)
    if cell_count == 0:
        raise ResolvedDropInputError("at least one cell is required")
    for name, values in (
        ("cell_volumes", cell_volumes),
        ("cell_centers", cell_centers),
        ("velocity", velocity),
    ):
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ResolvedDropInputError(f"{name} must be a sequence")
    if len(cell_volumes) != cell_count or len(cell_centers) != cell_count or len(velocity) != cell_count:
        raise ResolvedDropInputError(
            "alpha, cell_volumes, cell_centers, and velocity must have equal lengths"
        )

    threshold = _finite_number(alpha_threshold, "alpha_threshold")
    if threshold <= 0.0 or threshold > 1.0:
        raise ResolvedDropInputError("alpha_threshold must be in (0, 1]")

    fractions: list[float] = []
    volumes: list[float] = []
    centers: list[tuple[float, float, float]] = []
    velocities: list[tuple[float, float, float]] = []
    for cell in range(cell_count):
        fraction = _finite_number(alpha[cell], f"alpha[{cell}]")
        if fraction < 0.0 or fraction > 1.0:
            raise ResolvedDropInputError(f"alpha[{cell}] must be in [0, 1]")
        volume = _finite_number(cell_volumes[cell], f"cell_volumes[{cell}]")
        if volume <= 0.0:
            raise ResolvedDropInputError(f"cell_volumes[{cell}] must be positive")
        fractions.append(fraction)
        volumes.append(volume)
        centers.append(_vector3(cell_centers[cell], f"cell_centers[{cell}]"))
        velocities.append(_vector3(velocity[cell], f"velocity[{cell}]"))

    face_edges = _edge_list(edges, "edges", cell_count)
    seam_edges = _edge_list(periodic_edges, "periodic_edges", cell_count)
    if isinstance(inlet_cells, (str, bytes)):
        raise ResolvedDropInputError("inlet_cells must be an iterable of cell indices")
    try:
        inlet_iterator = iter(inlet_cells)
    except TypeError as error:
        raise ResolvedDropInputError("inlet_cells must be an iterable of cell indices") from error
    inlet = {
        _index(value, f"inlet_cells[{position}]", cell_count)
        for position, value in enumerate(inlet_iterator)
    }

    active = [fraction >= threshold for fraction in fractions]
    forest = _DisjointSet(cell_count)
    for left, right in (*face_edges, *seam_edges):
        if active[left] and active[right]:
            forest.union(left, right)

    by_root: dict[int, list[int]] = {}
    for cell, selected in enumerate(active):
        if selected:
            by_root.setdefault(forest.find(cell), []).append(cell)

    # A minimum cell index order makes labels deterministic across edge order.
    groups = sorted(by_root.values(), key=lambda cells: cells[0])
    labels = [0] * cell_count
    components: list[ResolvedComponent] = []
    inlet_labels: list[int] = []
    detached_labels: list[int] = []
    for label, cells in enumerate(groups, start=1):
        for cell in cells:
            labels[cell] = label
        liquid_weights = [fractions[cell] * volumes[cell] for cell in cells]
        liquid_volume = math.fsum(liquid_weights)
        # threshold > 0 and positive cell volumes make this strictly positive.
        centroid = tuple(
            math.fsum(weight * centers[cell][axis] for cell, weight in zip(cells, liquid_weights))
            / liquid_volume
            for axis in range(3)
        )
        velocity_moment = tuple(
            math.fsum(weight * velocities[cell][axis] for cell, weight in zip(cells, liquid_weights))
            for axis in range(3)
        )
        mean_velocity = tuple(value / liquid_volume for value in velocity_moment)
        connected = any(cell in inlet for cell in cells)
        component = ResolvedComponent(
            label=label,
            cell_count=len(cells),
            cells=tuple(cells),
            inlet_connected=connected,
            liquid_volume=liquid_volume,
            equivalent_diameter=(6.0 * liquid_volume / math.pi) ** (1.0 / 3.0),
            centroid=centroid,
            volume_velocity_moment=velocity_moment,
            mean_velocity=mean_velocity,
        )
        components.append(component)
        (inlet_labels if connected else detached_labels).append(label)

    return ResolvedDropResult(
        alpha_semantics=alpha_semantics,
        alpha_threshold=threshold,
        labels=tuple(labels),
        components=tuple(components),
        inlet_connected_labels=tuple(inlet_labels),
        detached_labels=tuple(detached_labels),
        input_liquid_volume=math.fsum(
            fraction * volume for fraction, volume in zip(fractions, volumes)
        ),
        labeled_liquid_volume=math.fsum(component.liquid_volume for component in components),
        unlabeled_liquid_volume=math.fsum(
            fractions[cell] * volumes[cell] for cell, selected in enumerate(active) if not selected
        ),
    )


def identify_from_mapping(data: Mapping[str, Any]) -> ResolvedDropResult:
    """Run the diagnostic from the JSON-compatible command-line schema."""

    allowed = {
        "alpha",
        "alpha_semantics",
        "alpha_threshold",
        "cell_volumes",
        "cell_centers",
        "edges",
        "periodic_edges",
        "velocity",
        "inlet_cells",
    }
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ResolvedDropInputError(f"unknown input keys: {', '.join(unknown)}")
    required = {"alpha", "alpha_semantics", "cell_volumes", "cell_centers", "edges", "velocity"}
    missing = sorted(required - set(data))
    if missing:
        raise ResolvedDropInputError(f"missing input keys: {', '.join(missing)}")
    return identify_resolved_drops(
        alpha=data["alpha"],
        alpha_semantics=data["alpha_semantics"],
        alpha_threshold=data.get("alpha_threshold", 0.5),
        cell_volumes=data["cell_volumes"],
        cell_centers=data["cell_centers"],
        edges=data["edges"],
        periodic_edges=data.get("periodic_edges", ()),
        velocity=data["velocity"],
        inlet_cells=data.get("inlet_cells", ()),
    )


def _read_json(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSON input path, or - for standard input")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation (default: 2)")
    arguments = parser.parse_args(argv)
    try:
        data = _read_json(arguments.input)
        if not isinstance(data, Mapping):
            raise ResolvedDropInputError("top-level JSON value must be an object")
        result = identify_from_mapping(data)
    except (OSError, json.JSONDecodeError, ResolvedDropInputError) as error:
        parser.exit(2, f"resolved_drops: {error}\n")
    json.dump(result.to_dict(), sys.stdout, indent=arguments.indent, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
