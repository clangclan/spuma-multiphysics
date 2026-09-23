#!/usr/bin/env python3

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from resolved_drops import (
    GEOMETRIC_ALPHA,
    ResolvedDropInputError,
    identify_from_mapping,
    identify_resolved_drops,
)


def fields(count):
    return dict(
        alpha=[1.0] * count,
        alpha_semantics=GEOMETRIC_ALPHA,
        cell_volumes=[1.0] * count,
        cell_centers=[[float(cell), 0.0, 0.0] for cell in range(count)],
        velocity=[[float(cell + 1), 0.0, 0.0] for cell in range(count)],
    )


class ResolvedDropsTests(unittest.TestCase):
    def test_metrics_labels_and_inlet_partition(self):
        data = fields(6)
        data.update(
            alpha=[1.0, 0.5, 0.1, 0.75, 1.0, 0.0],
            cell_volumes=[2.0, 4.0, 1.0, 2.0, 1.0, 3.0],
            cell_centers=[[0, 0, 0], [2, 0, 0], [3, 0, 0], [10, 0, 0], [12, 0, 0], [20, 0, 0]],
            velocity=[[1, 0, 0], [3, 0, 0], [100, 0, 0], [4, 2, 0], [10, 8, 0], [0, 0, 0]],
        )
        result = identify_resolved_drops(
            **data,
            edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
            inlet_cells=[0],
            alpha_threshold=0.5,
        )

        self.assertEqual(result.labels, (1, 1, 0, 2, 2, 0))
        self.assertEqual(result.inlet_connected_labels, (1,))
        self.assertEqual(result.detached_labels, (2,))
        first, second = result.components
        self.assertTrue(first.inlet_connected)
        self.assertEqual(first.cells, (0, 1))
        self.assertAlmostEqual(first.liquid_volume, 4.0)
        self.assertAlmostEqual(first.equivalent_diameter, (24.0 / math.pi) ** (1.0 / 3.0))
        self.assertEqual(first.centroid, (1.0, 0.0, 0.0))
        self.assertEqual(first.volume_velocity_moment, (8.0, 0.0, 0.0))
        self.assertEqual(first.mean_velocity, (2.0, 0.0, 0.0))
        self.assertFalse(second.inlet_connected)
        self.assertAlmostEqual(second.liquid_volume, 2.5)
        self.assertEqual(second.centroid, (10.8, 0.0, 0.0))
        self.assertEqual(second.mean_velocity, (6.4, 4.4, 0.0))
        self.assertAlmostEqual(result.input_liquid_volume, 6.6)
        self.assertAlmostEqual(result.labeled_liquid_volume, 6.5)
        self.assertAlmostEqual(result.unlabeled_liquid_volume, 0.1)

    def test_periodic_edge_merges_seam_component(self):
        data = fields(4)
        result = identify_resolved_drops(
            **data,
            edges=[(0, 1), (1, 2)],
            periodic_edges=[(0, 3)],
            inlet_cells=[3],
        )
        self.assertEqual(result.labels, (1, 1, 1, 1))
        self.assertEqual(result.inlet_connected_labels, (1,))
        self.assertEqual(result.detached_labels, ())

    def test_multiple_components_get_deterministic_distinct_labels(self):
        data = fields(7)
        data["alpha"] = [1, 0, 1, 1, 0, 1, 0]
        result = identify_resolved_drops(
            **data,
            edges=[(5, 3), (3, 2), (0, 1), (1, 2)],
            inlet_cells=[],
        )
        self.assertEqual(result.labels, (1, 0, 2, 2, 0, 2, 0))
        self.assertEqual([part.cells for part in result.components], [(0,), (2, 3, 5)])
        self.assertEqual(result.detached_labels, (1, 2))

    def test_inactive_inlet_does_not_attach_a_component(self):
        data = fields(2)
        data["alpha"] = [0.0, 1.0]
        result = identify_resolved_drops(**data, edges=[(0, 1)], inlet_cells=[0])
        self.assertEqual(result.detached_labels, (1,))

    def test_rejects_hem_semantics_and_bad_mesh_values(self):
        data = fields(2)
        base = dict(data, edges=[(0, 1)])
        for change in (
            {"alpha_semantics": "hem_flash_phase_fraction"},
            {"alpha": [1.0]},
            {"alpha": [1.1, 0.0]},
            {"alpha": [float("nan"), 0.0]},
            {"cell_volumes": [1.0, 0.0]},
            {"cell_centers": [[0, 0], [1, 0, 0]]},
            {"velocity": [[0, 0, 0], [math.inf, 0, 0]]},
            {"edges": [(0, 0)]},
            {"edges": [(0, 2)]},
            {"edges": [(0, 1.0)]},
            {"periodic_edges": [(1, 1)]},
            {"periodic_edges": None},
            {"inlet_cells": [False]},
            {"inlet_cells": None},
            {"cell_volumes": None},
            {"alpha_threshold": 0.0},
            {"alpha_threshold": 1.01},
        ):
            with self.subTest(change=change), self.assertRaises(ResolvedDropInputError):
                identify_resolved_drops(**(base | change))

    def test_mapping_is_strict_and_cli_emits_json(self):
        data = fields(2) | {"edges": [], "inlet_cells": []}
        with self.assertRaisesRegex(ResolvedDropInputError, "unknown input keys"):
            identify_from_mapping(data | {"alphaGas": [0.0, 0.0]})
        with self.assertRaisesRegex(ResolvedDropInputError, "missing input keys"):
            identify_from_mapping({"alpha_semantics": GEOMETRIC_ALPHA})

        script = Path(__file__).with_name("resolved_drops.py")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(script), str(path), "--indent", "0"],
                check=True,
                capture_output=True,
                text=True,
            )
        output = json.loads(completed.stdout)
        self.assertEqual(output["labels"], [1, 2])
        self.assertEqual(output["detached_labels"], [1, 2])


if __name__ == "__main__":
    unittest.main()
