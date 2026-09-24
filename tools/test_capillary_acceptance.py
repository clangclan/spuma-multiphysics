#!/usr/bin/env python3
"""Regression tests preventing invalid static-drop convergence claims."""
from copy import deepcopy
import unittest

from validate_capillary_solver import fit_mode_frequency, summarize_series


def series():
    return [dict(case=str(n), sigmaNPerM=.01,
                 geometry=dict(kind="sphere", spacingM=[.004/n]*3,
                               lengthsM=[.004]*3, radiusM=.001, temperatureK=270.,
                               pressurePa=3e6, phaseChange=False, surfaceTension="true"),
                 finalLaplaceErrorPa=1/n,
                 history=[dict(timeS=6e-8, maxSpeedMPerS=1/n,
                               volumeWeightedL2SpeedMPerS=.1/n)]) for n in (16, 24, 32)]


class Acceptance(unittest.TestCase):
    def gate(self, rows):
        return summarize_series(rows)["sphereGridConvergence"]

    def test_velocity_convergence_is_not_full_accuracy(self):
        result = self.gate(series())
        self.assertTrue(result["staticVelocityGatePassed"])
        self.assertFalse(result["fullPhysicsAccuracyValidated"])

    def test_actual_failed_velocity_trend_cannot_pass(self):
        rows = series()
        for row, speed in zip(rows, (5.02e-5, 8.34e-5, 1.16e-4)):
            row["history"][-1]["maxSpeedMPerS"] = speed
        self.assertFalse(self.gate(rows)["staticVelocityGatePassed"])

    def test_shorter_fine_run_cannot_fake_spatial_convergence(self):
        rows = series()
        rows[-1]["history"][-1]["timeS"] /= 2
        self.assertFalse(self.gate(rows)["comparisonValid"])

    def test_changed_physics_cannot_pass(self):
        for key, value in (("radiusM", .002), ("phaseChange", True), ("pressurePa", 4e6)):
            rows = series()
            rows[-1]["geometry"][key] = value
            self.assertFalse(self.gate(rows)["staticVelocityGatePassed"])

    def test_duplicate_mesh_cannot_pass(self):
        rows = series()
        rows[-1]["geometry"]["spacingM"] = deepcopy(rows[-2]["geometry"]["spacingM"])
        self.assertFalse(self.gate(rows)["staticVelocityGatePassed"])

    def test_l2_growth_cannot_hide_behind_umax(self):
        rows = series()
        rows[-1]["history"][-1]["volumeWeightedL2SpeedMPerS"] = 10
        self.assertFalse(self.gate(rows)["staticVelocityGatePassed"])

    def test_short_trace_cannot_measure_an_oscillation_period(self):
        import math
        history = [dict(timeS=i/20, amplitude=math.cos(2*math.pi*i/20)) for i in range(20)]
        self.assertEqual(fit_mode_frequency(history, "amplitude", 2*math.pi)["status"],
                         "insufficient duration or samples")


if __name__ == "__main__":
    unittest.main()
