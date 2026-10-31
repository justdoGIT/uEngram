"""Unit tests for channel diagnostics and domain specialist architecture."""
from __future__ import annotations

import unittest
import numpy as np

from uengram.diagnostics import ChannelDiagnosticReport, diagnose_channels
from uengram.domain_specialists import DomainSpecialistArchitecture, SpecialistScores


class ChannelDiagnosticsTests(unittest.TestCase):
    def test_well_conditioned_channels(self):
        rng = np.random.default_rng(42)
        # 3 independent random normal streams
        data = rng.standard_normal((500, 3))
        report = diagnose_channels(data, ["c1", "c2", "c3"])
        self.assertEqual(report.status, "well_conditioned")
        self.assertTrue(report.is_safe)
        self.assertEqual(report.matrix_rank, 3)
        self.assertEqual(report.full_rank, 3)
        self.assertEqual(len(report.redundant_groups), 0)
        self.assertEqual(len(report.colinear_pairs), 0)
        self.assertEqual(report.invalid_samples, 0)

    def test_singular_colinear_channels(self):
        rng = np.random.default_rng(42)
        c1 = rng.standard_normal(500)
        c2 = rng.standard_normal(500)
        # Exact linear combination
        c3 = c1 - c2
        data = np.column_stack([c1, c2, c3])
        report = diagnose_channels(data, ["a", "b", "diff"])
        self.assertEqual(report.status, "singular")
        self.assertFalse(report.is_safe)
        self.assertEqual(report.matrix_rank, 2)
        self.assertEqual(report.full_rank, 3)
        self.assertGreater(len(report.redundant_groups), 0)
        # Check that the redundant group includes the colinear channels
        flat_redundant = {ch for g in report.redundant_groups for ch in g}
        self.assertTrue({"a", "b", "diff"}.issubset(flat_redundant))

    def test_sentinel_counting(self):
        data = np.ones((50, 2))
        data[0, 0] = -99999.0
        data[1, 1] = np.nan
        report = diagnose_channels(data, ["x", "y"])
        self.assertEqual(report.invalid_counts["x"], 1)
        self.assertEqual(report.invalid_counts["y"], 1)
        self.assertEqual(report.invalid_samples, 2)
        self.assertEqual(report.valid_samples, 48)

    def test_insufficient_samples(self):
        data = np.ones((5, 3))
        report = diagnose_channels(data)
        self.assertEqual(report.status, "insufficient_valid_samples")
        self.assertFalse(report.is_safe)


class DomainSpecialistArchitectureTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(101)
        # 4 channels: 0 = cpu, 1 = mem_usage, 2 = mem_used, 3 = net
        # 100 windows, 4 channels, length 16
        n_windows = 120
        self.fit_windows = rng.standard_normal((n_windows, 4, 16))
        self.cal_windows = rng.standard_normal((n_windows, 4, 16))
        self.test_windows = rng.standard_normal((n_windows, 4, 16))

        self.domain_channels = {
            "cpu": [0],
            "memory": [1, 2],
            "network": [3],
        }
        self.channel_names = ["cpu_u", "mem_u", "mem_d", "net_t"]

    def test_fit_and_score_flow(self):
        arch = DomainSpecialistArchitecture(
            self.domain_channels,
            channel_names=self.channel_names,
            domain_d=4,
        )
        arch.fit(self.fit_windows)
        self.assertTrue(arch._fitted)
        self.assertEqual(len(arch.specialists), 3)
        self.assertIn("cpu", arch.prefit_diagnostics)

        arch.fit_calibration(self.cal_windows)
        scores = arch.score(self.test_windows)
        self.assertIsInstance(scores, SpecialistScores)
        self.assertEqual(scores.fused_score.shape, (120,))
        self.assertEqual(scores.fused_p_value.shape, (120,))
        self.assertEqual(set(scores.domain_scores.keys()), {"cpu", "memory", "network"})

    def test_subsystem_alarm_attribution(self):
        arch = DomainSpecialistArchitecture(
            self.domain_channels,
            channel_names=self.channel_names,
            domain_d=4,
        )
        arch.fit(self.fit_windows)
        arch.fit_calibration(self.cal_windows)

        # Inject a massive anomaly only into the network channel (channel 3)
        # Inject an anomaly with both level and dynamic disturbance into network channel
        anom_windows = self.test_windows.copy()
        rng = np.random.default_rng(202)
        anom_windows[10:15, 3, :] += 50.0 + rng.standard_normal((5, 16)) * 20.0

        scores = arch.score(anom_windows)
        fused_flags, attributions = arch.fuse(
            scores, alpha=0.01, branch_policy="dynamic_health", specialist_rule="or"
        )

        # At the anomalous windows, network must be in the attribution
        for i in range(10, 15):
            self.assertTrue(fused_flags[i])
            self.assertIn("network", attributions[i])
            # CPU and memory should not have triggered
            self.assertNotIn("cpu", attributions[i])
            self.assertNotIn("memory", attributions[i])
        route, meta = DomainSpecialistArchitecture.memory_route(
            alarm_open=False,
            active_domains=[],
            data_quality_ok=True,
            operator_confirmed_normal=True,
        )
        self.assertEqual(route, "SEMANTIC")
        self.assertEqual(meta["reason"], "operator_confirmed_normal")

        # 2. Domain exceedance with attribution
        route, meta = DomainSpecialistArchitecture.memory_route(
            alarm_open=False,
            active_domains=["network"],
            data_quality_ok=True,
        )
        self.assertEqual(route, "EPISODIC")
        self.assertEqual(meta["triggering_domains"], ["network"])

        # 3. Bad quality quarantine
        route, meta = DomainSpecialistArchitecture.memory_route(
            alarm_open=False,
            active_domains=["cpu"],
            data_quality_ok=False,
        )
        self.assertEqual(route, "QUARANTINE")
        self.assertEqual(meta["reason"], "data_quality_rejected")

    def test_strict_conditioning_rejects_singular(self):
        # Create windows with an exact colinear channel
        bad_windows = self.fit_windows.copy()
        bad_windows[:, 2, :] = bad_windows[:, 0, :] + bad_windows[:, 1, :]
        arch = DomainSpecialistArchitecture(
            self.domain_channels,
            strict_conditioning=True,
        )
        with self.assertRaises(ValueError) as ctx:
            arch.fit(bad_windows)
        self.assertIn("conditioning failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
