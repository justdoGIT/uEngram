"""Domain-specialist sensor architecture with calibrated multi-expert fusion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .diagnostics import ChannelDiagnosticReport, diagnose_channels
from .role_architecture import (
    EmpiricalCalibrator,
    RoleBasedArchitecture,
    RoleScores,
)


@dataclass(frozen=True)
class SpecialistScores:
    """Per-domain role scores and combined decision metrics."""

    domain_scores: dict[str, RoleScores]
    domain_p_values: dict[str, dict[str, np.ndarray]]
    fused_score: np.ndarray
    fused_p_value: np.ndarray

    def attribution_for_window(
        self, window_idx: int, alpha: float = 0.01, branch_policy: str = "dynamic_health"
    ) -> list[str]:
        """Return the names of domains whose branches triggered for this window."""
        triggered = []
        for domain, p_dict in self.domain_p_values.items():
            flags = RoleBasedArchitecture.fuse(
                {k: v[window_idx : window_idx + 1] for k, v in p_dict.items()},
                alpha=alpha,
                policy=branch_policy,
            )
            if bool(flags[0]):
                triggered.append(domain)
        return triggered


class DomainSpecialistArchitecture:
    """Multi-expert telemetry architecture with calibrated domain specialists.

    Instead of pooling all channels into one latent representation where local
    faults are diluted and redundant channels cause covariance singularity,
    telemetry is partitioned into domain experts (e.g. CPU, memory, network).
    Each expert runs calibrated level, dynamic, and health monitoring.
    """

    def __init__(
        self,
        domain_channels: dict[str, tuple[int, ...] | list[int]],
        channel_names: list[str] | tuple[str, ...] | None = None,
        domain_d: int = 4,
        strict_conditioning: bool = False,
    ):
        if not domain_channels:
            raise ValueError("domain_channels must contain at least one domain")
        self.domain_channels = {
            k: tuple(int(i) for i in v) for k, v in domain_channels.items()
        }
        for domain, channels in self.domain_channels.items():
            if not channels:
                raise ValueError(f"domain {domain} must have at least one channel")
        self.channel_names = tuple(channel_names) if channel_names else None
        self.domain_d = int(domain_d)
        self.strict_conditioning = bool(strict_conditioning)
        self.specialists: dict[str, RoleBasedArchitecture] = {}
        self.prefit_diagnostics: dict[str, ChannelDiagnosticReport] = {}
        self.overall_diagnostic: ChannelDiagnosticReport | None = None
        self._fitted = False

    def fit(self, windows, noise_lag: int = 1, groups=None, mode_ids=None):
        """Fit domain specialists after checking conditioning and redundancy."""
        x = np.asarray(windows, dtype=np.float64)
        if x.ndim != 3:
            raise ValueError("windows must have shape (n_windows, n_channels, length)")

        # Run overall telemetry diagnosis
        self.overall_diagnostic = diagnose_channels(
            x,
            list(self.channel_names) if self.channel_names else None,
        )
        if self.strict_conditioning and not self.overall_diagnostic.is_safe:
            raise ValueError(
                f"overall telemetry conditioning failed: status={self.overall_diagnostic.status}, "
                f"rank={self.overall_diagnostic.matrix_rank}/{self.overall_diagnostic.full_rank}, "
                f"redundant_groups={self.overall_diagnostic.redundant_groups}"
            )

        self.specialists.clear()
        self.prefit_diagnostics.clear()

        for domain, channels in self.domain_channels.items():
            sub_x = x[:, list(channels), :]
            sub_names = (
                [self.channel_names[i] for i in channels]
                if self.channel_names
                else [f"ch_{i}" for i in channels]
            )
            diag = diagnose_channels(sub_x, sub_names)
            self.prefit_diagnostics[domain] = diag
            if self.strict_conditioning and not diag.is_safe:
                raise ValueError(
                    f"domain '{domain}' conditioning failed: status={diag.status}, "
                    f"rank={diag.matrix_rank}/{diag.full_rank}"
                )

            specialist = RoleBasedArchitecture(d=self.domain_d)
            specialist.fit(sub_x, noise_lag=noise_lag, groups=groups, mode_ids=mode_ids)
            self.specialists[domain] = specialist

        self._fitted = True
        return self

    def fit_calibration(self, windows, mode_ids=None):
        """Calibrate each domain specialist on a separate healthy period."""
        if not self._fitted:
            raise RuntimeError("architecture must be fitted before calibration")
        x = np.asarray(windows, dtype=np.float64)
        for domain, specialist in self.specialists.items():
            channels = self.domain_channels[domain]
            sub_x = x[:, list(channels), :]
            specialist.fit_calibration(sub_x, mode_ids=mode_ids)
        return self

    def score(self, windows, mode_ids=None) -> SpecialistScores:
        """Compute per-specialist role scores and fused continuous signals."""
        if not self._fitted:
            raise RuntimeError("architecture must be fitted before score")
        x = np.asarray(windows, dtype=np.float64)
        domain_scores: dict[str, RoleScores] = {}
        domain_p: dict[str, dict[str, np.ndarray]] = {}
        all_continuous: list[np.ndarray] = []
        all_p: list[np.ndarray] = []

        for domain, specialist in self.specialists.items():
            channels = self.domain_channels[domain]
            sub_x = x[:, list(channels), :]
            scores = specialist.score(sub_x, mode_ids=mode_ids)
            p_values = specialist.p_values(scores)
            domain_scores[domain] = scores
            domain_p[domain] = p_values

            # Domain score: sum of branch logs
            branch_p = np.stack(
                [p_values[b] for b in RoleBasedArchitecture._ALARM_BRANCHES],
                axis=0,
            )
            # Surprisal from min p-value across branches for this domain
            domain_min_p = np.min(branch_p, axis=0)
            domain_surprisal = -np.log(np.maximum(domain_min_p, 1e-12))
            all_continuous.append(domain_surprisal)
            all_p.append(domain_min_p)

        # Fused continuous score is the sum of domain surprisals
        fused_score = np.sum(np.stack(all_continuous, axis=0), axis=0)
        fused_p = np.min(np.stack(all_p, axis=0), axis=0)

        return SpecialistScores(
            domain_scores=domain_scores,
            domain_p_values=domain_p,
            fused_score=fused_score,
            fused_p_value=fused_p,
        )

    def fuse(
        self,
        scores: SpecialistScores,
        alpha: float = 0.01,
        branch_policy: str = "dynamic_health",
        specialist_rule: Literal["or", "at_least_2", "all"] = "or",
    ) -> tuple[np.ndarray, list[list[str]]]:
        """Combine domain specialist alarms and produce root-cause attributions."""
        domain_flags: dict[str, np.ndarray] = {}
        for domain, specialist in self.specialists.items():
            p_dict = scores.domain_p_values[domain]
            flags = specialist.fuse(p_dict, alpha=alpha, policy=branch_policy)
            domain_flags[domain] = flags

        domains = list(self.specialists.keys())
        stacked = np.stack([domain_flags[d] for d in domains], axis=1)

        if specialist_rule == "or":
            fused_alarm = stacked.any(axis=1)
        elif specialist_rule == "at_least_2":
            fused_alarm = stacked.sum(axis=1) >= 2
        elif specialist_rule == "all":
            fused_alarm = stacked.all(axis=1)
        else:
            raise ValueError(f"unknown specialist_rule: {specialist_rule}")

        attributions: list[list[str]] = []
        for i in range(len(fused_alarm)):
            active = [domains[j] for j in range(len(domains)) if bool(stacked[i, j])]
            attributions.append(active)

        return fused_alarm, attributions

    @staticmethod
    def memory_route(
        alarm_open: bool,
        active_domains: list[str],
        *,
        data_quality_ok: bool,
        operator_confirmed_normal: bool = False,
    ) -> tuple[str, dict]:
        """Route window into memory with explicit provenance and attribution metadata."""
        if not data_quality_ok:
            return "QUARANTINE", {"reason": "data_quality_rejected"}
        if alarm_open or len(active_domains) > 0:
            return "EPISODIC", {
                "reason": "alarm_open" if alarm_open else "domain_exceedance",
                "triggering_domains": list(active_domains),
            }
        if operator_confirmed_normal:
            return "SEMANTIC", {"reason": "operator_confirmed_normal"}
        return "QUARANTINE", {"reason": "unconfirmed_baseline"}
