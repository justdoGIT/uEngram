"""Synthetic engine-sensor stream with full ground truth."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CHANNELS = ["rpm", "oil_psi", "coolant_temp", "vib_x", "vib_y", "vib_z"]
C = len(CHANNELS)
K = 64
S_AX = np.arange(K) / K

P_REGIME = np.array([[0.97, 0.03, 0.0], [0.015, 0.97, 0.015], [0.0, 0.03, 0.97]])
RPM_LEVEL = np.array([0.8, 1.8, 2.6])
PSI_LEVEL = np.array([2.0, 3.0, 3.5])
TEMP_TARGET = np.array([0.6, 0.85, 1.0])
VIB_AMP = np.array([0.2, 0.35, 0.5])
ANOM_P = {1: 0.003, 2: 0.003, 3: 0.002}
ANOM_START = 200

KNOCK, CAVITATION, THERMAL, SEIZURE = 1, 2, 3, 4


@dataclass
class SynthStream:
    windows: np.ndarray  # [T,C,K] float32
    regime: np.ndarray
    era: np.ndarray
    anomaly: np.ndarray
    event_id: np.ndarray
    wear: np.ndarray


def wear_curve(T: int, maint_frac: float = 0.6) -> tuple[np.ndarray, int]:
    t_m = int(maint_frac * T)
    t = np.arange(T, dtype=np.float64)
    w = np.where(
        t < t_m,
        0.9 * t / t_m,
        0.1 + 0.3 * (t - t_m) / max(T - 1 - t_m, 1),
    )
    return w, t_m


def make_window(rng, regime, wear, era, temp_prev, temp_now, anomaly) -> np.ndarray:
    rpm_level = RPM_LEVEL[regime]
    if anomaly == SEIZURE:
        rpm_level = 0.3
    psi_level = PSI_LEVEL[regime] - 0.6 * wear + (0.5 if era == 1 else 0.0)
    temp_now_eff = temp_now + 0.25 * wear
    temp_prev_eff = temp_prev + 0.25 * wear
    amp = VIB_AMP[regime] * (1.0 + 0.8 * wear)
    f = 4.0 * rpm_level

    psi_sigma, vz_sigma = 0.05, 0.08
    if anomaly == CAVITATION:
        psi_level -= 1.2
        psi_sigma = 0.3
        vz_sigma = 0.24
    if anomaly == SEIZURE:
        amp *= 4.0

    out = np.empty((C, K))
    out[0] = rpm_level + 0.05 * np.sin(2 * np.pi * S_AX) + rng.normal(0, 0.03, K)
    out[1] = psi_level + rng.normal(0, psi_sigma, K)
    temp = temp_prev_eff + (temp_now_eff - temp_prev_eff) * S_AX
    if anomaly == THERMAL:
        temp = temp + 0.4 * S_AX
    out[2] = temp + rng.normal(0, 0.01, K)
    harm = 0.3 * wear * np.sin(2 * np.pi * 2 * f * S_AX)
    out[3] = amp * np.sin(2 * np.pi * f * S_AX) + harm + rng.normal(0, 0.08, K)
    out[4] = amp * np.cos(2 * np.pi * f * S_AX) + harm + rng.normal(0, 0.08, K)
    out[5] = 0.5 * amp * np.sin(2 * np.pi * 2 * f * S_AX) + rng.normal(0, vz_sigma, K)
    if anomaly == KNOCK:
        out[3] += 1.5 * np.sin(2 * np.pi * 25 * S_AX)
        out[3, rng.integers(0, K, size=3)] += 2.0
    if anomaly == SEIZURE:
        out[3:6] += rng.normal(0, 4 * 0.08, (3, K))
    return out.astype(np.float32)


def generate(T: int, seed: int, maint_frac: float = 0.6) -> SynthStream:
    rng = np.random.default_rng(seed)
    wear, t_m = wear_curve(T, maint_frac)
    seizure_t = {int(0.25 * T), int(0.5 * T), int(0.85 * T)}

    windows = np.empty((T, C, K), dtype=np.float32)
    regime = np.zeros(T, dtype=np.int64)
    era = (np.arange(T) >= t_m).astype(np.int64)
    anomaly = np.zeros(T, dtype=np.int64)
    event_id = np.full(T, -1, dtype=np.int64)

    cur = 0
    level = TEMP_TARGET[0]
    active_type, remaining, next_event = 0, 0, 0
    for t in range(T):
        if t > 0:
            cur = int(rng.choice(3, p=P_REGIME[cur]))
        regime[t] = cur
        prev_level = level
        level = prev_level + 0.5 * (TEMP_TARGET[cur] - prev_level)

        if t in seizure_t:
            active_type, remaining = SEIZURE, 1
            event_id[t] = next_event
            next_event += 1
        elif remaining > 0:
            event_id[t] = next_event - 1
        elif t >= ANOM_START:
            u = rng.random()
            acc = 0.0
            for typ in (1, 2, 3):
                acc += ANOM_P[typ]
                if u < acc:
                    active_type, remaining = typ, int(rng.integers(1, 4))
                    event_id[t] = next_event
                    next_event += 1
                    break
        if remaining > 0:
            anomaly[t] = active_type
            remaining -= 1
        windows[t] = make_window(rng, cur, wear[t], int(era[t]), prev_level, level, int(anomaly[t]))
    return SynthStream(windows, regime, era, anomaly, event_id, wear)


def make_probe(seed: int, T: int, maint_frac: float = 0.6) -> SynthStream:
    """Probe set at the final machine state: 30 nominal per regime, 30 per anomaly type."""
    rng = np.random.default_rng(seed + 20_000)
    wear_all, _ = wear_curve(T, maint_frac)
    w_end = float(wear_all[-1])
    wins, reg, ano, eid = [], [], [], []
    ev = 0
    for r in range(3):
        for _ in range(30):
            wins.append(make_window(rng, r, w_end, 1, TEMP_TARGET[r], TEMP_TARGET[r], 0))
            reg.append(r); ano.append(0); eid.append(-1)
    for typ in (1, 2, 3, 4):
        for i in range(30):
            r = i % 3
            wins.append(make_window(rng, r, w_end, 1, TEMP_TARGET[r], TEMP_TARGET[r], typ))
            reg.append(r); ano.append(typ); eid.append(ev); ev += 1
    n = len(wins)
    return SynthStream(
        np.stack(wins), np.array(reg), np.ones(n, dtype=np.int64),
        np.array(ano), np.array(eid), np.full(n, w_end),
    )
