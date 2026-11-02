"""Report measures added with uEngram v3 (numpy).

* alarm_episodes: false-alarm episodes per week, as in upstream's MetroPT-3
  protocol (docs/EXPLAINED.md 16): alarms on normal windows, merged when less
  than an hour apart, per week of observed normal window time. On MetroPT-3 no
  detector held a one-week conformal calibration for six months; the window
  share bounds the rate, the episode count is what an operator sees.
* step_changes_per_day: supersessions by the step detector, the quantity the
  v3 step calibration (stepcal.py) exists to keep at real changes.
* health_ranking: how well the health score (distance to the frozen healthy
  reference) and the alarm score rank fault windows above normal ones, per
  fault type: AUROC per node, averaged over the nodes with both classes.
  Health is reported, not alarmed: this measures whether it should be.
"""
from __future__ import annotations

import numpy as np

from ..metrics import auroc

EPISODE_GAP_S = 3600.0
WEEK_S = 7 * 86400.0


def alarm_episodes(normal: list[dict], K: int, fired) -> float:
    t = sorted(r["t0"] for r in normal if fired(r))
    episodes = sum(1 for i, x in enumerate(t) if i == 0 or x - t[i - 1] >= EPISODE_GAP_S)
    exposure = len(normal) * K / WEEK_S
    return episodes / exposure if exposure > 0 else float("nan")


def step_changes_per_day(recs: list[dict], K: int) -> float:
    days = len(recs) * K / 86400.0
    return sum(r["action"] == "step_change" for r in recs) / days if days > 0 else float("nan")


def health_ranking(recs_by_node: dict[str, list[dict]], faults: list[dict], K: int,
                   near_fault, in_fault) -> dict[str, dict]:
    """fault type -> {"health": mean AUROC, "novelty": mean AUROC, "nodes": n}."""
    types = sorted({f["type"] for f in faults if f.get("status") == "done"})
    out = {}
    for ftype in types:
        per = {"health": [], "novelty": []}
        for name, recs in recs_by_node.items():
            mine = [f for f in faults if f.get("status") == "done" and f["type"] == ftype and f["node"] == name]
            pos = [r for r in recs if any(in_fault(r["t0"], f, K) for f in mine)]
            neg = [r for r in recs if not near_fault(r["t0"], faults, name, K)]
            if not pos or not neg:
                continue
            labels = np.array([True] * len(pos) + [False] * len(neg))
            for key, field in (("health", "persistent_health"), ("novelty", "persistent_score")):
                s = np.array([r.get(field, np.nan) for r in pos + neg], dtype=np.float64)
                ok = np.isfinite(s)
                if labels[ok].any() and (~labels[ok]).any():
                    per[key].append(auroc(s[ok], labels[ok]))
        out[ftype] = {k: float(np.mean(v)) if v else float("nan") for k, v in per.items()}
        out[ftype]["nodes"] = len(per["health"])
    return out


def health_table(rank: dict[str, dict]) -> list[str]:
    lines = ["| fault | nodes | health AUROC | alarm-score AUROC |", "|---|---|---|---|"]
    for ftype, r in rank.items():
        lines.append(f"| {ftype} | {r['nodes']} | {r['health']:.3f} | {r['novelty']:.3f} |")
    return lines
