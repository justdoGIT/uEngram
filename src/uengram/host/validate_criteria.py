"""Pre-registered pass/fail criteria of the validation campaign.

`campaign.plan` stores `criteria(alpha)` in campaign.json together with the
time of registration, before any test data exists; the report evaluates
exactly the stored values (`evaluate`) and prints one verdict per criterion.

The thresholds are reachable with the planned sample sizes. A fault type's
strongest level runs about 3 times per node (10 faults cycling over 2-3
levels), so 10-17 times fleet-wide: a perfect 10/10 has a Wilson 95 % lower
bound of 0.72, 9/10 of 0.60, 8/10 of 0.49. A clean week without any alarm
has a block-bootstrap upper bound of 3b/n (rule of three over n/b blocks),
about 0.001 for four days and the minimum block of 4 windows.

Stdlib only: campaign.plan imports this module without numpy.
"""
from __future__ import annotations

import math


def criteria(alpha: float) -> dict:
    return {
        # 1. clean week, per node: share of windows with an alarm (point, upper 95 % bound)
        "clean_latent_max": alpha / 2, "clean_latent_upper": alpha,
        "clean_event_max": alpha / 2, "clean_event_upper": alpha,
        # 2. strongest level of each fault type, target nodes pooled
        "detect_strongest_min": 0.8, "detect_strongest_lower": 0.5,
        # 3. the strongest level is detected at least as often as the weakest
        "monotone_strength": True,
        # 4. switching a component off lowers detection or raises false alarms
        "ablations_must_cost": True,
        # 5. the full model detects more than each baseline, or as much with fewer false alarms
        "beat_baselines": True,
        # 6. the online daemon computed what the offline replay computes
        "online_identical_min": 0.99,
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Share k/n with its Wilson score 95 % interval."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, c - h), min(1.0, c + h)


def _row(criterion: str, scope: str, value: str, required: str, ok: bool | None) -> dict:
    return {"criterion": criterion, "scope": scope, "value": value, "required": required,
            "verdict": "n/a" if ok is None else "PASS" if ok else "FAIL"}


def _rate(rows: list[dict]) -> tuple[int, int]:
    return sum(r["detected"] for r in rows), len(rows)


def evaluate(crit: dict, levels: dict, clean: dict, det: list[dict], cons: dict,
             summary: dict) -> list[dict]:
    """One row per criterion and scope. `summary` is validate_variants.variant_summary."""
    out = []
    for node, c in clean.items():
        for ch in ("latent", "event"):
            if "error" in c:
                out.append(_row(f"clean week: {ch} false alarms", node, c["error"], "", None))
                continue
            m, _, hi = c[ch]
            mx, up = crit[f"clean_{ch}_max"], crit[f"clean_{ch}_upper"]
            out.append(_row(f"clean week: {ch} false alarms", node, f"{m:.4f} (upper {hi:.4f})",
                            f"<= {mx:.4f} (upper <= {up:.4f})", m <= mx and hi <= up))
    target = [r for r in det if r["role"] == "target"]
    for ftype, lv in levels.items():
        k, n = _rate([r for r in target if r["type"] == ftype and r["level"] == lv[-1]])
        p, lo, _ = wilson(k, n)
        out.append(_row("strongest level detected", f"{ftype} {lv[-1]}", f"{k}/{n} (lower {lo:.2f})" if n else "0 run",
                        f">= {crit['detect_strongest_min']} (lower >= {crit['detect_strongest_lower']})",
                        None if n == 0 else p >= crit["detect_strongest_min"] and lo >= crit["detect_strongest_lower"]))
        if crit["monotone_strength"] and len(lv) > 1:
            ks, ns = _rate([r for r in target if r["type"] == ftype and r["level"] == lv[-1]])
            kw, nw = _rate([r for r in target if r["type"] == ftype and r["level"] == lv[0]])
            ok = None if not (ns and nw) else ks / ns >= kw / nw
            out.append(_row("strongest >= weakest", ftype, f"{ks}/{ns} vs {kw}/{nw}", "strongest share >= weakest", ok))
    full = summary.get("full")
    for name, s in summary.items():
        if full is None or name == "full":
            continue
        d_full, d, fa_full, fa = full["det"][0], s["det"][0], full["fa"][0], s["fa"][0]
        value = f"detected {d_full:.2f} vs {d:.2f}, false alarms {fa_full:.4f} vs {fa:.4f}"
        if name.startswith("no_") and crit["ablations_must_cost"]:
            out.append(_row("component earns its place", name, value, "full detects more or alarms less",
                            d_full > d or fa > fa_full))
        elif not name.startswith("no_") and crit["beat_baselines"]:
            out.append(_row("beats simpler baseline", name, value, "full detects more, or equal with fewer alarms",
                            d_full > d or (d_full == d and fa_full < fa)))
    for node, c in cons.items():
        ok = None if not c.get("matched") else c["identical"] >= crit["online_identical_min"]
        out.append(_row("online equals offline", node, f"{c.get('identical', float('nan')):.3f}",
                        f">= {crit['online_identical_min']}", ok))
    return out


def verdict_table(rows: list[dict]) -> list[str]:
    out = ["| criterion | scope | value | required | verdict |", "|---|---|---|---|---|"]
    out += [f"| {r['criterion']} | {r['scope']} | {r['value']} | {r['required']} | {r['verdict']} |" for r in rows]
    fails = sum(r["verdict"] == "FAIL" for r in rows)
    na = sum(r["verdict"] == "n/a" for r in rows)
    out += ["", f"{len(rows) - fails - na} passed, {fails} failed, {na} not evaluable."]
    return out
