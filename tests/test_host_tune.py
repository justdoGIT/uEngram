import numpy as np
import pytest

from uengram.host.tune import analyse, paired_change


def _rows(values: dict[str, tuple[float, float]], reps: int, noise: float = 0.01, seed: int = 0):
    """values: name -> (energy factor, time factor) relative to the baseline 'base'."""
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(reps):
        drift = 1 + 0.05 * rep  # e.g. warming up: shared by every value of a rep
        for v, (fe, ft) in {"base": (1.0, 1.0), **values}.items():
            rows.append({"value": v, "rep": rep, "rc": 0, "foreign_share": 0.06,  # the kernel's own share
                         "energy_j": 1000 * fe * drift * (1 + noise * rng.normal()),
                         "wall_s": 60 * ft * drift * (1 + noise * rng.normal())})
    return rows


def test_recommends_the_largest_clear_saving_within_the_time_tolerance():
    rows = _rows({"saves": (0.90, 1.02), "too_slow": (0.80, 1.20), "unclear": (0.995, 1.0)}, reps=6)
    res = analyse(rows, "base", tolerance=0.05)
    assert res["recommended"] == "saves"
    e = res["values"]["saves"]["energy_change_pct"]
    assert -12 < e["pct"] < -8 and e["lo"] < e["pct"] < e["hi"] < 0


def test_without_paired_evidence_the_baseline_stays():
    rows = _rows({"saves": (0.90, 1.0)}, reps=1)
    assert analyse(rows, "base", tolerance=0.05)["recommended"] == "base"


def test_runs_with_another_job_on_the_node_do_not_count():
    rows = _rows({"saves": (0.90, 1.0)}, reps=6)
    for r in rows:
        if r["value"] == "saves" and r["rep"] < 5:
            r["foreign_share"] = 0.35  # a job ran alongside these builds
    res = analyse(rows, "base", tolerance=0.05)
    assert res["values"]["saves"]["runs"] == 1
    assert res["recommended"] == "base"  # one paired rep is no evidence


def test_paired_change_is_a_percent_change_with_an_interval_on_logs():
    c = paired_change([100.0, 200.0, 400.0], [90.0, 180.0, 360.0])
    assert c["pct"] == pytest.approx(-10.0) and c["lo"] == pytest.approx(-10.0) and c["n"] == 3
