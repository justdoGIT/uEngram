import json
import sys

import pytest

from bench import synth_bootstrap


def _row(seed: int, encoder: str, method: str, retention: float) -> str:
    return json.dumps({
        "dataset": "synth", "seed": seed, "encoder": encoder, "method": method, "budget": 256,
        "metrics": {"event_recall": retention, "write_reduction_pct": 80.0 + seed,
                    "ap": 0.9 - seed * 0.01},
    })


def _write_jsonl(path, encoder: str, retentions: dict) -> None:
    lines = [_row(seed, encoder, "uengram", value) for seed, value in retentions.items()]
    path.write_text("\n".join(lines) + "\n")


def test_main_writes_means_and_paired_difference(tmp_path, monkeypatch):
    d16 = tmp_path / "synth_runs.jsonl"
    d36 = tmp_path / "synth_d36_runs.jsonl"
    _write_jsonl(d16, "dsp", {0: 1.0, 1: 1.0, 2: 1.0})
    _write_jsonl(d36, "dsp_d36", {0: 0.8, 1: 0.75, 2: 0.82})
    out = tmp_path / "ci.json"

    monkeypatch.setattr(sys, "argv", [
        "synth_bootstrap", "--d16", str(d16), "--d36", str(d36), "--out", str(out), "--rng", "1",
    ])
    synth_bootstrap.main()

    result = json.loads(out.read_text())
    assert result["draws"] == synth_bootstrap.DRAWS
    d16_entry = result["means"]["d16/uengram"]
    assert d16_entry["n_seeds"] == 3
    assert d16_entry["event_recall"]["mean"] == 1.0
    lo, hi = d16_entry["event_recall"]["ci95"]
    assert lo <= 1.0 <= hi

    d36_entry = result["means"]["d36/uengram"]
    assert d36_entry["event_recall"]["mean"] == pytest.approx((0.8 + 0.75 + 0.82) / 3)

    paired = result["paired_d36_minus_d16"]["uengram"]
    assert paired["n_seeds"] == 3
    expected_mean_diff = d36_entry["event_recall"]["mean"] - d16_entry["event_recall"]["mean"]
    assert paired["event_recall"]["mean"] == pytest.approx(expected_mean_diff)
