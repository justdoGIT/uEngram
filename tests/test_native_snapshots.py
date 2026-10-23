import json
from types import SimpleNamespace

import numpy as np

from bench.native_snapshots import archive_native


def _stream():
    ends = np.datetime64("2024-01-01T00:00:00") + np.array([0, 2, 4]).astype("timedelta64[s]")
    values = np.array([[[10.0]], [[20.0]], [[40.0]]], dtype=np.float32)
    return SimpleNamespace(
        name="fixture", W=values, cond=np.zeros((3, 4, 1), dtype=np.float32),
        test=np.array([True, True, True]), start=ends, end=ends,
        segment=np.zeros(3, dtype=np.int64), extra={},
    )


def _csv(path, rows):
    path.write_text(
        "timestamp,TP2,TP3,H1,DV_pressure,Reservoirs,Oil_temperature,Motor_current,COMP,DV_eletric,MPG,LPS\n"
        + "".join(",".join(map(str, row)) + "\n" for row in rows)
    )


def test_native_rows_keep_fractional_times_values_and_per_episode_causality(tmp_path):
    source = tmp_path / "source.csv"
    _csv(source, [
        ("2024-01-01 00:00:00.100", 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0),
        ("2024-01-01 00:00:02.500", 20, 21, 22, 23, 24, 25, 26, 0, 0, 0, 0),
        ("2024-01-01 00:00:03.300", 30, 31, 32, 33, 34, 35, 36, 0, 0, 0, 0),
        ("2024-01-01 00:00:04.200", 40, 41, 42, 43, 44, 45, 46, 0, 0, 0, 0),
        ("2024-01-01 00:00:06.000", 60, 61, 62, 63, 64, 65, 66, 0, 0, 0, 0),
    ])
    stream = _stream()
    # Two overlapping intervals; third is clipped at stream end (no fabricated future).
    episodes = [{"open_index": 1}, {"open_index": 2}]
    out = tmp_path / "native.npz"
    info = archive_native(stream, episodes, str(out), {"method": "fixture"}, radius=1,
                          source_path=str(source), progress_interval=0)
    with np.load(out) as z:
        rows = z["row_number"].tolist()
        assert rows == [0, 1, 2, 3]
        assert z["timestamp"].tolist()[1] == "2024-01-01T00:00:02.500000000"
        assert z["raw_values"][1, 1] == "20"
        assert z["numeric_values"][1, 1] == 20
        assert z["row_sha256"].shape == (4,)
        manifest = json.loads(str(z["manifest"]))
        records = manifest["episodes"]
        assert records[0]["model_final_sample_match"]
        assert records[0]["decision_timestamp_native"] == "2024-01-01T00:00:02.500000000"
        assert records[0]["post_decision_row_count"] == 2
        assert z["episode_causal_mask"][0].tolist() == [True, True, False, False]
        assert z["episode_model_window_map"][0].tolist() == [0, 1, -1, 2]
        assert records[0]["context_gaps"]
        assert records[0]["context_row_count"] == 4
        assert manifest["units"]["TP2"].startswith("unknown")
        assert records[1]["missing_requested_post_model_offsets"] == [1]
        assert records[1]["missing_post_context"] is True
    assert info["native_rows"] == 4
    assert info["source_sha256"]


def test_identical_endpoint_candidates_use_first_native_observation(tmp_path):
    source = tmp_path / "duplicates.csv"
    values = (20, 21, 22, 23, 24, 25, 26, 0, 0, 0, 0)
    _csv(source, [
        ("2024-01-01 00:00:02.500", *values),
        ("2024-01-01 00:00:02.800", *values),
    ])
    out = tmp_path / "duplicates.npz"
    archive_native(_stream(), [{"open_index": 1}], str(out), {}, radius=1,
                   source_path=str(source), progress_interval=0)
    with np.load(out) as z:
        record = json.loads(str(z["manifest"]))["episodes"][0]
        assert record["matching_final_native_rows"] == 2
        assert record["decision_timestamp_native"] == "2024-01-01T00:00:02.500000000"


def test_unmatched_final_sample_fails_closed_on_causal_boundary(tmp_path):
    source = tmp_path / "mismatch.csv"
    _csv(source, [("2024-01-01 00:00:02.500", 99, 21, 22, 23, 24, 25, 26, 0, 0, 0, 0)])
    out = tmp_path / "mismatch.npz"
    archive_native(_stream(), [{"open_index": 1}], str(out), {}, radius=1,
                   source_path=str(source), progress_interval=0)
    with np.load(out) as z:
        record = json.loads(str(z["manifest"]))["episodes"][0]
        assert record["decision_timestamp_native"] is None
        assert record["causal_boundary_known"] is False
        assert record["causal_row_count"] is None
        assert not z["episode_causal_mask"][0].any()
        assert not z["episode_causal_known"][0].any()


def test_missing_source_postcontext_is_reported_without_filling(tmp_path):
    source = tmp_path / "short.csv"
    _csv(source, [
        ("2024-01-01 00:00:00.100", 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0),
        ("2024-01-01 00:00:02.500", 20, 21, 22, 23, 24, 25, 26, 0, 0, 0, 0),
    ])
    out = tmp_path / "short.npz"
    archive_native(_stream(), [{"open_index": 1}], str(out), {}, radius=1,
                   source_path=str(source), progress_interval=0)
    with np.load(out) as z:
        records = json.loads(str(z["manifest"]))["episodes"]
        assert z["row_number"].tolist() == [0, 1]
        assert records[0]["post_decision_row_count"] == 0
        assert records[0]["missing_context"] is False
        assert records[0]["missing_post_context"] is True
        assert records[0]["model_final_sample_match"]
