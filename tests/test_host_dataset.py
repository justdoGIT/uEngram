import gzip
import json

import numpy as np
import pytest

from uengram.host import dataset
from uengram.host.collector import JsonlSink
from uengram.host.features import read_columns
from uengram.host.node import NodeMemory
from uengram.host.synth import make_ticks


def _model_and_ticks(tmp_path):
    ticks = make_ticks(4 * 3600, seed=7)
    source = tmp_path / "ticks.jsonl"
    sink = JsonlSink(str(source), {"host": "synth", "kernel": "test-kernel"})
    for row in ticks:
        sink(row["t"], {k: v for k, v in row.items() if k != "t"})
    sink.close()
    meta, cols = read_columns([str(source)])
    model = NodeMemory.commission(cols, meta)
    model_path = tmp_path / "model.pkl"
    model.save(str(model_path))
    return ticks, source, model_path


def test_load_incidents_rejects_bad_interval(tmp_path):
    path = tmp_path / "incidents.jsonl"
    path.write_text(json.dumps({"node": "synth", "start": 4, "end": 3}) + "\n")
    with pytest.raises(dataset.DatasetError, match="later than start"):
        dataset.load_incidents(str(path))


def test_export_preserves_raw_missing_values_and_label_provenance(tmp_path):
    ticks, source, model = _model_and_ticks(tmp_path)
    labels = [{"id": "maint-1", "node": "synth", "start": ticks[5000]["t"],
               "end": ticks[5035]["t"], "type": "cooling", "source": "operator",
               "operator_confirmed": True}]
    ticks[100]["pkg_temp_c"] = None
    # The source is intentionally written again so the exported raw row contains null.
    source = tmp_path / "ticks-with-null.jsonl"
    sink = JsonlSink(str(source), {"host": "synth", "kernel": "test-kernel"})
    for row in ticks:
        sink(row["t"], {k: v for k, v in row.items() if k != "t"})
    sink.close()

    out = tmp_path / "dataset"
    manifest = dataset.export_single(str(out), "synth", [str(source)], str(model), labels)
    assert manifest["uengram_outputs_are_predictions"] is True
    assert manifest["labels"]["records"] == 1
    assert manifest["nodes"][0]["windows"] > 0
    assert manifest["nodes"][0]["quality"]["missing_values"]["pkg_temp_c"] == 1

    with gzip.open(out / "nodes" / "synth" / "ticks.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    assert any(row.get("pkg_temp_c") is None for row in rows)
    with gzip.open(out / "nodes" / "synth" / "windows.jsonl.gz", "rt") as f:
        windows = [json.loads(line) for line in f]
    marked = [row for row in windows if row["labels"]["incidents"]]
    assert marked and marked[0]["labels"]["status"] == "confirmed"
    assert marked[0]["uengram_prediction"]["alarm"] == marked[0]["alarm"]



def test_campaign_export_combines_nodes_and_marks_faults_controlled(tmp_path):
    ticks, source, model = _model_and_ticks(tmp_path)
    campaign = tmp_path / "campaign"
    spool = campaign / "nodes" / "synth" / "spool"
    models = campaign / "models"
    spool.mkdir(parents=True)
    models.mkdir()
    import shutil
    shutil.copy2(source, spool / "ticks-synth-20261025.jsonl")
    shutil.copy2(model, models / "model-synth.pkl")
    (campaign / "campaign.json").write_text(json.dumps({
        "start": ticks[0]["t"], "clean_split": ticks[100]["t"],
        "commission": ticks[200]["t"], "end": ticks[-1]["t"],
        "alpha": 0.01, "nodes": {"synth": {"faults": ["cpu"]}},
    }))
    (campaign / "faults.jsonl").write_text(json.dumps({
        "id": "f1", "node": "synth", "peer": None, "type": "cpu",
        "level": 1, "start": ticks[1000]["t"], "end": ticks[1032]["t"],
        "status": "done",
    }) + "\n")
    out = tmp_path / "cluster"
    manifest = dataset.export_campaign(str(campaign), str(out))
    assert manifest["dataset_kind"] == "cluster_campaign"
    assert manifest["nodes"][0]["name"] == "synth"
    assert manifest["labels"]["records"] == 1


def test_controlled_labels_cannot_become_field_ground_truth(tmp_path):
    ticks, source, model = _model_and_ticks(tmp_path)
    incident = {"id": "fault-1", "node": "synth", "start": ticks[1000]["t"],
                "end": ticks[1032]["t"], "type": "cpu", "source": "controlled_campaign",
                "operator_confirmed": False, "ground_truth_eligible": False}
    out = tmp_path / "controlled"
    dataset.export_single(str(out), "synth", [str(source)], str(model), [incident], include_ticks=False)
    with gzip.open(out / "nodes" / "synth" / "windows.jsonl.gz", "rt") as f:
        windows = [json.loads(line) for line in f]
    marked = [row for row in windows if row["labels"]["incidents"]]
    assert marked and marked[0]["labels"]["status"] == "controlled"
    assert marked[0]["labels"]["incidents"][0]["ground_truth_eligible"] is False
    assert not (out / "nodes" / "synth" / "ticks.jsonl.gz").exists()


def test_json_value_converts_non_finite_values_to_null():
    value = dataset._json_value({"nan": np.nan, "inf": np.inf, "ok": np.float64(3)})
    assert value == {"nan": None, "inf": None, "ok": 3.0}
