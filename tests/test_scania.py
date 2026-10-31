from pathlib import Path

from bench.stream_scania import load_scania


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_scania_loader_keeps_missing_values_and_splits_healthy_rows(tmp_path):
    root = tmp_path / "2024-34-3"
    header = "vehicle_id,time_step,feature_a,feature_b\n"
    _write(root / "data/train_operational_readouts.csv", header +
           "1,0,1.0,2.0\n1,1,,2.5\n2,0,3.0,4.0\n")
    _write(root / "data/validation_operational_readouts.csv", header +
           "10,0,5.0,6.0\n10,1,5.5,\n")
    _write(root / "data/train_tte.csv", "vehicle_id,length_of_study_time_step,in_study_repair\n"
           "1,2,0\n2,1,1\n")
    _write(root / "data/validation_labels.csv", "vehicle_id,class_label\n10,2\n")

    stream = load_scania(root)
    assert stream.W.shape == (4, 2)
    assert stream.fit.sum() == 2  # vehicle 1 is healthy and not divisible by 5
    assert stream.cal.sum() == 0  # vehicle 2 is repaired and excluded
    assert stream.test.sum() == 2
    assert stream.extra["missing_any"].tolist() == [False, True, False, True]
    assert stream.extra["class_label"][stream.test].tolist() == [2, 2]
    assert stream.monitorable.tolist() == [False, True, False, True]
