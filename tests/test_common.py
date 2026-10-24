import json

import numpy as np

from bench.common import dump


def test_dump_serializes_numpy_scalars_arrays_and_datetime(tmp_path):
    path = tmp_path / "records.jsonl"
    dump(str(path), [{
        "integer": np.int64(3),
        "real": np.float64(1.25),
        "array": np.array([1, 2], dtype=np.int64),
        "time": np.datetime64("2026-01-02T03:04:05"),
    }])
    record = json.loads(path.read_text())
    assert record == {
        "integer": 3,
        "real": 1.25,
        "array": [1, 2],
        "time": "2026-01-02T03:04:05",
    }
