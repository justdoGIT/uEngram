import io
import zipfile

import numpy as np

from bench.moment_tsb_ad_holdout import Series, _read_series, _windows


def test_tsb_windows_keep_healthy_training_before_anomaly():
    values = np.arange(256, dtype=np.float32)
    labels = np.zeros(256, dtype=np.int8)
    labels[128:136] = 1
    series = Series("fixture.csv", values, labels, first_anomaly=128)

    train, test, test_labels = _windows(
        series, K=8, stride=8, train_fraction=0.5
    )

    assert train.shape == (8, 1, 8)
    assert test.shape == (24, 1, 8)
    assert test_labels.tolist()[0:8] == [False] * 8
    assert test_labels.tolist()[8] is True
    assert test_labels.tolist()[9:] == [False] * 15
    assert np.isfinite(test).all()


def test_tsb_csv_reader_validates_and_preserves_labels():
    payload = b"Data,Label\n1.0,0\n2.0,1\n3.0,0\n4.0,0\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("TSB-AD-U/fixture.csv", payload)
    buffer.seek(0)

    with zipfile.ZipFile(buffer) as archive:
        result = _read_series(archive, "fixture.csv")

    np.testing.assert_allclose(result.values, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(result.labels, [0, 1, 0, 0])
    assert result.first_anomaly == 1
