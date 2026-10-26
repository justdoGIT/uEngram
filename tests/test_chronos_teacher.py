import numpy as np
import pytest

from uengram.chronos_teacher import Chronos2TeacherAdapter
from uengram.teacher_transfer import TeacherSpec, WindowSchema


def test_chronos_adapter_flattens_quantiles_and_median_in_order():
    torch = pytest.importorskip("torch")

    class FakePipeline:
        def predict_quantiles(self, values, **kwargs):
            assert values.shape == (3, 1, 8)
            assert kwargs["prediction_length"] == 2
            q = [torch.arange(6, dtype=torch.float32).reshape(1, 2, 3) + i for i in range(3)]
            m = [torch.arange(2, dtype=torch.float32).reshape(1, 2) + i for i in range(3)]
            return q, m

    spec = TeacherSpec(
        model_name="fake-chronos",
        model_version="test",
        layer="forecast",
        feature_dim=8,
        input_schema=WindowSchema(("vibration",), 1.0, 8),
    )
    adapter = Chronos2TeacherAdapter(FakePipeline(), spec, prediction_length=2)
    result = adapter.encode(np.zeros((3, 1, 8), dtype=np.float32))

    assert result.shape == (3, 8)
    np.testing.assert_allclose(result[0], [0, 1, 2, 3, 4, 5, 0, 1])
    assert np.isfinite(result).all()


def test_chronos_adapter_rejects_schema_mismatch():
    class FakePipeline:
        def predict_quantiles(self, values, **kwargs):
            raise AssertionError("pipeline must not run")

    spec = TeacherSpec(
        model_name="fake-chronos",
        model_version="test",
        layer="forecast",
        feature_dim=8,
        input_schema=WindowSchema(("vibration",), 1.0, 8),
    )
    adapter = Chronos2TeacherAdapter(FakePipeline(), spec, prediction_length=2)

    with pytest.raises(ValueError, match="window_size"):
        adapter.encode(np.zeros((1, 1, 7), dtype=np.float32))
