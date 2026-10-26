import numpy as np
import pytest

from uengram import synth
from uengram.encoder_cnn import CNNEncoder
from uengram.teacher_transfer import (
    TeacherSpec,
    WindowSchema,
    extract_teacher_features,
)


def _spec(windows, *, causal=True, alignment="right"):
    return TeacherSpec(
        model_name="test-teacher",
        model_version="0.1",
        layer="middle",
        feature_dim=2,
        input_schema=WindowSchema(
            tuple(f"channel-{i}" for i in range(windows.shape[1])),
            sample_rate_hz=10.0,
            window_size=windows.shape[2],
        ),
        causal=causal,
        alignment=alignment,
    )


def _teacher(windows):
    return np.column_stack((
        windows.mean(axis=(1, 2)),
        windows.std(axis=(1, 2)),
    ))


def test_teacher_features_preserve_order_and_validate_contract():
    windows = synth.generate(24, seed=31).windows
    spec = _spec(windows)
    result = extract_teacher_features(_teacher, windows, spec, batch=7)

    np.testing.assert_allclose(result.values, _teacher(windows))
    assert result.spec == spec

    with pytest.raises(ValueError, match="causal and right-aligned"):
        extract_teacher_features(_teacher, windows, _spec(windows, causal=False))
    with pytest.raises(ValueError, match="window_size"):
        wrong = TeacherSpec(
            model_name=spec.model_name,
            model_version=spec.model_version,
            layer=spec.layer,
            feature_dim=spec.feature_dim,
            input_schema=WindowSchema(
                spec.input_schema.channel_names,
                spec.input_schema.sample_rate_hz,
                spec.input_schema.window_size + 1,
            ),
        )
        extract_teacher_features(_teacher, windows, wrong)


def test_distilled_cnn_is_deterministic_and_records_teacher_identity():
    pytest.importorskip("torch")
    windows = synth.generate(128, seed=32).windows
    spec = _spec(windows)
    first = CNNEncoder(seed=4, d=8, epochs=2, batch=32).fit_distilled(
        windows, _teacher, spec, noise_lag=1, groups=None, teacher_weight=0.5
    )
    second = CNNEncoder(seed=4, d=8, epochs=2, batch=32).fit_distilled(
        windows, _teacher, spec, noise_lag=1, groups=None, teacher_weight=0.5
    )

    np.testing.assert_allclose(first.encode(windows), second.encode(windows))
    assert first.teacher_spec == spec
    assert first.teacher_head is not None
    assert np.isfinite(first.encode(windows)).all()


def test_distillation_rejects_bad_teacher_output_and_weight():
    windows = synth.generate(24, seed=33).windows
    spec = _spec(windows)

    def wrong_shape(batch):
        return np.zeros((len(batch), 3))

    with pytest.raises(ValueError, match="teacher output shape"):
        extract_teacher_features(wrong_shape, windows, spec)
    with pytest.raises(ValueError, match="teacher_weight"):
        CNNEncoder().fit_distilled(
            windows, _teacher, spec, noise_lag=1, groups=None, teacher_weight=-1
        )


def test_moment_adapter_right_aligns_and_reduces_patch_embeddings():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from uengram.moment_teacher import MOMENTTeacherAdapter

    class FakeMoment:
        def __call__(self, *, x_enc, input_mask):
            assert tuple(x_enc.shape) == (2, 2, 8)
            assert tuple(input_mask.shape) == (2, 8)
            patches = x_enc.reshape(2, 2, 4, 2).mean(dim=-1)
            return SimpleNamespace(
                embeddings=patches.mean(dim=1).unsqueeze(-1).repeat(1, 1, 3)
            )

    windows = np.arange(16, dtype=np.float32).reshape(2, 2, 4)
    spec = TeacherSpec(
        model_name="fake-moment",
        model_version="test",
        layer="embedding-mean-patches",
        feature_dim=3,
        input_schema=WindowSchema(("a", "b"), 1.0, 4),
    )
    adapter = MOMENTTeacherAdapter(FakeMoment(), spec, context_length=8)
    result = adapter.encode(windows)

    assert result.shape == (2, 3)
    assert np.isfinite(result).all()
    assert torch is not None


def test_moment_adapter_linear_resampling_can_restore_instance_statistics():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from uengram.moment_teacher import MOMENTTeacherAdapter

    class FakeMoment:
        def __call__(self, *, x_enc, input_mask):
            assert tuple(x_enc.shape) == (2, 2, 8)
            assert tuple(input_mask.shape) == (2, 8)
            return SimpleNamespace(
                embeddings=torch.zeros((2, 1, 3), dtype=torch.float32)
            )

    windows = np.array(
        [
            [[0, 1, 2, 3], [4, 4, 4, 4]],
            [[2, 2, 2, 2], [0, 2, 4, 6]],
        ],
        dtype=np.float32,
    )
    spec = TeacherSpec(
        model_name="fake-moment",
        model_version="test",
        layer="embedding-mean-patches+instance-stats",
        feature_dim=7,
        input_schema=WindowSchema(("a", "b"), 1.0, 4),
    )
    adapter = MOMENTTeacherAdapter(
        FakeMoment(),
        spec,
        context_length=8,
        context_policy="linear",
        include_instance_stats=True,
        base_feature_dim=3,
    )

    result = adapter.encode(windows)

    assert result.shape == (2, 7)
    np.testing.assert_allclose(result[:, 3:5], [[1.5, 4.0], [2.0, 3.0]])
    np.testing.assert_allclose(
        result[:, 5:],
        [[np.sqrt(1.25), 0.0], [0.0, np.sqrt(5.0)]],
    )
