import numpy as np

from bench import admission_ablation, whitening_ablation
from uengram import encoders


def test_admission_switch_changes_results_at_a_saturating_budget():
    """Guards against both arms silently running identical code.

    ``run_synth.main`` uses a spawn pool whose workers ignore parent-process patches; the
    ablation must run in-process, where the switch demonstrably reaches the engine.
    """
    competitive = admission_ablation._run(4, 1.0, "0")
    accept_all = admission_ablation._run(4, 0.0, "0")
    assert competitive["event_recall"] > accept_all["event_recall"] + 0.05


def test_admission_arms_restore_the_original_store_factory():
    original = admission_ablation.rs.make_store
    admission_ablation._run(4, 0.0, "0")
    assert admission_ablation.rs.make_store is original


def test_identity_whitening_is_the_identity_and_is_restored():
    original = encoders.NoiseWhitener.fit
    whitener = encoders.NoiseWhitener()
    whitening_ablation._identity_fit(whitener, np.ones((5, 3)), 1, None)
    assert np.array_equal(whitener.W, np.eye(3))
    assert encoders.NoiseWhitener.fit is original
