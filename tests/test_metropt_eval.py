import numpy as np
import pytest

from bench.metropt_calibration import innovation_inflation
from bench.metropt_eval import episodes, evaluate
from uengram import metropt


@pytest.mark.parametrize("rho", [0.0, 0.6])
def test_innovation_scale_works_with_short_independent_state_runs(rho):
    rng = np.random.default_rng(44)
    # No run is long enough for the old lag-38 estimator. Each visit sequence
    # is stationary unit-variance noise; EWMA variance has a known AR(1) value.
    # Independent runs, not highly correlated visits, control sampling error.
    x = rng.normal(size=(4000, 4))
    runs = np.empty((4000, 36, 4))
    for i in range(36):
        x = rho * x + np.sqrt(1 - rho * rho) * rng.normal(size=x.shape)
        runs[:, i] = x
    scale, _ = innovation_inflation(runs, .1)
    a = .9 * rho
    np.testing.assert_allclose(scale, (1 + a) / (1 - a), rtol=.15)


def test_innovation_scale_does_not_invent_missing_calibration():
    with pytest.raises(ValueError, match="warmed-up"):
        innovation_inflation([np.zeros((3, 4)) for _ in range(30)], .1)


def test_episode_boundary_is_strictly_less_than_one_hour():
    t = np.datetime64("2020-03-01T00:00", "s")
    times = t + np.array([0, 3599, 7199, 7200], dtype="timedelta64[s]")
    assert episodes(times) == 2  # exact-hour separation starts a new episode
    assert episodes(np.array([], dtype="datetime64[s]")) == 0


def test_early_alerts_use_window_end_and_do_not_imply_report_detection():
    fs = np.datetime64(metropt.FAILURES[0][0], "s")
    end = fs + np.array([-7200, -3600, 600, 1800], dtype="timedelta64[s]")
    start = end - np.timedelta64(600, "s")
    fid = metropt.failure_id(start, end)
    # Pre-report alert is two hours before onset. Positives have no alert.
    scores = np.array([5., 0., 0., 0.])
    m, detail = evaluate(scores, np.full(4, np.nan), start, end, fid,
                         np.zeros(4, bool), 1.)
    assert m["failures_detected"] == 0
    assert detail["failures"][0]["lead_hours"] == -2.
    # The detection is available 10 minutes after onset, not at window start.
    scores = np.array([0., 0., 5., 0.])
    m, detail = evaluate(scores, np.full(4, np.nan), start, end, fid,
                         np.zeros(4, bool), 1.)
    assert m["failures_detected"] == 1
    assert detail["failures"][0]["lead_hours"] == pytest.approx(1/6)


def test_false_alarm_exposure_excludes_gaps_missing_scores_and_warmup():
    start = np.array(["2020-03-02T00:00", "2020-03-03T00:00",
                      "2020-03-10T00:00", "2020-03-17T00:00"], dtype="datetime64[s]")
    end = start + np.timedelta64(600, "s")
    m, detail = evaluate(np.array([2., np.nan, 2., 0.]), np.full(4, np.nan),
                         start, end, np.full(4, -1), np.ones(4, bool), 1., warmup=1)
    assert detail["false_alarm_episodes"] == 1
    assert detail["clean_observed_hours"] == pytest.approx(1/3)
    assert m["false_alarm_per_week"] == pytest.approx(504.)
