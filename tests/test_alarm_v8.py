import types

import numpy as np
import pytest

from bench.alarm_definition import Definition, actionability, fill_windows
from bench.modality_alarm import hold
from bench.streams import load_stream

T0 = np.datetime64("2020-01-02T00:00:00", "s")
STEP = np.timedelta64(3600, "s")


def _stream(n, event_start_h, lps_h=None):
    end = T0 + np.arange(1, n + 1) * STEP
    start = T0 + np.timedelta64(event_start_h * 3600, "s")
    events = ((str(start), str(start + 10 * STEP)),)
    eid = np.where((end >= start) & (end <= start + 10 * STEP), 0, -1)
    lps = None if lps_h is None else (str(T0 + np.timedelta64(lps_h * 3600, "s")),)
    return types.SimpleNamespace(end=end, events=events, event_id=lambda: eid, lps=lps), np.arange(n)


def _details(active, opened):
    n = len(active)
    return {"active": np.array(active, bool), "opened": np.array(opened, bool), "valid": np.ones(n, bool)}


def test_warning_needs_the_episode_that_catches_the_failure_to_open_before_the_report():
    s, t = _stream(30, 20)  # report starts at hour 20; windows end on the hour
    # Episode opens at hour 16 (end of window 15) and stands into the failure: warned 4 h early.
    a = [i >= 15 for i in range(30)]
    o = [i == 15 for i in range(30)]
    out = actionability(s, t, _details(a, o))
    assert out["events_warned_before_report"] == 1 and out["hours_before_report"] == pytest.approx(4.0)


def test_alarm_that_clears_before_the_failure_is_not_a_warning():
    s, t = _stream(30, 20)
    a = [12 <= i <= 14 or i >= 22 for i in range(30)]  # early burst clears, alarm reopens inside
    o = [i in (12, 22) for i in range(30)]
    out = actionability(s, t, _details(a, o))
    assert out["events_warned_before_report"] == 0


def test_two_hours_before_lps_uses_the_same_episode():
    s, t = _stream(30, 20, lps_h=23)
    a, o = [i >= 20 for i in range(30)], [i == 20 for i in range(30)]  # opens at hour 21
    assert actionability(s, t, _details(a, o))["events_warned_2h_before_lps"] == 1
    a, o = [i >= 21 for i in range(30)], [i == 21 for i in range(30)]  # opens at hour 22
    assert actionability(s, t, _details(a, o))["events_warned_2h_before_lps"] == 0


def test_three_valued_hold_pauses_on_undecided_windows_and_expires_when_stale():
    cond = np.array([1, 1, 0, 1, 1, 1, 1, 1], bool)
    valid = np.ones(8, bool)
    known = np.array([1, 1, 0, 1, 0, 0, 0, 1], bool)
    reset = hold(cond, valid, 3, 1)[0]
    paused = hold(cond, valid, 3, 1, known)[0]
    stale = hold(cond, valid, 3, 1, known, stale=2)[0]
    assert reset.tolist() == [0, 0, 0, 0, 0, 1, 1, 1]  # the false window breaks the run
    assert paused.tolist() == [0, 0, 0, 1, 1, 1, 1, 1]  # undecided window 2 neither adds nor resets
    assert stale.tolist() == [0, 0, 0, 1, 1, 1, 0, 0]  # a third undecided window in a row expires it


@pytest.fixture(scope="module")
def metropt3():
    return Definition(load_stream("metropt3"))


def test_healthy_startup_reaches_cut_in_within_one_window_and_failures_keep_low_pressure(metropt3):
    s, t = metropt3.stream, metropt3.test
    assert fill_windows(s) == 1
    raw, _, _ = metropt3.sources["dataset"]
    phys, known, _ = metropt3.sources["physical"]
    eligible = s.monitorable[t]
    assert (phys[eligible, 0] == raw[eligible, 0]).all() and known[eligible, 0].all()
    event = s.event_id()[t]
    assert phys[eligible & (event == 2), 0].any() and phys[eligible & (event == 3), 0].any()
