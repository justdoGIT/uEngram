import numpy as np

from bench.skab_teacher_time_gate import _episodes, _sustained


def test_sustained_requires_complete_persistence_window():
    flags = np.array([False, True, True])
    assert not _sustained(flags, persistence=3).any()
    flags = np.array([False, True, True, True])
    assert np.flatnonzero(_sustained(flags, persistence=3)).tolist() == [1]


def test_episode_count_does_not_repeat_one_alarm_run():
    flags = np.array([True, True, True, True, False, True, True, True])
    times = np.datetime64("1970-01-01T00:00:00", "s") + np.arange(len(flags), dtype="timedelta64[s]")
    assert _episodes(flags, times) == [
        np.datetime64(0, "s"), np.datetime64(5, "s")
    ]


def test_short_alarm_at_end_is_not_an_episode():
    flags = np.array([False, False, True, True])
    times = np.datetime64("1970-01-01T00:00:00", "s") + np.arange(len(flags), dtype="timedelta64[s]")
    assert _episodes(flags, times) == []
