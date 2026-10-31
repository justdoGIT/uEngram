from __future__ import annotations

import pytest

from bench.capsule_cost_study import worst_case_replay_count
from bench.capsule_kill_test import summarize as summarize_kills
from bench.capsule_restart_sweep import restart_positions
from bench.capsule_restart_sweep import summarize as summarize_sweep


@pytest.mark.parametrize("window_count,positions", [(256, 16), (64, 63), (8, 20), (4, 1)])
def test_restart_positions_are_distinct_sorted_and_strictly_inside_the_prefix(window_count, positions):
    cuts = restart_positions(window_count, positions)

    assert cuts == sorted(set(cuts))
    assert cuts and 1 <= cuts[0] and cuts[-1] <= window_count - 1  # evaluate() needs 1 <= cut <= n - 1


@pytest.mark.parametrize("count,cadence", [(256, 1), (256, 4), (256, 16), (64, 64), (100, 7), (9, 7)])
def test_worst_case_replay_leaves_exactly_cadence_minus_one_windows_past_a_checkpoint(count, cadence):
    prefix = worst_case_replay_count(count, cadence)

    assert prefix <= count
    assert prefix % cadence == cadence - 1  # a checkpoint after every `cadence` windows leaves this many
    assert prefix + cadence > count  # and it is the longest such prefix


def _cell(state=False, keys=False, missing_alarm=0, missing=0, extra=0):
    return {"state_equivalent": state, "keys_match": keys, "missing_alarm_open": missing_alarm,
            "missing_events": missing, "extra_events": extra}


def test_alarm_loss_is_judged_only_where_the_reference_actually_alarms():
    quiet = {"reference": {"events": 2, "alarm_open": 0},
             "positions": [{"restart_at": 5, **{a: _cell(missing_alarm=0) for a in ("lite", "sidecar", "capsule")}}]}
    noisy = {"reference": {"events": 9, "alarm_open": 4},
             "positions": [{"restart_at": 5, "lite": _cell(missing_alarm=2), "sidecar": _cell(),
                            "capsule": _cell(state=True, keys=True)}]}

    arms = summarize_sweep({"quiet": quiet, "noisy": noisy})["arms"]

    assert arms["lite"]["cells"] == 2
    assert arms["lite"]["cells_where_reference_alarms"] == 1  # the quiet machine cannot lose an alarm
    assert arms["lite"]["alarm_lost"] == 1
    assert arms["sidecar"]["alarm_lost"] == 0 and arms["capsule"]["state_equivalent"] == 1


def _kill(logged, checkpoint, vendor, finished=False, ok=True):
    return {"finished_before_kill": finished, "logged_windows_at_kill": logged,
            "checkpoint_step_at_kill": checkpoint, "vendor_event_step_at_kill": vendor,
            "recovery_ok": ok, "keys_match": ok, "no_duplicates": ok, "state_equivalent": ok,
            "integrity_ok": ok}


def test_kill_coverage_counts_the_boundaries_a_trial_actually_died_in():
    trials = [
        _kill(logged=0, checkpoint=-1, vendor=-1),                # died before the first window
        _kill(logged=5, checkpoint=3, vendor=3),                  # logged one window past the checkpoint
        _kill(logged=4, checkpoint=3, vendor=3),                  # logged == checkpoint + 1: nothing pending
        _kill(logged=6, checkpoint=3, vendor=5),                  # vendor committed ahead of the checkpoint
        _kill(logged=9, checkpoint=8, vendor=8, finished=True),   # finished before the kill landed
    ]

    summary = summarize_kills(trials)

    assert summary["killed_mid_stream"] == 4
    assert summary["killed_before_first_window"] == 1
    assert summary["killed_with_logged_but_uncheckpointed_input"] == 2
    assert summary["killed_with_vendor_ahead_of_checkpoint"] == 1
    assert summary["keys_match"] == 5 and summary["recovery_failed"] == 0


def test_storage_of_names_the_filesystem_backing_a_path(tmp_path):
    from pathlib import Path

    from bench.capsule_resilience import storage_of

    if not Path("/proc/mounts").exists():
        pytest.skip("needs Linux /proc/mounts")

    storage = storage_of(tmp_path)

    assert set(storage) == {"fstype", "source", "rotational"}
    assert isinstance(storage["fstype"], str) and storage["fstype"]  # a path always lies on some mount
    assert storage_of(tmp_path / "missing" / "dir")["fstype"] == storage["fstype"]  # resolved by mount
