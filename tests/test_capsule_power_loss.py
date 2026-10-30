from __future__ import annotations

import shutil
import sys

import pytest

from bench import capsule_power_loss as pl
from bench import capsule_resilience as cr
from bench.capsule_kill_test import load_windows, save_windows

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None or not sys.platform.startswith("linux"),
                                reason="the LD_PRELOAD shim needs gcc on Linux")

WINDOWS = 8


@pytest.fixture()
def fixture_run(tmp_path, monkeypatch):
    """pl.run over a few synthetic windows, in the exact form the recorded child reconstructs."""
    save_windows(tmp_path / "w.npz", cr._fixture_windows(WINDOWS))
    canonical = load_windows(tmp_path / "w.npz")
    monkeypatch.setattr(cr, "_smd_windows", lambda machine, count: canonical)

    def go(synchronous="FULL", checkpoint_every=1, control="none"):
        return pl.run("fixture", WINDOWS, synchronous, checkpoint_every, spread=8, combos_per_point=4,
                      cache_dir=tmp_path / "cache", control=control)["summary"]
    return go


def test_full_sync_loses_no_acknowledged_window_and_never_corrupts(fixture_run):
    summary = fixture_run("FULL", 1)

    assert summary["trials"] > 30
    assert summary["capsule_inconsistent"] == 0
    assert summary["images_losing_acknowledged_windows"] == 0
    assert summary["continues_to_completion_of_consistent"] == summary["capsule_consistent"]
    assert summary["final_vendor_consistent"] == summary["trials"]  # replay re-emits identical keys


def test_a_longer_checkpoint_interval_still_loses_no_acknowledged_window_under_full_sync(fixture_run):
    summary = fixture_run("FULL", 4)

    assert summary["capsule_inconsistent"] == 0
    assert summary["images_losing_acknowledged_windows"] == 0  # the input log commits every window


def test_normal_sync_stays_consistent_but_can_lose_acknowledged_windows(fixture_run):
    summary = fixture_run("NORMAL", 1)

    assert summary["capsule_inconsistent"] == 0
    assert summary["images_losing_acknowledged_windows"] > 0
    assert summary["final_vendor_consistent"] == summary["trials"]


def test_the_harness_detects_corruption_when_journaling_and_sync_are_disabled(fixture_run):
    """Negative control: without it, 'zero inconsistent' for the real configurations means nothing."""
    summary = fixture_run("FULL", 1, control="no-journal")

    assert summary["capsule_inconsistent"] > 0
