import gzip
import json
import os
import time

import numpy as np
import pytest

from uengram.host.collector import SpoolSink, spool_files
from uengram.host.features import read_columns
from uengram.host.node import NodeMemory
from uengram.host.synth import T0, make_ticks
from uengram.host.validate import commission_from_spool, write_report
from uengram.host.validate_criteria import criteria, evaluate

DAY = 86400


def _day(t):
    return time.strftime("%Y%m%d", time.gmtime(t))


def _touch(d, name, text="{}\n"):
    p = os.path.join(d, name)
    if name.endswith(".gz"):
        with gzip.open(p, "wt") as f:
            f.write(text)
    else:
        with open(p, "w") as f:
            f.write(text)
    return p


def test_spool_files_prefer_the_finished_day_and_skip_partial_copies(tmp_path):
    d = str(tmp_path)
    gz = _touch(d, "ticks-h-20261020.jsonl.gz")
    _touch(d, "ticks-h-20261020.jsonl")  # copied before compression finished
    _touch(d, "ticks-h-20261021.jsonl.gz.tmp")  # compression in progress
    today = _touch(d, "ticks-h-20261021.jsonl")
    _touch(d, "ticks-h2-20261021.jsonl")  # another host
    assert spool_files(d, "h") == [gz, today]


def test_spool_rotates_at_utc_midnight_and_reads_back_in_order(tmp_path):
    d = str(tmp_path / "spool")
    sink = SpoolSink(d, {"host": "h"}, keep_days=2)
    midnight = np.floor(T0 / DAY) * DAY
    ticks = make_ticks(200, seed=0, t_start=midnight - 100 + 0.5)
    for r in ticks:
        sink(r["t"], {k: v for k, v in r.items() if k != "t"})
    sink.close(wait=None)
    files = spool_files(d, "h")
    assert len(files) == 2 and files[0].endswith(".jsonl.gz") and files[1].endswith(".jsonl")
    meta, cols = read_columns(files)
    assert meta == {"host": "h"}
    np.testing.assert_array_equal(cols["t"], [r["t"] for r in ticks])
    # two days on, the day before that midnight is past keep_days and is dropped
    sink = SpoolSink(d, {"host": "h"}, keep_days=2)
    sink(midnight + 2 * DAY + 0.5, {})
    sink.close(wait=None)
    assert [os.path.basename(p) for p in spool_files(d, "h")] == [
        f"ticks-h-{_day(midnight)}.jsonl.gz", f"ticks-h-{_day(midnight + 2 * DAY)}.jsonl"]


# -- end to end: a synthetic one-node campaign ----------------------------------
ONSETS = {"spin": (1000, 1200), "thrash": (2000, 2200), "oom": (3000, 3000)}
TYPE = {"spin": ("cpu", 1.0), "thrash": ("mem", 2.5), "oom": ("oom", 1)}


@pytest.fixture(scope="module")
def campaign_dir(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("camp"))
    spool = os.path.join(d, "nodes", "synth", "spool")
    t_start = np.floor(T0 / 32) * 32 + 0.5
    comm = make_ticks(4 * 3600, seed=1, t_start=t_start)
    t_test = comm[-1]["t"] + 1
    test = make_ticks(3600, seed=2, t_start=t_test, anomalies=ONSETS)
    sink = SpoolSink(spool, {"host": "synth"})
    for r in comm + test:
        sink(r["t"], {k: v for k, v in r.items() if k != "t"})
    sink.close(wait=None)
    start, t_comm, end = comm[0]["t"] - 0.5, t_test - 0.5, test[-1]["t"] + 1
    faults = [{"id": i, "type": TYPE[a][0], "level": TYPE[a][1], "node": "synth", "peer": None,
               "status": "done", "start": t_test + lo, "end": t_test + hi + 1}
              for i, (a, (lo, hi)) in enumerate(ONSETS.items())]
    with open(os.path.join(d, "faults.jsonl"), "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in faults)
    camp = {"start": start, "clean_split": start + 3 * 3600, "commission": t_comm, "end": end, "alpha": 0.01,
            "nodes": {"synth": {"ssh": "synth", "ip": "", "faults": ["cpu", "mem", "oom"]}},
            "levels": {"cpu": [1.0], "mem": [2.5], "oom": [1]}, "criteria": criteria(0.01),
            "criteria_registered": start, "schedule": faults, "commissioned": {"synth": {"deployed": t_comm}},
            "report": None}
    with open(os.path.join(d, "campaign.json"), "w") as f:
        json.dump(camp, f)
    node = commission_from_spool(d, "synth", start, t_comm, 0.01)
    node.save(os.path.join(d, "models", "model-synth.pkl"))
    # the online daemon: the deployed model fed tick by tick
    online = NodeMemory.load(os.path.join(d, "models", "model-synth.pkl"))
    with open(os.path.join(d, "nodes", "synth", "events.jsonl"), "w") as f:
        for r in test:
            for rec in online.feed_ticks([r]):
                f.write(json.dumps(rec) + "\n")
    write_report(d, variants=("full", "no_ondelay", "saturation_only", "max_z", "single_gaussian"))
    with open(os.path.join(d, "report.json")) as f:
        rep = json.load(f)
    with open(os.path.join(d, "report.md")) as f:
        md = f.read()
    return rep, md


def test_offline_replay_equals_the_online_daemon(campaign_dir):
    rep, _ = campaign_dir
    c = rep["consistency"]["synth"]
    assert c["matched"] == c["online_windows"] > 100
    assert c["identical"] == 1.0


@pytest.mark.parametrize("part", ["clean_week", "calibration"])
def test_normal_windows_are_calibrated(campaign_dir, part):
    cal = campaign_dir[0][part]["synth"]
    assert cal["normal_windows"] > 50
    rate, lo, hi = cal["latent"]
    assert lo <= rate <= hi and rate <= 0.02


def test_injected_faults_are_found_by_the_right_channel(campaign_dir):
    det = {r["type"]: r for r in campaign_dir[0]["detection"]}
    assert det["cpu"]["detected"] and det["cpu"]["attribution_ok"]
    assert det["mem"]["detected"] and det["mem"]["latent"]
    assert det["oom"]["event_ok"]
    assert all(0 < r["delay_s"] <= 4 * 32 for r in det.values())


def test_report_lists_every_requested_variant(campaign_dir):
    md = campaign_dir[1]
    section4 = md.split("## 4.")[1].split("## 5.")[0]
    rows = [ln.split("|")[1].strip() for ln in section4.splitlines() if ln.startswith("| ")]
    assert rows[1:] == ["full", "no_ondelay", "saturation_only", "max_z", "single_gaussian"]


def test_the_report_evaluates_the_stored_criteria(campaign_dir):
    rep, md = campaign_dir
    assert "## 0. Pre-registered verdicts (registered " in md
    v = {(r["criterion"], r["scope"]): r["verdict"] for r in rep["verdicts"]}
    assert v[("online equals offline", "synth")] == "PASS"
    # an hour without a single alarm is too short to bound the rate under 1 %: not a pass
    assert campaign_dir[0]["clean_week"]["synth"]["latent"][0] == 0.0
    assert v[("clean week: latent false alarms", "synth")] == "FAIL"
    # one trial per fault type cannot clear the lower bound: too little evidence is not a pass
    assert v[("strongest level detected", "cpu 1.0")] == "FAIL"


def _det(ftype, level, hits, n):
    return [{"type": ftype, "level": level, "role": "target", "detected": i < hits} for i in range(n)]


@pytest.mark.parametrize("hits,verdict", [(10, "PASS"), (9, "PASS"), (8, "FAIL")])
def test_strongest_level_needs_80_percent_and_a_lower_bound_of_half(hits, verdict):
    rows = evaluate(criteria(0.01), {"cpu": [1.0]}, {}, _det("cpu", 1.0, hits, 10), {}, {})
    assert rows[0]["verdict"] == verdict  # 8/10: share 0.8 but Wilson lower bound 0.49


def test_detection_falling_with_strength_fails():
    det = _det("disk", 10, 9, 10) + _det("disk", 0, 8, 10)
    rows = {r["criterion"]: r["verdict"] for r in evaluate(criteria(0.01), {"disk": [10, 40, 0]}, {}, det, {}, {})}
    assert rows["strongest >= weakest"] == "FAIL"


def test_a_component_or_baseline_without_measured_benefit_fails():
    s = lambda det, fa: {"det": (det, 0, 1), "fa": (fa, 0, 1), "per_type": {}}
    summary = {"full": s(0.9, 0.004), "no_spe": s(0.9, 0.004), "no_events": s(0.8, 0.004),
               "max_z": s(0.9, 0.003), "single_gaussian": s(0.9, 0.006)}
    v = {r["scope"]: r["verdict"] for r in evaluate(criteria(0.01), {}, {}, [], {}, summary)}
    assert v == {"no_spe": "FAIL", "no_events": "PASS", "max_z": "FAIL", "single_gaussian": "PASS"}
