import json
import os

import pytest

from uengram.host import campaign as cp

NODES = {
    "a": {"ssh": "a", "ip": "10.0.0.1", "faults": ["cpu", "net"]},
    "b": {"ssh": "b", "ip": "10.0.0.2", "faults": ["cpu", "mem", "oom", "disk", "net"]},
    "c": {"ssh": "c", "ip": "10.0.0.3", "faults": ["cpu", "mem"]},
}
START = 1_800_000_000.0


@pytest.fixture
def camp(tmp_path):
    return str(tmp_path), cp.plan(str(tmp_path), START, 3, 4, 4, seed=3, nodes=NODES)


def test_plan_spaces_faults_and_stays_in_the_test_phase(camp):
    _, c = camp
    s = c["schedule"]
    assert len(s) > 0.8 * 3 * 4 * 4  # a few may not fit; most must
    t = sorted(f["planned"] for f in s)
    assert all(b - a >= cp.DURATION + cp.FAULT_GAP for a, b in zip(t, t[1:]))
    for name in NODES:
        mine = sorted(f["planned"] for f in s if name in (f["node"], f["peer"]))
        assert all(b - a >= cp.NODE_GAP for a, b in zip(mine, mine[1:]))
    assert min(t) >= c["commission"] + 3600 and max(t) <= c["end"] - 1800


def test_plan_respects_each_nodes_fault_list_and_pairs_network_faults(camp):
    _, c = camp
    for f in c["schedule"]:
        assert f["type"] in NODES[f["node"]]["faults"] and f["level"] in cp.LEVELS[f["type"]]
        if f["type"] == "net":
            assert f["peer"] != f["node"] and "net" in NODES[f["peer"]]["faults"]
        else:
            assert f["peer"] is None
    # node b has 5 fault types and 13 (type, level) cells: its 16 faults cover every cell
    cells = {(f["type"], f["level"]) for f in c["schedule"] if f["node"] == "b"}
    assert len(cells) == sum(len(cp.LEVELS[x]) for x in NODES["b"]["faults"])


class Fleet:
    """Stands in for ssh: gate answers per host, a fault takes `dur` seconds."""

    def __init__(self, monkeypatch, now, busy=()):
        self.now, self.busy, self.ran = now, set(busy), []
        monkeypatch.setattr(cp, "_now", lambda: self.now)
        monkeypatch.setattr(cp, "_push_helper", lambda host: None)
        monkeypatch.setattr(cp, "_gate", lambda host: (host not in self.busy, f"busy={host in self.busy}"))

        def execute(camp, f):
            self.ran.append(f["id"])
            self.now += cp.DURATION
            return {"rc": 0}
        monkeypatch.setattr(cp, "_execute", execute)


def _testing(d, c):
    c["commissioned"] = {n: {"deployed": c["commission"]} for n in NODES}
    cp._save(d, c)
    return c


def test_due_faults_run_one_per_tick_and_are_logged(camp, monkeypatch):
    d, c = camp
    c = _testing(d, c)
    first, second = c["schedule"][:2]
    fleet = Fleet(monkeypatch, second["planned"] + 1)  # both are due
    assert cp.tick(d) == "testing"
    assert fleet.ran == [first["id"]]
    fleet.now += cp.FAULT_GAP
    cp.tick(d)
    assert fleet.ran == [first["id"], second["id"]]
    with open(os.path.join(d, "faults.jsonl")) as f:
        log = [json.loads(x) for x in f]
    assert [x["id"] for x in log] == fleet.ran
    assert all(x["status"] == "done" and x["end"] - x["start"] == cp.DURATION for x in log)


def test_a_due_fault_waits_until_the_previous_one_has_faded(camp, monkeypatch):
    d, c = camp
    c = _testing(d, c)
    first, second = c["schedule"][:2]
    fleet = Fleet(monkeypatch, second["planned"] + 1)
    cp.tick(d)
    fleet.now += cp.FAULT_GAP - 60  # too soon after the first fault ended
    cp.tick(d)
    assert fleet.ran == [first["id"]]


def test_a_busy_node_postpones_then_skips(camp, monkeypatch):
    d, c = camp
    c = _testing(d, c)
    f0 = c["schedule"][0]
    fleet = Fleet(monkeypatch, f0["planned"], busy={"a", "b", "c"})
    for k in range(1, 8):
        cp.tick(d)
        f = next(x for x in cp.load(d)["schedule"] if x["id"] == f0["id"])
        assert f["postponed"] == k and f["status"] == ("pending" if k <= 6 else "skipped")
        assert f["planned"] == fleet.now + 600
        fleet.now = f["planned"]
    assert f0["id"] not in fleet.ran


def test_faults_on_a_node_that_failed_commissioning_are_skipped(camp, monkeypatch):
    d, c = camp
    c = _testing(d, c)
    f0 = c["schedule"][0]
    c["commissioned"][f0["node"]] = {"error": "commissioning too short"}
    cp._save(d, c)
    fleet = Fleet(monkeypatch, f0["planned"])
    cp.tick(d)
    f = next(x for x in cp.load(d)["schedule"] if x["id"] == f0["id"])
    assert f["status"] == "skipped" and "not commissioned" in f["reason"]
    assert f0["id"] not in fleet.ran
