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
    return str(tmp_path), cp.plan(str(tmp_path), START, record_days=7, test_days=7, per_type=10, seed=3,
                                  nodes=NODES, clean_days=3)


def test_plan_spaces_faults_and_stays_in_the_test_phase(camp):
    _, c = camp
    s = c["schedule"]
    asked = 10 * sum(len(cfg["faults"]) for cfg in NODES.values())
    assert len(s) > 0.9 * asked  # a few may not fit; most must
    t = sorted(f["planned"] for f in s)
    assert all(b - a >= cp.DURATION + cp.FAULT_GAP for a, b in zip(t, t[1:]))
    for name in NODES:
        mine = sorted(f["planned"] for f in s if name in (f["node"], f["peer"]))
        assert all(b - a >= cp.NODE_GAP for a, b in zip(mine, mine[1:]))
    assert min(t) >= c["commission"] + 3600 and max(t) <= c["end"] - 1800
    assert c["start"] < c["clean_split"] < c["commission"]  # the clean week is fault-free


def test_plan_respects_each_nodes_fault_list_and_pairs_network_faults(camp):
    _, c = camp
    for f in c["schedule"]:
        assert f["type"] in NODES[f["node"]]["faults"] and f["level"] in cp.LEVELS[f["type"]]
        if f["type"] == "net":
            assert f["peer"] != f["node"] and "net" in NODES[f["peer"]]["faults"]
        else:
            assert f["peer"] is None
    # every node runs each of its fault types per_type times, covering every strength
    for name, cfg in NODES.items():
        mine = [f for f in c["schedule"] if f["node"] == name]
        for ftype in cfg["faults"]:
            got = [f["level"] for f in mine if f["type"] == ftype]
            assert len(got) >= 8 and set(got) == set(cp.LEVELS[ftype])


class Fleet:
    """Stands in for ssh and kepr: gate answers per node, a fault takes DURATION seconds."""

    def __init__(self, monkeypatch, now, busy=(), locked=False):
        self.now, self.busy, self.ran, self.locked, self.alerts = now, set(busy), [], locked, []
        monkeypatch.setattr(cp, "_now", lambda: self.now)
        monkeypatch.setattr(cp, "push_helper", lambda host: None)
        monkeypatch.setattr(cp, "gate", lambda name, host, need=0.0: (name not in self.busy, f"busy={name in self.busy}"))
        monkeypatch.setattr(cp, "vault_unlocked", lambda: not self.locked)
        monkeypatch.setattr(cp, "alert", lambda *a: self.alerts.append(a))

        def execute(camp, f):
            self.ran.append(f["id"])
            self.now += cp.DURATION
            return {"rc": 0}
        monkeypatch.setattr(cp, "execute", execute)


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


def test_a_locked_vault_delays_sudo_faults_without_spending_retries(camp, monkeypatch):
    d, c = camp
    c = _testing(d, c)
    sudo = next(f for f in c["schedule"] if f["type"] in cp.SUDO_FAULTS)
    plain = next(f for f in c["schedule"] if f["type"] == "cpu")
    for f in c["schedule"]:
        if f["id"] not in (sudo["id"], plain["id"]):
            f["status"] = "skipped"
    plain["planned"] = sudo["planned"]
    cp._save(d, c)
    fleet = Fleet(monkeypatch, sudo["planned"], locked=True)
    cp.tick(d)
    fleet.now += cp.FAULT_GAP
    cp.tick(d)
    after = {f["id"]: f for f in cp.load(d)["schedule"]}
    assert fleet.ran == [plain["id"]]  # faults without sudo still run
    assert after[sudo["id"]]["status"] == "pending" and after[sudo["id"]]["postponed"] == 0
    assert len(fleet.alerts) == 1  # rate-limited
    fleet.locked, fleet.now = False, after[sudo["id"]]["planned"]
    cp.tick(d)
    assert fleet.ran == [plain["id"], sudo["id"]]
