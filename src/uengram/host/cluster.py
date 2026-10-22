"""Cluster view: the hub's snapshot of every node, as placement advice; stdlib only.

The hub (hub.py, on msi) follows each node's pushed stream; this module turns
its snapshot into advice at read time. A node is `avoid` when its latest
window alarmed (novel state or an error counter fired), a pressure exceeded
everything seen during commissioning (`saturation`, 32 s windows), a PSI
trigger fired within the last two trigger windows (`stalling`, psi.py), or its
current state was first learned during an alarm (a known-bad state uEngram has
since accepted as a regime). A node whose stream is down or whose last window
is older than STALE_S is `unknown`. The rest are ranked by headroom on the
last LOAD_TICKS seconds, with the same ratio-of-sums rule as the window levels:
    headroom = (1 - busy) * (1 - max(cpu, memory, io stall share))
Nothing is actuated.
"""
from __future__ import annotations

import json
import time

from .psi import WINDOW_S

LOAD_TICKS = 5  # seconds of load behind the headroom
STALE_S = 3 * 32 + 10  # three windows plus margin
STALL_HOLD_S = 2 * WINDOW_S + 1  # a sustained stall re-fires once per trigger window
LOAD_FIELDS = ("dt_s", "cpu_busy_j", "cpu_total_j", "psi_cpu_us", "psi_mem_us", "psi_io_us")
NAN = float("nan")


def load_now(ticks: list[dict]) -> dict:
    """Busy share and stall shares over the given ticks (ratios of sums)."""
    ok = [t for t in ticks if all(t.get(f) is not None for f in LOAD_FIELDS)]
    dt, tot = sum(t["dt_s"] for t in ok), sum(t["cpu_total_j"] for t in ok)
    if not ok or dt <= 0 or tot <= 0:
        return {"busy": NAN, "cpu_pressure": NAN, "mem_pressure": NAN, "io_pressure": NAN, "seconds": 0.0}
    stall = {n: min(1.0, sum(t[f] for t in ok) * 1e-6 / dt)
             for n, f in (("cpu_pressure", "psi_cpu_us"), ("mem_pressure", "psi_mem_us"), ("io_pressure", "psi_io_us"))}
    return {"busy": sum(t["cpu_busy_j"] for t in ok) / tot, **stall, "seconds": dt}


def headroom(load: dict) -> float:
    return (1.0 - load["busy"]) * (1.0 - max(load["cpu_pressure"], load["mem_pressure"], load["io_pressure"]))


def stalling(pressure: dict, now: float) -> list[str]:
    return sorted(r for r, t in pressure.items() if now - t <= STALL_HOLD_S)


def advice(node: dict, now: float) -> tuple[str, list[str]]:
    """('ok' | 'avoid' | 'unknown', reasons) for one node entry of the hub snapshot."""
    st = node.get("state")
    if not node["link"]["up"]:
        return "unknown", [f"stream down: {node['link']['error']}"]
    if st is None or now - st["t0"] > STALE_S:
        return "unknown", ["no recent window"]
    why = []
    if st["alarm"]:
        why.append("alarm")
    if st.get("saturation"):
        why.append("saturated: " + ", ".join(st["saturation"]))
    if stall := stalling(node.get("pressure", {}), now):
        why.append("stalling now: " + ", ".join(stall))
    if st.get("state_alarmed_at"):
        why.append("known-bad state")
    return ("avoid" if why else "ok"), why


def placement(snapshot: dict, now: float | None = None) -> list[dict]:
    """Nodes in placement order: ok by headroom, then avoid, then unknown."""
    now = time.time() if now is None else now
    rows = []
    for name, node in snapshot["nodes"].items():
        verdict, why = advice(node, now)
        load = load_now(node.get("ticks", []))
        h = headroom(load) if load["seconds"] else NAN
        rows.append({"node": name, "verdict": verdict, "why": why, "headroom": h, "load": load})
    order = {"ok": 0, "avoid": 1, "unknown": 2}
    rows.sort(key=lambda r: (order[r["verdict"]], -(r["headroom"] if r["headroom"] == r["headroom"] else -1)))
    return rows


def read_snapshot(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def format_node(st: dict, pressure: dict | None = None) -> str:
    lv, prof = st.get("levels", {}), st.get("state_profile") or {}
    now = time.time()
    age = now - st["t0"]
    lines = [
        f"{st['host']}: window {age:.0f}s ago, state {st['kind']} #{st['item_id']} ({st['action']})",
        f"  score D2 {st['score']:.1f} / threshold {st['threshold']:.1f}  alarm={st['alarm']}"
        f" (latent={st['alarm_latent']}, events={st['events'] or 'none'})",
        f"  health D2 {st.get('persistent_health', NAN):.1f} from the commissioned healthy states"
        + (f" (most from {_top(st['health_modality_d2'])})" if st.get("health_modality_d2") else ""),
        f"  busy {lv.get('cpu_busy', NAN):.0%}  ipc {lv.get('ipc', NAN):.2f}"
        f"  freq x{lv.get('freq_ratio', NAN):.2f}  pkg {lv.get('pkg_power', NAN):.1f} W"
        f"  thermal residual {lv.get('thermal_residual', NAN):+.1f} C",
        f"  pressure cpu {lv.get('cpu_pressure', 0):.1%} mem {lv.get('mem_pressure', 0):.1%}"
        f" io {lv.get('io_pressure', 0):.1%}  mem used {lv.get('mem_used', NAN):.0%}",
        f"  novelty driver: {_top(st.get('modality_d2'))}; slots {st['slots']}",
        *([f"  saturated beyond commissioning: "
           + ", ".join(f"{k} {v:.1%}" for k, v in st["saturation"].items())] if st.get("saturation") else []),
        *([f"  PSI trigger within {STALL_HOLD_S} s: " + ", ".join(stalling(pressure, now))]
          if pressure and stalling(pressure, now) else []),
        *([f"  this state was first seen during an alarm at "
           f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st['state_alarmed_at']))}"]
          if st.get("state_alarmed_at") else []),
    ]
    if prof:
        lines.append(f"  this state so far: {prof['windows']} windows, {prof['watts']:.1f} W,"
                     f" {prof['gips']:.2f} G inst/s, {prof['nj_per_inst']:.2f} nJ/inst")
    return "\n".join(lines)


def _top(mods: dict | None) -> str:
    if not mods:
        return "-"
    m = max(mods, key=mods.get)
    return f"{m} {mods[m] / max(sum(v for v in mods.values() if v > 0), 1e-12):.0%}"


def format_cluster(snapshot: dict, now: float | None = None) -> str:
    now = time.time() if now is None else now
    blocks = []
    for name, node in snapshot["nodes"].items():
        st = node.get("state")
        blocks.append(format_node(st, node.get("pressure")) if st else f"{name}: no state yet")
    rows = placement(snapshot, now)
    rank = ", ".join(f"{r['node']} ({r['headroom']:.2f})" if r["verdict"] == "ok"
                     else f"{r['node']} ({r['verdict']}: {'; '.join(r['why'])})" for r in rows)
    return "\n\n".join(blocks + [f"hub snapshot {now - snapshot['t']:.0f}s old; placement order: {rank}"])
