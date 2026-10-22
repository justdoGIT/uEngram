"""Cluster view: latest state of every node, read over ssh; stdlib only.

Placement advice is advisory and explainable, from the latest window of each
node: a node is listed as `avoid` when its window alarmed (novel state or an
error counter fired), a pressure signal exceeds everything seen during
commissioning (`saturation`), or its current state was first learned during an
alarm (a known-bad state that uEngram has since accepted as a regime); otherwise
headroom = idle share times the share of time no task stalled on CPU, memory or I/O:
    headroom = (1 - cpu_busy) * (1 - max(cpu_pressure, mem_pressure, io_pressure))
Nodes are ranked by headroom. Nothing is actuated.
"""
from __future__ import annotations

import json
import subprocess
import time


def collect(nodes: list[str], state_path: str, timeout: float = 8.0) -> dict[str, dict]:
    out = {}
    for n in nodes:
        try:
            r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", n, f"cat {state_path}"],
                               capture_output=True, text=True, timeout=timeout)
            out[n] = json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr.strip()[:120]}
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            out[n] = {"error": str(e)[:120]}
    return out


def headroom(st: dict) -> float:
    lv = st.get("levels", {})
    press = max(lv.get("cpu_pressure", 0.0), lv.get("mem_pressure", 0.0), lv.get("io_pressure", 0.0))
    return (1.0 - lv.get("cpu_busy", 1.0)) * (1.0 - press)


def _top_modality(st: dict) -> str:
    mods = st.get("modality_d2")
    if not mods:
        return "-"
    m = max(mods, key=mods.get)
    return f"{m} {mods[m] / max(sum(v for v in mods.values() if v > 0), 1e-12):.0%}"


def format_node(st: dict) -> str:
    if "error" in st:
        return f"unreachable: {st['error']}"
    lv, prof = st.get("levels", {}), st.get("state_profile") or {}
    age = time.time() - st["t0"]
    lines = [
        f"{st['host']}: window {age:.0f}s ago, state {st['kind']} #{st['item_id']} ({st['action']})",
        f"  score D2 {st['score']:.1f} / threshold {st['threshold']:.1f}  alarm={st['alarm']}"
        f" (latent={st['alarm_latent']}, events={st['events'] or 'none'})",
        f"  busy {lv.get('cpu_busy', float('nan')):.0%}  ipc {lv.get('ipc', float('nan')):.2f}"
        f"  freq x{lv.get('freq_ratio', float('nan')):.2f}  pkg {lv.get('pkg_power', float('nan')):.1f} W"
        f"  thermal residual {lv.get('thermal_residual', float('nan')):+.1f} C",
        f"  pressure cpu {lv.get('cpu_pressure', 0):.1%} mem {lv.get('mem_pressure', 0):.1%}"
        f" io {lv.get('io_pressure', 0):.1%}  mem used {lv.get('mem_used', float('nan')):.0%}",
        f"  novelty driver: {_top_modality(st)}; slots {st['slots']}",
        *([f"  saturated beyond commissioning: "
           + ", ".join(f"{k} {v:.1%}" for k, v in st["saturation"].items())] if st.get("saturation") else []),
        *([f"  this state was first seen during an alarm at "
           f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st['state_alarmed_at']))}"]
          if st.get("state_alarmed_at") else []),
    ]
    if prof:
        lines.append(f"  this state so far: {prof['windows']} windows, {prof['watts']:.1f} W,"
                     f" {prof['gips']:.2f} G inst/s, {prof['nj_per_inst']:.2f} nJ/inst")
    return "\n".join(lines)


def avoid(st: dict) -> bool:
    return bool(st["alarm"] or st.get("state_alarmed_at") or st.get("saturation"))


def format_cluster(states: dict[str, dict]) -> str:
    rows, blocks = [], []
    for n, st in states.items():
        blocks.append(format_node(st) if "error" not in st else f"{n}: unreachable ({st['error']})")
        if "error" in st:
            continue
        rows.append((n, "avoid" if avoid(st) else f"{headroom(st):.2f}", headroom(st), avoid(st)))
    rows.sort(key=lambda r: (r[3], -r[2]))
    rank = "placement order: " + ", ".join(f"{n} ({h})" for n, h, _, _ in rows)
    return "\n\n".join(blocks + [rank])
