"""Running one campaign fault on the fleet: ssh, the idle gate, the injection.

Stdlib only. Faults run through `fault-inject.sh`, copied to each node before
use. Faults in SUDO_FAULTS and model deploys need root on the node; the sudo
password goes from msi's kepr store straight into the ssh pipe, so the store
must be unlocked (`vault_unlocked`). msi's kepr daemon is started on demand
and a fresh daemon starts locked, so the campaign checks before each use.
"""
from __future__ import annotations

import os
import subprocess
import time

HELPER = "/tmp/uengram-fault-inject.sh"
PACKAGED_HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fault-inject.sh")
NETEM_DELAY_MS = 40
NETEM_MBPS = 4
PORT = 47123
SUDO_FAULTS = {"mem", "oom", "disk", "leak", "power", "netem"}
NOTIFY = os.path.expanduser("~/projects/dev-setup/scripts/notify.sh")


def run_ssh(host: str, cmd: str, sudo: bool = False, timeout: float = 600) -> subprocess.CompletedProcess:
    """Run cmd on host; with sudo, the password goes from kepr straight into the
    ssh pipe (never through this process's memory, argv or a file)."""
    if not sudo:
        return subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd], capture_output=True, text=True,
                              timeout=timeout)
    kepr = subprocess.Popen(["kepr", "get", "KK_SUDO_PASSWORD", "--quiet"], stdout=subprocess.PIPE)
    try:
        return subprocess.run(["ssh", "-o", "BatchMode=yes", host, f"sudo -S -p '' {cmd}"], stdin=kepr.stdout,
                              capture_output=True, text=True, timeout=timeout)
    finally:
        kepr.stdout.close()
        kepr.wait()


def vault_unlocked() -> bool:
    r = subprocess.run(["kepr", "status"], capture_output=True, text=True, timeout=30)
    return "Store: unlocked" in r.stdout + r.stderr


def alert(severity: str, title: str, body: str) -> None:
    """Fleet notification (durable log, desktop, Fleet API) when dev-setup is present."""
    if os.access(NOTIFY, os.X_OK):
        subprocess.run([NOTIFY, "--severity", severity, "--title", title, "--body", body],
                       capture_output=True, timeout=60)


def push_helper(host: str) -> None:
    subprocess.run(["scp", "-q", PACKAGED_HELPER, f"{host}:{HELPER}"], check=True, timeout=60)


def slurm_jobs(name: str) -> int:
    """Jobs on node `name`, asked from here (msi runs slurmctld; a worker's own
    squeue can stall a minute when its client config is missing)."""
    r = subprocess.run(["squeue", "-h", "-w", name], capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"squeue failed: {r.stderr.strip()[:200]}")  # unknown: postpone
    return len(r.stdout.splitlines())


def need_mb(f: dict, duration: float) -> float:
    """MemAvailable a fault needs on its node (twice what it can allocate)."""
    return 2.0 * f["level"] * duration if f["type"] == "leak" else 0.0


def gate(name: str, host: str, need: float = 0.0) -> tuple[bool, str]:
    r = run_ssh(host, f"sh {HELPER} check", timeout=60)
    if r.returncode != 0:
        return False, f"check failed: {r.stderr.strip()[:200]}"
    kv = dict(x.split("=") for x in r.stdout.split())
    jobs = slurm_jobs(name)
    why = f"{r.stdout.strip()} slurm_jobs={jobs} need_mb={need:.0f}"
    return float(kv["busy"]) <= 0.5 and jobs == 0 and float(kv["mem_avail_mb"]) >= need, why


def _stream(peer: dict, dur: float, send_cmd: str, host: str, sudo: bool) -> dict:
    """Run a TCP sink on the peer, then `send_cmd` on host (the stream's sender)."""
    rx = subprocess.Popen(["ssh", "-o", "BatchMode=yes", peer["ssh"], f"sh {HELPER} netrecv {PORT} {dur}"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    time.sleep(3)
    try:
        tx = run_ssh(host, send_cmd, sudo=sudo)
    finally:
        rx.wait(timeout=dur + 120)
    return {"rc": tx.returncode, "rx_rc": rx.returncode, "out": tx.stdout.strip()[-200:]}


def execute(camp: dict, f: dict) -> dict:
    nodes = camp["nodes"]
    host, dur, lv = nodes[f["node"]]["ssh"], camp["duration_s"], f["level"]
    if f["type"] == "cpu":
        return {"rc": run_ssh(host, f"sh {HELPER} spin {lv} {dur}").returncode}
    if f["type"] == "mem":
        r = run_ssh(host, f"sh {HELPER} mem {lv} {dur}", sudo=True)
        return {"rc": r.returncode, "out": r.stdout.strip()[-200:]}
    if f["type"] == "oom":
        r = run_ssh(host, f"sh {HELPER} oom", sudo=True)
        return {"rc": r.returncode, "out": r.stdout.strip()[-200:]}
    if f["type"] in ("leak", "power"):
        sub = {"leak": "leak", "power": "powercap"}[f["type"]]
        r = run_ssh(host, f"sh {HELPER} {sub} {lv} {dur}", sudo=True)
        return {"rc": r.returncode, "out": r.stdout.strip()[-200:]}
    if f["type"] == "disk":
        return {"rc": run_ssh(host, f"sh {HELPER} diskread {lv} {dur}", sudo=True).returncode}
    if f["type"] == "net":
        peer = nodes[f["peer"]]
        return _stream(peer, dur, f"sh {HELPER} netsend {peer['ip']} {PORT} {lv} {dur}", host, sudo=False)
    if f["type"] == "netem":
        peer = nodes[f["peer"]]
        return _stream(peer, dur, f"sh {HELPER} netem {lv} {NETEM_DELAY_MS} {peer['ip']} {PORT} "
                                  f"{NETEM_MBPS} {dur}", host, sudo=True)
    raise ValueError(f["type"])
