"""`laya-http` backend: the same Laya triage run against a `laya-serve` /v1/systemone endpoint.

Used for the lenovo instance, reached only through an SSH local forward the operator opens
(`ssh -N -L 127.0.0.1:18000:127.0.0.1:8000 lenovo-worker`). Requests are sequential (one in
flight), paced, 120 s client timeout, at most one retry; every failure is an explicit
unavailable record. The request pins `model: "english"` and every answer must report
`routing.model == "english"` and an untruncated `usage`; anything else is unavailable and
counted. Questions, state format, rule and thresholds are reused unchanged from the local
Laya preregistration (bench.laya_advisory, bench.advisory_eval, bench.advisory_experiment),
whose files are not modified so that preregistration's code hashes stay valid.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from bench import advisory_eval as ev
from bench import laya_advisory as la
from bench.advisory_experiment import _log, _pct, _streams, stream_row
from bench.jev_advisory import PROMPT_HASH, QUESTIONS

BACKEND = "laya-http"
ROUTE = "english"
TIMEOUT = 120.0
PACE_SECONDS = 0.25
CODE_FILES = ("bench/laya_http.py", "bench/laya_advisory.py", "bench/advisory_eval.py",
              "bench/advisory_experiment.py", "bench/jev_advisory.py")
WEIGHT_FILES = ("rl_agent_config.json", "model.safetensors", "encoder/config.json",
                "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json")


class RoutingError(ValueError):
    """The server answered with a checkpoint other than the pinned English one."""


class HttpLaya:
    """Sequential client for one laya-serve endpoint (no concurrency by construction)."""

    def __init__(self, base_url: str, timeout: float = TIMEOUT, retries: int = 1,
                 pace: float = PACE_SECONDS):
        self.base_url, self.timeout, self.retries, self.pace = base_url.rstrip("/"), timeout, retries, pace
        self.calls = self.retried = 0
        self._last = 0.0

    def health(self) -> dict:
        with urllib.request.urlopen(f"{self.base_url}/health", timeout=self.timeout) as r:
            return json.load(r)

    def _post(self, state, questions) -> dict:
        wait = self.pace - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        body = json.dumps({"model": ROUTE, "state": state, "questions": questions}).encode()
        req = urllib.request.Request(f"{self.base_url}/v1/systemone", data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        self.calls += 1
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                result = json.load(r)
                result["x_server_inference_ms"] = r.headers.get("X-Inference-Time-Ms")
        finally:
            self._last = time.monotonic()
        if not isinstance(result, dict):
            raise ValueError("response is not a JSON object")
        routing = result.get("routing") or {}
        if routing.get("model") != ROUTE:
            raise RoutingError(f"routed to {routing.get('model')!r} ({routing.get('reason')!r})")
        return result

    def predict(self, state, questions) -> dict:
        for attempt in range(self.retries + 1):
            try:
                return self._post(state, questions)
            except RoutingError:
                raise  # a deterministic routing answer is recorded, not retried
            except (urllib.error.URLError, TimeoutError, OSError, ValueError):
                if attempt == self.retries:
                    raise
                self.retried += 1
        raise AssertionError("unreachable")


def assess(client: HttpLaya, state: str) -> dict:
    """One request per assessment; any failure is explicit `available: False`, never healthy."""
    start = time.perf_counter()
    try:
        result = client.predict(state, QUESTIONS)
        parsed = la.parse_result(result)
    except Exception as exc:  # noqa: BLE001 - every failure must surface as unavailable
        return {"available": False, "answers": None, "elevated": None, "usage": None,
                "routing": None, "elapsed_ms": (time.perf_counter() - start) * 1000,
                "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    return {"available": True, "answers": parsed["answers"], "elevated": la.elevates(parsed["answers"]),
            "usage": parsed["usage"], "routing": result.get("routing"),
            "server_inference_ms": result.get("x_server_inference_ms"),
            "laya_model_field": parsed["model"], "elapsed_ms": (time.perf_counter() - start) * 1000,
            "error": None}


def cache_key(prov: dict, state: str) -> str:
    return la.cache_key({"revision": prov["revision"], "laya_version": f"{BACKEND}:{prov['laya_version']}"}, state)


def cacheable(res: dict) -> bool:
    """Answers and deterministic refusals (routing, truncation) are cached; transport
    failures are reported for this run but retried on resume."""
    err = res.get("error") or ""
    return res["available"] or err.startswith(("RoutingError", "ValueError: truncated"))


def run_assessments(client, prov, due, cache_dir, log, label, check_health, every=30.0):
    """Sequential, resume-safe assessments; health is re-checked at every progress line."""
    results, fresh, last = {}, 0, time.monotonic()
    for k, (i, state) in enumerate(due.items(), 1):
        path = cache_dir / f"{cache_key(prov, state)}.json"
        if path.exists():
            results[i] = json.loads(path.read_text())
        else:
            res = assess(client, state)
            if cacheable(res):
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(res))
                os.replace(tmp, path)
            results[i], fresh = res, fresh + 1
        if time.monotonic() - last >= every:
            check_health()
            _log(log, f"[{label}] {k}/{len(due)} assessed ({fresh} new requests, {client.retried} retries)")
            last = time.monotonic()
    return results


def remote_facts(host: str, revision: str) -> dict:
    """Read-only: checkpoint file digests and package versions on the serving host."""
    files = " ".join(WEIGHT_FILES)
    cmd = (f"cd ~/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/{revision} && "
           f"for f in {files}; do printf '%s ' \"$f\"; sha256sum < \"$(readlink -f \"$f\")\" | cut -d' ' -f1; done; "
           "~/.local/share/laya/.venv/bin/python -c \"import importlib.metadata as m; "
           "print('versions', m.version('laya'), m.version('torch'))\"")
    out = subprocess.run(["ssh", host, cmd], capture_output=True, text=True, timeout=180, check=True).stdout
    digests, versions = {}, {}
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0] == "versions":
            versions = {"laya": parts[1], "torch": parts[2]}
        elif len(parts) == 2:
            digests[parts[0]] = parts[1]
    return {"weights_sha256": digests, "versions": versions}


def frozen_spec(local_prereg_path) -> dict:
    return {"backend": BACKEND, "route": ROUTE, "prompt_hash": PROMPT_HASH, "questions": QUESTIONS,
            "state_format": la.STATE_FORMAT, "code_sha256": {f: ev.sha256_file(f) for f in CODE_FILES},
            "local_preregistration": {"path": str(local_prereg_path), "sha256": ev.sha256_file(local_prereg_path)},
            "transport": {"timeout_s": TIMEOUT, "retries": 1, "concurrency": 1, "pace_s": PACE_SECONDS}}


def preregister(args, log):
    local = json.loads(Path(args.local_prereg).read_text())
    if local["baseline_freeze"]["sha256"] != ev.sha256_file(args.freeze):
        raise SystemExit("baseline freeze changed since the local preregistration")
    health = HttpLaya(args.endpoint).health()
    revision = health["revisions"][ROUTE]
    remote = remote_facts(args.host, revision)
    doc = {**frozen_spec(args.local_prereg), "artifact": "preregistration-addendum",
           "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "recorded_before_model_answers_on_machine_cases": True,
           "instance": {"label": "lenovo", "endpoint": args.endpoint,
                        "endpoint_note": "SSH local forward to lenovo-worker 127.0.0.1:8000 (laya-serve)",
                        "health_at_preregistration": health, "english_revision": revision,
                        "remote": remote,
                        "weights_identical_to_local": remote["weights_sha256"] == local["model"]["weights_sha256"]},
           "unchanged_from_local": ["rule", "questions", "prompt_hash", "state_format", "thresholds",
                                    "input_exclusions", "caveats"],
           "rule": local["rule"], "per_request_requirements": {
               "routing.model": ROUTE, "usage.truncated": False, "usage.state_tokens_dropped": 0,
               "otherwise": "explicit unavailable, counted"},
           "abort_if": "/health english revision differs from the preregistered one at any check",
           "comparison": "per-assessment agreement of the thresholded decision and |dP| vs the local "
                         "(convaiinnovations/laya 55cf4c4e) run on identical state text"}
    Path(args.prereg_path).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    _log(log, f"addendum written {args.prereg_path} sha256={ev.sha256_file(args.prereg_path)}")


def compare(local_prov, http_prov, due, local_cache, http_cache) -> dict:
    """Agreement of the frozen decision and probability differences on identical requests."""
    agree = both = 0
    diffs = {"review": [], "condition": []}
    for state in due.values():
        lp = local_cache / f"{la.cache_key(local_prov, state)}.json"
        hp = http_cache / f"{cache_key(http_prov, state)}.json"
        if not (lp.exists() and hp.exists()):
            continue
        a, b = json.loads(lp.read_text()), json.loads(hp.read_text())
        if not (a["available"] and b["available"]):
            continue
        both += 1
        agree += a["elevated"] == b["elevated"]
        for q, opt in (("review", "review"), ("condition", "fault_consistent")):
            diffs[q].append(abs(a["answers"][q]["probabilities"][opt] - b["answers"][q]["probabilities"][opt]))
    stats = {q: {"max_abs_diff": max(v) if v else None, "mean_abs_diff": float(np.mean(v)) if v else None}
             for q, v in diffs.items()}
    return {"paired_assessments": both, "decision_agreement": agree / both if both else None, "probability": stats}


def run(args, log):
    prereg = json.loads(Path(args.prereg_path).read_text())
    spec = frozen_spec(args.local_prereg)
    drift = [k for k in spec if prereg.get(k) != spec[k]]
    if drift:
        raise SystemExit(f"code/rule changed after the addendum: {drift}")
    client = HttpLaya(args.endpoint)
    revision = prereg["instance"]["english_revision"]
    health_log = []

    def check_health():
        try:
            h = client.health()
        except Exception as exc:  # noqa: BLE001 - tunnel/server down: stop, do not burn requests
            h = {"error": f"{type(exc).__name__}: {str(exc)[:200]}", "revisions": {}}
        health_log.append({"utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                           "english_revision": h["revisions"].get(ROUTE), "error": h.get("error")})
        if h["revisions"].get(ROUTE) != revision:
            Path(args.out).with_suffix(".aborted.json").write_text(json.dumps({"health": h, "log": health_log}))
            raise SystemExit(f"health check failed or english revision changed: {h}")
        return h

    start_health = check_health()
    remote = prereg["instance"]["remote"]
    prov = {"revision": revision, "laya_version": remote["versions"].get("laya", "unknown")}
    local = json.loads(Path(args.local_prereg).read_text())
    local_prov = {"revision": local["model"]["revision"], "laya_version": local["model"]["laya_version"]}
    cache = Path(args.cache).expanduser(); cache.mkdir(parents=True, exist_ok=True)
    rows, prereg_sha = [], ev.sha256_file(args.prereg_path)
    for name, record, z, due in _streams(json.loads(Path(args.freeze).read_text())):
        _log(log, f"[{name}] {len(due)} assessments via {args.endpoint}")
        results = run_assessments(client, prov, due, cache, log, name, check_health)
        row = stream_row(name, record, z, results, prov, prereg_sha)
        ok = [r for r in results.values() if r["available"]]
        server_ms = [float(r["server_inference_ms"]) for r in ok if r.get("server_inference_ms")]
        row.update({"backend": BACKEND, "instance": "lenovo", "endpoint": args.endpoint,
                    "routing_models": sorted({(r.get("routing") or {}).get("model") for r in ok}),
                    "routing_errors": sum(1 for r in results.values() if (r["error"] or "").startswith("RoutingError")),
                    "max_input_tokens": max((r["usage"]["input_tokens"] for r in ok), default=None),
                    "max_state_tokens": max((r["usage"]["state_tokens"] for r in ok), default=None),
                    "server_inference_ms": {"p50": _pct(server_ms, 50), "p95": _pct(server_ms, 95),
                                            "n": len(server_ms)},
                    "vs_local_msi": compare(local_prov, prov, due, Path(args.local_cache).expanduser(), cache)})
        row["vs_local_msi"]["note"] = "local cache may still be filling while the local run is in progress"
        rows.append(row)
        _log(log, f"[{name}] done coverage={row['coverage']} elevations={row['elevations']}")
    end_health = check_health()
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    keys = ("events_detected", "false_alarm_per_week", "alarm_time_share_clean",
            "notification_episodes_per_week_all", "window_balanced_accuracy", "window_f1", "lead_hours")
    Path(args.summary).write_text(json.dumps({
        "backend": BACKEND, "instance": "lenovo", "english_revision": revision,
        "addendum_sha256": prereg_sha, "results": args.out, "health_start": start_health,
        "health_end": end_health, "health_checks": len(health_log), "requests": client.calls,
        "retries": client.retried, "caveats": rows[0]["caveats"] if rows else [],
        "streams": {r["stream"]: {"coverage": r["coverage"], "elevations": r["elevations"],
                                  "unavailable_errors": r["unavailable_errors"], "latency_ms": r["latency_ms"],
                                  "routing_models": r["routing_models"], "max_input_tokens": r["max_input_tokens"],
                                  "truncated_requests": r["truncated_requests"], "vs_local_msi": r["vs_local_msi"],
                                  **{k: {m: r["metrics"][k].get(m) for m in keys} for k in r["metrics"]}}
                    for r in rows}}, indent=2, sort_keys=True) + "\n")
    _log(log, f"wrote {args.out} and {args.summary}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoint", default="http://127.0.0.1:18000")
    p.add_argument("--host", default="lenovo-worker", help="ssh alias for read-only provenance")
    p.add_argument("--freeze", default="results/v3_normal_freeze.json")
    p.add_argument("--local-prereg", default="results/v3_laya_preregistration.json")
    p.add_argument("--prereg", action="store_true", help="write the addendum only")
    p.add_argument("--prereg-path", default="results/v3_laya_lenovo_preregistration.json")
    p.add_argument("--out", default="results/v3_laya_lenovo.jsonl")
    p.add_argument("--summary", default="results/v3_laya_lenovo_summary.json")
    p.add_argument("--cache", default="~/.local/state/uengram/laya_lenovo_cache")
    p.add_argument("--local-cache", default="~/.local/state/uengram/laya_cache")
    p.add_argument("--log", default=str(Path.home() / ".local/state/uengram/laya_lenovo.log"))
    a = p.parse_args()
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    with open(a.log, "a") as log:
        (preregister if a.prereg else run)(a, log)


if __name__ == "__main__":
    main()
