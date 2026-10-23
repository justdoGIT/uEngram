"""Direct TypeSafe Jev decisions client and label-free, causal Jev advisory inputs."""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

MODEL = "typesafe/jev-1.14"
# OpenRouter Decisions API: the contract every output_modalities=decisions
# model declares, and the route the chat/completions error directs Jev callers
# to. Takes the full model `id` (unlike /api/v1/systemone, which wants the
# bare vendor-suffix id).
ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
CHOICE_CRITERIA = {
    "fault_consistent": "Observed evidence supports an abnormal/fault-consistent condition",
    "expected": "Observed evidence is consistent with expected operation",
    "insufficient": "Available evidence cannot distinguish expected from abnormal",
}
QUESTIONS = {
    "evidence": {"type": "choice", "instructions": "Is the provided physical and numerical evidence sufficient to assess this window? Choose insufficient if key evidence is missing or conflicting.", "criteria": {"sufficient": "Enough relevant evidence to make an assessment", "insufficient": "Missing, unsupported, or conflicting evidence prevents an assessment"}},
    "condition": {"type": "choice", "instructions": "Which condition best fits the provided current and prior window evidence?", "criteria": CHOICE_CRITERIA},
    "review": {"type": "choice", "instructions": "Should this candidate window receive operator review based only on the evidence shown?", "criteria": {"review": "Evidence warrants operator review", "no_review": "Evidence does not warrant operator review", "uncertain": "Cannot determine whether review is warranted"}},
}
PROMPT_VERSION = "jev-advisory-1"
PROMPT_HASH = hashlib.sha256(json.dumps(QUESTIONS, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Assessment:
    available: bool
    answers: dict[str, Any] | None
    model: str | None
    usage: dict[str, Any] | None
    elapsed_ms: float
    error: str | None = None


def _features(summary: dict[str, Any]) -> dict[str, Any]:
    """Return only numerical evidence and bounded relative time; omit identifiers/labels."""
    safe = {}
    excluded = {"lps_share", "lps", "oil_level", "pressure_switch"}
    for key in ("novelty", "health", "novelty_threshold", "health_threshold",
                "monitorable", "active", "decision_state", "support",
                "baseline_scores", "missingness", "physical"):
        if key not in summary:
            continue
        value = summary[key]
        if key == "physical" and isinstance(value, dict):
            value = {name: item for name, item in value.items()
                     if str(name).lower().replace(" ", "_") not in excluded}
        safe[key] = value
    # Deliberately no calendar dates, event/fault labels, file paths, stream names,
    # asset IDs, future windows, or post-decision records.
    return safe


def causal_state(current: dict[str, Any], prior: dict[str, Any] | None, relative_hour: float) -> dict[str, Any]:
    if not isinstance(relative_hour, (int, float)) or relative_hour < 0:
        raise ValueError("relative_hour must be nonnegative")
    return {"relative_hour_from_test_start": float(relative_hour),
            "prior_observed_window": None if prior is None else _features(prior),
            "current_candidate_window": _features(current)}


def parse_response(payload: Any) -> dict[str, Any]:
    """Validate official TypeSafe Choice answer contract and normalized probabilities."""
    if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
        raise ValueError("missing answers object")
    model = payload.get("model")
    if not isinstance(model, str) or not model.startswith(MODEL):
        raise ValueError("response model does not match pinned Jev model")
    answers = payload["answers"]
    parsed = {}
    for name in QUESTIONS:
        item = answers.get(name)
        if not isinstance(item, dict) or item.get("type") != "choice":
            raise ValueError(f"invalid Choice answer: {name}")
        allowed = set(QUESTIONS[name]["criteria"])
        probabilities = item.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != allowed:
            raise ValueError(f"invalid probability keys: {name}")
        vals = list(probabilities.values())
        if any(not isinstance(x, (int, float)) or x < 0 or x > 1 for x in vals) or abs(sum(vals) - 1) > 1e-5:
            raise ValueError(f"invalid probability distribution: {name}")
        choice, confidence = item.get("choice"), item.get("confidence")
        if choice not in allowed or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError(f"invalid choice or confidence: {name}")
        parsed[name] = {"choice": choice, "probabilities": probabilities, "confidence": float(confidence)}
    return {"model": payload.get("model"), "answers": parsed, "usage": payload.get("usage")}


def assess(state: dict[str, Any], timeout: float = 30) -> Assessment:
    """Call the documented OpenRouter Decisions API. Fail closed on unavailability."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return Assessment(False, None, None, None, 0, "missing OPENROUTER_API_KEY")
    body = json.dumps({"model": MODEL, "state": state, "questions": QUESTIONS}, separators=(",", ":")).encode()
    req = urllib.request.Request(ENDPOINT, data=body, method="POST", headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            parsed = parse_response(json.load(response))
        return Assessment(True, parsed["answers"], parsed["model"], parsed["usage"], (time.perf_counter() - start) * 1000)
    except urllib.error.HTTPError as exc:
        # Do not persist potentially sensitive provider response bodies or headers.
        return Assessment(False, None, None, None, (time.perf_counter() - start) * 1000, f"HTTP {exc.code}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return Assessment(False, None, None, None, (time.perf_counter() - start) * 1000, type(exc).__name__)
    except (ValueError, json.JSONDecodeError) as exc:
        return Assessment(False, None, None, None, (time.perf_counter() - start) * 1000, f"invalid response: {type(exc).__name__}")


def cache_key(state: dict[str, Any], baseline_hash: str, trace_hash: str) -> str:
    encoded = json.dumps({"prompt_hash": PROMPT_HASH, "model": MODEL, "state": state,
                          "baseline_hash": baseline_hash, "trace_hash": trace_hash}, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
