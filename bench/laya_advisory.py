"""Local open-weight Laya backend (ConvAI Innovations, Apache-2.0) for candidate triage.

Laya is NOT Jev; its outputs are never labelled Jev. It reuses the Jev typed question schema
(`bench.jev_advisory.QUESTIONS`, same prompt hash) so the two transports stay comparable.
Request text is built only from causal trace inputs: no labels, event/fault dates, IDs,
stream/file names, absolute timestamps, future windows or existing warning channels.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from bench.advisory_eval import CANDIDATE_STATES
from bench.jev_advisory import PROMPT_HASH, QUESTIONS

MODEL_ID = "convaiinnovations/laya"
REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"  # laya.PINNED_REVISIONS reviewed SHA
BACKEND = "laya"
STATE_FORMAT = "laya-state-1"
REVIEW_PROBABILITY = 0.80
REVIEW_CONFIDENCE = 0.50  # TypeSafe Choice formula over Laya probabilities, not Laya `confidence`
REFERENCE_WINDOWS = 36
PROB_TOLERANCE = 1e-3  # Laya rounds probabilities to 4 decimals
# Allow-list: anything not named here (lps_share, oil level, pressure switch, ...) is never sent.
FEATURES = {
    "load_share": ("loaded-time share", "", 2),
    "comp_switches": ("compressor on/off switches", "", 0),
    "motor_current_a": ("motor current", " A", 2),
    "oil_temp_c": ("oil temperature", " C", 1),
    "tp2_bar": ("TP2 pressure", " bar", 2),
    "tp3_bar": ("TP3 pressure", " bar", 2),
    "h1_bar": ("H1 pressure", " bar", 2),
    "dv_pressure_bar": ("DV pressure", " bar", 2),
    "reservoirs_min_bar": ("reservoir minimum pressure", " bar", 2),
}
BANNED_FEATURES = ("lps_share", "lps", "oil_level", "pressure_switch")
MODES = {0: "idle or offloaded", 1: "cycling load", 2: "sustained load"}
STATUS = {0: "unobserved", 1: "normal reference", 2: "unresolved candidate (not an alarm)",
          3: "alarm"}


def _num(x, digits):
    return "missing" if not np.isfinite(x) else f"{float(x):.{digits}f}"


def _ratio(score, threshold):
    if not (np.isfinite(score) and np.isfinite(threshold)) or threshold <= 0:
        return None
    return float(score) / float(threshold)


class StateBuilder:
    """Renders one assessment's English state text from a frozen trace (inputs only)."""

    def __init__(self, z: dict):
        self.z = z
        self.names = [str(x) for x in z["feature_names"]]
        self.cols = {n: self.names.index(n) for n in FEATURES if n in self.names}
        self.support = json.loads(str(z["baseline_counts"]))
        self.per_state = json.loads(str(z["config"])).get("per_state", False)
        ds, st = z["decision_state"], z["state"]
        self.reference = {int(s): np.flatnonzero((ds == 1) & (st == s)) for s in np.unique(st)}

    def _recent_normal(self, i):
        idx = self.reference.get(int(self.z["state"][i]), np.empty(0, int))
        stop = bisect.bisect_left(idx, i)  # strictly before i: causal
        window = idx[max(0, stop - REFERENCE_WINDOWS):stop]
        if not len(window):
            return None
        return np.nanmedian(self.z["features"][window], axis=0)

    def _readings(self, i, reference):
        parts = []
        for name, (label, unit, digits) in FEATURES.items():
            if name not in self.cols:
                continue
            value = self.z["features"][i, self.cols[name]]
            text = f"{label} {_num(value, digits)}{unit if np.isfinite(value) else ''}"
            if reference is not None and np.isfinite(reference[self.cols[name]]):
                text += f" (recent normal {_num(reference[self.cols[name]], digits)})"
            parts.append(text)
        return "; ".join(parts)

    def _scores(self, i):
        z = self.z
        nov = _ratio(z["novelty"][i], z["novelty_threshold"][i])
        hea = _ratio(z["health"][i], z["health_threshold"][i])
        nov_text = "not scored" if nov is None else f"{nov:.2f} times its calibrated threshold"
        hea_text = "not scored" if hea is None else f"{hea:.2f} times its calibrated threshold"
        return f"Novelty score {nov_text}. Health score {hea_text} (health is not used for alarms)."

    def _missing(self, i):
        z, missing = self.z, []
        if not np.isfinite(z["novelty"][i]):
            missing.append("novelty score")
        missing += [FEATURES[n][0] for n, c in self.cols.items() if not np.isfinite(z["features"][i, c])]
        return ", ".join(missing) if missing else "none"

    def _support(self, i):
        key = str(int(self.z["state"][i])) if self.per_state else "-1"
        counts = self.support.get(key, {})
        ok = counts.get("novelty_threshold_supported", False)
        n = int(counts.get("monitorable_windows", 0))
        return (f"calibrated from {n} reference windows" if ok
                else f"not calibrated (only {n} reference windows)")

    def build(self, i: int, entry: int) -> str:
        z = self.z
        minutes = int(round(float(z["end"][i] - z["start"][i]) / 60))
        hours = max(0.0, float(z["end"][i] - z["end"][entry]) / 3600)
        mode = MODES.get(int(z["state"][i]), "unknown")
        text = [
            f"Air compressor monitoring summary for one {minutes}-minute window flagged by a "
            "numerical anomaly detector.",
            f"Detector status: {STATUS[int(z['decision_state'][i])]}. "
            f"Hours since this flagged period began: {hours:.1f}.",
            f"Operating mode: {mode}; its reference is {self._support(i)}.",
            self._scores(i),
            f"Missing values: {self._missing(i)}.",
            f"Current readings: {self._readings(i, self._recent_normal(i))}.",
        ]
        if i > 0:
            p = i - 1
            nov = _ratio(z["novelty"][p], z["novelty_threshold"][p])
            text.append(f"Previous window: status {STATUS[int(z['decision_state'][p])]}; novelty "
                        f"{'not scored' if nov is None else f'{nov:.2f} times threshold'}; "
                        f"readings: {self._readings(p, None)}.")
        return " ".join(text)


def due_states(z: dict, due: np.ndarray) -> dict[int, str]:
    """State text for every scheduled assessment index of one trace."""
    builder, out, entry = StateBuilder(z), {}, None
    for i, state in enumerate(z["decision_state"]):
        if int(state) not in CANDIDATE_STATES:
            entry = None
            continue
        entry = i if entry is None else entry
        if due[i]:
            out[i] = builder.build(i, entry)
    return out


def typesafe_confidence(probabilities: dict) -> float:
    """TypeSafe's documented Choice confidence, (p_max - 1/n) / (1 - 1/n)."""
    n = len(probabilities)
    return (max(probabilities.values()) - 1 / n) / (1 - 1 / n)


def elevates(answers: dict) -> bool:
    """Frozen rule: review chosen, P(review) >= 0.80 and Choice confidence >= 0.50."""
    review = answers["review"]
    p = review["probabilities"]
    return (review["choice"] == "review" and p["review"] >= REVIEW_PROBABILITY
            and typesafe_confidence(p) >= REVIEW_CONFIDENCE)


def parse_result(result: Any) -> dict:
    """Validate Laya's typed Choice answers and require an untruncated request."""
    usage = result.get("usage") if isinstance(result, dict) else None
    if not isinstance(usage, dict):
        raise ValueError("missing usage")
    if usage.get("truncated") or usage.get("state_tokens_dropped", 0) or usage.get("truncated_questions"):
        raise ValueError("truncated request")
    parsed = {}
    for name, q in QUESTIONS.items():
        item = result.get("answers", {}).get(name)
        if not isinstance(item, dict) or item.get("type") != "choice":
            raise ValueError(f"invalid Choice answer: {name}")
        probs = item.get("probabilities")
        if not isinstance(probs, dict) or set(probs) != set(q["criteria"]):
            raise ValueError(f"invalid probability keys: {name}")
        vals = [float(v) for v in probs.values()]
        if any(not 0 <= v <= 1 for v in vals) or abs(sum(vals) - 1) > PROB_TOLERANCE:
            raise ValueError(f"invalid probability distribution: {name}")
        if item.get("choice") not in probs:
            raise ValueError(f"invalid choice: {name}")
        parsed[name] = {"choice": item["choice"], "probabilities": {k: float(v) for k, v in probs.items()},
                        "laya_confidence": item.get("confidence"),
                        "laya_answer_confidence": item.get("answer_confidence")}
    return {"answers": parsed, "model": result.get("model"), "usage": usage}


def assess(agent, state: str) -> dict:
    """One predict per assessment; any failure is explicit `available: False`, never healthy."""
    start = time.perf_counter()
    try:
        parsed = parse_result(agent.predict(state, QUESTIONS))
    except Exception as exc:  # noqa: BLE001 - every failure must surface as unavailable
        return {"available": False, "answers": None, "elevated": None, "usage": None,
                "elapsed_ms": (time.perf_counter() - start) * 1000,
                "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    return {"available": True, "answers": parsed["answers"], "elevated": elevates(parsed["answers"]),
            "usage": parsed["usage"], "laya_model_field": parsed["model"],
            "elapsed_ms": (time.perf_counter() - start) * 1000, "error": None}


def cache_key(provenance: dict, state: str) -> str:
    blob = {"backend": BACKEND, "model": MODEL_ID, "revision": provenance["revision"],
            "laya_version": provenance["laya_version"], "prompt_hash": PROMPT_HASH,
            "state_format": STATE_FORMAT, "state": state}
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()


def budget(agent, states) -> dict:
    """Token budget of every (state, question) sequence with Laya's own tokenizer."""
    from laya.common import _encode_question_text, build_head, encode_text, render_options, serialize_state

    tok, cfg = agent.tok, agent.cfg
    max_len, head_max = int(cfg.get("max_len", 512)), int(cfg.get("head_max_len", 192))
    heads, head_truncated = {}, []
    for qid, qdef in QUESTIONS.items():
        q = agent._to_internal(qdef)
        ids, _, _ = build_head(tok, q, head_max)
        opts = [len(_encode_question_text(tok, " " + o, add_special_tokens=False))
                for o in render_options(q)]
        ins = len(_encode_question_text(tok, "%s question: %s" % (q["t"], q["ins"]),
                                        add_special_tokens=False))
        if max(opts) > 48 or ins > head_max - sum(1 + n for n in opts):
            head_truncated.append(qid)
        heads[qid] = len(ids)
    room = max_len - max(heads.values()) - 1
    lengths = [len(encode_text(tok, serialize_state(s).replace(tok.mask_token, " "),
                               add_special_tokens=False)["input_ids"]) for s in states]
    longest = max(lengths, default=0)
    return {"max_len": max_len, "head_max_len": head_max, "head_tokens": heads,
            "head_truncated_questions": head_truncated, "state_room": room,
            "requests": len(lengths), "max_state_tokens": longest,
            "max_sequence_tokens": max(heads.values()) + longest + 1,
            "truncated_requests": int(sum(n > room for n in lengths)) + (len(lengths) if head_truncated else 0)}


def load_agent(expected_sha256: dict | None = None):
    """Pinned CPU agent (2 threads) plus provenance of the weights actually loaded."""
    import importlib.metadata

    import torch
    from huggingface_hub import snapshot_download

    import laya
    from laya.revisions import PINNED_REVISIONS, snapshot_revision

    if PINNED_REVISIONS.get(MODEL_ID) != REVISION:
        raise RuntimeError("laya package reviewed revision differs from the preregistered one")
    torch.set_num_threads(2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = laya.load(MODEL_ID, device="cpu", revision=REVISION, expected_sha256=expected_sha256)
    snapshot = Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True,
                                      allow_patterns=["rl_agent_config.json", "model.safetensors",
                                                      "tokenizer/*", "encoder/*"]))
    digests = {}
    for f in sorted(p for p in snapshot.rglob("*") if p.is_file()):
        with open(f, "rb") as fh:
            digests[str(f.relative_to(snapshot))] = hashlib.file_digest(fh, "sha256").hexdigest()
    provenance = {"backend": BACKEND, "model_id": MODEL_ID, "revision": REVISION,
                  "snapshot_revision": snapshot_revision(str(snapshot)), "weights_sha256": digests,
                  "laya_version": importlib.metadata.version("laya"), "torch_version": torch.__version__,
                  "device": "cpu", "torch_threads": torch.get_num_threads(),
                  "max_len": int(agent.cfg.get("max_len", 512)),
                  "head_max_len": int(agent.cfg.get("head_max_len", 192)),
                  "load_warnings": [f"{w.category.__name__}: {w.message}" for w in caught]}
    if provenance["snapshot_revision"] != REVISION:
        raise RuntimeError("loaded snapshot is not the preregistered revision")
    return agent, provenance
