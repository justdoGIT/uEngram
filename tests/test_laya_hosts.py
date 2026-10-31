import json

import pytest

from bench import laya_advisory as la
from bench import laya_hosts as lh
from bench.jev_advisory import QUESTIONS


def _answers(review=0.9):
    out = {}
    for name, q in QUESTIONS.items():
        opts = list(q["criteria"])
        rest = (1 - review) / (len(opts) - 1)
        out[name] = {"type": "choice", "choice": opts[0],
                     "probabilities": {o: (review if o == opts[0] else rest) for o in opts}}
    return out


def _rec(available=True, review=0.9):
    return {"available": available, "answers": _answers(review) if available else None,
            "elevated": la.elevates(_answers(review)) if available else None,
            "usage": {"input_tokens": 300, "truncated": False, "state_tokens_dropped": 0},
            "elapsed_ms": 10.0, "error": None, "host": "msi"}


def test_hp_key_distinguishes_hosts_and_models():
    p = {"revision": "abc", "laya_version": "x"}
    assert lh.hp_key(p, "state") != la.cache_key({"revision": "abc", "laya_version": "x"}, "state")


def test_load_cached_records_and_rejects_host_mismatch(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps(_rec()))
    assert lh.load_cached(p, "msi")["host"] == "msi"
    with pytest.raises(ValueError):
        lh.load_cached(p, "hp")


def test_merge_prefers_msi_and_labels_hosts(tmp_path):
    local, hp = tmp_path / "local", tmp_path / "hp"
    local.mkdir(), hp.mkdir()
    lp = {"revision": "rL", "laya_version": "0.4.13"}
    hpr = {"revision": "rH", "laya_version": "0.4.13"}
    states = {1: "only-msi", 2: "both", 3: "only-hp"}
    (local / f"{la.cache_key(lp, 'only-msi')}.json").write_text(json.dumps(_rec(True, 0.85)))
    (local / f"{la.cache_key(lp, 'both')}.json").write_text(json.dumps({**_rec(True, 0.85), "host": "msi"}))
    (hp / f"{lh.hp_key(hpr, 'both')}.json").write_text(json.dumps({**_rec(True, 0.95), "host": "hp"}))
    (hp / f"{lh.hp_key(hpr, 'only-hp')}.json").write_text(json.dumps({**_rec(True, 0.95), "host": "hp"}))
    out = lh.merge(states, lp, hpr, local, hp)
    assert out[1]["host"] == "msi" and out[2]["host"] == "msi" and out[3]["host"] == "hp"
    assert out[2]["answers"]["review"]["probabilities"]["review"] == 0.85  # msi wins


def test_merge_missing_host_raises(tmp_path):
    with pytest.raises(ValueError):
        lh.merge({1: "x"}, {"revision": "r", "laya_version": "0"},
                 {"revision": "h", "laya_version": "0"}, tmp_path, tmp_path)


def test_compare_pair_detects_flips_and_diffs():
    a = _rec(True, 0.9); a["elevated"] = la.elevates(a["answers"])
    b = _rec(True, 0.85); b["elevated"] = la.elevates(b["answers"])
    c = lh.compare_pair(a, b)
    assert c["d_review_probability"] == pytest.approx(0.05)
    assert c["flip"] is False
    flips = lh.compare_pair(_rec(True, 0.95), _rec(True, 0.6))
    assert flips["flip"] is True


def test_summarize_pairs_reports_flips_and_max():
    pairs = [lh.compare_pair(_rec(True, 0.9), _rec(True, 0.8)),
             lh.compare_pair(_rec(True, 0.95), _rec(True, 0.6))]
    s = lh.summarize_pairs(pairs)
    assert s["pairs"] == 2 and s["decision_flips"] == 1
    assert s["d_review_probability"]["max"] == pytest.approx(0.35)


def test_compare_pair_max_any_covers_all_question_options():
    a = _rec(); b = _rec()
    b["answers"]["condition"]["probabilities"]["fault_consistent"] += 0.04
    b["answers"]["condition"]["probabilities"]["expected"] -= 0.04
    assert lh.compare_pair(a, b)["d_max_any"] == pytest.approx(0.04)
