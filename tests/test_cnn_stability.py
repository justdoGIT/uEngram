import copy

import numpy as np
import pytest

from bench.alarm_eval import ACTIONS
from bench.cnn_stability import empirical_surprisal, score_only_copy
from bench.cnn_stability_merge import merge


def _component(value: int) -> dict:
    return {
        "native": {
            "metrics": {
                "events_detected": value,
                "auroc": value / 10,
                "ap": value / 20,
                "alarm_time_share_clean": value / 100,
            },
            "false_alarm_episodes": value + 1,
        }
    }


def _components(seeds: range) -> dict:
    return {
        f"cnn_{variant}_seed{seed}": _component(seed)
        for variant in ("base", "actcorr", "swa")
        for seed in seeds
    }


def test_empirical_surprisal_orders_large_scores_and_preserves_nan():
    score = np.array([1.0, 2.0, 3.0, np.nan, 4.0])
    calibration = np.array([True, True, True, False, False])
    surprisal = empirical_surprisal(score, calibration)
    assert surprisal[4] > surprisal[2] > surprisal[1] > surprisal[0]
    assert np.isnan(surprisal[3])


def test_score_only_copy_replaces_actions_without_mutating_input():
    original = {
        "novelty": np.array([1.0, 2.0]),
        "action": np.array([0, 1], dtype=np.int8),
        "health": np.array([0.1, 0.2]),
    }
    result = score_only_copy(original, np.array([4.0, 5.0]))
    assert np.array_equal(
        result["action"], np.full(2, ACTIONS.index("score"), dtype=np.int8)
    )
    assert np.array_equal(result["novelty"], [4.0, 5.0])
    assert np.array_equal(original["action"], [0, 1])
    assert result["health"] is original["health"]


def test_merge_combines_seed_groups_and_computes_native_summary():
    primary = {
        "study": "test",
        "protocol": {"selection": "all retained"},
        "datasets": {
            "skab": {
                "role": "consumed development data",
                "components": _components(range(5)),
            }
        },
    }
    extra = {
        "datasets": {
            "skab": {
                "role": "consumed development data",
                "components": {
                    f"cnn_base_seed{seed}": _component(seed)
                    for seed in range(5, 10)
                },
            }
        }
    }
    metro = {
        "datasets": {
            "metropt3": {
                "role": "consumed development data",
                "components": _components(range(3)),
            }
        }
    }
    result = merge(primary, extra, metro)
    summary = result["derived_seed_summaries"]["skab"]["base_10_seeds"]
    assert result["datasets"]["skab"]["seed_groups"]["base"] == list(range(10))
    assert summary["events_detected"] == {
        "median": 4.5,
        "minimum": 0,
        "maximum": 9,
        "values": list(range(10)),
    }
    assert set(result["datasets"]) == {"skab", "metropt3"}


def test_merge_rejects_an_unmarked_evaluation_dataset():
    primary = {
        "study": "test",
        "protocol": {},
        "datasets": {
            "skab": {
                "role": "consumed development data",
                "components": _components(range(5)),
            }
        },
    }
    extra = {
        "datasets": {
            "skab": {
                "role": "consumed development data",
                "components": {
                    f"cnn_base_seed{seed}": _component(seed)
                    for seed in range(5, 10)
                },
            }
        }
    }
    metro = {
        "datasets": {
            "metropt3": {
                "role": "independent confirmation",
                "components": _components(range(3)),
            }
        }
    }
    with pytest.raises(ValueError, match="consumed"):
        merge(copy.deepcopy(primary), extra, metro)
