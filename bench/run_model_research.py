"""Compare opt-in encoders without selecting on real test labels.

The first 80% of commissioning windows fits each encoder. The remaining
commissioning windows are label-free input to synthetic-fault model selection.
Real event labels are accessed only afterward for the descriptive evaluation.
All repository streams are consumed development data; this command is not
independent confirmation.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bench.alarm_eval import ACTIONS, apply_policy
from bench.physical_residuals import PhysicalResiduals, observation_times, physical_features
from bench.run_v2 import FROZEN
from bench.stream_skab import skab_stream
from bench.streams import load_stream
from bench.v2_scoring import score_embeddings
from bench.model_selection import select_by_synthetic_challenges
from uengram.alarm import AlarmConfig
from uengram.encoder_cnn import CNNEncoder
from uengram.encoder_multiscale import MultiScaleEncoder
from uengram.encoders import DSPEncoder


class _ComponentView:
    def __init__(self, encoder, name: str):
        self.encoder = encoder
        self.name = name
        self.d = encoder.level_d if name == "level" else encoder.dynamics_d
        self.sigma0 = encoder.sigma0_level if name == "level" else encoder.sigma0_dynamics

    def encode(self, windows, sequential=True):
        del sequential
        return self.encoder.encode_components(windows)[self.name]

def _stream(name: str):
    return skab_stream(K=10) if name == "skab" else load_stream(name)


def _encoder(name: str, seed: int, epochs: int):
    if name == "dsp":
        return DSPEncoder(d=16)
    if name == "multiscale":
        return MultiScaleEncoder(d=16)
    if name == "cnn":
        return CNNEncoder(seed=seed, d=16, epochs=epochs)
    if name == "cnnssl":
        from uengram.encoder_cnn_ssl import SSLTemporalEncoder

        return SSLTemporalEncoder(seed=seed, d=16, epochs=epochs)
    raise ValueError(f"unknown encoder candidate: {name}")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _evaluate(stream, scores: dict, config: AlarmConfig) -> dict:
    metrics, details, _ = apply_policy(stream, scores, config)
    return {
        "metrics": _json_safe(metrics),
        "events_detected": [
            int(item["event"]) for item in details["events"] if item["detected"]
        ],
        "false_alarm_episodes": int(details["false_alarm_episodes"]),
    }


def run(stream_name: str, names: list[str], seeds: list[int], epochs: int) -> dict:
    stream = _stream(stream_name)
    fit_rows = np.flatnonzero(stream.fit)
    if len(fit_rows) < 25:
        raise ValueError("need at least 25 commissioning windows for train/select split")
    split = max(10, int(len(fit_rows) * 0.8))
    if split >= len(fit_rows):
        split = len(fit_rows) - 1
    encoder_rows, selection_rows = fit_rows[:split], fit_rows[split:]
    selection_windows = stream.W[selection_rows]
    print(
        f"dataset={stream_name} role=consumed-development fit={len(encoder_rows)} "
        f"selection={len(selection_rows)} calibration={int(stream.cal.sum())} "
        f"test={int(stream.test.sum())}",
        flush=True,
    )
    candidates, selection = {}, {}
    for name in names:
        for seed in (seeds if name in ("cnn", "cnnssl") else [0]):
            key = name if name in ("dsp", "multiscale") else f"{name}_seed{seed}"
            print(f"candidate={key} stage=fit", flush=True)
            encoder = _encoder(name, seed, epochs)
            encoder.fit(
                stream.W[encoder_rows], 1, stream.segment[encoder_rows]
            )
            candidates[key] = encoder
    selection_candidates = dict(candidates)
    if "multiscale" in candidates:
        encoder = candidates["multiscale"]
        for block in ("level", "dynamics"):
            selection_candidates[f"multiscale_{block}"] = _ComponentView(encoder, block)
    print("stage=label_free_synthetic_selection", flush=True)
    selected = select_by_synthetic_challenges(
        selection_candidates, selection_windows, alpha=0.01
    )
    selection["synthetic_challenges"] = _json_safe(selected)
    selection["uses_real_test_labels"] = False
    selection["selection_window_indices"] = selection_rows.tolist()

    native_config = FROZEN["uengram"]
    health_config = replace(
        native_config, k_novelty=0, k_health=3, new_state_alert=False,
        health_needs_baseline=False,
    )
    rows = {}
    for key, encoder in candidates.items():
        print(f"candidate={key} stage=streaming-score", flush=True)
        components = (
            encoder.encode_components(stream.W)
            if isinstance(encoder, MultiScaleEncoder) else None
        )
        latent = components["combined"] if components else encoder.encode(stream.W)
        score = score_embeddings(
            stream, "uengram", key, 0, encoder, latent,
            f"[model-research {stream_name} {key}]", record_encoding=False,
        )
        rows[key] = {
            "native_frozen_policy": _evaluate(stream, score, native_config),
            "separate_health_only": _evaluate(stream, score, health_config),
            "latent_dimension": int(encoder.d),
            "sigma0_shape": list(np.asarray(encoder.sigma0).shape),
        }
        if components:
            for block in ("level", "dynamics"):
                view = SimpleNamespace(
                    d=components[block].shape[1],
                    sigma0=getattr(encoder, f"sigma0_{block}"),
                )
                branch_score = score_embeddings(
                    stream, "uengram", f"multiscale_{block}", 0, view,
                    components[block],
                    f"[model-research {stream_name} multiscale_{block}]",
                    record_encoding=False,
                )
                rows[f"multiscale_{block}"] = {
                    "native_frozen_policy": _evaluate(
                        stream, branch_score, native_config
                    ),
                    "separate_health_only": _evaluate(
                        stream, branch_score, health_config
                    ),
                    "latent_dimension": int(view.d),
                    "sigma0_shape": list(np.asarray(view.sigma0).shape),
                    "role": "separate evidence branch; not fused",
                }

    if stream_name == "metropt3":
        print("branch=physical_residual stage=fit-score", flush=True)
        context, targets = physical_features(stream, observation_times(stream))
        residual = PhysicalResiduals(conditional=True).fit(
            context, targets, stream.fit & stream.monitorable
        )
        physical_score, component_scores = residual.score(context, targets)
        physical = {
            "novelty": physical_score,
            "health": np.full(len(physical_score), np.nan),
            "action": np.full(len(physical_score), ACTIONS.index("score"), dtype=np.int8),
            "meta": {"method": "physical_residual", **residual.metadata()},
        }
        physical_policy = AlarmConfig(
            alpha=0.01, k_novelty=1, k_health=0, per_state=False,
            new_state_alert=False,
        )
        rows["physical_residual"] = {
            "fixed_policy": _evaluate(stream, physical, physical_policy),
            "components_supported": np.isfinite(component_scores).any(axis=0).tolist(),
            "role": "separate physical evidence; not fused into novelty",
        }

    return {
        "protocol": "candidate-research-v1",
        "dataset": stream_name,
        "dataset_role": "consumed development data; not independent confirmation",
        "selection": selection,
        "evaluation": rows,
        "configs": {"cnn_epochs": epochs, "cnn_seeds": seeds, "alarm": native_config.__dict__},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", default="skab")
    parser.add_argument("--encoders", default="dsp,multiscale,cnn,cnnssl")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite research output: {destination}")
    result = run(
        args.stream,
        [item.strip() for item in args.encoders.split(",") if item.strip()],
        [int(item) for item in args.seeds.split(",") if item.strip()],
        args.epochs,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n")
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
