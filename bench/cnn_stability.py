#!/usr/bin/env python3
"""Evaluate CNN stability ablations without changing the production encoder.

SKAB and MetroPT-3 are consumed development datasets. Results from this
exploratory harness are not independent confirmation.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from bench.alarm_eval import ACTIONS, apply_policy
from bench.run_v2 import FROZEN
from bench.stream_skab import skab_stream
from bench.streams import load_stream
from bench.v2_scoring import score_embeddings
from uengram.alarm import AlarmConfig
from uengram.encoder_cnn import CNNEncoder
from uengram.encoders import DSPEncoder, fit_projection


class ResearchCNN(CNNEncoder):
    """CNN ablations that leave the production class unchanged."""

    def __init__(self, *, seed: int, variant: str, d: int = 32, epochs: int = 90):
        super().__init__(seed=seed, d=d, epochs=epochs, pool="maxavg", ortho_beta=0.05)
        if variant not in ("actcorr", "swa"):
            raise ValueError(f"unsupported research variant: {variant}")
        self.variant = variant
        self.activation_corr_beta = 0.05 if variant == "actcorr" else 0.0
        self.swa_start = 60 if variant == "swa" else None
        self.training_trace: list[dict[str, float]] = []

    def fit(self, train_windows, noise_lag: int, groups: np.ndarray | None) -> "ResearchCNN":
        import torch

        previous_threads = torch.get_num_threads()
        previous_determinism = torch.are_deterministic_algorithms_enabled()
        torch.manual_seed(self.seed)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        try:
            x = np.asarray(train_windows, dtype=np.float32)
            n, channels, samples = x.shape
            self.ch_mean = x.mean(axis=(0, 2), keepdims=True)[0].astype(np.float32)
            self.ch_std = np.maximum(
                x.std(axis=(0, 2), keepdims=True)[0], 1e-6
            ).astype(np.float32)
            xt = torch.from_numpy(self._std(x))
            decoder = self._build(channels, samples)
            params = [*self.conv.parameters(), *self.fc.parameters(), *decoder.parameters()]
            optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
            generator = torch.Generator().manual_seed(self.seed)
            identity = torch.eye(self.d)
            averaged: list[torch.Tensor] | None = None
            averaged_count = 0
            inference_params = [*self.conv.parameters(), *self.fc.parameters()]
            self.conv.train()
            self.fc.train()
            decoder.train()
            for epoch in range(self.epochs):
                permutation = torch.randperm(n, generator=generator)
                epoch_recon = 0.0
                epoch_corr = 0.0
                batches = 0
                for start in range(0, n, self.batch):
                    xb = xt[permutation[start : start + self.batch]]
                    z = self._forward_enc(xb)
                    reconstruction = torch.mean(
                        (decoder(z).reshape(-1, channels, samples) - xb) ** 2
                    )
                    weight_ortho = torch.mean(
                        (self.fc.weight @ self.fc.weight.T - identity) ** 2
                    )
                    correlation = torch.zeros(())
                    if self.activation_corr_beta and len(z) > 1:
                        centered = z - z.mean(dim=0, keepdim=True)
                        normalized = centered / (
                            centered.std(dim=0, unbiased=False, keepdim=True) + 1e-4
                        )
                        corr = normalized.T @ normalized / len(z)
                        off_diagonal = corr - torch.diag(torch.diag(corr))
                        correlation = off_diagonal.square().sum() / (
                            self.d * (self.d - 1)
                        )
                    loss = reconstruction + 0.05 * weight_ortho
                    loss = loss + self.activation_corr_beta * correlation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_recon += float(reconstruction.detach())
                    epoch_corr += float(correlation.detach())
                    batches += 1
                if self.swa_start is not None and epoch >= self.swa_start:
                    if averaged is None:
                        averaged = [p.detach().clone() for p in inference_params]
                    else:
                        for avg, param in zip(averaged, inference_params, strict=True):
                            avg.add_((param.detach() - avg) / (averaged_count + 1))
                    averaged_count += 1
                record = {
                    "epoch": float(epoch + 1),
                    "reconstruction": epoch_recon / batches,
                    "activation_correlation": epoch_corr / batches,
                }
                self.training_trace.append(record)
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(
                        f"train variant={self.variant} seed={self.seed} "
                        f"epoch={epoch + 1}/{self.epochs} "
                        f"recon={record['reconstruction']:.6f} "
                        f"corr={record['activation_correlation']:.6f}",
                        flush=True,
                    )
            if averaged is not None:
                for param, avg in zip(inference_params, averaged, strict=True):
                    param.data.copy_(avg)
            self.conv.eval()
            self.fc.eval()
            latent = self._latent(x)
            self.whitener, self.noise, self.sigma0 = fit_projection(latent, self.d, noise_lag, groups)
            return self
        finally:
            torch.set_num_threads(previous_threads)
            torch.use_deterministic_algorithms(previous_determinism)


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def empirical_surprisal(scores: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(scores[calibration], dtype=float))
    reference = reference[np.isfinite(reference)]
    out = np.full(len(scores), np.nan)
    finite = np.isfinite(scores)
    at_or_above = len(reference) - np.searchsorted(reference, scores[finite], side="left")
    p_value = (at_or_above + 1.0) / (len(reference) + 1.0)
    out[finite] = -np.log(p_value)
    return out


def score_only_copy(sc: dict, novelty: np.ndarray | None = None) -> dict:
    return {
        **sc,
        "novelty": np.asarray(
            sc["novelty"] if novelty is None else novelty,
            dtype=float,
        ),
        "action": np.full(len(sc["action"]), ACTIONS.index("score"), dtype=np.int8),
    }


def summarize(stream, metrics: dict, details: dict) -> dict:
    result = {"metrics": clean_json(metrics)}
    detected = [row["event"] for row in details["events"] if row["detected"]]
    names = stream.extra.get("experiment_names")
    result["detected_events"] = [names[i] if names is not None else int(i) for i in detected]
    result["false_alarm_episodes"] = int(details["false_alarm_episodes"])
    result["clean_observed_hours"] = float(details["clean_observed_hours"])
    return result


def evaluate(stream, sc: dict, cfg: AlarmConfig) -> dict:
    metrics, details, _ = apply_policy(stream, sc, cfg)
    return summarize(stream, metrics, details)


def model_bytes(encoder) -> int:
    parameters = [*encoder.conv.parameters(), *encoder.fc.parameters()]
    return sum(p.numel() * p.element_size() for p in parameters)


def build_encoder(variant: str, seed: int):
    common = {"seed": seed, "d": 32}
    if variant == "plain30":
        return CNNEncoder(**common, epochs=30)
    if variant == "middle30":
        return CNNEncoder(**common, epochs=30, representation="middle")
    if variant == "multiscale30":
        return CNNEncoder(**common, epochs=30, representation="multiscale")
    if variant == "plain":
        return CNNEncoder(**common, epochs=90)
    if variant == "plain_multiscale":
        return CNNEncoder(**common, epochs=90, representation="multiscale")
    if variant == "base":
        return CNNEncoder(**common, epochs=90, pool="maxavg", ortho_beta=0.05)
    if variant == "multiscale":
        return CNNEncoder(
            **common,
            epochs=90,
            pool="maxavg",
            ortho_beta=0.05,
            representation="multiscale",
        )
    return ResearchCNN(seed=seed, variant=variant)


def load_dataset(name: str):
    if name == "skab":
        return skab_stream(K=10), 48
    if name == "metropt3":
        return load_stream("metropt3"), 42
    raise ValueError(name)


def run_dataset(name: str, variants: list[str], seeds: list[int]) -> dict:
    stream, dsp_d = load_dataset(name)
    print(
        f"dataset={name} windows={len(stream.W)} fit={stream.fit.sum()} "
        f"cal={stream.cal.sum()} test={stream.test.sum()}",
        flush=True,
    )
    native_cfg = FROZEN["uengram"]
    score_cfg = AlarmConfig(**{**native_cfg.__dict__, "new_state_alert": False})
    scores: dict[str, dict] = {}
    components: dict[str, dict] = {}

    print(f"dataset={name} stage=dsp_d{dsp_d}", flush=True)
    dsp = DSPEncoder(d=dsp_d).fit(
        stream.W[stream.fit], 1, stream.segment[stream.fit]
    )
    dsp_z = dsp.encode(stream.W)
    dsp_sc = score_embeddings(
        stream,
        "uengram",
        f"dsp_d{dsp_d}",
        0,
        dsp,
        dsp_z,
        f"[{name} dsp_d{dsp_d}]",
        record_encoding=False,
    )
    scores[f"dsp_d{dsp_d}"] = dsp_sc
    components[f"dsp_d{dsp_d}"] = {
        "native": evaluate(stream, dsp_sc, native_cfg),
        "score_only": evaluate(stream, score_only_copy(dsp_sc), score_cfg),
        "latent_dimension": dsp_d,
    }

    for variant in variants:
        for seed in seeds:
            key = f"cnn_{variant}_seed{seed}"
            print(f"dataset={name} stage={key}", flush=True)
            started = time.perf_counter()
            encoder = build_encoder(variant, seed)
            encoder.fit(stream.W[stream.fit], 1, stream.segment[stream.fit])
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            latent = encoder.encode(stream.W)
            encode_seconds = time.perf_counter() - started
            sc = score_embeddings(
                stream,
                "uengram",
                f"cnn_{variant}",
                seed,
                encoder,
                latent,
                f"[{name} {variant} seed={seed}]",
                record_encoding=False,
            )
            scores[key] = sc
            record = {
                "native": evaluate(stream, sc, native_cfg),
                "score_only": evaluate(stream, score_only_copy(sc), score_cfg),
                "fit_seconds": fit_seconds,
                "encode_seconds": encode_seconds,
                "encoder_bytes": model_bytes(encoder),
                "latent_dimension": encoder.d,
                "epochs": encoder.epochs,
                "pool": encoder.pool,
                "representation": encoder.representation,
            }
            if isinstance(encoder, ResearchCNN):
                record["training_final"] = encoder.training_trace[-1]
            components[key] = record
            print(
                f"dataset={name} done={key} fit_s={fit_seconds:.2f} "
                f"encode_s={encode_seconds:.2f} "
                f"native_events={record['native']['metrics']['events_detected']}",
                flush=True,
            )

    surprisals = {
        key: empirical_surprisal(sc["novelty"], stream.cal)
        for key, sc in scores.items()
    }
    fusions: dict[str, dict] = {}
    dsp_key = f"dsp_d{dsp_d}"
    for variant in variants:
        keys = [f"cnn_{variant}_seed{seed}" for seed in seeds]
        matrix = np.column_stack([surprisals[key] for key in keys])
        valid_count = np.isfinite(matrix).sum(axis=1)
        cnn_median = np.full(len(matrix), np.nan)
        valid = valid_count > 0
        cnn_median[valid] = np.nanmedian(matrix[valid], axis=1)
        cnn_fisher = np.nansum(matrix, axis=1)
        cnn_fisher[~valid] = np.nan
        candidates = {
            f"cnn_{variant}_median": cnn_median,
            f"cnn_{variant}_fisher": cnn_fisher,
            f"dsp_plus_{variant}_max": np.fmax(surprisals[dsp_key], cnn_median),
            f"dsp_plus_{variant}_mean": (surprisals[dsp_key] + cnn_median) / 2.0,
        }
        template = scores[keys[0]]
        for fusion_name, fused in candidates.items():
            fusions[fusion_name] = evaluate(stream, score_only_copy(template, fused), score_cfg)

    return {
        "role": "consumed development data; not independent confirmation",
        "shape": {
            "windows": len(stream.W),
            "fit": int(stream.fit.sum()),
            "calibration": int(stream.cal.sum()),
            "test": int(stream.test.sum()),
        },
        "dsp_dimension": dsp_d,
        "cnn": {
            "d": 32,
            "epochs": 90,
            "pool": "maxavg",
            "ortho_beta": 0.05,
            "representation": "bottleneck or multiscale, as named",
        },
        "score_only_config": clean_json(score_cfg.__dict__),
        "components": clean_json(components),
        "fusions": clean_json(fusions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="skab")
    parser.add_argument("--variants", default="base,actcorr,swa")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    result = {
        "study": "CNN internal representations, stability, ensembling, and DSP fusion",
        "protocol": {
            "selection": "exploratory; all reported variants retained",
            "component_p_values": "empirical upper-tail ranks on the existing calibration split",
            "fusion": "combined scores are recalibrated on the same calibration split; no independence guarantee claimed",
            "variants": {
                "base": "production maxavg plus orthogonal-weight CNN",
                "plain": "average-pooled bottleneck CNN at d=32 and 90 epochs",
                "plain_multiscale": "plain CNN using pooled first- and second-convolution activations",
                "plain30": "average-pooled bottleneck CNN at d=32 and 30 epochs",
                "middle30": "first-convolution readout at d=32 and 30 epochs",
                "multiscale30": "first- and second-convolution readout at d=32 and 30 epochs",
                "actcorr": "base plus off-diagonal activation-correlation penalty",
                "swa": "base plus arithmetic mean of encoder weights from epochs 60 through 90; SWA-inspired, not a canonical SWA schedule",
                "multiscale": "pooled first- and second-convolution activations projected without test-label layer selection",
            },
        },
        "datasets": {},
    }
    names = [item.strip() for item in args.datasets.split(",") if item.strip()]
    for name in names:
        result["datasets"][name] = run_dataset(name, variants, seeds)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(clean_json(result), indent=2, sort_keys=True) + "\n")
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
