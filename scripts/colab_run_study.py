#!/usr/bin/env python3
"""Run the GPU-accelerated EngineAD CNN seed study on Colab (needs colab_study_lib.py alongside)."""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from colab_study_lib import (
    BATCH, CAL_ALPHA, DIMENSIONS, EPOCHS, NOISE_LAG, SEEDS, SPLIT_SALT,
    _sha256, _split, auroc, average_precision, conformal_threshold,
    fit_projection, load_segments, normal_training_groups,
)

# GPU CNN Encoder
class CNNEncoderGPU:
    def __init__(self, seed: int = 0, d: int = 16, epochs: int = 30, batch: int = 512,
                 pool: str = "maxavg", ortho_beta: float = 0.05, device: str = "cuda"):
        self.seed, self.d, self.epochs, self.batch = seed, d, epochs, batch
        self.pool, self.ortho_beta = pool, ortho_beta
        self.device = torch.device(device)

    def _build(self, C: int, K: int):
        self.conv = nn.Sequential(
            nn.Conv1d(C, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GELU(),
        ).to(self.device)
        in_dim = 128 if self.pool == "maxavg" else 64
        self.fc = nn.Linear(in_dim, self.d, bias=self.ortho_beta == 0.0).to(self.device)
        dec = nn.Sequential(nn.Linear(self.d, 128), nn.GELU(), nn.Linear(128, C * K)).to(self.device)
        return dec

    def _pool(self, h):
        if self.pool == "maxavg":
            return torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)
        return h.mean(dim=-1)

    def _forward_enc(self, xb):
        return self.fc(self._pool(self.conv(xb)))

    def _std(self, windows):
        return (np.asarray(windows, dtype=np.float32) - self.ch_mean) / self.ch_std

    def fit(self, train_windows, noise_lag: int, groups: np.ndarray | None):
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        x = np.asarray(train_windows, dtype=np.float32)
        N, C, K = x.shape
        self.ch_mean = x.mean(axis=(0, 2), keepdims=True)[0].astype(np.float32)
        self.ch_std = np.maximum(x.std(axis=(0, 2), keepdims=True)[0], 1e-6).astype(np.float32)

        xt_cpu = torch.from_numpy(self._std(x))
        dec = self._build(C, K)
        params = list(self.conv.parameters()) + list(self.fc.parameters()) + list(dec.parameters())
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        self.conv.train(); self.fc.train(); dec.train()
        I = torch.eye(self.d, device=self.device) if self.ortho_beta > 0.0 else None

        for _ in range(self.epochs):
            perm = torch.randperm(N, generator=gen)
            for i in range(0, N, self.batch):
                batch_indices = perm[i : i + self.batch]
                xb = xt_cpu[batch_indices].to(self.device, non_blocking=True)
                z = self._forward_enc(xb)
                recon = torch.mean((dec(z).reshape(-1, C, K) - xb) ** 2)
                loss = recon
                if self.ortho_beta > 0.0:
                    W = self.fc.weight
                    loss = loss + self.ortho_beta * torch.mean((W @ W.T - I) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()

        self.conv.eval(); self.fc.eval(); dec.eval()
        del dec, opt, xt_cpu
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        self.whitener, self.noise, self.sigma0 = fit_projection(self._latent(x), self.d, noise_lag, groups)
        return self

    def _latent(self, windows) -> np.ndarray:
        x_std = self._std(windows)
        out = []
        with torch.no_grad():
            for i in range(0, len(x_std), 2048):
                xb = torch.from_numpy(x_std[i : i + 2048]).to(self.device, non_blocking=True)
                out.append(self._forward_enc(xb).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float64)

    def encode(self, windows, sequential: bool = True) -> np.ndarray:
        return self.noise.transform(self.whitener.transform(self._latent(windows)))


def _binary_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    predicted = scores > threshold
    positive = labels.astype(bool)
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & ~positive))
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / n_pos if n_pos else math.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else math.nan
    return {
        "segments": int(len(labels)),
        "anomaly_segments": n_pos,
        "normal_segments": n_neg,
        "auroc": auroc(scores, positive),
        "average_precision": average_precision(scores, positive),
        "calibrated_threshold": float(threshold),
        "precision_at_threshold": float(precision),
        "recall_at_threshold": float(recall),
        "f1_at_threshold": float(f1),
        "segment_fpr_at_threshold": fp / n_neg if n_neg else math.nan,
    }


def _metric_range(values: list[float]) -> list[float]:
    finite = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if not len(finite):
        return [math.nan, math.nan, math.nan]
    return [float(np.median(finite)), float(np.min(finite)), float(np.max(finite))]


def _macro_metric(per_asset: dict, rule: str, metric: str) -> float:
    values = [asset[rule][metric] for asset in per_asset.values() if math.isfinite(asset[rule][metric])]
    return float(np.mean(values)) if values else math.nan


def _seed_range(rows: list[dict], group: str, rule: str, metric: str) -> list[float]:
    return _metric_range([row[group][rule][metric] for row in rows])


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    text = json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False)
    temporary.write_text(text + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    data_dir = Path("/content/enginead/2.2")
    out_path = Path("/content/cnn_seed_stability.json")
    manifest_path = Path("/content/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["files"]
    splits = _split([r["name"] for r in records])
    print(f"Frozen split: train={len(splits['train'])} cal={len(splits['calibration'])} test={len(splits['test'])}")

    label_audit = []
    train_x: list[np.ndarray] = []
    train_groups: list[np.ndarray] = []
    train_summary = []
    group_id = 0
    for name in splits["train"]:
        print(f"Loading training asset {name}...")
        x, first, any_positive, clean, audit = load_segments(data_dir / name)
        indices, asset_groups, group_id = normal_training_groups(clean, group_id)
        train_x.append(x[indices])
        train_groups.append(asset_groups)
        train_summary.append({
            "file": name, "all_normal_segments_used": len(indices),
            "normal_run_count": int(np.unique(asset_groups).size),
        })
        label_audit.append(audit)
        del x, first, any_positive, clean, audit, indices, asset_groups

    x_train = np.concatenate(train_x, axis=0)
    groups = np.concatenate(train_groups)
    del train_x, train_groups
    print(f"Total training windows: {len(x_train)} shape={x_train.shape}")

    eval_sets = {"calibration": [], "test": []}
    for split_name in ("calibration", "test"):
        for name in splits[split_name]:
            print(f"Loading {split_name} asset {name}...")
            x, first, any_positive, clean, audit = load_segments(data_dir / name)
            eval_sets[split_name].append((name, x, first, any_positive, clean))
            label_audit.append(audit)
    label_audit.sort(key=lambda row: row["file"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")

    base = {
        "dataset": manifest["dataset"],
        "dataset_manifest_sha256": _sha256(manifest_path),
        "script_sha256": "colab-gpu-sweep-v1",
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "torch": torch.__version__, "device": device},
        "segment_label_audit": label_audit,
        "protocol": {
            "scope": "exploratory segment-level asset-held-out encoder study; not event-level validation",
            "split_salt": SPLIT_SALT,
            "access_note": "The iSMART Lab page states that EngineAD is public and no access request is required; all published version 2.2 files were unrestricted.",
            "split_files": splits,
            "label_convention": {
                "primary": "first point label in each segment, matching the upstream baseline.py convention",
                "sensitivity": "positive if any of the 300 point labels is anomalous",
                "training_and_calibration": "only segments whose 300 point labels are all normal",
                "source": "https://github.com/Armanfard-Lab/EngineAD/blob/main/baseline.py",
            },
            "training_rule": "all-point-normal segments only; projection groups break at any mixed/anomalous segment and vehicle boundary",
            "training_assets": train_summary,
            "dimensions": list(DIMENSIONS), "seeds": list(SEEDS), "epochs": EPOCHS,
            "batch_size": BATCH, "pool": "maxavg", "ortho_beta": 0.05,
            "noise_lag": NOISE_LAG,
            "score": "squared L2 norm of the projected, noise-whitened CNN embedding",
            "primary_metric": "segment average precision using first-row labels; any-positive sensitivity also reported",
            "threshold": "95% order statistic calibrated only on all-point-normal calibration segments; empirical, no exchangeability guarantee claimed",
            "caveats": [
                "Some 300-point segments contain mixed normal/anomaly labels; alternate any-positive metrics expose this aggregation sensitivity.",
                "Labels are segment/point labels, not event onset/offset intervals; false alarms per week and event recall are not estimable.",
                "The published eight-component PCA transform was fit at fleet level and includes held-out vehicles, so raw-sensor representation is not independently refit.",
                "Segments within a truck are serially dependent; pooled calibration quantiles do not provide a distribution-free 5% false-alarm guarantee.",
                "Original timestamps and engine-on block identifiers are absent; lag differences may bridge unobserved real-world gaps between adjacent serialized segments.",
                "Only five test vehicles; report per-vehicle outcomes and seed ranges, not population-level certainty.",
                "This is a separate exploratory benchmark and does not satisfy the preregistered real-field event-level cohort gate.",
            ],
        },
        "results": [],
    }
    _write_json(out_path, base)

    total_fits = len(DIMENSIONS) * len(SEEDS)
    fit_idx = 0
    sweep_start = time.perf_counter()

    for d in DIMENSIONS:
        for seed in SEEDS:
            fit_idx += 1
            print(f"[{fit_idx}/{total_fits}] Fit start d={d} seed={seed} epochs={EPOCHS} batch={BATCH}...")
            t0 = time.perf_counter()
            encoder = CNNEncoderGPU(seed=seed, d=d, epochs=EPOCHS, batch=BATCH,
                                    pool="maxavg", ortho_beta=0.05, device=device)
            encoder.fit(x_train, NOISE_LAG, groups)
            fit_seconds = time.perf_counter() - t0

            cal_normal = []
            for _, x, _, _, clean in eval_sets["calibration"]:
                z = encoder.encode(x)
                scores = np.einsum("ij,ij->i", z, z)
                cal_normal.append(scores[clean])
            threshold = conformal_threshold(np.concatenate(cal_normal), CAL_ALPHA)

            per_asset = {}
            pooled_scores, pooled_first, pooled_any = [], [], []
            for name, x, first, any_positive, _ in eval_sets["test"]:
                z = encoder.encode(x)
                scores = np.einsum("ij,ij->i", z, z)
                per_asset[name] = {
                    "first_row_label": _binary_metrics(scores, first, threshold),
                    "any_positive_label": _binary_metrics(scores, any_positive, threshold),
                }
                pooled_scores.append(scores)
                pooled_first.append(first)
                pooled_any.append(any_positive)
            all_scores = np.concatenate(pooled_scores)
            pooled = {
                "first_row_label": _binary_metrics(all_scores, np.concatenate(pooled_first), threshold),
                "any_positive_label": _binary_metrics(all_scores, np.concatenate(pooled_any), threshold),
            }
            macro = {
                rule: {metric: _macro_metric(per_asset, rule, metric) for metric in ("auroc", "average_precision")}
                for rule in ("first_row_label", "any_positive_label")
            }
            row = {
                "d": d, "seed": seed, "fit_seconds": fit_seconds,
                "calibration_all_normal_segments": int(sum(len(s) for s in cal_normal)),
                "calibration_threshold": float(threshold),
                "pooled_test": pooled, "macro_test": macro, "per_test_asset": per_asset,
            }
            base["results"].append(row)
            _write_json(out_path, base)
            del encoder

            f_ap = pooled["first_row_label"]["average_precision"]
            f_auc = pooled["first_row_label"]["auroc"]
            elapsed = time.perf_counter() - sweep_start
            eta = (elapsed / fit_idx) * (total_fits - fit_idx)
            print(f"[{fit_idx}/{total_fits}] Done d={d} seed={seed} in {fit_seconds:.1f}s | AUROC={f_auc:.4f} AP={f_ap:.4f} | ETA: {eta/60:.1f}m")

    # Final summary across seeds
    summaries = []
    for d in DIMENSIONS:
        rows = [r for r in base["results"] if r["d"] == d]
        summaries.append({
            "d": d, "seed_count": len(rows),
            "first_row_pooled_auroc": _seed_range(rows, "pooled_test", "first_row_label", "auroc"),
            "first_row_pooled_ap": _seed_range(rows, "pooled_test", "first_row_label", "average_precision"),
            "first_row_macro_auroc": _seed_range(rows, "macro_test", "first_row_label", "auroc"),
            "first_row_macro_ap": _seed_range(rows, "macro_test", "first_row_label", "average_precision"),
            "first_row_segment_fpr": _seed_range(rows, "pooled_test", "first_row_label", "segment_fpr_at_threshold"),
            "any_positive_pooled_auroc": _seed_range(rows, "pooled_test", "any_positive_label", "auroc"),
            "any_positive_pooled_ap": _seed_range(rows, "pooled_test", "any_positive_label", "average_precision"),
            "any_positive_macro_auroc": _seed_range(rows, "macro_test", "any_positive_label", "auroc"),
            "any_positive_macro_ap": _seed_range(rows, "macro_test", "any_positive_label", "average_precision"),
            "any_positive_segment_fpr": _seed_range(rows, "pooled_test", "any_positive_label", "segment_fpr_at_threshold"),
        })
    base["seed_summaries"] = summaries
    base["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    base["total_sweep_seconds"] = time.perf_counter() - sweep_start
    _write_json(out_path, base)
    print(f"FULL STUDY COMPLETE in {base['total_sweep_seconds']:.1f}s ({base['total_sweep_seconds']/60:.1f}m)!")
    return 0


if __name__ == "__main__":
    sys.exit(main())