"""Exploratory CNN3 normalization runs on the previously consumed MetroPT3 stream.

Artifacts are deliberately separate from v2: ``{Z, sigma0, metadata}`` in
``.collab/v3`` and trusted local CPU-portable pickle checkpoints.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

from bench.encoder_gate import accept_gate, gate_stats
from bench.encoding_cache import input_hash
from bench.streams import load_stream
from uengram.encoder_cnn3 import CNN3Encoder


def _fingerprint(array):
    a = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.sha256(a.tobytes()).hexdigest()


def _rank_stats(z):
    centered = np.asarray(z, dtype=np.float64) - np.mean(z, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = singular * singular
    total = float(variance.sum())
    if total <= 0:
        return {"raw_latent_effective_rank": 0.0, "raw_latent_first_pc_share": 0.0}
    p = variance / total
    return {"raw_latent_effective_rank": float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0])))),
            "raw_latent_first_pc_share": float(p[0])}


def train_one(stream, mode, seed, device="auto", output_dir=".collab/v3"):
    """Fit one complete CNN3 model; return ``(encoder, Z, record)``.

    Stream arrays are selected using its commissioning and calibration masks.
    Only COMP/DV_eletric/MPG are provided as conditions, never LPS.
    """
    import torch
    cond = None if stream.cond is None else np.asarray(stream.cond[:, :3], dtype=np.float32)
    if mode == "state" and (cond is None or cond.shape[1] < 3):
        raise ValueError("state mode requires COMP, DV_eletric, MPG conditions")
    fit_kw = {} if cond is None else {"cond": cond[stream.fit]}
    enc = CNN3Encoder(seed=int(seed), mode=mode, device=device).fit(
        stream.W[stream.fit], 1, stream.segment[stream.fit], **fit_kw)
    fit_cond = None if cond is None else cond[stream.fit]
    cal_cond = None if cond is None else cond[stream.cal]
    stats = gate_stats(enc, stream.W[stream.fit], stream.W[stream.cal], stream.segment[stream.fit],
                       fit_cond, cal_cond)
    raw = enc._latent(stream.W[stream.fit], cond=fit_cond) if fit_cond is not None else enc._latent(stream.W[stream.fit])
    record = {
        "stream": stream.name, "dataset_role": "development/exploratory; previously consumed",
        "encoder": "cnn3", "mode": mode, "seed": int(seed),
        "input_hash": input_hash(stream, f"cnn3_{mode}"),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "training": {"epochs_max": enc.epochs, "members": enc.members, "patience": enc.patience,
                     "batch": enc.batch, "latent_dim": enc.d, "noise_lag": 1},
        "condition_channels": ["COMP", "DV_eletric", "MPG"] if cond is not None else [],
        "condition_fingerprint": None if cond is None else _fingerprint(cond),
        "fit_windows": int(np.sum(stream.fit)), "cal_windows": int(np.sum(stream.cal)),
        "fit_seconds": float(enc.fit_seconds), "member_training_seconds": enc.member_training_seconds.tolist(),
        "backend": str(enc.device), "medoid": int(enc.medoid),
        "normalization": enc.normalization_diagnostics,
        **stats, **_rank_stats(raw),
        "accepted_by_old_gate": bool(accept_gate(stats)),
    }
    all_cond = None if cond is None else cond
    Z = enc.encode(stream.W, cond=all_cond) if all_cond is not None else enc.encode(stream.W)
    stem = Path(output_dir) / f"{stream.name}_cnn3_{mode}_{int(seed)}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(stem) + ".npz", Z=Z, sigma0=enc.sigma0,
                        metadata=json.dumps(record, sort_keys=True))
    saved = copy.deepcopy(enc)
    saved.enc = saved.enc.to("cpu")
    saved.device = torch.device("cpu")
    with open(str(stem) + ".pkl", "wb") as f:
        pickle.dump(saved, f, protocol=pickle.HIGHEST_PROTOCOL)
    return enc, Z, record


def run(seeds=range(5), modes=("range", "state"), device="auto", output_dir=".collab/v3"):
    if any(mode not in ("range", "state") for mode in modes):
        raise ValueError("modes must contain only range and state")
    if any(seed not in range(5) for seed in seeds):
        raise ValueError("MetroPT3 development runs use seeds 0 through 4")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    stream = load_stream("metropt3")
    records = []
    for mode in modes:
        for seed in seeds:
            try:
                _, _, record = train_one(stream, mode, seed, device=device, output_dir=output_dir)
            except Exception as error:
                record = {"stream": stream.name, "dataset_role": "development/exploratory; previously consumed",
                          "encoder": "cnn3", "mode": mode, "seed": int(seed),
                          "accepted_by_old_gate": False,
                          "failure": f"{type(error).__name__}: {error}"}
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            path = Path(output_dir) / f"{stream.name}_cnn3_{mode}_gate.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps([r for r in records if r["mode"] == mode], indent=2))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--modes", nargs="+", choices=("range", "state"), default=["range", "state"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=".collab/v3")
    args = parser.parse_args()
    run(args.seeds, args.modes, args.device, args.output_dir)


if __name__ == "__main__":
    main()
