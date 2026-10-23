"""Policy-independent encodings, reusable across detectors and GPU/CPU training.

Imported archives must carry the same input fingerprint. They contain only
embeddings from an encoder fitted on commissioning data, never fault labels.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from types import SimpleNamespace

import numpy as np


def input_hash(stream, name):
    h = hashlib.sha256()
    for array in (np.ascontiguousarray(stream.W, dtype=np.float32), stream.fit,
                  stream.cal, stream.segment, stream.end.astype(np.int64)):
        h.update(np.ascontiguousarray(array))
    if name == "cnn2s":
        h.update(np.ascontiguousarray(stream.cond, dtype=np.float32))
    return h.hexdigest()


def cached_encoding(stream, name, seed, directory):
    from bench.v2_scoring import encode
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{stream.name}_{name}_{seed}.npz")
    fingerprint = input_hash(stream, name)
    if os.path.exists(path):
        with np.load(path) as z:
            meta = json.loads(str(z["metadata"]))
            if (meta.get("stream"), meta.get("encoder"), meta.get("seed")) != (stream.name, name, seed):
                raise ValueError(f"misidentified encoding cache: {path}")
            if name.startswith("cnn2") and not meta.get("accepted", False):
                raise ValueError(f"encoder gate was not passed: {path}")
            if meta["input_hash"] != fingerprint:
                raise ValueError(f"stale encoding cache: {path}; choose a new scores directory")
            return SimpleNamespace(sigma0=z["sigma0"], encoding_meta=meta, encoding_path=path), z["Z"]
    started = time.monotonic()
    enc, Z = encode(stream, name, seed)
    meta = {"stream": stream.name, "encoder": name, "seed": seed, "input_hash": fingerprint,
            "backend": str(getattr(enc, "device", "cpu")),
            "fit_and_encode_seconds": time.monotonic() - started}
    if name.startswith("cnn2"):
        from bench.encoder_gate import accept_gate, gate_stats
        kw = dict(cond_fit=stream.cond[stream.fit], cond_cal=stream.cond[stream.cal]) if name == "cnn2s" else {}
        stats = gate_stats(enc, stream.W[stream.fit], stream.W[stream.cal], stream.segment[stream.fit], **kw)
        meta.update(stats, accepted=accept_gate(stats))
        if not meta["accepted"]:
            raise ValueError(f"encoder rejected by commissioning/calibration gate: {json.dumps(stats)}")
    temp = path + f".{os.getpid()}.npz"
    np.savez_compressed(temp, Z=Z, sigma0=enc.sigma0, metadata=json.dumps(meta))
    os.replace(temp, path)
    return SimpleNamespace(sigma0=enc.sigma0, encoding_meta=meta, encoding_path=path), Z
