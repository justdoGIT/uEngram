"""Archive observed input windows around alarm openings, not predicted futures.

The detector never sees post-opening context. This offline recorder runs after
streaming has completed; missing post-context at stream end remains -1. Context
never crosses a logging/operating-status segment boundary. Archives are outside
the model's fixed memory budget; their measured bytes are reported separately.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np


def archive_frames(stream, episodes, path, run, reference=None, latent=None, radius=2):
    test = np.flatnonzero(stream.test)
    frames = np.full((len(episodes), 2 * radius + 1), -1, np.int64)
    manifest = []
    for row, ep in enumerate(episodes):
        center = int(test[ep["open_index"]])
        for col, offset in enumerate(range(-radius, radius + 1)):
            j = center + offset
            if 0 <= j < len(stream.W) and stream.segment[j] == stream.segment[center]:
                frames[row, col] = j
        rec = dict(ep)
        rec["input_index"] = center
        rec["context_input_indices"] = frames[row].tolist()
        observed = frames[row][frames[row] >= 0]
        rec["context_complete_at"] = str(stream.end[observed[-1]])
        if reference is not None and latent is not None and len(reference[0]):
            mu, var = reference
            distances = np.sum((latent[center] - mu) ** 2 / var, axis=1)
            rec["nearest_reference_index"] = int(np.argmin(distances))
            rec["reference_distance_squared"] = float(distances.min())
        manifest.append(rec)
    keep = np.unique(frames[frames >= 0])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = dict(input_index=keep, windows=stream.W[keep],
                  start=stream.start[keep].astype(np.int64), end=stream.end[keep].astype(np.int64),
                  features=stream.features[keep], feature_names=np.asarray(stream.feature_names),
                  state=stream.state[keep], context_indices=frames,
                  manifest=json.dumps(manifest), run=json.dumps(run))
    if stream.cond is not None:
        arrays["condition_windows"] = stream.cond[keep]
    if reference is not None:
        arrays["reference_mean"], arrays["reference_variance"] = reference
    np.savez_compressed(path, **arrays)
    with open(path, "rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    return {"path": path, "sha256": digest, "bytes": os.path.getsize(path),
            "episodes": len(episodes), "unique_input_windows": len(keep),
            "incomplete_context_episodes": int(np.any(frames < 0, axis=1).sum())}
