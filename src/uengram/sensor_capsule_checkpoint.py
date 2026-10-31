"""Versioned, pickle-free checkpoint encoding for a uEngram engine."""
from __future__ import annotations

import io
import json
from dataclasses import asdict

import numpy as np

from .confirmed_adaptation import AdapterSnapshot, ConfirmedLatentAdapter
from .engine import Engram

_SCHEMA = 1
_ARRAYS = (
    "mu", "S", "ckpt_mu", "n", "kind", "t_created", "t_last", "surpr",
    "item_id", "superseded_by", "mode_key", "ewma_e", "ref_mu", "ref_var", "hits",
)


def dump_checkpoint(engine: Engram, adapter: ConfirmedLatentAdapter | None) -> bytes:
    """Return a self-describing NumPy archive with no executable payload."""
    metadata = {
        "schema": _SCHEMA,
        "config": asdict(engine.cfg),
        "d": engine.d,
        "name": engine.name,
        "raw_window_bytes": engine.raw_window_bytes,
        "next_id": engine._next_id,
        "exemplars": engine.exemplars,
        "lineage": _jsonable(list(engine.lineage)),
        "events": _jsonable(engine.events),
        "protected": engine._protect_count,
        "ledger": {
            "flush_every": engine.ledger.flush_every,
            "writes": engine.ledger.writes,
            "bytes_written": engine.ledger.bytes_written,
            "dirty": engine.ledger._dirty,
        },
        "adapter": _adapter_data(adapter),
    }
    arrays = {name: getattr(engine, name) for name in _ARRAYS}
    arrays["metadata"] = np.frombuffer(json.dumps(metadata, separators=(",", ":")).encode(), dtype=np.uint8)
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def restore_checkpoint(
    engine: Engram, adapter: ConfirmedLatentAdapter | None, payload: bytes
) -> None:
    """Restore a checkpoint only into an engine with the same immutable contract."""
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        metadata = json.loads(bytes(archive["metadata"]).decode())
        if metadata.get("schema") != _SCHEMA or metadata.get("config") != asdict(engine.cfg):
            raise ValueError("checkpoint engine configuration does not match")
        if metadata.get("d") != engine.d or metadata.get("name") != engine.name:
            raise ValueError("checkpoint engine identity does not match")
        if metadata.get("raw_window_bytes") != engine.raw_window_bytes:
            raise ValueError("checkpoint raw-window size does not match")
        for name in _ARRAYS:
            saved, current = archive[name], getattr(engine, name)
            if saved.shape != current.shape or saved.dtype != current.dtype:
                raise ValueError(f"checkpoint {name} shape or dtype does not match")
            setattr(engine, name, saved.copy())
    engine._next_id = int(metadata["next_id"])
    engine.exemplars = {int(key): [int(value) for value in values]
                        for key, values in metadata["exemplars"].items()}
    engine.lineage.clear()
    engine.lineage.extend(metadata["lineage"])
    engine.events = [tuple(event) for event in metadata["events"]]
    engine._protect_count = {int(key): int(value) for key, value in metadata["protected"].items()}
    ledger = metadata["ledger"]
    engine.ledger.flush_every = int(ledger["flush_every"])
    engine.ledger.writes = int(ledger["writes"])
    engine.ledger.bytes_written = int(ledger["bytes_written"])
    engine.ledger._dirty = {int(key): int(value) for key, value in ledger["dirty"].items()}
    _restore_adapter(adapter, metadata["adapter"])


def _adapter_data(adapter: ConfirmedLatentAdapter | None) -> dict | None:
    if adapter is None:
        return None
    snapshot = adapter.snapshot()
    return {
        "offsets": {str(key): value.tolist() for key, value in snapshot.offsets.items()},
        "quarantine": {str(key): list(rows) for key, rows in snapshot.quarantine.items()},
    }


def _restore_adapter(adapter: ConfirmedLatentAdapter | None, data: dict | None) -> None:
    if data is None:
        if adapter is not None:
            adapter.rollback(AdapterSnapshot({}, {}))
        return
    if adapter is None:
        raise ValueError("checkpoint requires a confirmed latent adapter")
    adapter.rollback(AdapterSnapshot(
        offsets={int(key): np.asarray(value, dtype=np.float64) for key, value in data["offsets"].items()},
        quarantine={int(key): tuple(tuple(float(v) for v in row) for row in rows)
                    for key, rows in data["quarantine"].items()},
    ))


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
