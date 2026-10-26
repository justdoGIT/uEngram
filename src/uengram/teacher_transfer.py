"""Contracts and validation for offline teacher-to-student transfer.

The module deliberately knows nothing about a particular foundation model. A
caller supplies an adapter around its weights and must declare the input
schema, feature layer, and causal alignment before features can be distilled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class WindowSchema:
    """Versioned sensor-window contract used by a teacher adapter."""

    channel_names: tuple[str, ...]
    sample_rate_hz: float
    window_size: int
    version: str = "1"

    def validate(self, windows: np.ndarray) -> None:
        values = np.asarray(windows)
        if values.ndim != 3:
            raise ValueError("windows must have shape (batch, channels, samples)")
        channels, samples = values.shape[1:]
        if channels != len(self.channel_names):
            raise ValueError(
                f"teacher schema expects {len(self.channel_names)} channels; got {channels}"
            )
        if samples != self.window_size:
            raise ValueError(
                f"teacher schema expects window_size={self.window_size}; got {samples}"
            )
        if self.sample_rate_hz <= 0 or not np.isfinite(self.sample_rate_hz):
            raise ValueError("sample_rate_hz must be a positive finite number")
        if not self.version:
            raise ValueError("window schema version must not be empty")


@dataclass(frozen=True)
class TeacherSpec:
    """Identity and safety contract for one teacher representation."""

    model_name: str
    model_version: str
    layer: str
    feature_dim: int
    input_schema: WindowSchema
    causal: bool = True
    alignment: str = "right"
    schema_version: str = "1"

    def validate(self) -> None:
        if not self.model_name or not self.model_version or not self.layer:
            raise ValueError("teacher name, version, and layer are required")
        if self.feature_dim <= 0:
            raise ValueError("teacher feature_dim must be positive")
        if not self.causal or self.alignment != "right":
            raise ValueError(
                "teacher features must be causal and right-aligned for online use"
            )
        if not self.schema_version:
            raise ValueError("teacher schema_version must not be empty")
        self.input_schema.validate(
            np.zeros(
                (1, len(self.input_schema.channel_names), self.input_schema.window_size),
                dtype=np.float32,
            )
        )


@dataclass(frozen=True)
class TeacherFeatures:
    """Validated feature matrix and the contract that produced it."""
    values: np.ndarray
    spec: TeacherSpec
    block_slices: tuple[tuple[int, int], ...] = ()
    block_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.spec.validate()
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.spec.feature_dim:
            raise ValueError(
                "teacher features must have shape (batch, spec.feature_dim)"
            )
        if not np.isfinite(values).all():
            raise ValueError("teacher features must be finite")
        if bool(self.block_slices) != bool(self.block_names):
            raise ValueError("teacher block names and slices must be both empty or set")
        if self.block_slices:
            if len(self.block_slices) != len(self.block_names):
                raise ValueError("teacher block names and slices must have equal length")
            if self.block_slices[0][0] != 0 or self.block_slices[-1][1] != values.shape[1]:
                raise ValueError("teacher blocks must cover the full feature width")
            if any(
                start < 0 or end <= start or start != previous
                for (start, end), previous in zip(
                    self.block_slices, (0, *[end for _, end in self.block_slices[:-1]])
                )
            ):
                raise ValueError("teacher blocks must be contiguous and non-empty")
        object.__setattr__(self, "values", values)


class TeacherAdapter(Protocol):
    """Minimal adapter around a loaded teacher model or exported weights."""

    def encode(self, windows: np.ndarray) -> np.ndarray | Any:
        """Return one feature row per causal input window."""


def _call_teacher(teacher: TeacherAdapter | Any, windows: np.ndarray) -> np.ndarray:
    result = teacher.encode(windows) if hasattr(teacher, "encode") else teacher(windows)
    if hasattr(result, "detach"):
        result = result.detach().cpu().numpy()
    return np.asarray(result, dtype=np.float64)


def extract_teacher_features(
    teacher: TeacherAdapter | Any,
    windows: np.ndarray,
    spec: TeacherSpec,
    batch: int = 1024,
) -> TeacherFeatures:
    """Run a declared teacher over windows without changing their order."""

    spec.validate()
    if batch <= 0:
        raise ValueError("teacher batch must be positive")
    values = np.asarray(windows, dtype=np.float32)
    spec.input_schema.validate(values)
    chunks = []
    for start in range(0, len(values), batch):
        chunk = _call_teacher(teacher, values[start : start + batch])
        if chunk.ndim != 2 or chunk.shape != (len(values[start : start + batch]), spec.feature_dim):
            raise ValueError(
                "teacher output shape must be (batch, spec.feature_dim)"
            )
        chunks.append(chunk)
    matrix = np.concatenate(chunks, axis=0) if chunks else np.empty((0, spec.feature_dim))
    return TeacherFeatures(matrix, spec)


def extract_multi_teacher_features(
    teachers: Sequence[tuple[TeacherAdapter | Any, TeacherSpec]],
    windows: np.ndarray,
    batch: int = 1024,
) -> TeacherFeatures:
    """Concatenate equal-contract teacher features for multi-teacher training."""

    if not teachers:
        raise ValueError("at least one teacher is required")
    first_spec = teachers[0][1]
    matrices = []
    slices = []
    names = []
    offset = 0
    for teacher, spec in teachers:
        if spec.input_schema != first_spec.input_schema:
            raise ValueError("all teachers must use the same input schema")
        features = extract_teacher_features(teacher, windows, spec, batch=batch)
        matrices.append(features.values)
        end = offset + spec.feature_dim
        slices.append((offset, end))
        names.append(f"{spec.model_name}:{spec.layer}")
        offset = end
    combined = TeacherSpec(
        model_name="multi-teacher",
        model_version="|".join(spec.model_version for _, spec in teachers),
        layer="concat(" + ",".join(spec.layer for _, spec in teachers) + ")",
        feature_dim=offset,
        input_schema=first_spec.input_schema,
        causal=all(spec.causal for _, spec in teachers),
        alignment="right",
    )
    return TeacherFeatures(
        np.concatenate(matrices, axis=1),
        combined,
        block_slices=tuple(slices),
        block_names=tuple(names),
    )
