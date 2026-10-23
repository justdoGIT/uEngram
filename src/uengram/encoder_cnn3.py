"""CNN2 training with explicit healthy-range normalization ablations."""
from __future__ import annotations

import numpy as np

from .encoder_cnn2 import CNN2Encoder


class CNN3Encoder(CNN2Encoder):
    """CNN2 architecture with range- or observed-state-conditioned scales.

    ``mode='range'`` scales each analog channel by the larger of its robust
    healthy scale and full healthy operating-range scale. ``mode='state'``
    uses per-DV-eletric-state robust scales, floored by that same full range.
    A single global center is retained so absolute level changes remain visible.
    """

    def __init__(self, *args, mode="range", **kwargs):
        if mode not in ("range", "state"):
            raise ValueError("mode must be 'range' or 'state'")
        super().__init__(*args, **kwargs)
        self.mode = mode

    def _fit_normalizer(self, x, split):
        analog = x[:split, :self.analog_channels]
        self.center = np.median(analog, axis=(0, 2), keepdims=True).astype(np.float32)
        q25, q75 = np.percentile(analog, [25, 75], axis=(0, 2), keepdims=True)
        robust = (q75 - q25) / 1.349
        full_range = ((np.max(analog, axis=(0, 2), keepdims=True)
                       - np.min(analog, axis=(0, 2), keepdims=True)) / 6)
        self.scale_floor = np.maximum(full_range, 1e-6).astype(np.float32)
        global_scale = np.maximum(robust, self.scale_floor)
        self.cond_center = (np.mean(x[:split, self.analog_channels:], axis=(0, 2), keepdims=True).astype(np.float32)
                            if x.shape[1] > self.analog_channels else np.zeros((1, 0, 1), np.float32))
        self.state_values = np.empty(0, dtype=np.float32)
        self.state_scales = None
        if self.mode == "state":
            if x.shape[1] < self.analog_channels + 2:
                raise ValueError("state normalization requires cond[:, 1] (DV_eletric)")
            dv = self._state_ids(x)
            states = np.unique(dv[:split])
            if not len(states):
                raise ValueError("state normalization has no observed fit-prefix states")
            self.state_values = states.astype(np.float32)
            scales = []
            for state in states:
                selected = analog[:split][dv[:split] == state]
                if not len(selected):
                    scales.append(global_scale[0, :, 0])
                    continue
                a, b = np.percentile(selected, [25, 75], axis=(0, 2))
                scales.append(np.maximum((b - a) / 1.349, self.scale_floor[0, :, 0]))
            self.state_scales = np.asarray(scales, dtype=np.float32)
        self.scale = global_scale.astype(np.float32)
        self.normalization_diagnostics = {
            "mode": self.mode,
            "fit_prefix_windows": int(split),
            "scale_floor": self.scale_floor.reshape(-1).astype(float).tolist(),
            "global_scale": self.scale.reshape(-1).astype(float).tolist(),
            "state_values": self.state_values.astype(float).tolist(),
            "state_scales": None if self.state_scales is None else self.state_scales.astype(float).tolist(),
            "unseen_state_handling": "global healthy-range scale; not a healthy-state assertion",
        }

    def _state_ids(self, x):
        # Conditioning order is the caller's [COMP, DV_eletric, MPG] contract.
        dv = x[:, self.analog_channels + 1]
        if not np.all(np.isfinite(dv)) or np.any((dv < 0) | (dv > 1)):
            raise ValueError("unsupported DV_eletric values outside [0, 1]")
        # Match the observed duty bins in bench.streams, including mixed cycles.
        return np.digitize(dv.mean(axis=1), (.05, .4)).astype(np.float32)

    def _scaled(self, x):
        analog = x[:, :self.analog_channels]
        if self.mode == "range":
            scale = self.scale
        else:
            ids = self._state_ids(x)
            lookup = {float(v): i for i, v in enumerate(self.state_values)}
            indices = np.asarray([lookup.get(float(v), -1) for v in ids])
            scale = np.broadcast_to(self.scale, analog.shape[:2] + (1,)).copy()
            supported = indices >= 0
            scale[supported] = self.state_scales[indices[supported], :, None]
        normalized = (analog - self.center) / scale
        if self.analog_channels < x.shape[1]:
            states = x[:, self.analog_channels:] - self.cond_center
            return np.concatenate((normalized, states), axis=1).astype(np.float32, copy=False)
        return normalized.astype(np.float32, copy=False)
