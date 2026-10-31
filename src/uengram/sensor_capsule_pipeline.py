"""uEngram runner that makes SensorMemoryCapsule recovery automatic."""
from __future__ import annotations

from .confirmed_adaptation import ConfirmedLatentAdapter
from .engine import Engram
from .sensor_memory import SensorStateMemory
from .sensor_memory_capsule import SensorMemoryCapsule
from .sensor_memory_types import (
    EdgeMemoryPort,
    RawEvidencePins,
    SensorIngestReceipt,
    SensorWindow,
)


class CapsuleSensorStateMemory:
    """Record input, use uEngram, then checkpoint; replay only a WAL suffix."""

    def __init__(
        self,
        capsule: SensorMemoryCapsule,
        engine: Engram,
        memory: EdgeMemoryPort,
        *,
        raw_evidence: RawEvidencePins | None = None,
        adapter: ConfirmedLatentAdapter | None = None,
        checkpoint_every: int = 1,
    ) -> None:
        if isinstance(checkpoint_every, bool) or not isinstance(checkpoint_every, int) or checkpoint_every < 1:
            raise ValueError("checkpoint_every must be an integer >= 1")
        self.capsule = capsule
        self.checkpoint_every = checkpoint_every
        self._since_checkpoint = 0
        restored = capsule.restore(engine, adapter)
        self.pipeline = SensorStateMemory(
            engine,
            memory,
            raw_evidence=raw_evidence,
            outbox=capsule,
            vector_index=capsule,
            last_step=-1 if restored is None else restored.last_step,
            last_alarm=None if restored is None else restored.last_alarm,
            adapter=adapter,
        )

    def ingest(self, window: SensorWindow) -> SensorIngestReceipt:
        self.capsule.record_window(window)
        receipt = self.pipeline.ingest(window)
        self._since_checkpoint += 1
        if self._since_checkpoint >= self.checkpoint_every:
            self.capsule.checkpoint(
                self.pipeline.engine,
                last_step=receipt.step_index,
                last_alarm=receipt.decision.flagged,
                adapter=self.pipeline.adapter,
            )
            self._since_checkpoint = 0
        return receipt

    def recover(self) -> tuple[SensorIngestReceipt, ...]:
        """Replay only recorded windows after the latest valid checkpoint."""
        return tuple(self.ingest(window) for window in self.capsule.uncheckpointed_windows())
