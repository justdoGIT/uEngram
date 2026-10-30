"""The real EdgeMemory Lite shared library behind the sensor adapter's memory port.

The vendor binding (``edge_memory_gateway``) and ``cffi`` are not project dependencies. Provide
them for a run with, for example::

    PYTHONPATH=<vendor>/bindings/python/src EDGE_MEMORY_LIBRARY=<vendor>/lib/x64/libedge_memory_extremedb.so \
        uv run --with cffi python -m bench.capsule_resilience --vendor real ...

``recall`` does not expose the idempotency key, so records are identified by their content,
which carries the event kind, step and item and is unique per lifecycle event.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

KEYWORD = "uengram_sensor"
MAX_RECORDS = 4096


def available() -> bool:
    """True when the vendor binding, cffi and the shared library can all be loaded."""
    if not os.environ.get("EDGE_MEMORY_LIBRARY"):
        return False
    try:
        import cffi  # noqa: F401
        import edge_memory_gateway  # noqa: F401
    except ImportError:
        return False
    return True


class RealLiteMemory:
    """Persistent EdgeMemory Lite store; ``reopen`` simulates a process restart of the vendor."""

    def __init__(self, storage_path: str | os.PathLike[str], scope_id: str = "uengram-bench") -> None:
        from edge_memory_gateway import EdgeMemory

        self._factory = lambda: EdgeMemory(
            library_path=os.environ["EDGE_MEMORY_LIBRARY"],
            storage_path=Path(storage_path), scope_id=scope_id)
        self._pause_s = float(os.environ.get("UENGRAM_VENDOR_POST_COMMIT_MS", "0")) / 1000
        self.memory = self._factory()

    def remember(self, content: str, **kwargs: object) -> object:
        result = self.memory.remember(content, **kwargs)
        if self._pause_s:  # fault injection: widen "vendor committed, checkpoint still pending"
            time.sleep(self._pause_s)
        return result

    def recall(self, **kwargs: object) -> object:
        return self.memory.recall(**kwargs)

    def forget(self, memory_id: int) -> int:
        return self.memory.forget(memory_id)

    def reopen(self) -> RealLiteMemory:
        self.memory.close()
        self.memory = self._factory()
        return self

    def close(self) -> None:
        self.memory.close()

    def records(self) -> dict[str, str]:
        """Every stored lifecycle record, keyed (and valued) by its content."""
        response = self.memory.recall(keywords=(KEYWORD,), max_results=MAX_RECORDS)
        if len(response.items) >= MAX_RECORDS:
            raise RuntimeError("record enumeration truncated; raise MAX_RECORDS")
        return {item.content: item.content for item in response.items}
