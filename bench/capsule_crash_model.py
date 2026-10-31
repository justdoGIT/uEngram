"""Model a power cut from a recorded I/O trace, in the ALICE/CrashMonkey tradition.

The companion shim (`bench/crash_shim.c`, built as `libcrashshim.so`) intercepts every write,
fsync/fdatasync, truncate, create and unlink a process performs under one directory and appends
each to a binary log. This module replays that log to build the on-disk images a crash could
leave at any point, then lets the caller point real SQLite at that image to see whether the
capsule's own recovery actually copes with it.

Model (explicit, so its limits are visible):

- Per file, the state as of a log index splits into a **durable** base -- the content guaranteed
  to survive because it preceded that file's most recent `fsync`/`fdatasync`, or, for a file's
  existence, its most recent ancestor-directory `fsync` -- and a **pending** list of writes,
  truncates, creates and unlinks issued since. A crash image takes durable plus some *prefix* of
  pending, applied in issue order.
- Prefix, not an arbitrary subset: the model assumes a file's own writes are not reordered past
  each other by the time they would reach stable storage. This is weaker than full atomicity and
  does not model out-of-order completion below the page cache; it is a standard, tractable crash
  model, not a hardware fault injector.
- Files are independent: a crash image is a prefix choice per file, combined. When the full cross
  product is large it is sampled, always including "nothing pending applied" and "everything
  pending applied" for every file.
- fsync's own guarantee is trusted: once a byte is durable it is never un-done. This tests the
  *window* between syncs (why `synchronous` and `checkpoint_every` matter), not whether the
  kernel's fsync primitive itself is honest.
"""
from __future__ import annotations

import random
import struct
from dataclasses import dataclass, field
from pathlib import Path

OP_WRITE, OP_SYNC, OP_TRUNC, OP_CREATE, OP_UNLINK, OP_DIRSYNC, OP_MARK = range(1, 8)
_HEADER = struct.Struct("<BIQQI")


@dataclass(frozen=True)
class Op:
    kind: int
    path: str
    a: int
    b: int
    data: bytes


def read_log(path: Path) -> list[Op]:
    raw, ops, i = Path(path).read_bytes(), [], 0
    while i < len(raw):
        kind, plen = raw[i], int.from_bytes(raw[i + 1:i + 5], "little")
        i += 5
        p = raw[i:i + plen].decode()
        i += plen
        a, b, dlen = struct.unpack_from("<QQI", raw, i)
        i += 20
        d = raw[i:i + dlen]
        i += dlen
        ops.append(Op(kind, p, a, b, d))
    return ops


@dataclass
class _FileState:
    exists: bool = False
    data: bytearray = field(default_factory=bytearray)
    pending: list[Op] = field(default_factory=list)


def _apply(state: _FileState, op: Op) -> None:
    if op.kind == OP_CREATE:
        state.exists = True
    elif op.kind == OP_UNLINK:
        state.exists, state.data = False, bytearray()
    elif op.kind == OP_WRITE:
        end = op.a + len(op.data)
        if end > len(state.data):
            state.data.extend(b"\0" * (end - len(state.data)))
        state.data[op.a:end] = op.data
    elif op.kind == OP_TRUNC:
        if op.a < len(state.data):
            del state.data[op.a:]
        else:
            state.data.extend(b"\0" * (op.a - len(state.data)))


def _commit_directory_entries(state: _FileState) -> None:
    """A directory fsync makes the file's latest create/unlink durable, not its unsynced data."""
    last = max((i for i, op in enumerate(state.pending) if op.kind in (OP_CREATE, OP_UNLINK)), default=None)
    if last is None:
        return
    kind = state.pending[last].kind
    state.exists = kind == OP_CREATE
    state.data = bytearray()  # a created file starts empty; an unlinked file's content is gone
    del state.pending[:last + 1]


def durable_state(ops: list[Op], index: int) -> dict[str, _FileState]:
    """Per-file durable base and pending list as of ``ops[:index]``.

    An ``fsync`` of a file credits every pending operation on it, including its own create (ext4's
    practical behaviour; POSIX itself only guarantees the data). Other directory-entry changes need
    a directory fsync.
    """
    files: dict[str, _FileState] = {}
    for op in ops[:index]:
        if op.kind == OP_MARK:
            continue
        if op.kind == OP_DIRSYNC:
            for state in files.values():
                _commit_directory_entries(state)
            continue
        state = files.setdefault(op.path, _FileState())
        state.pending.append(op)
        if op.kind == OP_SYNC:
            pending, state.pending = state.pending, []
            for pend in pending:
                if pend.kind != OP_SYNC:
                    _apply(state, pend)
    return files


def acknowledged_count(ops: list[Op], index: int) -> int:
    """Number of ingests the recorded process had finished (MARK records) before ``ops[:index]``."""
    return sum(1 for op in ops[:index] if op.kind == OP_MARK)


def crash_images(ops: list[Op], index: int, *, max_combos: int = 64,
                 rng: random.Random | None = None) -> list[dict[str, tuple[bool, bytes]]]:
    """Candidate on-disk images a crash at ``index`` could leave, each ``{path: (exists, bytes)}``."""
    files = durable_state(ops, index)
    paths = [p for p, s in files.items() if s.pending]
    option_counts = [len(files[p].pending) + 1 for p in paths]
    total = 1
    for count in option_counts:
        total *= count
        if total > max_combos:
            break
    rng = rng or random.Random(0)

    def render(prefixes: dict[str, int]) -> dict[str, tuple[bool, bytes]]:
        image = {}
        for path, state in files.items():
            exists, data = state.exists, bytearray(state.data)
            scratch = _FileState(exists, data)
            for op in state.pending[:prefixes.get(path, 0)]:
                _apply(scratch, op)
            image[path] = (scratch.exists, bytes(scratch.data))
        return image

    if total <= max_combos:
        combos, choice = [], [0] * len(paths)

        def recurse(i: int) -> None:
            if i == len(paths):
                combos.append(dict(zip(paths, choice)))
                return
            for value in range(option_counts[i]):
                choice[i] = value
                recurse(i + 1)
        recurse(0)
        return [render(c) for c in combos]

    extremes = [{p: 0 for p in paths}, {p: len(files[p].pending) for p in paths}]
    samples = [{p: rng.randint(0, len(files[p].pending)) for p in paths}
               for _ in range(max_combos - len(extremes))]
    return [render(c) for c in extremes + samples]


def materialize(image: dict[str, tuple[bool, bytes]], real_root: Path, crash_root: Path) -> None:
    """Write a candidate image into real files under ``crash_root``, mirroring ``real_root``'s layout."""
    crash_root.mkdir(parents=True, exist_ok=True)
    for path, (exists, data) in image.items():
        target = crash_root / Path(path).relative_to(real_root)
        if not exists:
            target.unlink(missing_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
