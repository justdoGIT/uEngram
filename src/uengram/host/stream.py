"""Push the node's state as it changes; stdlib only.

The daemon listens on a Unix socket (`<state dir>/stream.sock`, owner-only)
and writes one JSON line per message to every connected reader:

  {"type": "state", ...}     the latest window record and the memory's view
                             (node.NodeMemory.state), sent on connect and at
                             the end of every window
  {"type": "tick", ...}      the last second's CPU and stall counts, for
                             placement on current load (cluster.headroom)
  {"type": "pressure", ...}  a PSI trigger fired (psi.py), within ~2 s of the
                             stall, between window ends

`uengram-host follow` relays the socket to stdout, so a hub on another node
reads the stream through ssh: authenticated by the fleet's existing ssh keys,
no listening network port and no shared secret on the nodes.
A reader that cannot keep up (its socket buffer is full) is dropped; the
daemon never waits for a reader.
"""
from __future__ import annotations

import json
import os
import select
import socket
import sys

MAX_CLIENTS = 8


def atomic_json(path: str, obj, mode: int = 0o600) -> None:
    """Write JSON so readers see the old or the new file, never a torn one."""
    tmp = f"{path}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


class StateStream:
    def __init__(self, path: str):
        self.path = path
        if os.path.exists(path):  # left by a killed daemon (runit runs one daemon per node)
            os.unlink(path)
        old = os.umask(0o177)
        try:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.bind(path)
        finally:
            os.umask(old)
        self.sock.listen(MAX_CLIENTS)
        self.sock.setblocking(False)
        self.clients: dict[int, socket.socket] = {}
        self.last_state: bytes | None = None

    def register(self, poller: select.poll) -> None:
        self.poller = poller
        poller.register(self.sock.fileno(), select.POLLIN)

    def handle(self, fd: int, events: int) -> bool:
        """Poll event on one of our fds; False if the fd is not ours."""
        if fd == self.sock.fileno():
            try:
                c, _ = self.sock.accept()
            except BlockingIOError:
                return True
            if len(self.clients) >= MAX_CLIENTS:
                c.close()
                return True
            c.setblocking(False)
            self.clients[c.fileno()] = c
            self.poller.register(c.fileno(), select.POLLIN)  # readers never send: input = hangup
            if self.last_state is not None:
                self._send(c.fileno(), self.last_state)
            return True
        if fd in self.clients:
            self._drop(fd)
            return True
        return False

    def publish(self, msg: dict) -> None:
        line = (json.dumps(msg, separators=(",", ":")) + "\n").encode()
        if msg.get("type") == "state":
            self.last_state = line
        for fd in list(self.clients):
            self._send(fd, line)

    def _send(self, fd: int, line: bytes) -> None:
        try:
            if self.clients[fd].send(line) != len(line):
                self._drop(fd)  # partial write: the reader is behind, a torn line would follow
        except OSError:  # BlockingIOError (buffer full), BrokenPipeError, ...
            self._drop(fd)

    def _drop(self, fd: int) -> None:
        c = self.clients.pop(fd)
        self.poller.unregister(fd)
        c.close()

    def close(self) -> None:
        for fd in list(self.clients):
            self._drop(fd)
        self.sock.close()
        if os.path.exists(self.path):
            os.unlink(self.path)


def follow(path: str) -> int:
    """Relay the daemon's stream to stdout until either side closes."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(path)
    except OSError as e:
        print(f"uengram-host follow: {path}: {e} (is uengram-hostd running a model?)", file=sys.stderr)
        return 1
    out = sys.stdout.buffer
    try:
        while chunk := s.recv(65536):
            out.write(chunk)
            out.flush()
    except (BrokenPipeError, ConnectionResetError):
        pass
    return 0
