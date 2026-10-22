import threading
import time

from uengram.host.collector import AsyncWriter


def test_a_stalled_write_does_not_delay_the_caller_and_order_is_kept():
    gate, out = threading.Event(), []
    w = AsyncWriter()
    w.submit(lambda: gate.wait(5))  # a disk that blocks, like hp's USB stick
    t0 = time.monotonic()
    for i in range(100):
        w.submit(out.append, i)
    assert time.monotonic() - t0 < 0.1 and out == []
    gate.set()
    w.close()
    assert out == list(range(100))


def test_a_failing_write_is_counted_and_later_writes_continue():
    out = []

    def fail():
        raise OSError(28, "No space left on device")

    w = AsyncWriter()
    w.submit(fail)
    w.submit(out.append, "after")
    w.close()
    assert w.failed == 1 and out == ["after"]


def test_beyond_the_queue_bound_writes_are_dropped_and_counted():
    gate = threading.Event()
    w = AsyncWriter(maxsize=3)
    w.submit(gate.wait, 5)
    time.sleep(0.05)  # the writer has taken the blocking job off the queue
    for i in range(5):
        w.submit(print, i)
    assert w.dropped == 2
    gate.set()
    w.close()
