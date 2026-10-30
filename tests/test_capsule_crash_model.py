from __future__ import annotations

import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from bench import capsule_crash_model as cm

ROOT = "/r"
F = "/r/f"


def op(kind, path=F, a=0, data=b""):
    return cm.Op(kind, path, a, 0, data)


def create(path=F):
    return op(cm.OP_CREATE, path)


def write(data, at=0, path=F):
    return op(cm.OP_WRITE, path, at, data)


def sync(path=F):
    return op(cm.OP_SYNC, path)


def images_of(ops, index=None):
    return cm.crash_images(ops, len(ops) if index is None else index, max_combos=1024)


def file_states(images, path=F):
    return {image[path] for image in images if path in image}


def test_an_unsynced_write_may_or_may_not_survive_and_may_be_missing_entirely():
    states = file_states(images_of([create(), write(b"x")]))

    assert states == {(False, b""), (True, b""), (True, b"x")}


def test_fsync_makes_everything_before_it_durable_in_every_image():
    states = file_states(images_of([create(), write(b"x"), sync()]))

    assert states == {(True, b"x")}


def test_a_write_after_the_last_fsync_is_the_only_thing_a_crash_can_drop():
    states = file_states(images_of([create(), write(b"x"), sync(), write(b"y", at=1)]))

    assert states == {(True, b"x"), (True, b"xy")}


def test_a_crash_can_apply_a_prefix_of_pending_writes_but_never_a_later_one_alone():
    states = file_states(images_of([create(), write(b"a"), sync(), write(b"b", 1), write(b"c", 2)]))

    assert (True, b"a\0c") not in states  # c without b would reorder two writes to the same file
    assert states == {(True, b"a"), (True, b"ab"), (True, b"abc")}


def test_an_unlink_without_a_directory_fsync_may_be_lost():
    states = file_states(images_of([create(), write(b"x"), sync(), op(cm.OP_UNLINK)]))

    assert states == {(True, b"x"), (False, b"")}


def test_a_directory_fsync_makes_an_unlink_durable_even_after_unsynced_writes():
    ops = [create(), write(b"x"), sync(), write(b"y", 1), op(cm.OP_UNLINK), op(cm.OP_DIRSYNC, ROOT)]

    assert file_states(images_of(ops)) == {(False, b"")}


def test_a_directory_fsync_makes_a_new_file_exist_but_not_its_unsynced_data():
    states = file_states(images_of([create(), write(b"x"), op(cm.OP_DIRSYNC, ROOT)]))

    assert states == {(True, b""), (True, b"x")}


def test_truncate_is_replayed_and_extends_with_zeroes():
    states = file_states(images_of([create(), write(b"abcd"), sync(), op(cm.OP_TRUNC, F, 2),
                                    op(cm.OP_TRUNC, F, 4), sync()]))

    assert states == {(True, b"ab\0\0")}


def test_files_are_independent_so_one_files_fsync_does_not_credit_another():
    g = "/r/g"
    ops = [create(), create(g), write(b"x"), write(b"y", path=g), sync()]

    states_g = file_states(images_of(ops), g)

    assert (True, b"y") in states_g and (False, b"") in states_g  # g was never synced


def test_marks_count_acknowledged_ingests_and_do_not_affect_file_state():
    ops = [create(), cm.Op(cm.OP_MARK, "", 0, 0, b"ingested 1"), write(b"x"),
           cm.Op(cm.OP_MARK, "", 0, 0, b"ingested 2"), sync()]

    assert cm.acknowledged_count(ops, 1) == 0
    assert cm.acknowledged_count(ops, 2) == 1
    assert cm.acknowledged_count(ops, len(ops)) == 2
    assert file_states(images_of(ops)) == {(True, b"x")}


def test_sampling_always_includes_the_nothing_applied_and_everything_applied_extremes():
    ops = [create(), write(b"a"), sync()]
    for n in range(12):
        ops.append(write(bytes([n]), at=1 + n))
    states = file_states(cm.crash_images(ops, len(ops), max_combos=4))

    assert (True, b"a") in states  # nothing pending applied
    assert any(len(data) == 13 for _, data in states)  # everything pending applied


def test_read_log_round_trips_every_record_field(tmp_path):
    raw = b""
    for kind, path, a, b, data in [(cm.OP_WRITE, "/r/f", 7, 0, b"hello"), (cm.OP_SYNC, "/r/f", 0, 0, b""),
                                   (cm.OP_MARK, "", 0, 0, b"m")]:
        encoded = path.encode()
        raw += bytes([kind]) + struct.pack("<I", len(encoded)) + encoded
        raw += struct.pack("<QQI", a, b, len(data)) + data
    log = tmp_path / "log.bin"
    log.write_bytes(raw)

    ops = cm.read_log(log)

    assert [(o.kind, o.path, o.a, o.data) for o in ops] == [
        (cm.OP_WRITE, "/r/f", 7, b"hello"), (cm.OP_SYNC, "/r/f", 0, b""), (cm.OP_MARK, "", 0, b"m")]


needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None or not sys.platform.startswith("linux"),
                               reason="the LD_PRELOAD shim needs gcc on Linux")


@needs_gcc
def test_the_real_shim_records_creates_writes_fsyncs_truncates_unlinks_and_marks(tmp_path):
    from bench.capsule_power_loss import compiled_shim

    shim = compiled_shim(tmp_path / "cache")
    root = tmp_path / "root"
    root.mkdir()
    log = tmp_path / "trace.log"
    script = (
        "import ctypes, os, sys\n"
        f"p = {str(root / 'a.bin')!r}\n"
        "mark = ctypes.CDLL(None).uengram_cc_mark\n"
        "fd = os.open(p, os.O_CREAT | os.O_WRONLY, 0o600)\n"
        "os.write(fd, b'abcdef'); os.pwrite(fd, b'ZZ', 1); os.fsync(fd); mark(b'one')\n"
        "os.ftruncate(fd, 3); os.close(fd)\n"
        "fd = os.open(p, os.O_WRONLY | os.O_TRUNC); os.close(fd)\n"
        "os.unlink(p)\n"
        f"open({str(tmp_path / 'outside.txt')!r}, 'w').write('not tracked')\n")
    subprocess.run([sys.executable, "-c", script], check=True,
                   env={"LD_PRELOAD": str(shim), "UENGRAM_CC_ROOT": str(root), "UENGRAM_CC_LOG": str(log)})

    ops = cm.read_log(log)
    got = [(o.kind, Path(o.path).name if o.path else "", o.a, o.data) for o in ops]

    assert got == [
        (cm.OP_CREATE, "a.bin", 0, b""),
        (cm.OP_WRITE, "a.bin", 0, b"abcdef"),
        (cm.OP_WRITE, "a.bin", 1, b"ZZ"),
        (cm.OP_SYNC, "a.bin", 0, b""),
        (cm.OP_MARK, "", 0, b"one"),
        (cm.OP_TRUNC, "a.bin", 3, b""),
        (cm.OP_TRUNC, "a.bin", 0, b""),  # O_TRUNC of an existing file
        (cm.OP_UNLINK, "a.bin", 0, b""),
    ]  # and nothing for the untracked file outside the root
