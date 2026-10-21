import numpy as np

from uengram import cmapss


def _fixture(tmp_path, cycles=(20, 150)):
    rows = []
    for u, n in enumerate(cycles, start=1):
        for c in range(1, n + 1):
            rows.append([u, c, 0, 0, 100] + [float(u * 1000 + c * 10 + s) for s in range(1, 22)])
    p = tmp_path / "train.txt"
    p.write_text("\n".join(" ".join(str(v) for v in r) for r in rows))
    return cmapss.load(str(p))


def test_windows_count_rul_cap_and_sensor_columns(tmp_path):
    data = _fixture(tmp_path)
    W, unit, rul = cmapss.windows(data, K=16)
    assert W.shape == (5 + 135, 14, 16)
    assert (unit == 1).sum() == 5 and (unit == 2).sum() == 135
    # unit 1: windows end at cycles 16..20 -> RUL 4..0
    np.testing.assert_array_equal(rul[unit == 1], [4, 3, 2, 1, 0])
    # unit 2: first windows end at cycles 16.. -> RUL 134.. capped at 125
    assert rul[unit == 2][0] == 125 and rul[unit == 2][-1] == 0
    # first kept sensor is s2: value u*1000 + c*10 + 2 at the first cycle of window 0
    assert W[0, 0, 0] == 1000 + 10 + 2
    assert W[0, 0, 15] == 1000 + 160 + 2


def test_last_only_takes_final_window(tmp_path):
    data = _fixture(tmp_path)
    W, unit, rul = cmapss.windows(data, K=16, last_only=True)
    assert W.shape == (2, 14, 16) and list(unit) == [1, 2]
    assert W[0, 0, 15] == 1000 + 200 + 2
