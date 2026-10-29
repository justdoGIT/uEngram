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


def _two_condition_data(rng):
    # one unit alternating between two operating conditions whose sensor levels differ by 1000
    rows = []
    for c in range(1, 61):
        cond = 0.0 if c % 2 else 25.0
        level = 1000.0 if cond else 0.0
        rows.append([1, c, cond, 0.62, 100] + list(level + rng.normal(size=21)))
    return np.array(rows)


def test_condition_scaler_removes_operating_condition_offset():
    data = _two_condition_data(np.random.default_rng(0))
    raw, _, _ = cmapss.windows(data, K=16)
    scaled, _, _ = cmapss.windows(data, K=16, scaler=cmapss.fit_condition_scaler(data))
    # raw windows swing by ~1000 between consecutive cycles; scaled ones stay O(1)
    assert np.ptp(raw[0, 0]) > 900 and np.ptp(scaled[0, 0]) < 10


def test_condition_scaler_uses_only_fit_rows_for_unseen_data():
    rng = np.random.default_rng(1)
    fit, other = _two_condition_data(rng), _two_condition_data(rng)
    other[:, 5:] += 7.0  # drift present only in the data being scored
    scaler = cmapss.fit_condition_scaler(fit)
    scaled, _, _ = cmapss.windows(other, K=16, scaler=scaler)
    # a scaler refit on `other` would absorb the +7 shift; the fit-only scaler must keep it
    assert scaled.mean() > 3.0


def test_files_rejects_unknown_subset():
    assert cmapss.files("FD004") == ("train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt")
    try:
        cmapss.files("FD005")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
