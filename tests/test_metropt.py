import numpy as np

from uengram import metropt


def _fixture_csv(path, t0, n, gap_after=None, gap_s=600):
    cols = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature",
            "Motor_current", "COMP", "DV_eletric", "Towers", "MPG", "LPS",
            "Pressure_switch", "Oil_level", "Caudal_impulses"]
    t = np.datetime64(t0, "s")
    with open(path, "w") as fh:
        fh.write("," + "timestamp," + ",".join(cols) + "\n")
        for i in range(n):
            if gap_after is not None and i == gap_after:
                t = t + np.timedelta64(gap_s, "s")
            vals = [i + 0.5 * j for j in range(len(cols))]
            fh.write(f"{10 * i},{str(t).replace('T', ' ')}," + ",".join(f"{v}" for v in vals) + "\n")
            t = t + np.timedelta64(10, "s")


def test_fetch_parses_csv_and_windows_never_span_gaps(tmp_path):
    _fixture_csv(tmp_path / metropt.CSV, "2020-02-01T00:00:00", 25, gap_after=12)
    t, X, cols = metropt.load(metropt.fetch(str(tmp_path)))
    assert len(t) == 25 and X.shape == (25, 15) and cols[0] == "TP2"
    assert t[0] == np.datetime64("2020-02-01T00:00:00")
    W, S, E, G = metropt.windows(t, X, cols, K=5)
    # 12 samples before the gap -> 2 windows, 13 after -> 2 windows; remainders dropped
    assert W.shape == (4, 7, 5) and list(G) == [0, 0, 1, 1]
    assert np.all((E - S).astype(int) == 40)
    # channel order follows ANALOG: column 0 is TP2 (value i), Motor_current is 6 * 0.5 + i
    np.testing.assert_allclose(W[0, 0], np.arange(5))
    np.testing.assert_allclose(W[0, 6], np.arange(5) + 3.0)
    # the second segment starts after the gap, so its first window starts at sample 12
    assert S[2] == t[12]


def test_failure_labels_and_clean_negative_buffer():
    fs = np.datetime64("2020-07-15T14:30")
    fe = np.datetime64("2020-07-15T19:00")
    h = np.timedelta64(1, "h")
    start = np.array([fs - 25 * h, fs - 2 * h, fs - h / 6, fe + h, fe + 47 * h, fe + 49 * h])
    end = start + np.timedelta64(10, "m")
    fid = metropt.failure_id(start, end)
    # only the window straddling the report start overlaps failure 4 (index 3)
    assert list(fid) == [-1, -1, 3, -1, -1, -1]
    neg = metropt.clean_negative(start, end)
    assert list(neg) == [True, False, False, False, False, True]
