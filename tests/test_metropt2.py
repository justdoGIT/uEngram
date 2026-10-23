import numpy as np

from uengram import metropt2


def _csv(path, header_cols, n=35, t0=0):
    with open(path, "w") as fh:
        fh.write("timestamp," + ",".join(header_cols) + "\n")
        for i in range(n):
            s = t0 + i
            stamp = f"2022-04-28 12:{s // 60:02d}:{s % 60:02d}.{(7 * i) % 1000:03d}"
            vals = {c: float(i + j) for j, c in enumerate(header_cols)}
            fh.write(stamp + "," + ",".join(str(vals[c]) for c in header_cols) + "\n")


def test_decimates_to_first_sample_of_each_10s_bin_and_maps_columns_by_name(tmp_path):
    # dataset_train.csv lists gpsLong before gpsLat; columns must map by name.
    cols = [c for c in metropt2.COLS if c not in ("gpsLat", "gpsLong")] + ["gpsLong", "gpsLat"]
    _csv(tmp_path / "MetroPT2.csv", cols, n=35, t0=3)  # seconds 3..37
    t, X, names = metropt2.parse_csv(str(tmp_path / "MetroPT2.csv"))
    # bins [0,10) [10,20) [20,30) [30,40): first samples are seconds 3, 10, 20, 30
    assert t.astype("datetime64[s]").astype(str).tolist() == [
        "2022-04-28T12:00:03", "2022-04-28T12:00:10", "2022-04-28T12:00:20", "2022-04-28T12:00:30"]
    row_of_second = np.array([3, 10, 20, 30]) - 3
    lat = names.index("gpsLat")
    np.testing.assert_allclose(X[:, lat], row_of_second + cols.index("gpsLat"))
    np.testing.assert_allclose(X[:, names.index("TP2")], row_of_second + cols.index("TP2"))


def test_park_polygons_contain_their_centres_and_exclude_open_line():
    centres = np.array([np.mean(p, axis=0) for p in metropt2.PARKS])
    assert metropt2.in_park(centres[:, 0], centres[:, 1]).all()
    # Porto city centre (Trindade) is on the line, not in a park polygon.
    assert not metropt2.in_park(np.array([41.1525]), np.array([-8.6093]))[0]
