"""MetroPT2 Zenodo record (10.5281/zenodo.7766691, CC BY 4.0): two 1 Hz logs of
Metro do Porto air production units, used as fresh confirmation streams for
protocol v2 (docs/EXPLAINED.md section 17).

- metropt2_jan: dataset_train.csv, 2022-01-01 to 2022-06-02. Failures from
  Veloso et al., Sci Data 9:764 (2022), Table 2.
- metropt2_apr: MetroPT2.csv, 2022-04-28 to 2022-07-28. Failures and the
  train's low-pressure (LPS) warning times from Jakobs, Veloso & Gama,
  arXiv:2502.07394 (2025), Table 1.

At equal timestamps the files hold different readings (oil temperature, motor
current). They are distinct recordings, not interchangeable copies; physical
machine independence is unverified. Both use DV_eletric = 1 for loaded
operation. Retain the first sample of each 10 s bin, stamped with its actual
measurement time to whole seconds, not the earlier bin start.

The splits mirror MetroPT-3 (three weeks commissioning, one week calibration)
and were fixed before any detector saw these files.
"""
from __future__ import annotations

import os

import numpy as np

DEST = "data/metropt2"
COLS = ("TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current",
        "COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level",
        "Caudal_impulses", "Flowmeter", "gpsLat", "gpsLong", "gpsSpeed", "gpsQuality")
STREAMS = {
    "metropt2_jan": {
        "csv": "dataset_train.csv", "fit_end": "2022-01-22T00:00", "train_end": "2022-01-29T00:00",
        "events": (("2022-02-28T21:53", "2022-03-01T02:00"),  # air leak, clients
                   ("2022-03-23T14:54", "2022-03-23T15:24"),  # air leak, air dryer
                   ("2022-05-30T12:00", "2022-06-02T06:18")),  # oil leak, compressor
        "lps": None},
    "metropt2_apr": {
        "csv": "MetroPT2.csv", "fit_end": "2022-05-19T00:00", "train_end": "2022-05-26T00:00",
        "events": (("2022-06-04T10:19:24", "2022-06-04T14:22:39"),  # air leak
                   ("2022-07-11T10:10:18", "2022-07-14T10:22:08")),  # oil leak
        "lps": ("2022-06-04T11:26:01", "2022-07-13T19:43:52")},
}
# Workshop and parking polygons (lat, long), Veloso et al. 2022, Table 1.
PARKS = (
    ((41.212693, -8.661192), (41.213258, -8.657555), (41.215243, -8.660516), (41.213375, -8.662785)),
    ((41.213604, -8.660474), (41.212829, -8.661654), (41.210198, -8.657824), (41.211126, -8.656569)),
    ((41.376419, -8.757253), (41.378367, -8.758960), (41.378528, -8.758405), (41.376558, -8.756909)),
    ((41.269800, -8.614621), (41.272303, -8.613077), (41.272428, -8.613600), (41.271245, -8.614321)),
    ((41.168919, -8.543092), (41.170735, -8.542713), (41.170798, -8.543335), (41.168919, -8.543646)),
    ((41.157216, -8.584129), (41.156949, -8.583357), (41.155669, -8.583958), (41.155766, -8.584537)),
    ((41.113698, -8.607218), (41.113636, -8.606551), (41.115258, -8.606347), (41.115251, -8.606870)),
)


def parse_csv(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Decimate a 1 Hz log to 10 s bins (first sample of each bin)."""
    ts, rows = [], []
    with open(path) as fh:
        header = fh.readline().strip().split(",")
        idx = [header.index(c) for c in COLS]
        last = None
        for line in fh:
            key = line[:18]  # "YYYY-MM-DD HH:MM:S" identifies the 10 s bin
            if key == last:
                continue
            last = key
            parts = line.rstrip("\n").split(",")
            ts.append(parts[0][:19].replace(" ", "T"))
            rows.append([float(parts[i]) for i in idx])
    return np.array(ts, dtype="datetime64[s]"), np.array(rows, dtype=np.float64), list(COLS)


def load(name: str, dest: str = DEST) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(t [n] datetime64[s], X [n, 20], columns), cached as compressed npz."""
    cache = os.path.join(dest, f"{name}_actual_time.npz")
    if not os.path.exists(cache):
        t, X, cols = parse_csv(os.path.join(dest, STREAMS[name]["csv"]))
        np.savez_compressed(cache, t=t.astype(np.int64), X=X, cols=np.array(cols))
    z = np.load(cache)
    return z["t"].astype("datetime64[s]"), z["X"], [str(c) for c in z["cols"]]


def in_park(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """True where (lat, lon) lies inside any published workshop/parking polygon."""
    inside = np.zeros(np.shape(lat), dtype=bool)
    for poly in PARKS:
        hit = np.zeros(np.shape(lat), dtype=bool)
        for (y1, x1), (y2, x2) in zip(poly, poly[1:] + poly[:1]):
            cross = (y1 > lat) != (y2 > lat)
            with np.errstate(divide="ignore", invalid="ignore"):
                x_at = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
            hit ^= cross & (lon < x_at)
        inside |= hit
    return inside
