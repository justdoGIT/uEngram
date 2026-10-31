"""CARE to Compare as protocol v2 streams (docs/EXPLAINED.md section 17).

Guck, Roelofs & Faulstich (2024), Zenodo 10.5281/zenodo.15846963, CC BY-SA
4.0: 95 event datasets from 36 wind turbines in 3 farms, 10-minute SCADA rows,
one training year per dataset. Supplied event tables label 45 anomaly / 50
normal events; the original protocol text's 44 / 51 count was incorrect.
Each dataset is one stream. Preprocessing follows the reference implementation
(Fraunhofer IEE EnergyFaultDetector, care2compare and CAREScore): 10-minute
averages only, counters dropped, angles as sin/cos. Status 0 identifies
commissioning/calibration normals and evaluation rows; prediction ignores
retrospective test statuses. This status-0 convention differs from the paper's
table that also lists idling (status 2) as normal.

Splits, fixed before any detector saw the data: commissioning = status-0
training rows before the last CAL_DAYS of the training period; calibration =
the training rows of those last CAL_DAYS; test = the prediction rows. CARE has
no discrete operating-state input comparable to compressor load, so state 0
uses the same thresholds as the global baseline. Retrospective evaluation
masks are not availability inputs to the deployed policy.
"""
from __future__ import annotations

import csv
import io
import os
import zipfile
from array import array

import numpy as np

from bench.streams import Stream

ZIP = "data/care/CARE_To_Compare.zip"
CACHE = "data/care/cache"
ROOT = "CARE_To_Compare/Wind Farm {farm}"
FARMS = ("A", "B", "C")
CAL_DAYS = 28
NAN = float("nan")


def _table(z, member) -> list[dict]:
    with z.open(member) as fh:
        return list(csv.DictReader(io.TextIOWrapper(fh, encoding="latin-1"), delimiter=";"))


def event_table(path: str = ZIP) -> list[dict]:
    rows = []
    with zipfile.ZipFile(path) as z:
        for farm in FARMS:
            for r in _table(z, f"{ROOT.format(farm=farm)}/event_info.csv"):
                rows.append({"farm": farm, "event_id": int(r["event_id"]), "label": r["event_label"],
                             "start": r["event_start"].replace(" ", "T"),
                             "end": r["event_end"].replace(" ", "T"),
                             "asset": r.get("asset_id", r.get("asset")),
                             "description": r["event_description"]})
    return rows


def care_stream_names(path: str = ZIP) -> list[str]:
    return [f"care_{e['farm']}_{e['event_id']}" for e in event_table(path)]


def _sensors(z, farm) -> list[tuple[str, str, bool]]:
    """(sensor_name, description, is_angle) of non-counter sensors with averages."""
    out = []
    for r in _table(z, f"{ROOT.format(farm=farm)}/feature_description.csv"):
        if "average" in r["statistics_type"] and r["is_counter"].strip() != "True":
            out.append((r["sensor_name"], r["description"], r["is_angle"].strip() == "True"))
    return out


def parse_dataset(farm: str, event_id: int, path: str = ZIP) -> dict:
    with zipfile.ZipFile(path) as z:
        sensors = _sensors(z, farm)
        with z.open(f"{ROOT.format(farm=farm)}/datasets/{event_id}.csv") as fh:
            lines = io.TextIOWrapper(fh, encoding="latin-1")
            header = lines.readline().strip().split(";")
            pos = {c: i for i, c in enumerate(header)}
            cols = [(pos.get(f"{s}_avg", pos.get(s)), d, a) for s, d, a in sensors]
            cols = [c for c in cols if c[0] is not None]
            meta = [pos[k] for k in ("time_stamp", "id", "train_test", "status_type_id")]
            ts, ids, train, status = [], [], [], []
            X = array("f")  # 4 bytes per value; a farm C file has ~16M values
            take = [i for i, _, _ in cols]
            for line in lines:
                p = line.rstrip("\n").split(";")
                ts.append(p[meta[0]].replace(" ", "T"))
                ids.append(int(p[meta[1]]))
                train.append(p[meta[2]] == "train")
                status.append(int(float(p[meta[3]])))
                X.extend(float(p[i]) if p[i] not in ("", "nan", "NaN") else NAN for i in take)
    X = np.frombuffer(X, dtype=np.float32).reshape(len(ids), len(cols)).astype(np.float64)
    names, out = [], []
    for j, (_, desc, angle) in enumerate(cols):
        if angle:
            rad = np.deg2rad(X[:, j])
            out += [np.sin(rad), np.cos(rad)]
            names += [f"{desc} (sin)", f"{desc} (cos)"]
        else:
            out.append(X[:, j])
            names.append(desc)
    order = np.argsort(np.array(ids))
    return {"t": np.array(ts, dtype="datetime64[s]")[order], "train": np.array(train)[order],
            "status": np.array(status, dtype=np.int64)[order],
            "X": np.column_stack(out)[order].astype(np.float32), "names": np.array(names)}


def load_dataset(name: str, path: str = ZIP, cache: str = CACHE) -> dict:
    _, farm, eid = name.split("_")
    f = os.path.join(cache, f"{name}.npz")
    if not os.path.exists(f):
        os.makedirs(cache, exist_ok=True)
        d = parse_dataset(farm, int(eid), path)
        np.savez_compressed(f, t=d["t"].astype(np.int64), train=d["train"], status=d["status"],
                            X=d["X"], names=d["names"])
    z = np.load(f)
    return {"t": z["t"].astype("datetime64[s]"), "train": z["train"], "status": z["status"],
            "X": z["X"], "names": [str(n) for n in z["names"]]}


def care_stream(name: str, path: str = ZIP, cache: str = CACHE) -> Stream:
    _, farm, eid = name.split("_")
    ev = next(e for e in event_table(path) if e["farm"] == farm and e["event_id"] == int(eid))
    d = load_dataset(name, path, cache)
    t, train, status = d["t"], d["train"], d["status"]
    # A SCADA value is observed at its supplied row time. Nominal 10-minute
    # exposure is an accounting convention; event labels use that exact time.
    start = t - np.timedelta64(600, "s")
    # Only TRAINING statuses may define noise-estimation boundaries. Test
    # statuses include retrospective fault annotations and are evaluation-only.
    boundary = ((np.diff(t).astype(np.int64) > 600) |
                (train[1:] & train[:-1] & ((status[1:] == 0) != (status[:-1] == 0))))
    segment = np.cumsum(np.r_[0, boundary])
    cal = train & (t >= t[train].max() - np.timedelta64(CAL_DAYS, "D"))
    fit = train & ~cal & (status == 0)
    events = ((ev["start"], ev["end"]),) if ev["label"] == "anomaly" else ()
    X = d["X"]
    monitorable = ~train | (status == 0)
    s = Stream(name, X, start, t, segment, np.zeros(len(t), np.int64), monitorable, fit, cal, ~train,
               events, X.shape[1] * 4, X.astype(np.float64), tuple(d["names"]), point_labels=True,
               metric_mask=status == 0)
    s.extra.update({"care_label": ev["label"], "care_event_end": ev["end"], "farm": farm,
                    "asset": ev["asset"], "event_id": int(eid), "description": ev["description"]})
    return s
