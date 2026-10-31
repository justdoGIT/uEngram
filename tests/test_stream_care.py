import zipfile

import numpy as np
import pytest

from bench import stream_care as sc

ROOT = "CARE_To_Compare/Wind Farm A"


def _zip(path, n=80):
    t0 = np.datetime64("2023-01-01T00:00")
    rows = ["time_stamp;asset_id;id;train_test;status_type_id;sensor_0_avg;sensor_0_max;"
            "sensor_1_avg;sensor_2_avg;power_3_avg"]
    for i in range(n):
        t = str(t0 + np.timedelta64(10 * i, "m")).replace("T", " ")
        split = "train" if i < 60 else "prediction"
        status = 4 if i in (5, 70) else 0
        # sensor_0 normal, sensor_1 a counter (dropped), sensor_2 an angle in degrees
        rows.append(f"{t};11;{i};{split};{status};{i}.5;{i + 9};{1000 + i};{90 if i % 2 else 0};"
                    + ("" if i == 7 else f"{2 * i}"))
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{ROOT}/datasets/68.csv", "\n".join(rows) + "\n")
        z.writestr(f"{ROOT}/event_info.csv",
                   "asset;event_id;event_label;event_start;event_start_id;event_end;event_end_id;"
                   "event_description\n11;68;anomaly;2023-01-01 11:00:00;66;2023-01-01 12:00:00;72;Leak\n")
        z.writestr(f"{ROOT}/feature_description.csv",
                   "sensor_name;statistics_type;description;unit;is_angle;is_counter\n"
                   "sensor_0;average,maximum;Ambient temperature;C;False;False\n"
                   "sensor_1;average;Energy counter;kWh;False;True\n"
                   "sensor_2;average;Nacelle direction;deg;True;False\n"
                   "power_3;average;Active power;kW;False;False\n")
        for farm in ("B", "C"):
            z.writestr(f"CARE_To_Compare/Wind Farm {farm}/event_info.csv",
                       "asset_id;event_id;event_label;event_start;event_start_id;event_end;"
                       "event_end_id;event_description\n")


def test_care_dataset_follows_reference_preprocessing_and_splits(tmp_path, monkeypatch):
    path = str(tmp_path / "care.zip")
    _zip(path)
    s = sc.care_stream("care_A_68", path, str(tmp_path / "cache"))
    # averages only, counter dropped, angle as sin/cos, missing value kept as NaN
    assert s.feature_names == ("Ambient temperature", "Nacelle direction (sin)",
                               "Nacelle direction (cos)", "Active power")
    np.testing.assert_allclose(s.W[1, 1:3], [1.0, 0.0], atol=1e-6)
    assert np.isnan(s.W[7, 3]) and s.W[0, 0] == pytest.approx(0.5)
    # normal behaviour = status 0; test = prediction rows; calibration = last
    # 28 days of the training rows (all 60 here); commissioning excludes status != 0
    assert not s.monitorable[5] and s.monitorable[70] and s.monitorable[6]
    assert not s.metric_mask[70]  # retrospective test status filters metrics only
    assert s.test.sum() == 20 and not (s.test & (s.fit | s.cal)).any()
    assert s.cal.sum() == 60 and s.fit.sum() == 0
    # SCADA labels use the supplied row time, not the previous exposure interval.
    assert np.flatnonzero(s.event_id() == 0).tolist() == list(range(66, 73))
    assert s.extra["care_label"] == "anomaly" and sc.care_stream_names(path) == ["care_A_68"]
    assert s.end[66] == np.datetime64("2023-01-01T11:00:00")
    assert s.segment[4] < s.segment[5] < s.segment[6]
