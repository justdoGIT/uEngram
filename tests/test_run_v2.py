import numpy as np

from bench.run_v2 import _care_events


class _CareStream:
    test = np.array([False, True, True])
    metric_mask = np.array([True, False, False])
    end = np.array([0, 1, 2], dtype="datetime64[s]")
    extra = {"care_label": "anomaly"}

    def event_id(self):
        return np.zeros(3, dtype=int)


def test_care_event_without_normal_rows_is_marked_ineligible():
    result = _care_events(_CareStream(), {"active": np.zeros(2, dtype=bool)})
    assert result == {
        "label": "anomaly",
        "status": "not_evaluable",
        "normal_rows": 0,
        "reason": "CARE event has no normal-status rows to evaluate",
    }
