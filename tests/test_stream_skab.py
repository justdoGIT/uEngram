import numpy as np

from bench.stream_skab import skab_stream
from tests.test_skab import _fixture_root


def test_skab_stream_builds_one_event_per_fault_file(tmp_path):
    root = _fixture_root(tmp_path, n_af=200, n_fault=100, anomaly_at=(40, 70))
    stream = skab_stream(root, K=20, fit_frac=0.7)
    assert len(stream.events) == 34
    assert stream.state.tolist() == [0] * len(stream.W)
    assert stream.before_h == 0 and stream.after_h == 0
    eid = stream.event_id()
    test = stream.test
    # every test window overlapping its file's labelled interval gets that
    # file's event id, and commissioning/calibration windows never do.
    assert (eid[~test] == -1).all()
    assert set(eid[test & (eid >= 0)].tolist()) == set(range(34))
    neg = stream.clean_negative()
    # a window inside the labelled interval is never a clean negative
    assert not neg[test & (eid >= 0)].any()
    assert neg[test & (eid == -1)].all()
    assert np.isfinite(stream.features).all()
