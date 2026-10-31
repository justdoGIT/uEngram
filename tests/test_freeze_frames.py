import json
from types import SimpleNamespace

import numpy as np

from bench.freeze_frames import archive_frames


def test_evidence_maps_test_indices_and_never_fills_missing_context_across_gaps(tmp_path):
    time = np.arange(6).astype('datetime64[h]').astype('datetime64[s]')
    stream = SimpleNamespace(
        W=np.arange(12).reshape(6, 2), features=np.arange(12).reshape(6, 2),
        feature_names=('a', 'b'), test=np.array([False, False, True, True, True, True]),
        start=time, end=time + np.timedelta64(10, 'm'),
        segment=np.array([0, 0, 0, 0, 1, 1]), state=np.zeros(6), cond=None)
    episodes = [{'open_index': 1}, {'open_index': 3}]
    path = str(tmp_path / 'frames.npz')
    latent = np.arange(6, dtype=float)[:, None]
    info = archive_frames(stream, episodes, path, {}, (np.array([[0.0], [3.0]]), np.ones((2, 1))), latent)
    with np.load(path) as z:
        assert z['context_indices'].tolist() == [[1, 2, 3, -1, -1], [-1, 4, 5, -1, -1]]
        np.testing.assert_array_equal(z['windows'], stream.W[z['input_index']])
        records = json.loads(str(z['manifest']))
        assert records[0]['input_index'] == 3
        assert records[0]['nearest_reference_index'] == 1
        assert records[0]['reference_distance_squared'] == 0
        assert records[0]['context_complete_at'] == str(stream.end[3])
    assert info['incomplete_context_episodes'] == 2
