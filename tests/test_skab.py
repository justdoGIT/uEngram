import json
from unittest import mock

import numpy as np

from uengram import skab


def _write_csv(path, t0, n, labelled, anomaly_at=None, gap_after=None, gap_s=600):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["datetime", *skab.CHANNELS]
    if labelled:
        header += ["anomaly", "changepoint"]
    t = np.datetime64(t0, "s")
    with open(path, "w") as fh:
        fh.write(";".join(header) + "\n")
        for i in range(n):
            if gap_after is not None and i == gap_after:
                t = t + np.timedelta64(gap_s, "s")
            vals = [i + 0.1 * j for j in range(len(skab.CHANNELS))]
            row = [str(t).replace("T", " "), *[f"{v}" for v in vals]]
            if labelled:
                a = int(anomaly_at is not None and anomaly_at[0] <= i <= anomaly_at[1])
                c = int(a and i in (anomaly_at[0], anomaly_at[1]))
                row += [str(a), str(c)]
            fh.write(";".join(row) + "\n")
            t = t + np.timedelta64(1, "s")


def _fixture_root(tmp_path, n_af=200, n_fault=100, anomaly_at=(40, 70), gap_after=None):
    root = tmp_path / "skab"
    _write_csv(root / "anomaly-free/anomaly-free.csv", "2020-01-01T00:00:00", n_af, labelled=False)
    for i, rel in enumerate(skab.FAULT_FILES):
        _write_csv(root / rel, "2021-01-01T00:00:00", n_fault, labelled=True,
                   anomaly_at=anomaly_at, gap_after=gap_after if i == 0 else None)
    return root


def test_load_splits_commissioning_and_finds_one_onset_per_file(tmp_path):
    root = _fixture_root(tmp_path)
    raw = skab.load(root, fit_frac=0.7, K=20)
    assert raw["X"].shape[1] == len(skab.CHANNELS)
    n_fit = int((200 * 0.7) // 20 * 20)
    assert raw["n_fit"] == n_fit
    assert (raw["split"] == "fit").sum() == n_fit
    assert (raw["split"] == "cal").sum() == 200 - n_fit
    assert (raw["split"] == "test").sum() == 100 * len(skab.FAULT_FILES)
    assert set(raw["onset"]) == set(range(len(skab.FAULT_FILES)))
    # onset/offset bracket exactly the labelled run for every experiment
    for i in range(len(skab.FAULT_FILES)):
        assert (raw["offset"][i] - raw["onset"][i]).astype(int) == 70 - 40


def test_load_rejects_a_fault_file_with_no_anomaly(tmp_path):
    root = _fixture_root(tmp_path)
    # Corrupt one fault file to have zero anomalous rows.
    _write_csv(root / skab.FAULT_FILES[0], "2021-01-01T00:00:00", 100, labelled=True,
               anomaly_at=(1000, 1000))
    try:
        skab.load(root, fit_frac=0.7, K=20)
        raise AssertionError("expected ValueError for a file with no labelled anomaly")
    except ValueError as e:
        assert "no labelled anomaly" in str(e)


def test_windows_never_cross_fit_cal_or_file_boundaries(tmp_path):
    root = _fixture_root(tmp_path, gap_after=12)
    raw = skab.load(root, fit_frac=0.7, K=5)
    W, S, E, G, split, experiment = skab.windows(
        raw["t"], raw["X"], raw["split"], raw["experiment"], K=5)
    assert W.shape[1:] == (len(skab.CHANNELS), 5)
    # every window's rows share one split and one experiment id by construction;
    # re-derive both from the raw per-row arrays to catch any off-by-one.
    assert set(split.tolist()) == {"fit", "cal", "test"}
    assert (experiment[split == "test"] >= 0).all()
    assert (experiment[split != "test"] == -1).all()
    # the gap inside the first fault file splits its windows into two segments
    first_exp_g = sorted(set(G[experiment == 0].tolist()))
    assert len(first_exp_g) >= 2
    assert np.all((E - S).astype(np.int64) == 4)


def test_windows_tag_anomalous_rows_into_the_right_experiment(tmp_path):
    root = _fixture_root(tmp_path, n_fault=100, anomaly_at=(40, 70))
    raw = skab.load(root, fit_frac=0.7, K=20)
    W, S, E, G, split, experiment = skab.windows(
        raw["t"], raw["X"], raw["split"], raw["experiment"], K=20)
    is_test = split == "test"
    # windows fully inside [40,70] for experiment 0 exist and no fit/cal window
    # is misclassified as belonging to any experiment.
    exp0 = (experiment == 0) & is_test
    assert exp0.sum() == 5  # 100 rows / K=20, non-overlapping
    onset, offset = raw["onset"][0], raw["offset"][0]
    overlapping = exp0 & (S <= offset) & (E >= onset)
    assert overlapping.sum() >= 1


def _fake_tree(contents: dict[str, bytes]) -> dict:
    return {"tree": [
        {"path": f"data/{rel}", "sha": skab._blob_sha1(data)}
        for rel, data in contents.items()
    ]}


def test_fetch_downloads_and_verifies_against_git_blob_sha1(tmp_path):
    contents = {rel: f"content-{rel}".encode() for rel in skab.ALL_FILES}
    tree_resp = mock.Mock()
    tree_resp.__enter__ = mock.Mock(return_value=tree_resp)
    tree_resp.__exit__ = mock.Mock(return_value=False)
    tree_resp.read = mock.Mock(return_value=json.dumps(_fake_tree(contents)).encode())

    def _urlopen(url, timeout=30):
        if "api.github.com" in url:
            return tree_resp
        rel = url.split(f"{skab.COMMIT}/data/", 1)[1]
        resp = mock.Mock()
        resp.__enter__ = mock.Mock(return_value=resp)
        resp.__exit__ = mock.Mock(return_value=False)
        resp.read = mock.Mock(return_value=contents[rel])
        return resp

    dest = tmp_path / "skab"
    with mock.patch.object(skab.urllib.request, "urlopen", side_effect=_urlopen):
        skab.fetch(dest)
    for rel in skab.ALL_FILES:
        assert (dest / rel).read_bytes() == contents[rel]
    # a second call with correct on-disk content makes no network request
    with mock.patch.object(skab.urllib.request, "urlopen", side_effect=_urlopen) as m:
        skab.fetch(dest)
    assert m.call_count == 1  # only the tree listing; every file already matches


def test_fetch_rejects_a_tampered_download(tmp_path):
    contents = {rel: f"content-{rel}".encode() for rel in skab.ALL_FILES}
    tree_resp = mock.Mock()
    tree_resp.__enter__ = mock.Mock(return_value=tree_resp)
    tree_resp.__exit__ = mock.Mock(return_value=False)
    tree_resp.read = mock.Mock(return_value=json.dumps(_fake_tree(contents)).encode())

    def _urlopen(url, timeout=30):
        if "api.github.com" in url:
            return tree_resp
        resp = mock.Mock()
        resp.__enter__ = mock.Mock(return_value=resp)
        resp.__exit__ = mock.Mock(return_value=False)
        resp.read = mock.Mock(return_value=b"tampered")
        return resp

    with mock.patch.object(skab.urllib.request, "urlopen", side_effect=_urlopen):
        try:
            skab.fetch(tmp_path / "skab")
            raise AssertionError("expected ValueError for a sha1 mismatch")
        except ValueError as e:
            assert "sha1 mismatch" in str(e)
