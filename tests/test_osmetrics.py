from pathlib import Path

import numpy as np

from bench.osmetrics import causal_windows, clean_values, event_overlap_mask, fit_fill_values, load


ROOT = Path("/home/kk/.cache/uengram/osmetrics/extracted")


def test_osmetrics_archive_is_aligned_and_causally_annotated():
    data = load(ROOT)
    assert data.values.shape == (40320, 6)
    assert data.timestamps.shape == (40320,)
    assert len(data.events) == 42
    assert data.events[0].start_point <= data.events[-1].start_point


def test_invalid_values_use_fit_prefix_only():
    data = load(ROOT)
    fill = fit_fill_values(data, 5760)
    clean = clean_values(data, fill)
    assert clean.shape == data.values.shape
    assert np.isfinite(clean).all()
    expected = np.array([
        np.median(data.values[:5760, column][
            np.isfinite(data.values[:5760, column])
            & (data.values[:5760, column] > -90000.0)
        ])
        for column in range(data.values.shape[1])
    ])
    assert np.all(fill == expected)


def test_causal_windows_do_not_include_future_samples():
    values = np.arange(24, dtype=float).reshape(12, 2)
    windows, starts, ends = causal_windows(values, length=4, stride=2)
    assert windows.shape == (5, 2, 4)
    assert starts.tolist() == [0, 2, 4, 6, 8]
    assert ends.tolist() == [3, 5, 7, 9, 11]
    assert windows[2, 0].tolist() == [8.0, 10.0, 12.0, 14.0]


def test_event_overlap_mask_marks_only_intersecting_windows():
    data = load(ROOT)
    fill = fit_fill_values(data, 5760)
    values = clean_values(data, fill)
    _, starts, ends = causal_windows(values, length=16, stride=16)
    mask = event_overlap_mask(starts, ends, data.events)
    assert mask.dtype == bool
    assert mask.any()
