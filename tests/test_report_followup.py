import hashlib
from pathlib import Path
import pytest

from bench.report_followup import _trace_hash, select_development


def row(stream='metropt3', method='uengram', encoder='physical', variant='v2_frozen', seed=0,
        detected=2, fa=.2, alarm=.01, balanced=.8, config=None, status='ok'):
    return {'stream': stream, 'method': method, 'encoder': encoder, 'variant': variant,
            'seed': seed, 'config': config or {'k_novelty': 3, 'k_health': 0, 'per_state': False},
            'status': status, 'metrics': {'events_detected': detected, 'false_alarm_per_week': fa,
                'alarm_time_share_clean': alarm, 'window_balanced_accuracy': balanced},
            'details': {'events': [{}, {}]}}


def test_incomplete_cnn_seed_set_cannot_qualify():
    rows = [row(encoder='cnn3_range', seed=i) for i in range(4)]
    selected, candidates, tradeoff = select_development(rows)
    assert selected is None
    assert not candidates[0]['complete']
    assert tradeoff is None


def test_quiet_zero_detection_is_not_a_successful_tradeoff():
    selected, candidates, tradeoff = select_development([row(detected=0)])
    assert selected is None
    assert tradeoff is None
    assert not candidates[0]['qualifies']


def test_high_recall_outside_false_alarm_budget_is_reported_not_selected():
    result = row(detected=2, fa=2, alarm=.03)
    selected, candidates, tradeoff = select_development([result])
    assert selected is None
    assert tradeoff['variant'] == 'v2_frozen'
    assert not candidates[0]['qualifies']


def test_holdout_rows_cannot_change_development_selection():
    good = row(variant='candidate_a', detected=2, fa=.1)
    weak = row(variant='candidate_b', detected=1, fa=.05)
    before, _, _ = select_development([good, weak])
    after, _, _ = select_development([good, weak,
        row(stream='metropt2_jan', variant='candidate_b', detected=99, fa=0),
        row(stream='metropt2_apr', variant='candidate_b', detected=99, fa=0)])
    assert before['variant'] == after['variant'] == 'candidate_a'


def test_every_seed_must_meet_event_and_alarm_constraints():
    rows = [row(encoder='cnn3_state', seed=i, detected=2) for i in range(5)]
    rows[4]['metrics']['events_detected'] = 0
    selected, _, tradeoff = select_development(rows)
    assert selected is None
    assert tradeoff is not None
    assert tradeoff['encoder'] == 'cnn3_state'


def test_duplicate_seed_attempts_invalidate_configuration():
    rows = [row(), row()]
    selected, candidates, _ = select_development(rows)
    assert selected is None
    assert not candidates[0]['complete']


def test_freeze_trace_hash_must_match_recorded_provenance(tmp_path: Path):
    trace_path = tmp_path / 'trace.npz'
    trace_path.write_bytes(b'original')
    trace = {'path': str(trace_path), 'sha256': hashlib.sha256(b'original').hexdigest(),
             'bytes': len(b'original')}
    trace_path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='trace hash mismatch'):
        _trace_hash(trace)
