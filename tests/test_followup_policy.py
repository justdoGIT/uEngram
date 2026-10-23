import numpy as np

from bench.alarm_eval import ACTIONS
from bench.followup_policy import evaluate_followup
from bench.streams import Stream
from uengram.alarm import AlarmConfig


def case():
    n = 110
    t = (np.arange(n) * 600).astype('datetime64[s]')
    cal = np.arange(n) < 100
    s = Stream('case', np.ones((n, 1)), t - np.timedelta64(600, 's'), t,
               np.zeros(n, int), np.zeros(n, int), np.ones(n, bool), np.zeros(n, bool),
               cal, ~cal, ((str(t[102]), str(t[105])),), 4, np.ones((n, 1)), ('pressure',))
    sc = {'novelty': np.r_[np.zeros(100), np.full(10, 10.)], 'health': np.zeros(n),
          'action': np.full(n, ACTIONS.index('new_candidate'))}
    return s, sc


def test_new_candidate_is_unresolved_and_does_not_veto_sustained_evidence():
    s, sc = case()
    cfg = AlarmConfig(k_novelty=3, k_health=0, new_state_alert=False, per_state=False)
    metrics, details, _ = evaluate_followup(s, sc, cfg)
    assert details['decision_state'].tolist() == [2, 2] + [3] * 8
    assert metrics['events_detected'] == 1
    assert details['window_confusion']['tp'] > 0


def test_unsupported_state_is_not_reported_in_reference_or_detected_by_advisory():
    s, sc = case()
    s.state[s.test] = 7
    sc['novelty'][s.test] = 0
    sc['action'][:] = ACTIONS.index('score')
    cfg = AlarmConfig(k_novelty=3, k_health=0, new_state_alert=False, per_state=True)
    metrics, details, _ = evaluate_followup(s, sc, cfg)
    assert np.all(details['decision_state'] == 2)
    assert metrics['events_detected'] == 0
    assert not np.any(details['active'])


def test_unobserved_windows_are_not_healthy_and_do_not_count_toward_accuracy():
    s, sc = case()
    s.monitorable[s.test] = False
    cfg = AlarmConfig(k_novelty=3, k_health=0, new_state_alert=False, per_state=False)
    _, details, _ = evaluate_followup(s, sc, cfg)
    assert np.all(details['decision_state'] == 0)
    assert details['window_confusion'] == {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
