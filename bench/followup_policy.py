"""Exploratory alarm comparisons with explicit unresolved/unsupported states.

Unresolved is not a fault label, a benign classification, or a detected event.
The v2 counter implementation is reused; new-state veto removal is an ablation.
"""
from __future__ import annotations

import numpy as np

from bench.alarm_eval import ACTIONS, apply_policy
from bench.metropt_eval import episodes


STATE_NAMES = ('unobserved', 'in_reference', 'unresolved', 'alarm')


def evaluate_followup(stream, scores, config):
    metrics, details, policy = apply_policy(stream, scores, config)
    test = np.flatnonzero(stream.test)
    novelty, health = scores['novelty'][test], scores['health'][test]
    observed = stream.monitorable[test] & (
        ((config.k_novelty > 0) & np.isfinite(novelty)) |
        ((config.k_health > 0) & np.isfinite(health)))
    states = np.zeros(len(test), np.int8)
    states[observed] = 1
    for j, i in enumerate(test):
        if not observed[j]:
            continue
        key = int(stream.state[i]) if config.per_state else -1
        support = policy.baseline_counts.get(key, {})
        known = ((config.k_novelty == 0 or support.get('novelty_threshold_supported', False)) and
                 (config.k_health == 0 or support.get('health_threshold_supported', False)))
        nt, ht = policy.thresholds(int(stream.state[i]))
        above = ((config.k_novelty > 0 and novelty[j] > nt) or
                 (config.k_health > 0 and health[j] > ht))
        candidate = int(scores['action'][i]) == ACTIONS.index('new_candidate')
        if not known or above or candidate:
            states[j] = 2
    states[details['active']] = 3
    unresolved = states == 2
    unresolved_opened = unresolved & ~np.r_[False, unresolved[:-1]]
    clean = stream.clean_negative()[test] & observed
    if stream.metric_mask is not None:
        clean &= stream.metric_mask[test]
    durations = (stream.end[test] - stream.start[test]).astype(np.int64)
    seconds = float(durations[clean].sum())
    weeks = seconds / (7 * 86400)
    metrics['unresolved_share_clean'] = float(durations[clean & unresolved].sum() / seconds) if seconds else np.nan
    metrics['unresolved_episodes_per_week'] = episodes(stream.end[test][unresolved_opened & clean]) / weeks if weeks else np.nan
    metrics['operator_events_with_unresolved_per_week'] = episodes(
        stream.end[test][(unresolved_opened | details['opened']) & clean]) / weeks if weeks else np.nan
    event = stream.event_id()[test] >= 0
    labeled = observed & (event | clean)
    if stream.metric_mask is not None:
        labeled &= stream.metric_mask[test]
    pred = details['active']
    tp, fp = int((pred & event & labeled).sum()), int((pred & ~event & labeled).sum())
    fn, tn = int((~pred & event & labeled).sum()), int((~pred & ~event & labeled).sum())
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    metrics.update({'window_sensitivity': sensitivity, 'window_specificity': specificity,
                    'window_balanced_accuracy': (sensitivity + specificity) / 2,
                    'window_precision': tp / (tp + fp) if tp + fp else np.nan,
                    'window_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan})
    details.update({'decision_state': states, 'unresolved_opened': unresolved_opened,
                    'decision_state_names': STATE_NAMES,
                    'window_confusion': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}})
    return metrics, details, policy
