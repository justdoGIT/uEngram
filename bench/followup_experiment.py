"""Exploratory normalization, physical-residual and no-veto ablations.

All three compressor recordings are consumed. Select on MetroPT3 only;
MetroPT2 replay is exploratory, never independent confirmation.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bench.alarm_eval import ACTIONS
from bench.common import dump
from bench.encoding_cache import input_hash
from bench.followup_policy import evaluate_followup
from bench.physical_residuals import PhysicalResiduals, observation_times, physical_features
from bench.run_v2 import FROZEN
from bench.streams import load_stream
from bench.v2_scoring import cached_scores, score_embeddings
from uengram.alarm import AlarmConfig

PROTOCOL = 'v3-exploratory-followup'
STREAMS = ('metropt3', 'metropt2_jan', 'metropt2_apr')


def policies():
    base = FROZEN['uengram']
    return [('v2_frozen', base), ('no_veto', replace(base, new_state_alert=False)),
            ('no_veto_health', replace(base, new_state_alert=False, k_health=3,
                                       health_needs_baseline=False))]


def physical_policies():
    for alpha in (.01, .003):
        for k in (1, 3, 6):
            for per_state in (False, True):
                yield (f'a{alpha}_k{k}_state{int(per_state)}', AlarmConfig(
                    alpha=alpha, k_novelty=k, k_health=0, per_state=per_state,
                    new_state_alert=False))


def _sha(path):
    with open(path, 'rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def save_trace(stream, scores, cfg, details, policy, path):
    test = np.flatnonzero(stream.test)
    thresholds = np.asarray([policy.thresholds(int(s)) for s in stream.state[test]])
    np.savez_compressed(path, test_input_index=test, start=stream.start[test].astype(np.int64),
                        end=stream.end[test].astype(np.int64), state=stream.state[test],
                        monitorable=stream.monitorable[test], novelty=scores['novelty'][test],
                        health=scores['health'][test], novelty_threshold=thresholds[:, 0],
                        health_threshold=thresholds[:, 1], active=details['active'],
                        opened=details['opened'], decision_state=details['decision_state'],
                        features=stream.features[test], feature_names=np.asarray(stream.feature_names),
                        config=json.dumps(cfg.__dict__), baseline_counts=json.dumps(policy.baseline_counts),
                        episodes=json.dumps(policy.episodes))
    return {'path': str(path), 'sha256': _sha(path), 'bytes': path.stat().st_size}


def evaluate(stream, scores, method, encoder, seed, variants, trace_dir):
    for name, cfg in variants:
        try:
            metrics, details, policy = evaluate_followup(stream, scores, cfg)
            path = trace_dir / f'{stream.name}_{method}_{encoder}_{seed}_{name}.npz'
            trace = save_trace(stream, scores, cfg, details, policy, path)
            small = {k: v for k, v in details.items()
                     if k not in ('active', 'opened', 'decision_state', 'unresolved_opened')}
            row = {'protocol': PROTOCOL, 'dataset_role': 'development' if stream.name == 'metropt3' else 'consumed exploratory replay',
                   'stream': stream.name, 'method': method, 'encoder': encoder, 'seed': seed,
                   'variant': name, 'config': cfg.__dict__, 'metrics': metrics,
                   'details': small, 'trace': trace, 'run': scores['meta']}
            print(f"[{stream.name}] {method}/{encoder}/{seed} {name}: "
                  f"detected={metrics['events_detected']:.0f}/{len(stream.events)} "
                  f"FA/week={metrics['false_alarm_per_week']:.3f} "
                  f"clean_alarm={metrics['alarm_time_share_clean']:.3%} "
                  f"balanced_accuracy={metrics['window_balanced_accuracy']:.3f}", flush=True)
        except ValueError as error:
            row = {'protocol': PROTOCOL, 'stream': stream.name, 'method': method, 'encoder': encoder,
                   'seed': seed, 'variant': name, 'config': cfg.__dict__, 'status': 'failed',
                   'error': f'{type(error).__name__}: {error}'}
            print(json.dumps(row), flush=True)
        yield row


def physical_runs(stream, trace_dir):
    base_dir = (Path.home() / '.local/state/uengram/v2scores' if stream.name == 'metropt3'
                else Path('data/metropt2/v2_cddfbf1'))
    dsp = cached_scores(stream, 'uengram', 'dsp', 0, str(base_dir), '[followup]')
    yield from evaluate(stream, dsp, 'uengram', 'dsp', 0, policies(), trace_dir)
    context, targets = physical_features(stream, observation_times(stream))
    for conditional in (False, True):
        name = 'physical_conditioned' if conditional else 'physical_marginal'
        model = PhysicalResiduals(conditional).fit(context, targets, stream.fit & stream.monitorable)
        values, residuals = model.score(context, targets)
        raw = trace_dir / f'{stream.name}_{name}_components.npz'
        np.savez_compressed(raw, context=context, targets=targets, residuals=residuals,
                            score=values, metadata=json.dumps(model.metadata()))
        sc = {'novelty': values, 'health': np.full(len(values), np.nan),
              'action': np.full(len(values), ACTIONS.index('score'), np.int8),
              'meta': {'stream': stream.name, 'method': name, 'components': str(raw),
                       'input_hash': input_hash(stream, 'dsp'), **model.metadata()}}
        yield from evaluate(stream, sc, name, 'physical', 0, list(physical_policies()), trace_dir)
        if conditional:
            fused = dict(dsp, health=values, meta=dict(dsp['meta'],
                         health_channel='conditional physical residual, not frozen-reference distance'))
            variants = [(f'union_k{k}', AlarmConfig(alpha=.005, k_novelty=k, k_health=k,
                         per_state=False, new_state_alert=False)) for k in (1, 3, 6)]
            yield from evaluate(stream, fused, 'dsp_physical_union', 'dsp', 0, variants, trace_dir)


def cnn_runs(stream, trace_dir, cnn_dir, modes, seeds):
    for mode in modes:
        for seed in seeds:
            name = f'cnn3_{mode}'
            path = Path(cnn_dir) / f'{stream.name}_{name}_{seed}.npz'
            if not path.is_file():
                yield {'protocol': PROTOCOL, 'stream': stream.name, 'method': 'uengram',
                       'encoder': name, 'seed': seed, 'status': 'failed',
                       'error': f'missing or failed training artifact: {path}'}
                continue
            with np.load(path) as z:
                Z, sigma0, metadata = z['Z'], z['sigma0'], json.loads(str(z['metadata']))
            if metadata.get('input_hash') != input_hash(stream, name):
                raise ValueError(f'CNN artifact does not match source inputs/conditions: {path}')
            cache = trace_dir / f'{stream.name}_uengram_{name}_{seed}_scores.npz'
            if cache.exists():
                with np.load(cache) as z:
                    sc = {k: z[k] for k in ('novelty', 'health', 'action', 'reference_mu', 'reference_var')}
                    sc['meta'] = json.loads(str(z['meta']))
            else:
                enc = SimpleNamespace(sigma0=sigma0, encoding_meta=metadata, encoding_path=str(path))
                sc = score_embeddings(stream, 'uengram', name, seed, enc, Z, '[followup]')
                np.savez_compressed(cache, **{k: v for k, v in sc.items() if k != 'meta'},
                                    meta=json.dumps(sc['meta']))
            yield from evaluate(stream, sc, 'uengram', name, seed, policies(), trace_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kind', choices=('physical', 'cnn'), required=True)
    ap.add_argument('--streams', nargs='+', default=list(STREAMS))
    ap.add_argument('--modes', nargs='+', default=['range', 'state'])
    ap.add_argument('--seeds', nargs='+', type=int, default=list(range(5)))
    ap.add_argument('--cnn-dir', default='.collab/v3')
    ap.add_argument('--trace-dir', default='data/metropt3/v3_followup_traces')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    if Path(args.out).exists():
        raise FileExistsError(f'refusing to overwrite an experiment result: {args.out}')
    for name in args.streams:
        print(f'[followup] loading {name}', flush=True)
        stream = load_stream(name)
        rows = (physical_runs(stream, trace_dir) if args.kind == 'physical' else
                cnn_runs(stream, trace_dir, args.cnn_dir, args.modes, args.seeds))
        for row in rows:
            dump(args.out, [row])


if __name__ == '__main__':
    main()
