"""Summarize exploratory v3 runs and freeze a development-selected baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

RESULT_FILES = (Path('results/v3_physical.jsonl'), Path('results/v3_cnn.jsonl'))
SUMMARY_PATH = Path('results/v3_followup_summary.json')
FREEZE_PATH = Path('results/v3_normal_freeze.json')
STREAMS = ('metropt3', 'metropt2_jan', 'metropt2_apr')
EXPECTED_SEEDS = {'physical': (0,), 'cnn': (0, 1, 2, 3, 4)}


def _finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _key(row):
    return (row.get('method'), row.get('encoder'), row.get('variant'),
            json.dumps(row.get('config', {}), sort_keys=True, separators=(',', ':')))


def load_results(paths=RESULT_FILES):
    """Read existing result streams once, retaining malformed lines as failures."""
    rows, inputs, absent = [], {}, []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            absent.append(str(path))
            continue
        raw = path.read_bytes()
        inputs[str(path)] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        for number, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('JSONL row is not an object')
            except (json.JSONDecodeError, ValueError) as error:
                row = {'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                       'source': str(path), 'line': number}
            row['_source'] = str(path)
            row['_line'] = number
            rows.append(row)
    return rows, inputs, absent


def _expected(row):
    encoder = str(row.get('encoder', ''))
    return EXPECTED_SEEDS['cnn'] if encoder.startswith('cnn3_') else EXPECTED_SEEDS['physical']


def _group_summary(key, rows, expected):
    seed_counts = defaultdict(int)
    for row in rows:
        seed_counts[row.get('seed')] += 1
    by_seed = {r.get('seed'): r for r in rows}
    unique = all(seed_counts.get(seed) == 1 for seed in expected)
    complete = unique and set(by_seed) == set(expected)
    failed = [r for r in rows if r.get('status') == 'failed']
    valid = [by_seed[s] for s in expected if seed_counts.get(s) == 1 and s in by_seed
             and by_seed[s].get('status') != 'failed' and isinstance(by_seed[s].get('metrics'), dict)]
    metrics = ('events_detected', 'false_alarm_per_week', 'alarm_time_share_clean',
               'alerts_per_week', 'operator_events_per_week', 'notifications_per_week_all',
               'unresolved_episodes_per_week', 'operator_events_with_unresolved_per_week',
               'window_precision', 'window_sensitivity', 'window_specificity', 'window_balanced_accuracy')
    aggregate = {}
    for name in metrics:
        vals = [float(r['metrics'][name]) for r in valid if _finite(r['metrics'].get(name))]
        if vals:
            aggregate[name] = {'mean': sum(vals) / len(vals), 'min': min(vals), 'max': max(vals)}
    expected_rows = [by_seed[s] for s in expected if s in by_seed]
    event_counts = [len(r.get('details', {}).get('events', [])) for r in expected_rows]
    detected = [float(r['metrics']['events_detected']) for r in valid
                if _finite(r.get('metrics', {}).get('events_detected'))]
    balanced = [float(r['metrics']['window_balanced_accuracy']) for r in valid
                if _finite(r.get('metrics', {}).get('window_balanced_accuracy'))]
    fa = [float(r['metrics']['false_alarm_per_week']) for r in valid
          if _finite(r.get('metrics', {}).get('false_alarm_per_week'))]
    alarm_share = [float(r['metrics']['alarm_time_share_clean']) for r in valid
                   if _finite(r.get('metrics', {}).get('alarm_time_share_clean'))]
    all_valid = complete and not failed and len(valid) == len(expected)
    qualifies = (all_valid and bool(event_counts) and len(detected) == len(expected)
                 and min(detected) >= 1 and len(fa) == len(expected) and max(fa) <= 1
                 and len(alarm_share) == len(expected) and max(alarm_share) <= .02
                 and len(balanced) == len(expected))
    summary = {'method': key[0], 'encoder': key[1], 'variant': key[2],
               'config': json.loads(key[3]), 'expected_seeds': list(expected),
               'present_seeds': sorted(s for s in by_seed if isinstance(s, int)),
               'complete': complete, 'valid': all_valid, 'failures': [r for r in failed],
               'attempt_rows': len(rows), 'event_count': event_counts[0] if event_counts else None,
               'qualifies': qualifies, 'metrics': aggregate}
    if all_valid:
        summary['per_seed'] = {str(s): {'metrics': by_seed[s]['metrics'],
                                        'trace': by_seed[s].get('trace')}
                               for s in expected}
    return summary


def _complexity(item):
    cfg = item['config']
    persistence = int(cfg.get('k_novelty', 0) or 0) + int(cfg.get('k_health', 0) or 0)
    return (persistence, bool(cfg.get('per_state', False)), item['method'] or '',
            item['encoder'] or '', item['variant'] or '',
            json.dumps(cfg, sort_keys=True, separators=(',', ':')))


def _rank(item):
    m = item['metrics']
    return (-m['events_detected']['mean'],
            max(m['false_alarm_per_week']['max'], 50 * m['alarm_time_share_clean']['max']),
            -m['window_balanced_accuracy']['mean'], _complexity(item))


def select_development(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get('stream') == 'metropt3':
            groups[_key(row)].append(row)
    items = [_group_summary(k, rs, _expected(rs[0])) for k, rs in groups.items()]
    usable = [x for x in items if x['qualifies']]
    tradeoffs = [x for x in items if x['valid'] and not x['qualifies']
                 and x['metrics'].get('events_detected', {}).get('mean', 0) > 0]
    return (min(usable, key=_rank) if usable else None,
            sorted(items, key=lambda x: (not x['valid'], -x['metrics'].get('events_detected', {}).get('mean', -1), _complexity(x))),
            min(tradeoffs, key=_rank) if tradeoffs else None)


def _row_coverage(rows):
    out = defaultdict(lambda: {'attempts': 0, 'successes': 0, 'failures': 0})
    for r in rows:
        key = (r.get('stream', 'unknown'), r.get('method', 'unknown'), r.get('encoder', 'unknown'), r.get('variant', 'unknown'))
        out['/'.join(map(str, key))]['attempts'] += 1
        out['/'.join(map(str, key))]['failures' if r.get('status') == 'failed' else 'successes'] += 1
    return dict(sorted(out.items()))


def _trace_hash(trace):
    if not isinstance(trace, dict) or not trace.get('path'):
        raise ValueError('selected baseline row has no trace path')
    path = Path(trace['path'])
    if not path.is_file():
        raise ValueError(f'selected baseline trace missing: {path}')
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if trace.get('sha256') != digest or trace.get('bytes') != len(raw):
        raise ValueError(f'selected baseline trace hash mismatch: {path}')
    return {'path': str(path), 'sha256': digest, 'bytes': len(raw)}


def _freeze(rows, inputs, selected):
    if selected is None:
        raise ValueError('no qualifying MetroPT3 development candidate; baseline freeze unavailable')
    chosen = (selected['method'], selected['encoder'], selected['variant'],
              json.dumps(selected['config'], sort_keys=True, separators=(',', ':')))
    frozen = {}
    for stream in STREAMS:
        matches = [r for r in rows if r.get('stream') == stream and _key(r) == chosen and r.get('seed') == 0]
        successful = [r for r in matches if r.get('status') != 'failed' and isinstance(r.get('metrics'), dict)]
        if len(successful) != 1:
            raise ValueError(f'baseline freeze unavailable: expected one successful seed-0 row for {stream}, found {len(successful)}')
        r = successful[0]
        frozen[stream] = {'stream': stream, 'method': r['method'], 'encoder': r['encoder'],
                          'variant': r['variant'], 'config': r.get('config', {}), 'seed': 0,
                          'expected_seeds': selected['expected_seeds'],
                          'event_count': len(r.get('details', {}).get('events', [])) or selected.get('event_count'),
                          'trace': _trace_hash(r.get('trace')), 'metrics': r['metrics']}
    rule = ('Select only complete MetroPT3 configurations: all expected seeds, >=1 event detected per seed, '
            'false_alarm_per_week <= 1 and alarm_time_share_clean <= 0.02 per seed. Rank by mean event detections, '
            'then worst max(FA/week, 50*clean alarm share), mean window balanced accuracy, then lower persistence/complexity. '
            'Use fixed seed 0 and the identical method/encoder/variant/config on MetroPT2 Jan and Apr; never tune holdouts.')
    return {'schema': 1, 'selection_rule': rule,
            'selection_rule_sha256': hashlib.sha256(rule.encode()).hexdigest(), 'inputs': inputs,
            'selected_candidate': {k: selected[k] for k in ('method', 'encoder', 'variant', 'config', 'expected_seeds', 'event_count', 'metrics')},
            'selected': frozen}


def render(summary):
    lines = ['## Exploratory v3 follow-up', '',
             'Development-only selection on MetroPT3; MetroPT2 Jan/Apr are consumed exploratory replays, not confirmation. '
             'No deployment-safety or revenue guarantee is implied.', '',
             f"Rows: {summary['row_count']} total; {summary['failure_count']} failed; missing input files: {', '.join(summary['missing_inputs']) or 'none'}.",
             '', '| stream/method/encoder/variant | attempts | successes | failures |', '|---|---:|---:|---:|']
    for key, c in summary['coverage'].items():
        lines.append(f"| {key} | {c['attempts']} | {c['successes']} | {c['failures']} |")
    lines += ['', '### MetroPT3 development selection', '']
    best = summary.get('selected_candidate')
    if best:
        lines.append(f"Selected `{best['method']}/{best['encoder']}/{best['variant']}` ({best['event_count']} events; full seed-wise budget passed).")
    else:
        lines.append('No usable candidate: no complete configuration detected at least one event on every expected seed while staying within both false-alarm constraints.')
    trade = summary.get('best_recall_tradeoff')
    if trade:
        m = trade['metrics']
        lines.append(f"Best complete non-quiet recall tradeoff outside/against budget: `{trade['method']}/{trade['encoder']}/{trade['variant']}`; mean detected {m['events_detected']['mean']:.3g}, FA/week max {m.get('false_alarm_per_week', {}).get('max', float('nan')):.3g}, clean alarm-time max {m.get('alarm_time_share_clean', {}).get('max', float('nan')):.2%}. Not a qualifying candidate.")
    lines += ['', 'Metrics below report mean [min, max] across seeds; ranges describe encoder-seed variation, not independent-machine confidence intervals.', '']
    cols = ('events_detected', 'false_alarm_per_week', 'alarm_time_share_clean', 'alerts_per_week',
            'operator_events_per_week', 'unresolved_episodes_per_week',
            'operator_events_with_unresolved_per_week', 'window_precision',
            'window_sensitivity', 'window_balanced_accuracy')
    lines += ['| method/encoder/variant | complete | budget | ' + ' | '.join(cols) + ' |', '|' + '---|' * (len(cols) + 3)]
    for item in summary['development_candidates']:
        cells = []
        for col in cols:
            vals = item['metrics'].get(col)
            cells.append('n/a' if not vals else f"{vals['mean']:.3g} [{vals['min']:.3g}, {vals['max']:.3g}]")
        lines.append(f"| {item['method']}/{item['encoder']}/{item['variant']} | {item['complete']} | {item['qualifies']} | " + ' | '.join(cells) + ' |')
    for stream in ('metropt2_jan', 'metropt2_apr'):
        lines += ['', f'### {stream} (fixed development selection; no tuning)', '']
        matches = [r for r in summary['_rows'] if r.get('stream') == stream and best and _key(r) ==
                   (best['method'], best['encoder'], best['variant'], json.dumps(best['config'], sort_keys=True, separators=(',', ':')))]
        if not matches:
            lines.append('Selected method/config absent; holdout comparison unavailable.')
        else:
            for r in matches:
                if r.get('status') == 'failed':
                    lines.append(f"Seed {r.get('seed')}: FAILED ({r.get('error', 'unspecified failure')}).")
                elif r.get('metrics'):
                    m = r['metrics']
                    lines.append(f"Seed {r.get('seed')}: events {m.get('events_detected', 'n/a')}/{len(r.get('details', {}).get('events', []))}; FA/week {m.get('false_alarm_per_week', 'n/a')}; clean alarm {m.get('alarm_time_share_clean', 'n/a')}; precision {m.get('window_precision', 'n/a')}; recall {m.get('window_sensitivity', 'n/a')}; balanced accuracy {m.get('window_balanced_accuracy', 'n/a')}; alerts/operator/unresolved episodes per week {m.get('alerts_per_week', 'n/a')}/{m.get('operator_events_per_week', 'n/a')}/{m.get('unresolved_episodes_per_week', 'n/a')}.")
    for stream, groups in summary.get('holdout_summaries', {}).items():
        lines += ['', f'### {stream} ablations (posthoc descriptive comparisons)', '',
                  'These are not selected or tuned holdout results; they retain the normal-run variants for context.',
                  '', '| method/encoder/variant | complete | events | FA/week | clean alarm | precision | recall | balanced accuracy |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for item in groups:
            m = item['metrics']
            def fmt(name, pct=False):
                val = m.get(name, {}).get('mean')
                return 'n/a' if val is None else (f'{val:.2%}' if pct else f'{val:.3g}')
            lines.append(f"| {item['method']}/{item['encoder']}/{item['variant']} | {item['complete']} | {fmt('events_detected')} | {fmt('false_alarm_per_week')} | {fmt('alarm_time_share_clean', True)} | {fmt('window_precision')} | {fmt('window_sensitivity')} | {fmt('window_balanced_accuracy')} |")
    if summary.get('native_snapshots_present'):
        lines += ['', 'Native snapshot comparison data is present in `results/v3_native_snapshots.json`; it is archived context, not a new confirmation.']
    return '\n'.join(lines) + '\n'


def build_summary(rows, inputs, missing):
    selected, candidates, tradeoff = select_development(rows)
    holdout_summaries = {}
    for stream in ('metropt2_jan', 'metropt2_apr'):
        groups = defaultdict(list)
        for row in rows:
            if row.get('stream') == stream:
                groups[_key(row)].append(row)
        holdout_summaries[stream] = [
            _group_summary(key, group, _expected(group[0])) for key, group in groups.items()]
        holdout_summaries[stream].sort(
            key=lambda item: (item['method'] or '', item['encoder'] or '', item['variant'] or ''))
    return {'schema': 1, 'row_count': len(rows), 'failure_count': sum(r.get('status') == 'failed' for r in rows),
            'missing_inputs': missing, 'input_hashes': inputs, 'coverage': _row_coverage(rows),
            'selected_candidate': selected, 'best_recall_tradeoff': tradeoff,
            'development_candidates': candidates, 'holdout_summaries': holdout_summaries,
            'native_snapshots_present': Path('results/v3_native_snapshots.json').is_file(), '_rows': rows}


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physical', default=str(RESULT_FILES[0]))
    parser.add_argument('--cnn', default=str(RESULT_FILES[1]))
    parser.add_argument('--summary', default=str(SUMMARY_PATH))
    parser.add_argument('--freeze', action='store_true', help='write immutable normal baseline manifest; requires all three seed-0 traces')
    args = parser.parse_args(argv)
    rows, inputs, missing = load_results((Path(args.physical), Path(args.cnn)))
    summary = build_summary(rows, inputs, missing)
    output = {k: v for k, v in summary.items() if k != '_rows'}
    output['attempt_rows'] = rows
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False) + '\n')
    section = render(summary)
    print(section)
    if args.freeze:
        freeze_path = Path(FREEZE_PATH)
        if freeze_path.exists():
            raise FileExistsError(f'refusing to overwrite normal baseline freeze: {freeze_path}')
        if missing:
            raise ValueError(f'baseline freeze unavailable; result input files missing: {missing}')
        freeze = _freeze(rows, inputs, summary['selected_candidate'])
        freeze_path.parent.mkdir(parents=True, exist_ok=True)
        freeze_path.write_text(json.dumps(_json_safe(freeze), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
