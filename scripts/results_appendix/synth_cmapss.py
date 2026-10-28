from pathlib import Path
import json
import math


def _load(repo: Path, name: str) -> list[dict]:
    with (repo / "results" / name).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _fmt(value, digits=3):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


def _mean(values):
    values = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return sum(values) / len(values) if values else float("nan")


def _summarize(records, keys, group):
    buckets = {}
    for record in records:
        buckets.setdefault(group(record), []).append(record)
    result = {}
    for key, rows in buckets.items():
        result[key] = {field: _mean([r["metrics"].get(field) for r in rows]) for field in keys}
        result[key]["n"] = len(rows)
    return result


def _table(title, outcome, source, columns, rows, note):
    return {"title": title, "outcome": outcome, "source": source,
            "columns": columns, "rows": rows, "note": note}


def build(repo: Path) -> list[dict]:
    tables = []

    # Main synthetic configuration result: repeated seeds are averaged.
    records = _load(repo, "synth_runs.jsonl")
    stats = _summarize(records, ["event_recall", "write_reduction_pct"],
                       lambda r: (r["encoder"], r["method"]))
    dsp_methods = ("naive", "fifo", "reservoir", "reference_labelfree", "uengram", "uengram_fixedvar", "uengram_nopromo")
    dsp_rows = [[method, _fmt(stats[("dsp", method)]["event_recall"]),
                 _fmt(stats[("dsp", method)]["write_reduction_pct"]), str(stats[("dsp", method)]["n"])]
                for method in dsp_methods]
    dsp_uengram = stats[("dsp", "uengram")]
    tables.append(_table("Synthetic DSP encoder: lifecycle against controls", "positive", "results/synth_runs.jsonl",
                         ["method", "event recall", "write reduction %", "seeds"], dsp_rows,
                         f"Across seeds, DSP uengram reaches event recall {_fmt(dsp_uengram['event_recall'])} with write reduction {_fmt(dsp_uengram['write_reduction_pct'])}%, while fifo and reservoir have lower event recall."))

    encoder_rows = [[encoder, _fmt(stats[(encoder, "uengram")]["event_recall"]),
                     _fmt(stats[(encoder, "uengram")]["write_reduction_pct"]), str(stats[(encoder, "uengram")]["n"])]
                    for encoder in ("dsp", "ctx", "cnn")]
    tables.append(_table("Synthetic encoder comparison (dsp, ctx, cnn)", "mixed", "results/synth_runs.jsonl",
                         ["encoder", "event recall", "write reduction %", "seeds"], encoder_rows,
                         f"uengram event recall is {_fmt(stats[('dsp', 'uengram')]['event_recall'])} for dsp, {_fmt(stats[('ctx', 'uengram')]['event_recall'])} for ctx, and {_fmt(stats[('cnn', 'uengram')]['event_recall'])} for cnn."))

    # Dimension-36 seed spread, retaining the main method and the failed promotion ablation.
    records = _load(repo, "synth_d36_runs.jsonl")
    methods = ("naive", "reservoir", "uengram", "uengram_nopromo")
    by_seed = _summarize(records, ["event_recall", "write_reduction_pct"], lambda r: (r["seed"], r["method"]))
    seeds = sorted({r["seed"] for r in records})
    rows = []
    for method in methods:
        vals = [by_seed[(seed, method)] for seed in seeds if (seed, method) in by_seed]
        er = [v["event_recall"] for v in vals]
        wr = [v["write_reduction_pct"] for v in vals]
        rows.append([method, str(len(vals)), f"{_fmt(_mean(er))} [{_fmt(min(er))}, {_fmt(max(er))}]",
                     f"{_fmt(_mean(wr))} [{_fmt(min(wr))}, {_fmt(max(wr))}]"])
    d16_uengram_recall = stats[("dsp", "uengram")]["event_recall"]
    tables.append(_table("Synthetic d=36 seed spread", "negative", "results/synth_d36_runs.jsonl, results/synth_runs.jsonl",
                         ["method", "seeds", "event recall mean [min, max]", "write reduction % mean [min, max]"], rows,
                         f"Across {len(seeds)} seeds, d=36 uengram mean event recall is {_fmt(_mean([by_seed[(s, 'uengram')]['event_recall'] for s in seeds if (s, 'uengram') in by_seed]))}, below the d=16 value of {_fmt(d16_uengram_recall)}."))


    # Slope feature ablation compared with the non-slope synthetic DSP encoder.
    base = _load(repo, "synth_runs.jsonl")
    slope = _load(repo, "synth_slope_runs.jsonl")
    combined = [("without_slope", r) for r in base if r["encoder"] == "dsp"] + [("with_slope", r) for r in slope]
    selected = ("naive", "reservoir", "uengram", "uengram_nopromo")
    groups = {}
    for variant, record in combined:
        if record["method"] in selected:
            groups.setdefault((variant, record["method"]), []).append(record)
    rows = [[variant, method, str(len(items)), _fmt(_mean([r["metrics"]["event_recall"] for r in items])),
             _fmt(_mean([r["metrics"]["write_reduction_pct"] for r in items]))]
            for (variant, method), items in sorted(groups.items())]
    tables.append(_table("Synthetic slope-feature ablation", "negative", "results/synth_runs.jsonl, results/synth_slope_runs.jsonl",
                         ["slope features", "method", "seeds", "event recall", "write reduction %"], rows,
                         f"The table compares matching DSP and DSP-slope runs; uengram event recall falls from {_fmt(_mean([r['metrics']['event_recall'] for v,r in combined if v=='without_slope' and r['method']=='uengram']))} without slope features to {_fmt(_mean([r['metrics']['event_recall'] for v,r in combined if v=='with_slope' and r['method']=='uengram']))} with them."))

    # Memory-budget sweep, concise selection across all budgets.
    records = _load(repo, "synth_sweep.jsonl")
    stats = _summarize(records, ["event_recall", "write_reduction_pct"], lambda r: (r["budget"], r["method"]))
    budgets = sorted({r["budget"] for r in records})
    keys = [(b, "uengram") for b in budgets]
    keys += [(b, "fifo") for b in (min(budgets), max(budgets))]
    keys += [(b, "reservoir") for b in (min(budgets), max(budgets))]
    rows = [[str(b), m, _fmt(stats[(b,m)]["event_recall"]), _fmt(stats[(b,m)]["write_reduction_pct"]), str(stats[(b,m)]["n"])] for b,m in keys]
    tables.append(_table("Synthetic memory-budget sweep", "positive", "results/synth_sweep.jsonl",
                         ["budget", "method", "event recall", "write reduction %", "seeds"], rows,
                         f"The table shows uengram at every budget and fifo/reservoir at the minimum and maximum budgets. uengram event recall ranges from {_fmt(min(stats[(b,'uengram')]['event_recall'] for b in budgets))} to {_fmt(max(stats[(b,'uengram')]['event_recall'] for b in budgets))}, above fifo at each displayed endpoint."))

    # CMAPSS by dimension: dimension identified by the encoder suffix in each file.
    files = [("d=16", "cmapss_runs.jsonl"), ("d=24", "cmapss_d24_runs.jsonl")]
    selected_methods = ("naive", "reservoir", "uengram")
    rows = []
    metric_keys = ["health_auroc", "health_ap", "rul_rmse", "rul_mae", "rul_rmse_health", "rul_mae_health"]
    for dim, filename in files:
        records = _load(repo, filename)
        for encoder in sorted({r["encoder"] for r in records}):
            for method in selected_methods:
                selected = [r for r in records if r["encoder"] == encoder and r["method"] == method]
                if not selected:
                    continue
                means = {k: _mean([r["metrics"].get(k) for r in selected]) for k in metric_keys}
                rows.append([dim, encoder, method, str(len(selected))] + [_fmt(means[k]) for k in metric_keys])
    tables.append(_table("CMAPSS health and RUL metrics by dimension", "mixed", "results/cmapss_runs.jsonl, results/cmapss_d24_runs.jsonl",
                         ["dimension", "encoder", "method", "seeds", "health AUROC", "health AP", "RUL RMSE", "RUL MAE", "RUL RMSE health", "RUL MAE health"], rows,
                         "The table reports seed means for health and RUL metrics by dimension and method; em dashes mark unavailable metrics in the source files."))

    # CMAPSS slope features versus ordinary DSP: run metrics are identical across seeds per method.
    base = _load(repo, "cmapss_runs.jsonl")
    slope = _load(repo, "cmapss_slope_runs.jsonl")
    rows = []
    for variant, dataset in (("without_slope", base), ("with_slope", slope)):
        encoder = "dsp" if variant == "without_slope" else "dsp_slope"
        for method in ("naive", "reservoir", "uengram", "uengram_frozen"):
            selected = [r for r in dataset if r["encoder"] == encoder and r["method"] == method]
            if selected:
                rows.append([variant, method, str(len(selected))] + [_fmt(_mean([r["metrics"].get(k) for r in selected])) for k in ("health_auroc", "rul_rmse", "rul_rmse_health", "write_reduction_pct")])
    tables.append(_table("CMAPSS slope-feature ablation", "mixed", "results/cmapss_runs.jsonl, results/cmapss_slope_runs.jsonl",
                         ["slope features", "method", "seeds", "health AUROC", "RUL RMSE", "RUL RMSE health", "write reduction %"], rows,
                         f"The table compares standard and slope DSP runs; uengram health AUROC changes from {_fmt(_mean([r['metrics']['health_auroc'] for r in base if r['encoder']=='dsp' and r['method']=='uengram']))} without slopes to {_fmt(_mean([r['metrics']['health_auroc'] for r in slope if r['method']=='uengram']))} with slopes, while the displayed RUL metrics also differ."))

    gate = json.loads((repo / "results" / "cmapss_two_teacher_gate.json").read_text())
    results = gate["results"]
    metric_names = ["event_recall_at_eol_horizon", "event_recall_before_failure", "window_auroc",
                    "window_average_precision", "pre_eol_alarms_per_100_healthy_cycles", "warning_cycles_before_eol_horizon_median"]
    display_names = ["event recall at EOL horizon", "event recall before failure", "window AUROC",
                     "window average precision", "pre-EOL alarms per 100 healthy cycles", "median warning cycles before EOL horizon"]
    rows = [[model] + [_fmt(results[model].get(metric)) for metric in metric_names]
            for model in ("baseline_cnn", "chronos_student", "moment_student", "two_teacher_student")]
    baseline_recall = results["baseline_cnn"]["event_recall_at_eol_horizon"]
    two_teacher_recall = results["two_teacher_student"]["event_recall_at_eol_horizon"]
    tables.append(_table("CMAPSS two-teacher lifecycle gate", "negative", "results/cmapss_two_teacher_gate.json",
                         ["model"] + display_names, rows,
                         f"The two-teacher student event recall at EOL horizon is {_fmt(two_teacher_recall)}, below baseline CNN at {_fmt(baseline_recall)}."))

    return tables
