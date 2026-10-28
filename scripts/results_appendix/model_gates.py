from __future__ import annotations

import json
import math
from pathlib import Path


def _load(repo: Path, name: str):
    return json.loads((repo / "results" / name).read_text())


def _f(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, float) and not math.isfinite(value):
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _table(title, outcome, source, columns, rows, note):
    return {"title": title, "outcome": outcome, "source": source,
            "columns": columns, "rows": rows, "note": note}



def build(repo: Path) -> list[dict]:
    out = []
    # CNN distillation smoke and intermediate-layer screening.
    d = _load(repo, "cnn_distillation_smoke.json")
    b, x = d["student"]["baseline"], d["student"]["distilled"]
    out.append(_table("CNN teacher distillation smoke", "positive", "results/cnn_distillation_smoke.json",
        ["student", "AUROC", "average precision", "fit seconds"],
        [["baseline", _f(b["auroc"]), _f(b["average_precision"]), _f(b["fit_seconds"])],
         ["distilled", _f(x["auroc"]), _f(x["average_precision"]), _f(x["fit_seconds"])]],
        f"A development smoke comparison over {b['test_windows']} test windows; teacher distillation raises both reported ranking metrics."))

    d = _load(repo, "cnn_intermediate_screen.json")
    rows = [[a["representation"], str(a["dimension"]), str(a["epochs"]), _f(a["median"]), _f(a["minimum"]), _f(a["maximum"])] for a in d["aggregates"]]
    best = max(d["aggregates"], key=lambda a: a["median"])
    out.append(_table("CNN intermediate-layer representation screen", "mixed", "results/cnn_intermediate_screen.json",
        ["representation", "dimension", "epochs", "median clean-to-fault displacement, 10th percentile", "minimum", "maximum"], rows,
        f"The screen uses label-free synthetic faults over {d['data']['selection_windows']} selection windows. The best aggregate median is {_f(best['median'])} for {best['representation']} at dimension {best['dimension']} and {best['epochs']} epochs."))

    for filename, title in [("cnn_intermediate_pipeline.json", "CNN intermediate pipeline and DSP fusion"),
                            ("cnn_stability_ablation.json", "CNN stability ablation")]:
        d = _load(repo, filename)
        rows = []
        for ds, data in d["datasets"].items():
            variants = {}
            for name, entry in data["components"].items():
                if name.startswith("dsp_"):
                    continue
                variant = name.rsplit("_seed", 1)[0]
                metrics = entry.get("native", {}).get("metrics", {})
                variants.setdefault(variant, []).append(metrics)
            for variant, metrics in variants.items():
                aurocs = [m["auroc"] for m in metrics if "auroc" in m]
                aps = [m["ap"] for m in metrics if "ap" in m]
                if aurocs and aps:
                    rows.append([ds, "single encoder", variant, str(len(metrics)),
                                 _f(sum(aurocs) / len(aurocs)), _f(sum(aps) / len(aps))])
            dsp = data["components"].get("dsp_d42" if ds == "metropt3" else "dsp_d48")
            if dsp:
                m = dsp.get("native", {}).get("metrics", {})
                rows.append([ds, "DSP baseline", "DSP", "1", _f(m.get("auroc")), _f(m.get("ap"))])
            fusion_metrics = [(name, entry.get("metrics", {})) for name, entry in data["fusions"].items()]
            fusion_metrics = [(name, m) for name, m in fusion_metrics if "auroc" in m and "ap" in m]
            if fusion_metrics:
                for kind, pair in (("fusion best", max(fusion_metrics, key=lambda item: item[1]["auroc"])),
                                   ("fusion worst", min(fusion_metrics, key=lambda item: item[1]["auroc"]))):
                    name, m = pair
                    rows.append([ds, kind, name, "—", _f(m["auroc"]), _f(m["ap"])])
        out.append(_table(title, "mixed", f"results/{filename}",
            ["dataset", "kind", "variant", "seed records", "mean AUROC", "mean AP"], rows[:12],
            f"Component rows show mean scores across seeds, while fusion rows show the highest and lowest AUROC per dataset. The table covers {min(len(rows), 12)} compact results."))

    # MOMENT policy comparison; retain baseline and every policy/configuration.
    d = _load(repo, "cnn_moment_context_study.json")
    rows = [["CNN baseline", _f(d["baseline"]["auroc"]), _f(d["baseline"]["average_precision"]), "—", "—"]]
    for v in d["variants"]:
        rows.append([f"{v['context_policy']} / stats={v['include_instance_stats']}", _f(v["student"]["auroc"]), _f(v["student"]["average_precision"]), str(v["teacher_feature_dim"]), _f(v["fit_seconds"])])
    out.append(_table("MOMENT context-policy study", "negative", "results/cnn_moment_context_study.json",
        ["policy", "AUROC", "average precision", "teacher features", "fit seconds"], rows,
        f"Controlled synthetic context-policy study with {d['data']['probe_windows']} probe windows; all MOMENT variants score below the CNN baseline AUROC of {_f(d['baseline']['auroc'])}."))

    d = _load(repo, "cnn_moment_distillation_crossdomain.json")
    s = d["student"]
    out.append(_table("MOMENT distillation cross-domain transfer", "mixed", "results/cnn_moment_distillation_crossdomain.json",
        ["student", "AUROC", "average precision", "fit seconds", "writes"],
        [[k, _f(s[k]["auroc"]), _f(s[k]["average_precision"]), _f(s[k]["fit_seconds"]), str(s[k]["writes"])] for k in ("baseline", "distilled")],
        f"Development transfer comparison on {s['baseline']['test_windows']} windows; distillation changes AUROC from {_f(s['baseline']['auroc'])} to {_f(s['distilled']['auroc'])} and average precision from {_f(s['baseline']['average_precision'])} to {_f(s['distilled']['average_precision'])}."))

    d = _load(repo, "cnn_moment_distillation_metropt_stats.json")
    s = d["student"]
    out.append(_table("MOMENT instance-statistics distillation check", "negative", "results/cnn_moment_distillation_metropt_stats.json",
        ["student", "AUROC", "average precision", "fit seconds", "writes"],
        [[k, _f(s[k]["auroc"]), _f(s[k]["average_precision"]), _f(s[k]["fit_seconds"]), str(s[k]["writes"])] for k in ("baseline", "distilled")],
        f"MetroPT-3 development transfer check on {s['baseline']['test_windows']} windows; adding the instance-statistics MOMENT teacher feature path lowers both AUROC and average precision."))

    # Recording-level teacher gates preserve each comparator and gate failure signal.
    for fname, dataset in [("cwru_two_teacher_gate.json", "CWRU"), ("mfpt_two_teacher_gate.json", "MFPT")]:
        d = _load(repo, fname)
        rows = []
        for name, r in d["results"].items():
            rows.append([name, _f(r["file_auroc"]), _f(r["window_auroc"]), _f(r["positive_file_recall"]), _f(r["normal_file_false_alarm_rate"]), str(r["test_windows"])])
        out.append(_table(f"{dataset} two-teacher recording gate (normal recording flagged: yes)", "mixed", f"results/{fname}",
            ["student", "file AUROC", "window AUROC", "fault recall", "normal false-alarm rate", "test windows"], rows,
            f"On the {dataset} recording holdout, the CNN, each teacher, and their combination flag the sole normal test recording."))

    # Encoder v2 name is not the excluded v2_ prefix.
    d = _load(repo, "encoder_v2_gate.json")
    rows = []
    for encoder in sorted({r["encoder"] for r in d["cpu_gate_history"]}):
        entries = [r for r in d["cpu_gate_history"] if r["encoder"] == encoder]
        failures = sum(not r["gate_passed"] for r in entries)
        rows.append(["historical CPU", encoder, f"{len(entries) - failures}/{len(entries)}",
                     str(failures), "—", "gate-history aggregate"])
    for r in d["gpu_analog"]:
        rows.append(["GPU analog", r["name"], f"{int(bool(r['accepted']))}/1",
                     "0" if r["accepted"] else "1", _f(r.get("mean_member_cka")),
                     "seed " + str(r["seed"])])
    out.append(_table("Encoder v2 gate history", "mixed", "results/encoder_v2_gate.json",
        ["record", "encoder", "accepted/total", "failed", "mean member CKA", "detail"], rows,
        f"Included despite its v2 name because encoder_v2_gate.json does not use the excluded top-level v2_ prefix; GPU analog accepts {sum(bool(r['accepted']) for r in d['gpu_analog'])}/{len(d['gpu_analog'])} seeds, and failures are retained by CPU encoder and GPU seed."))

    # Candidate screens: aggregate event detections and false-alarm count for every candidate.
    for fname, ds in [("model_research_candidates_metropt3_q10.json", "MetroPT-3"), ("model_research_candidates_skab_q10.json", "SKAB")]:
        d = _load(repo, fname)
        rows = []
        for name, entry in d["evaluation"].items():
            p = entry.get("native_frozen_policy", entry)
            events = p.get("events_detected", [])
            if isinstance(events, list):
                events = len(events)
            m = p.get("metrics", {})
            rows.append([name, _f(events, 0), _f(m.get("auroc")), _f(m.get("ap")), _f(p.get("false_alarm_episodes", m.get("false_alarm_episodes")), 0)])
        out.append(_table(f"{ds} q10 model-research candidates", "mixed", f"results/{fname}",
            ["candidate", "detected events", "AUROC", "AP", "false-alarm episodes"], rows,
            f"Consumed development candidate screen using synthetic challenges at q={_f(d['selection']['synthetic_challenges_quantile'])}; real test labels were not used for selection and all listed candidates are retained."))

    d = _load(repo, "moment_tsb_ad_holdout.json")
    rows = [["CNN baseline", _f(d["baseline"]["overall"]["auroc"]), _f(d["baseline"]["overall"]["average_precision"]), _f(d["baseline"]["fit_seconds"])]]
    for v in d["variants"]:
        m = v["scores"]["overall"]
        rows.append([f"{v['context_policy']} / stats={v['include_instance_stats']}", _f(m["auroc"]), _f(m["average_precision"]), _f(v["fit_seconds"])])
    out.append(_table("MOMENT TSB-AD-U cross-domain holdout", "null", "results/moment_tsb_ad_holdout.json",
        ["model", "AUROC", "average precision", "fit seconds"], rows,
        f"The frozen holdout covers {d['dataset']['test_windows']} windows from {len(d['dataset']['series'])} series. AUROC values of {_f(d['baseline']['overall']['auroc'])}, {_f(d['variants'][0]['scores']['overall']['auroc'])}, and {_f(d['variants'][1]['scores']['overall']['auroc'])} show no distinguishable change."))

    d = _load(repo, "or_rule_development.json")
    rows = []
    for name in ("and", "or"):
        r, m = d[name], d[name]["metrics"]
        rows.append([name.upper(), _f(m["events_detected"], 0), _f(m["false_alarm_per_week"]), _f(m["alarm_time_share_clean"]), _f(m["lead_hours"])])
    out.append(_table("OR-rule development comparison", "mixed", "results/or_rule_development.json",
        ["combine rule", "events detected", "false alarms/week", "clean alarm share", "lead hours"], rows,
        f"The OR rule detects {_f(d['or']['metrics']['events_detected'], 0)} events but incurs {_f(d['or']['metrics']['false_alarm_per_week'])} false alarms/week; AND detects {_f(d['and']['metrics']['events_detected'], 0)} with {_f(d['and']['metrics']['false_alarm_per_week'])} false alarms/week."))

    d = _load(repo, "osmetrics_role_gate.json")
    rows = [["DSP baseline", _f(d["baseline_dsp_d16"]["window_auroc"]), _f(d["baseline_dsp_d16"]["strict_event_recall"]), _f(d["baseline_dsp_d16"]["two_hour_event_recall"]), _f(d["baseline_dsp_d16"]["false_alarm_episodes_per_week"])]]
    for name, r in d["role_architecture"]["policies"].items():
        rows.append([name, _f(r["window_auroc"]), _f(r["strict_event_recall"]), _f(r["two_hour_event_recall"]), _f(r["false_alarm_episodes_per_week"])])
    out.append(_table("OSMetrics role-architecture gate", "negative", "results/osmetrics_role_gate.json",
        ["policy", "window AUROC", "strict event recall", "two-hour event recall", "false alarms per week"], rows,
        f"No role policy beats the DSP baseline AUROC of {_f(d['baseline_dsp_d16']['window_auroc'])} or its false-alarm rate of {_f(d['baseline_dsp_d16']['false_alarm_episodes_per_week'])} per week on this workload."))

    d = _load(repo, "osmetrics_specialist_gate.json")
    rows = [["DSP baseline", _f(d["baseline_dsp_d16"]["window_auroc"]), _f(d["baseline_dsp_d16"]["strict_event_recall"]), _f(d["baseline_dsp_d16"]["two_hour_event_recall"]), _f(d["baseline_dsp_d16"]["false_alarm_episodes_per_week"])]]
    for name, r in d["domain_specialists"]["policies"].items():
        rows.append([name, _f(r["window_auroc"]), _f(r["strict_event_recall"]), _f(r["two_hour_event_recall"]), _f(r["false_alarm_episodes_per_week"])])
    out.append(_table("OSMetrics domain-specialist gate", "mixed", "results/osmetrics_specialist_gate.json",
        ["policy", "window AUROC", "strict event recall", "two-hour event recall", "false alarms per week"], rows[:12],
        f"Across {d['protocol']['events_total']} events in one workload, domain-specialist policies vary in event recall and false-alarm rate."))

    d = _load(repo, "role_skab_gate.json")
    rows = [[name, _f(r["window_auroc"]), _f(r["window_average_precision"]), _f(r["strict_event_recall"]), f"{r['events_detected']}/{r['events_total']}", _f(r["false_alarms_per_hour"])] for name, r in d["policies"].items()]
    out.append(_table("SKAB role gate", "negative", "results/role_skab_gate.json",
        ["policy", "window AUROC", "window average precision", "strict event recall", "events", "false alarms per hour"], rows,
        f"On the fixed SKAB split with {d['same_split']['test_windows']} test windows, all policies have window AUROC {_f(next(iter(d['policies'].values()))['window_auroc'])} and strict event recall remains low."))
    return out
