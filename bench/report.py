"""Generate docs/RESULTS.md tables: python -m bench.report --out docs/RESULTS.md"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from uengram.metrics import bootstrap_ci, paired_bootstrap_diff

MARK = "<!-- FINDINGS BELOW: hand-written, preserved by bench.report -->"
SYNTH_COLS = ["items_end", "writes", "write_reduction_pct", "event_recall", "seizure_recall",
              "type_recall", "auroc", "ap", "f1", "false_flag_rate_era0", "false_flag_rate_era1",
              "stale_at1", "type_prec_at1", "supersede_delay_maint", "mmd2_current", "lat_p99_us"]
CMAPSS_COLS = ["items_end", "writes", "auroc", "ap", "health_auroc", "health_ap", "rul_rmse",
               "rul_rmse_health", "eol_retention", "mmd2_current"]
PAIRED = ["event_recall", "auroc", "ap", "false_flag_rate_era1", "stale_at1", "mmd2_current", "writes"]
SWEEP_COLS = ["items_end", "writes", "event_recall", "seizure_recall", "auroc", "stale_at1", "mmd2_current"]
PREV, CUR = "v2", "v3"  # the previous benchmark lives in results/<PREV>/
PREV_SYNTH = ["auroc", "ap", "f1", "event_recall", "seizure_recall", "false_flag_rate_era0",
              "false_flag_rate_era1", "stale_at1", "type_prec_at1", "supersede_delay_maint",
              "write_reduction_pct", "items_end", "mmd2_current"]
PREV_CMAPSS = ["auroc", "ap", "rul_rmse", "eol_retention", "items_end", "writes", "mmd2_current"]
ORACLE_NOTE = ("`reference_oracle` receives ground-truth anomaly labels as its severity input. It is "
               "an upper bound for the doc's engine, not a deployable configuration.")


def load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def fmt(x):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.0f}" if abs(x) >= 1000 else f"{x:.3g}"


def cell(vals, method, metric):
    m, lo, hi = bootstrap_ci(vals)
    if not np.isfinite(m):
        return "n/a (sphere space)" if (metric.startswith("mmd2") and method.startswith("reference")) else "n/a"
    return f"{fmt(m)} [{fmt(lo)}, {fmt(hi)}]"


def group(rows):
    """{(dataset, encoder): {method: {metric: [values by seed]}}}, plus seeds per method."""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in rows:
        for k, v in r["metrics"].items():
            out[(r["dataset"], r["encoder"])][r["method"]][k][r["seed"]] = v
    return out


def table(methods_metrics, cols, order):
    lines = ["| method | n | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for meth in order:
        if meth not in methods_metrics:
            continue
        mm = methods_metrics[meth]
        n = len(next(iter(mm.values())))
        cells = [cell(list(mm[c].values()), meth, c) if c in mm else "n/a" for c in cols]
        lines.append(f"| {meth} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def paired_table(mm, order):
    ref = "uengram"
    lines = ["| uengram minus | " + " | ".join(PAIRED) + " |", "|" + "---|" * (len(PAIRED) + 1)]
    for meth in order:
        if meth == ref or meth not in mm or ref not in mm:
            continue
        cells = []
        for c in PAIRED:
            seeds = sorted(set(mm[ref][c]) & set(mm[meth][c]))
            a = [mm[ref][c][s] for s in seeds]
            b = [mm[meth][c][s] for s in seeds]
            d, lo, hi = paired_bootstrap_diff(a, b)
            if not np.isfinite(d):
                cells.append("n/a")
                continue
            star = "*" if (lo > 0 or hi < 0) else ""
            cells.append(f"{fmt(d)} [{fmt(lo)}, {fmt(hi)}]{star}")
        lines.append(f"| {meth} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def sweep_table(rows):
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in rows:
        for k, v in r["metrics"].items():
            by[(r["method"], r["budget"])][k][r["seed"]] = v
    lines = ["| method | budget | n | " + " | ".join(SWEEP_COLS) + " |", "|" + "---|" * (len(SWEEP_COLS) + 3)]
    for (meth, b) in sorted(by, key=lambda x: (x[0], x[1])):
        mm = by[(meth, b)]
        n = len(mm["items_end"])
        lines.append(f"| {meth} | {b} | {n} | " + " | ".join(cell(list(mm[c].values()), meth, c) for c in SWEEP_COLS) + " |")
    return "\n".join(lines)


def prev_table(prev_rows, cur_rows, enc, metrics, method="uengram"):
    """uEngram PREV vs CUR on the same seeds; diff = CUR - PREV."""
    g1, g2 = group(prev_rows), group(cur_rows)
    keys = [k for k in g2 if k[1] == enc and k in g1]
    if not keys or method not in g1[keys[0]] or method not in g2[keys[0]]:
        return None
    m1, m2 = g1[keys[0]][method], g2[keys[0]][method]
    lines = [f"| metric | {PREV} | {CUR} | {CUR} - {PREV} |", "|---|---|---|---|"]
    for c in metrics:
        if c not in m1 or c not in m2:
            continue
        seeds = sorted(set(m1[c]) & set(m2[c]))
        a, b = [m2[c][s] for s in seeds], [m1[c][s] for s in seeds]
        d, lo, hi = paired_bootstrap_diff(a, b)
        star = "*" if np.isfinite(d) and (lo > 0 or hi < 0) else ""
        diff = f"{fmt(d)} [{fmt(lo)}, {fmt(hi)}]{star}" if np.isfinite(d) else "n/a"
        lines.append(f"| {c} | {cell(b, method, c)} | {cell(a, method, c)} | {diff} |")
    return "\n".join(lines)


def build(runs_dir: str) -> str:
    synth, cmapss = load(f"{runs_dir}/synth_runs.jsonl"), load(f"{runs_dir}/cmapss_runs.jsonl")
    sweep = load(f"{runs_dir}/synth_sweep.jsonl")
    prev_synth = load(f"{runs_dir}/{PREV}/synth_runs.jsonl")
    prev_cmapss = load(f"{runs_dir}/{PREV}/cmapss_runs.jsonl")
    out = ["# Results", "",
           f"Generated by `bench.report` from `results/*.jsonl` (engine {CUR}; earlier benchmarks are kept "
           "in `results/v1/` and `results/v2/`). Cells are `mean [95% bootstrap CI over seeds]`, 3 "
           "significant figures. `n` is the number of seeds. All bounded methods use B=256 items "
           "(uEngram: 192 semantic + 64 episodic) except in the budget sweep.", "",
           "uEngram v2: novelty scored against normal (semantic) states only; candidate probation 5-of-8, "
           "episodic promotion after a sustained dwell (20-of-30); alarm threshold calibrated conformally "
           "(alpha = 0.01) on a separate fault-free 4,000-window commissioning run; MEWMA step detector "
           "with supersession; encoders noise-whitened with a robust full noise covariance. Encoders: "
           "`dsp` spectral features, `ctx` spectral features with a learned thermal-lag residual, `cnn` "
           "autoencoder latent. `alpha_gate = 1e-4` as in v1.", "",
           "uEngram v3 adds: the step detector's input is scaled by the EWMA variance inflation measured "
           "on the encoder's training latents (within segments free of abrupt changes), so correlated "
           "within-state noise no longer reads as steps; on CMAPSS, a frozen healthy reference from a "
           "commissioning stream (first 30 windows of each training unit) gives a health score "
           "(`health_auroc`, `health_ap`).", "", ORACLE_NOTE, ""]
    syn_order = ["naive", "fifo", "reservoir", "reference_labelfree", "reference_oracle", "uengram",
                 "uengram_fixedvar", "uengram_nopromo", "uengram_frozen", "uengram_nostep"]
    cm_order = ["naive", "fifo", "reservoir", "reference_labelfree", "uengram", "uengram_frozen"]
    for title, rows, cols, order in (("Synthetic stream", synth, SYNTH_COLS, syn_order),
                                     ("CMAPSS FD001", cmapss, CMAPSS_COLS, cm_order)):
        g = group(rows)
        for (ds, enc) in sorted(g):
            out += [f"## {title}, encoder `{enc}`", "", table(g[(ds, enc)], cols, order), "",
                    f"### Paired differences ({title}, `{enc}`)", "",
                    "`*` marks a 95% paired-bootstrap CI that excludes 0.", "",
                    paired_table(g[(ds, enc)], order), ""]
        if not g:
            out += [f"## {title}", "", "not run", ""]
    out += ["## Budget sweep (synthetic, DSP encoder)", "", sweep_table(sweep) if sweep else "not run", ""]
    for title, prev, cur, metrics in (("synthetic", prev_synth, synth, PREV_SYNTH),
                                      ("CMAPSS", prev_cmapss, cmapss, PREV_CMAPSS)):
        for enc in ("dsp", "ctx", "cnn"):
            tab = prev_table(prev, cur, enc, metrics)
            if tab:
                out += [f"## uEngram {PREV} vs {CUR} ({title}, `{enc}`)", "",
                        "Same seeds; `*` marks a paired 95% CI that excludes 0.", "", tab, ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/RESULTS.md")
    ap.add_argument("--runs-dir", default="results")
    a = ap.parse_args()
    body = build(a.runs_dir)
    tail = "## Findings\n\n(to be written from the tables above)\n"
    if os.path.exists(a.out):
        old = open(a.out).read()
        if MARK in old:
            tail = old.split(MARK, 1)[1].lstrip("\n")
    with open(a.out, "w") as f:
        f.write(body + "\n" + MARK + "\n\n" + tail)


if __name__ == "__main__":
    main()
