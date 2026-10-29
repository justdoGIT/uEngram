# Reproducing the paper

Run commands from the repository root. Use Python 3.12 or newer (the project metadata supports Python 3.12–3.14). Install `uv`, then create the environment and install the base and research dependencies:

```bash
uv sync --extra cnn --extra baselines
```

This project configures the `cnn` extra to install the CPU PyTorch wheel. For a manual virtual environment, use Python 3.12+ and install the project plus `.[cnn,baselines]`. The optional `docs` extra is only needed to regenerate the results ledger and paper PDF. Some teacher experiments download pretrained MOMENT and Chronos models on first use; their adapters and dependencies are included in `cnn`.

## Public data placement

Commands below expect to run from the repository root. Keep public raw data outside version control; the scripts do not copy datasets into `results/`.

| Dataset | Expected local layout | Source recorded in this repository |
|---|---|---|
| NASA CMAPSS FD001 | `data/cmapss/train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt`; the standard archive can be fetched by the loader. | NASA C-MAPSS; source citation is in the paper references. |
| MetroPT-3 | Loader downloads/extracts the source CSV and caches it under `data/metropt3/`. | Davari et al., dataset DOI [10.24432/C5VW3R](https://doi.org/10.24432/C5VW3R); description DOI [10.1038/s41597-022-01877-3](https://doi.org/10.1038/s41597-022-01877-3). |
| MetroPT-2 | `data/metropt2/dataset_train.csv` (January recording) and `data/metropt2/MetroPT2.csv` (April recording). | [Zenodo 7766691](https://doi.org/10.5281/zenodo.7766691). |
| SKAB | `data/skab/anomaly-free/anomaly-free.csv` plus the repository's `data/` subdirectories and their CSVs; the loader can fetch the pinned files. | [SKAB repository](https://github.com/waico/SKAB). |
| OS-Metrics | Place `cpu_usage.csv`, `memory_usage.csv`, `memory_used.csv`, `network_response_time_a.csv`, `network_response_time_b.csv`, `network_response_time_diff.csv`, and `event_logs/event_logs_cpu.csv`, `event_logs/event_logs_memory.csv`, `event_logs/event_logs_network.csv` directly under one root directory and pass it as `--root`. | [Zenodo 17981352](https://doi.org/10.5281/zenodo.17981352). |
| SCANIA Component X | Root must contain `data/train_operational_readouts.csv`, `data/validation_operational_readouts.csv`, and `data/train_tte.csv`. | Source recorded in `docs/research-scania-validation.md`. |
| MFPT | Pass an archive containing the `MFPT Fault Data Sets/` directory as `--archive`, and the extracted/raw recordings root as `--raw-root`. | MFPT source and pinned archive are listed in `docs/research-neural-encoders.md`. |
| CWRU | `--raw-root` must contain the requested numbered `.mat` files (including the pinned healthy/fault subset used by the gate). | Case Western Reserve University Bearing Data Center, cited in the paper. |
| TSB-AD-U | Supply the dataset ZIP as `--archive`; required series are selected from that archive. | The dataset URL is recorded by the holdout benchmark: <https://www.thedatum.org/datasets/TSB-AD-U.zip>. |
| EngineAD | Not a local reproduction target; the committed artifact was produced on a Colab GPU by `scripts/colab_run_study.py`. | See the paper's citation to the EngineAD dataset. |

For OS-Metrics, MFPT, CWRU and TSB-AD-U the paths are explicit command-line inputs because raw downloads are not bundled. The loaders validate expected source files. Consult the respective dataset protocol docs for preprocessing and split details.

## Experiment commands and committed artifacts
Runtime is listed only where repository notes document a bound or measurement; otherwise it is **not documented**. “Regenerated at current code” refers only to the repository's audit record, not a promise that data or model downloads are identical forever. The d=36 synthetic run and SKAB dimension sweep are explicitly recorded as regenerated at HEAD; the other rows are committed historical outputs unless stated otherwise.

| Experiment | Command | Output in `results/` | Approx. runtime | Regenerated at current code? |
|---|---|---|---|---|
| Synthetic lifecycle, d=16 | `python -m bench.run_synth --seeds 0-9 --encoders dsp --out results/synth_runs.jsonl` | `synth_runs.jsonl` | Not documented | No (committed output; no current-code rerun recorded) |
| Synthetic lifecycle, d=36 | `python -m bench.run_synth --seeds 0-9 --encoders dsp_d36 --out results/synth_d36_runs.jsonl` | `synth_d36_runs.jsonl` | Not documented | Yes, recorded regenerated at HEAD |
| CMAPSS lifecycle | `python -m bench.run_cmapss --seeds 0-4 --encoders dsp,cnn --out results/cmapss_runs.jsonl` | `cmapss_runs.jsonl` | Not documented | No |
| MetroPT-3 lifecycle | `python -m bench.run_metropt --seeds 0-4 --encoders dsp,cnn --jobs 2 --out results/metropt_runs.jsonl` | `metropt_runs.jsonl` | Not documented; first data download is about 218 MB | Yes (October 2026 audit): the committed CNN-encoder row was stale relative to the current slot-admission code; regenerated at HEAD |
| MetroPT-2 confirmation (μEngram, half-space trees, MemStream) | `python -m bench.run_v2 --streams metropt2 --methods uengram:dsp,hst:dsp,memstream:dsp --seeds 0-4 --out results/v2_metropt2.jsonl` | `v2_metropt2.jsonl` | About 1 minute | Yes (October 2026 audit): regenerated at HEAD; event counts unchanged, the April false-alarm rate moved from 14.9 to 17.8 per week |
| MetroPT-3 confirmation (μEngram, half-space trees, MemStream, DSP encoder) | `python -m bench.run_v2 --streams metropt3 --methods uengram:dsp,hst:dsp,memstream:dsp --seeds 0-4 --out results/v2_development.jsonl` | `v2_development.jsonl` (DSP rows only; the file's CNN-encoder rows were not rechecked and are not cited in the paper) | About 1 minute for the DSP rows | Yes (October 2026 audit): μEngram's DSP row was stale, like the CNN row above; half-space trees and MemStream were unaffected |
| SKAB baselines (μEngram, half-space trees, MemStream) | `python -m bench.skab_baselines --methods uengram,hst,memstream --seeds 0,1,2,3,4 --out results/skab_baselines.json` | `skab_baselines.json` | About 30 seconds | Yes (October 2026 audit): new script and artifact |
| Synthetic bootstrap intervals | `python -m bench.synth_bootstrap --out results/synth_bootstrap_ci.json` | `synth_bootstrap_ci.json` | A few seconds | Yes (October 2026 audit): new script and artifact |
| Competitive-admission ablation | `python -m bench.admission_ablation --budgets 2,4,8,16,256 --seeds 0-4 --out results/admission_ablation.json` | `admission_ablation.json` | Not timed (minutes) | Yes (October 2026 audit): new script and artifact |
| Noise-whitening ablation | `python -m bench.whitening_ablation --seeds 0-9 --out results/whitening_ablation.json` | `whitening_ablation.json` | Not timed (minutes) | Yes (October 2026 audit): new script and artifact |
| Alarm component profile | `python -m bench.alarm_components` | `v8_component_profile.json` | About 25 seconds | Yes (October 2026 audit): the module reads a score cache under `~/.local/state/uengram/v2scores` that was stale relative to the current code; regenerated at HEAD. The cited MetroPT-3 pre-report AUROC values (0.07 health, 0.35 novelty) did not change beyond rounding. |
| SKAB frozen validation | `python -m bench.validate_skab --out results/skab_validation.json` | `skab_validation.json` | Not documented | No |
| SKAB embedding-dimension sweep | `python -m bench.skab_d_sweep --encoders dsp_d16,dsp_d24,dsp_d32,dsp_d48 --seeds 0 --out results/skab_d_sweep.json` | `skab_d_sweep.json` | Not documented | Yes, recorded regenerated at HEAD |
| SKAB d=48 seed check | `python -m bench.skab_d_sweep --encoders dsp_d48 --seeds 0,1,2,3,4,5,6,7,8,9 --out results/skab_d48_seeds.json` | `skab_d48_seeds.json` | Not documented | No |
| SCANIA frozen validation | `python -m bench.validate_scania --root /path/to/scania --out results/scania_validation.json --scores results/scania_scores.npz` | `scania_validation.json` (plus score cache) | Not documented | No |
| OS-Metrics specialist gate | `python -m bench.osmetrics_specialist_gate --root /path/to/osmetrics --out results/osmetrics_specialist_gate.json` | `osmetrics_specialist_gate.json` | Not documented | No |
| OS-Metrics role gate | `python -m bench.osmetrics_role_gate --root /path/to/osmetrics --out results/osmetrics_role_gate.json` | `osmetrics_role_gate.json` | Not documented | No |
| MFPT two-teacher gate | `python -m bench.mfpt_two_teacher_gate --archive /path/to/mfpt.zip --raw-root /path/to/mfpt --out results/mfpt_two_teacher_gate.json` | `mfpt_two_teacher_gate.json` | Not documented | No |
| CWRU two-teacher gate | `python -m bench.cwru_two_teacher_gate --raw-root /path/to/cwru --out results/cwru_two_teacher_gate.json` | `cwru_two_teacher_gate.json` | Not documented | No |
| CMAPSS two-teacher gate | `python -m bench.cmapss_two_teacher_gate --root data/cmapss --out results/cmapss_two_teacher_gate.json` | `cmapss_two_teacher_gate.json` | Not documented | No |
| SKAB teacher time gate | `python -m bench.skab_teacher_time_gate --root data/skab --out results/skab_teacher_time_gate.json` | `skab_teacher_time_gate.json` | Not documented (no enforced CPU cap exists in the code) | No |
| TSB-AD-U MOMENT holdout | `python -m bench.moment_tsb_ad_holdout --archive /path/to/TSB-AD-U.zip --out results/moment_tsb_ad_holdout.json --device cpu` | `moment_tsb_ad_holdout.json` | Not documented | No |
| CNN intermediate readouts | `python -m bench.cnn_stability --datasets skab,metropt3 --variants plain30,middle30,multiscale30 --seeds 0,1,2,3,4 --out results/cnn_intermediate_pipeline.json` | `cnn_intermediate_pipeline.json` | Model fitting totalled about 4 minutes on 4 CPU cores (October 2026 run) | Yes (October 2026 audit): 146 of 150 per-seed values identical, four differ by less than 4e-6 |
| Alarm-policy definition sweep | `python -m bench.alarm_definition_sweep sweep` | `v7_alarm_definition_sweep_metropt3.jsonl` | Not timed (minutes with 4 jobs) | Yes (October 2026 audit): regenerated at HEAD from the rebuilt score cache. Unlike the stale file, it contains one setting that meets all three operational targets; Section 5.5 reports it. |
| Error-gated alarm sweep (two-valued handling) | `python -m bench.error_gated_sweep sweep` | `v6_error_gated_sweep_metropt3.jsonl` | Not timed (minutes) | Yes (October 2026 audit): regenerated at HEAD; backs the 1.06% / 0.95 false-alarm figure in Section 5.5 |
| Calibrated-error coverage by operating state | `python -m bench.state_error_coverage --out results/state_error_coverage.json` | `state_error_coverage.json` | A few seconds | Yes (October 2026 audit): new script and artifact; backs the state-2 counts in Section 5.5 |
| Alarm-policy interaction sweep | `python -m bench.alarm_interaction sweep` | `v8_interaction_sweep_metropt3.jsonl` | Not documented | No; reads the same score cache as the sweeps above and was not rerun. It is not cited in the paper. |
| EngineAD GPU artifact | `python scripts/colab_run_study.py` (run in Colab GPU environment) | `enginead/cnn_seed_stability.json` | About 6.3 GB dataset; GPU required | No; not reproducible locally |

The alarm component and sweep modules do not expose `--out`; their committed filenames are shown explicitly. All other commands use their actual parser flags. Long lifecycle runs can be expensive; quick smoke options are not equivalent to the paper experiments.

## Paper tables and figures

| Paper item | Experiment above |
|---|---|
| Table 4 (synthetic controls), Section 5.1 and 5.2 | Synthetic lifecycle, d=16 and d=36; synthetic bootstrap intervals |
| Section 5.2 (width regressions) | Synthetic lifecycle d=36; SKAB embedding-dimension sweep |
| Table 5, CMAPSS row | CMAPSS lifecycle |
| Table 5, MetroPT-3 row | MetroPT-3 lifecycle; alarm component profile |
| Table 5, MetroPT-2 row | MetroPT-2 confirmation |
| Table 5, SKAB row | SKAB frozen validation |
| Table 5, SCANIA row | SCANIA frozen validation |
| Table 5, EngineAD row, Figure 5 | EngineAD GPU artifact (Colab) |
| Table 6 (SKAB width) | SKAB embedding-dimension sweep and d=48 seed check |
| Section 5.5 (alarm tuning) | Alarm-policy definition and interaction sweeps |
| Table 7 (CNN readouts) | CNN intermediate readouts |
| Section 5.7 items 1–5, Figure 6 | Distillation: MOMENT cross-domain and context studies (`bench.cnn_moment_distillation`, `bench.cnn_moment_context_study`), TSB-AD-U holdout, MFPT and CWRU gates, CMAPSS two-teacher gate, SKAB teacher time gate |
| Section 5.8, Figure 7B | SKAB sensor-fusion gate (`bench.skab_sensor_fusion_gate`) and role gate (`bench.role_skab_gate`) |
| Tables 8 and 9, Figure 7A | OS-Metrics specialist and role gates |
| Table 10 (external baselines), Section 5.10 | MetroPT-3 confirmation; SKAB baselines |
| Section 5.1, competitive admission ablation | Competitive-admission ablation |
| Section 5.4, noise-whitening ablation | Noise-whitening ablation |

**Pitfall for anyone writing a new ablation.** `bench.run_synth.main` always uses a `spawn` multiprocessing pool, even with `--jobs 1`. Spawned workers re-import their modules, so a monkeypatch applied in the parent process (for example to a config default) is silently ignored and both arms of the comparison run identical code. Call `bench.run_synth.run_one` in-process instead, as `bench.admission_ablation` and `bench.whitening_ablation` do.
