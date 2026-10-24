# uEngram

A reference implementation and benchmark of a deterministic, fixed-budget memory for embedded sensor streams ("Mem0 for sensors"). Routine windows are absorbed into running Gaussian statistics at no extra write cost, rare anomalies are pinned with their raw window, wear drift is tracked with checkpoints, repairs and new regimes are detected by a sequential step test (calibrated to the measured time correlation of the encoder's noise) and supersede the old state, and eviction is Ward-merge / importance-based inside a hard budget. Novelty is scored against normal states only, alarms use a threshold calibrated conformally on a fault-free commissioning run, and a frozen healthy reference from commissioning gives a health score that keeps growing with wear the adaptive memory accepts; a small lookup table fitted on run-to-failure data turns health into remaining useful life. The engine works on a whitened latent space in which within-state noise is decorrelated, so correlated sensors count once; encoders are a DSP+PCA encoder, the same with a learned thermal-lag residual (`ctx`), and a 1D-CNN autoencoder. It is benchmarked against naive append-only, FIFO, reservoir sampling and the engine from `docs/EdgeAIMemoryArchitecture.md` on a labelled synthetic stream and on NASA CMAPSS FD001, with multi-seed bootstrap confidence intervals.

- `docs/PLAN.md`: the original implementation plan.
- `docs/EXPLAINED.md`: every component in plain words and math, and where it lives in code.
- `docs/RESULTS.md`: benchmark tables and findings (v3; v1 and v2 kept in `results/v1/`, `results/v2/`).
- `docs/IMPROVEMENT-ANALYSIS.md`: the diagnostics that motivated v2; v3's are in `docs/EXPLAINED.md` sections 7c and 10c.
- `docs/research-*.md`: literature notes (machine-state mathematics, anomaly-detection evaluation, condition-monitoring features, CMAPSS practice).
- `docs/HOST-TELEMETRY.md`: uEngram on running Linux hosts (the Void cluster): the 1 Hz snapshot, the attribute schema with one owner per measured field, the per-signal mathematics, deployment, and cluster results; `docs/research-host-state.md` is its literature note.

## Environment

`/home` is small on the development machine, so the uv cache and venv live on `/home/kk/Backup`:

```bash
export UV_CACHE_DIR=/home/kk/Backup/.cache/uv UV_PROJECT_ENVIRONMENT=/home/kk/Backup/venvs/uengram
uv sync --extra cnn --extra baselines
```

Run everything from the repository root. Long runs are tee'd to `~/.local/state/uengram/`.

## Commands

```bash
mkdir -p ~/.local/state/uengram

# tests (cnn supplies PyTorch for CNN/MemStream; baselines supplies River/HST)
uv run pytest -q

# smoke runs
uv run python -m bench.run_synth --quick --out /tmp/uengram_smoke.jsonl
uv run python -m bench.run_cmapss --quick --out /tmp/uengram_cmapss_smoke.jsonl
uv run python -m bench.run_metropt --quick --out ~/.local/state/uengram/metropt_smoke.jsonl

# full synthetic benchmark: 10 seeds x 3 encoders x 10 methods
uv run python -m bench.run_synth --seeds 0-9 --encoders dsp,ctx,cnn --jobs 2 --out results/synth_runs.jsonl 2>&1 | tee ~/.local/state/uengram/synth.log

# budget sweep
uv run python -m bench.run_synth --seeds 0-4 --encoders dsp --sweep-budgets 64,128,256,512 --out results/synth_sweep.jsonl 2>&1 | tee ~/.local/state/uengram/sweep.log

# CMAPSS FD001 (first run downloads ~12 MB into data/cmapss/)
uv run python -m bench.run_cmapss --seeds 0-4 --encoders dsp,cnn --out results/cmapss_runs.jsonl 2>&1 | tee ~/.local/state/uengram/cmapss.log

# MetroPT-3: 5 seeds x 2 encoders x 8 methods; first run downloads ~218 MB
# The extracted data and cache require additional space (see EXPLAINED section 16).
uv run python -m bench.run_metropt --seeds 0-4 --encoders dsp,cnn --jobs 2 --out results/metropt_runs.jsonl 2>&1 | tee ~/.local/state/uengram/metropt.log

# tables
uv run python -m bench.report --out docs/RESULTS.md
```

Use `--jobs 2` as a conservative default. CARE workers materialize full datasets and embeddings; measure RAM headroom before increasing concurrency.

## Host telemetry (cluster nodes)

```bash
# probe all sources on this node (needs CAP_PERFMON + CAP_DAC_READ_SEARCH, see docs/HOST-TELEMETRY.md)
python3 -m uengram.host probe
# record (a single file, or daily spool files), commission, replay
python3 -m uengram.host record --spool ~/.local/state/uengram-host/spool
uv run python -m uengram.host commission --ticks ~/.local/state/uengram-host/spool/ticks-*.jsonl* --model model.pkl
uv run python -m uengram.host replay --model model.pkl --ticks more.jsonl --events events.jsonl
# on msi: follow every node's pushed state (the uengram-hub service), print the cluster view
python3 -m uengram.host hub --nodes msi-worker hp asus-worker lenovo-worker gigabyte-worker
python3 -m uengram.host cluster
# on msi, outside the campaign: measure a power/scheduler knob on one node's reference build
python3 -m uengram.host tune --node asus-worker --knob pl1 --values 40 30
# synthetic validation of the host pipeline
uv run python -m bench.run_host_synth --seeds 0-19 --jobs 2 --out results/host_synth.jsonl
# install the runit service on nodes (sudo via kepr); --campaign and --hub add the campaign runner and
# the hub (msi only); --recommission adds uengram-recommission (every node)
deploy/install-node.sh --numpy --recommission NODE...
# 14-day live validation campaign: record a clean week, commission on it, fault week, report
python3 -m uengram.host campaign plan --dir ~/Backup/uengram-host/campaign
python3 -m uengram.host campaign status --dir ~/Backup/uengram-host/campaign
```

## Layout

```
src/uengram/   library: alarm, baselines, baseline_reference, cmapss, encoders, encoder_{cnn,cnn2,cnn3,ctx,tab}, engine, health, ledger, lifecycle, metrics, metropt, metropt2, persistence, stats, store, stream_baselines, synth
src/uengram/host/  host telemetry: schema, perf, sources, collector, stream, hub, cluster, psi (stdlib); features, encoder, events, stepcal, node, daemon, synth (numpy); campaign, campaign_exec, tune, knob.sh, validate, validate_criteria, validate_variants, validate_signals, cli, cli_model
bench/         harness: advisory_{experiment,eval,report}, alarm_{components,definition,definition_sweep,dev,eval,interaction,math,sensitivity}, care_score, colab_train, common, cnn3_experiment, domain_dataset, encoding_cache, encoder_gate, error_gated_{alarm,sweep}, evaluate, followup_{experiment,policy}, freeze_frames, jev_{advisory,experiment}, laya_{advisory,hosts,http}, leak_duty, metropt_{calibration,eval}, modality_{alarm,sweep}, native_snapshots, physical_residuals, provenance, report, report_{followup,v2}, run_{cmapss,metropt,synth,v2}, run_host_synth, stream_care, streams, v2_scoring
deploy/        runit services (uengram-hostd, uengram-campaign, uengram-hub, uengram-recommission) and the node installer
tests/         unit and behaviour tests
results/       committed run outputs (JSONL)
docs/          plan, explainer, results, improvement analysis, research notes, architecture notes
```

## MetroPT-3 validation

The real compressor benchmark has a fixed commissioning/calibration/test split,
gap-aware windows, four reported leak intervals, detector baselines (Half-Space
Trees and a shared-feature MemStream core), and predeclared detection,
false-alarm and retention criteria. See [EXPLAINED section 16](docs/EXPLAINED.md#16-real-data-benchmark-metropt-3-protocol-fixed-before-any-result)
for the protocol and its training-only calibration amendment. This is one
machine with coarse reports, not a field-validation or revenue claim. Result:
uEngram (DSP) keeps all 4 reports in live provenance (P3 pass) but detects 3 of
4 and raises 14.1 false-alarm episodes per observed week (P1 and P2 fail); see
[RESULTS](docs/RESULTS.md#metropt-3-real-compressor-protocol-in-docsexplainedmd-section-16).

MetroPT-3 is distributed under CC BY 4.0: Davari et al., dataset
[DOI:10.24432/C5VW3R](https://doi.org/10.24432/C5VW3R), description
[DOI:10.1038/s41597-022-01877-3](https://doi.org/10.1038/s41597-022-01877-3).
The adapted MemStream core is based on
[Stream-AD/MemStream](https://github.com/Stream-AD/MemStream), Apache-2.0;
its license is included in `src/uengram/MemStream-LICENSE.txt`.

## Alarm-policy confirmation and GPU training

Protocol v2 separates novelty from actionable alarms, records input snapshots
around each episode, and measures standing-alarm time and advisory workload as
well as false-alarm openings. MetroPT-3 is now development data.
[EXPLAINED section 17](docs/EXPLAINED.md#17-alarm-policy-confirmation-protocol-v2)
freezes the MetroPT2 and CARE confirmation splits, configurations and criteria
before their detector scores are generated.

```bash
uv run python -m bench.run_v2 --streams metropt2 --methods uengram:dsp,hst:dsp,memstream:dsp --seeds 0-4 --jobs 2 --out results/v2_metropt2.jsonl
uv run python -m bench.run_v2 --streams care --methods uengram:tab,hst:tab,memstream:tab --seeds 0 --jobs 2 --out results/v2_care.jsonl
uv run python -m bench.report
```

Data must be downloaded into the layout documented in section 17. MetroPT2:
[Zenodo 7766691](https://doi.org/10.5281/zenodo.7766691), CC BY 4.0.
CARE: [Zenodo 15846963](https://doi.org/10.5281/zenodo.15846963), CC BY-SA 4.0.
The NumPy CARE event scorer adapts
[EnergyFaultDetector](https://github.com/AEFDI/EnergyFaultDetector), MIT;
its license is in `src/uengram/EnergyFaultDetector-LICENSE.txt`.
Retrospective CARE status annotations filter evaluation, never runtime alarms.

`bench.colab_train.prepare(stream, path)` exports a label-free input archive;
`bench.colab_train.train(input_path, output_dir)` trains CNN2 and exports gated
embeddings plus a CPU-device checkpoint. Colab CLI was used with a T4; observed
fit times and the restored-checkpoint check are recorded in
`results/encoder_v2_gate.json`. Sessions were stopped after downloading results.
Keep the preinstalled CUDA PyTorch on Colab rather than installing this
project's local CPU-wheel source. Use only authorized uploads and never load
an untrusted pickle checkpoint.

Faster training did not make CNN2 the better detector: its five development
seeds were less reliable than DSP. DSP was selected for compressor
confirmation; CARE uses tabular features, not invented waveforms.

**Confirmation rejected the v2 alarm policy for deployment:** it detected
none of the five published MetroPT2 failure intervals. Low critical-alarm
counts and preserved provenance do not establish useful fault warning.
CARE attempted all 95 datasets; six have no eligible evaluation rows, so no
full-cohort CARE score or confidence interval is reported.
See [RESULTS](docs/RESULTS.md) for the complete comparison and limitations.

## Follow-up experiments (exploratory, not confirmation)

Healthy-range and state-conditioned normalization, demand-conditioned
pressure/duty/thermal residuals, unresolved-state handling and native-source
snapshots were added after the v2 rejection. On the consumed recordings no
new configuration improves both recall and the clean-alarm-time budget at
once; the frozen baseline remains the selected fallback. Native snapshots
now retain original columns and fractional timestamps with explicit
causal/post-decision boundaries. The typed Jev advisory comparison is
preregistered but blocked pending OpenRouter credits or a TypeSafe key.
Details: [RESULTS](docs/RESULTS.md) and [EXPLAINED section 18](docs/EXPLAINED.md).

The separate English-base Laya comparison completed all 2,741 assessments,
using hp for remaining inference and msi as its client. No assessment met
the frozen escalation rule: preserving the baseline did not improve
detection, and Laya-only decisions lost the MetroPT3 detections. Use
`results/v3_laya_evaluated_summary.json` for corrected notification burden;
raw advisory outputs omitted baseline policy alerts. The measured hp
four-thread setting was faster than eight threads on the synthetic timing
check. See [the completed comparison](docs/RESULTS.md#laya-paired-comparison-completed-no-diagnostic-improvement)
and [reproduction commands](docs/EXPLAINED.md#laya-comparison-and-hp-inference).

A follow-up public-dataset audit found **no independent, compressor-matched,
asset-disjoint cohort with verified pointwise physical-fault labels**: the
MetroPT recordings are already consumed, and every other reviewed source is
a different system, lab-seeded, or has anonymized components. A
development-only weak-supervision export from the existing frozen traces
(2,503 cases, zero truncation against Laya's tokenizer) is available for
future fine-tuning experiments; it explicitly carries
`independent_evaluation_eligible: false`. See
[the dataset audit](docs/EXPLAINED.md#domain-dataset-readiness-v4-no-training)
and `results/v4_dataset_audit.json`.
