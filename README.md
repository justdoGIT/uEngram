# uEngram

A reference implementation and benchmark of a deterministic, fixed-budget memory for embedded sensor streams ("Mem0 for sensors"). Routine windows are absorbed into running Gaussian statistics at no extra write cost, rare anomalies are pinned with their raw window, wear drift is tracked with checkpoints, repairs and new regimes are detected by a sequential step test (calibrated to the measured time correlation of the encoder's noise) and supersede the old state, and eviction is Ward-merge / importance-based inside a hard budget. Novelty is scored against normal states only, alarms use a threshold calibrated conformally on a fault-free commissioning run, and a frozen healthy reference from commissioning gives a health score that keeps growing with wear the adaptive memory accepts; a small lookup table fitted on run-to-failure data turns health into remaining useful life. The engine works on a whitened latent space in which within-state noise is decorrelated, so correlated sensors count once; encoders are a DSP+PCA encoder, the same with a learned thermal-lag residual (`ctx`), and a 1D-CNN autoencoder. It is benchmarked against naive append-only, FIFO, reservoir sampling and the engine from `docs/EdgeAIMemoryArchitecture.md` on a labelled synthetic stream and on NASA CMAPSS FD001, with multi-seed bootstrap confidence intervals.

- `docs/PLAN.md`: the original implementation plan.
- `docs/EXPLAINED.md`: every component in plain words and math, and where it lives in code.
- `docs/RESULTS.md`: benchmark tables and findings (v3; v1 and v2 kept in `results/v1/`, `results/v2/`).
- `docs/IMPROVEMENT-ANALYSIS.md`: the diagnostics that motivated v2; v3's are in `docs/EXPLAINED.md` sections 7c and 10c.
- `docs/research-*.md`: literature notes (machine-state mathematics, anomaly-detection evaluation, condition-monitoring features, CMAPSS practice).

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

## Layout

```
src/uengram/   library: alarm, baselines, baseline_reference, cmapss, encoders, encoder_{cnn,cnn2,cnn3,ctx,tab}, engine, health, ledger, lifecycle, metrics, metropt, metropt2, persistence, skab, stats, store, stream_baselines, synth
bench/         harness: advisory_{experiment,eval,report}, alarm_{components,definition,definition_sweep,dev,eval,interaction,math,sensitivity}, care_score, colab_train, common, cnn3_experiment, domain_dataset, encoding_cache, encoder_gate, error_gated_{alarm,sweep}, evaluate, followup_{experiment,policy}, freeze_frames, jev_{advisory,experiment}, laya_{advisory,hosts,http}, leak_duty, metropt_{calibration,eval}, modality_{alarm,sweep}, native_snapshots, physical_residuals, provenance, report, report_{followup,v2}, run_{cmapss,metropt,synth,v2}, stream_{care,scania,skab}, streams, v2_scoring, validate_{scania,skab}
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

## Current model decision and validation boundary

The selected compressor fallback remains `uengram/dsp/v2_frozen`: the
deterministic DSP encoder with the frozen state-aware alarm policy. The v8
component and interaction sweeps tested 1,728 alarm configurations; none met
the joint development target of 4/4 events, at most 2% clean alarm time, and
at most one false-alarm episode per week. The selection therefore falls back
to the frozen candidate rather than claiming a tuning win. See
`results/v8_interaction_selection.json` and
`docs/research-remediation-plan.md`.

The optional `AlarmLogic(combine="or", error_k=3)` path is an evidence
integration mode, not a new encoder or an autonomous replacement. On
MetroPT-3 it detected 2/4 events while producing 1.217 false alarms per week
and 15.15% clean alarm time. The SCANIA run is an independent fleet
validation of the frozen detector, but its coarse repair classes do not
support compressor lead-time claims: AND/control reached 22.794% recall for
classes 1--4 with a 19.287% class-0 alarm rate. The missingness-based OR
tripwire is data-quality monitoring, not a failure detector.

SKAB (`https://github.com/waico/SKAB`, GPL-3.0) is a different fresh
validation: a water-circulation testbed, not a compressor and not a
multi-asset cohort, but one with real, independently verified pointwise
fault timestamps that MetroPT/SCANIA/CARE cannot provide. The byte-identical
frozen detector and policy detected 5/34 labelled faults (14.7%), all
vibration-forcing faults, 28-114 seconds after onset (never before); every
valve-throttling and non-vibration fault (24/29) went undetected. Row-level
novelty AUROC was 0.550. A follow-up ablation (`DSPSlopeEncoder`, adding
each channel's windowed-mean slope) rescued exactly the three gradual
non-vibration faults it targeted (8/34, zero regressions on SKAB) at
roughly double the false-alarm episodes; an opt-in two-tier alarm
architecture (`shock_mult` in `AlarmConfig`) decouples immediate 1-window
shocks from persistent drift ($k=3$), rescuing valve-throttling faults (up
to 15/34 detected at `shock_mult=1.2x`) on assets with transient dynamics.
Revalidated project-wide (synthetic 10 seeds, CMAPSS 5 seeds, MetroPT-3 5
seeds, same alarm policy): `DSPSlopeEncoder` is **rejected for general
adoption.** It is a severe regression on the synthetic benchmark
(event/seizure recall collapse to 0.23/0.00, AP 0.995 -> 0.068) and a mixed,
not clearly beneficial result on CMAPSS and MetroPT-3. No single encoder
here is a universal anomaly/wear-and-tear standard; the right features are
measurably domain- and fault-specific. See `docs/research-skab-validation.md`.

A follow-up found a simpler fix for SKAB than any feature engineering:
plain `DSPEncoder(d=48)` (full raw dimension, no PCA reduction) rescues
the same four gradual faults `DSPSlopeEncoder`/`CtxDSPEncoder` do, at the
baseline's exact false-alarm count (5, not 10+). Raising `d` project-wide
is not adopted either: it's a consistent win on MetroPT-3 (AUROC/AP up,
false-alarm rate roughly halved, every method) but a regression on the
synthetic benchmark (event_recall 1.00 -> 0.36) and hits a real numerical
ceiling on CMAPSS (`d>24` fails: near-duplicate band-energy features make
the noise-covariance estimate ill-conditioned). The matrix is now complete
across all seven benchmarks: full dimension is also a strict win on
MetroPT2 (`d=42`, matching MetroPT-3), a marginal recall-for-precision
trade on CARE (`tab_d32`), and a recall-collapse trap on SCANIA (higher
`d` lifts novelty AUROC to 0.571 but unregularized noise whitening
inflates minor PCA directions, pushing calibration thresholds to 772M;
resolved via `TabularEncoder(rcond=1e-4)`). Separately,
widening and longer-training the plain `CNNEncoder` (not the
already-rejected CNN2/CNN3) closes most of its gap to DSP on MetroPT-3 in
a single-seed test; CNN2 hits its own noise-covariance wall at the same
widening. See `docs/research-encoder-capacity.md` for all three
investigations plus the full per-dataset matrix (section 4),
`docs/research-neural-encoders.md` for the neural latent stabilization
and multi-scale pooling (`pool="maxavg"`, `ortho_beta=0.05`) that eliminate
CNN seed variance, and the real infrastructure bugs (two hardcoded `D=16`
constants) found and fixed while testing them.

This is a research baseline, not a production safety alarm. The next
evidence-gated improvements are operator-confirmed commissioning for unseen
load states, rate/residual features for pre-failure changes, seasonal
recalibration, and a fresh asset-disjoint evaluation cohort. The full
trade-offs and falsifiable tests are in
`docs/research-remediation-plan.md`.
