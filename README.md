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

Keep `--jobs` at 2 or below; the host has about 1 GB of RAM headroom.

## Layout

```
src/uengram/   library: stats, metrics, synth, encoders, ledger, store, baselines, stream_baselines, engine, lifecycle, cmapss, metropt
bench/         harness: provenance, evaluate, common, run_synth, run_cmapss, run_metropt, metropt_calibration, metropt_eval, report
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
machine with coarse reports, not a field-validation or revenue claim.

MetroPT-3 is distributed under CC BY 4.0: Davari et al., dataset
[DOI:10.24432/C5VW3R](https://doi.org/10.24432/C5VW3R), description
[DOI:10.1038/s41597-022-01877-3](https://doi.org/10.1038/s41597-022-01877-3).
The adapted MemStream core is based on
[Stream-AD/MemStream](https://github.com/Stream-AD/MemStream), Apache-2.0;
its license is included in `src/uengram/MemStream-LICENSE.txt`.
