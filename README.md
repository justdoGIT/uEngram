# uEngram

A reference implementation and benchmark of a deterministic, fixed-budget memory for embedded sensor streams ("Mem0 for sensors"). Routine windows are absorbed into running Gaussian statistics at no extra write cost, rare anomalies are pinned with their raw window, wear drift is tracked with checkpoints, repairs and new regimes are detected by a sequential step test and supersede the old state, and eviction is Ward-merge / importance-based inside a hard budget. Novelty is scored against normal states only, and alarms use a threshold calibrated conformally on a fault-free commissioning run. The engine works on a whitened latent space in which within-state noise is decorrelated, so correlated sensors count once; encoders are a DSP+PCA encoder, the same with a learned thermal-lag residual (`ctx`), and a 1D-CNN autoencoder. It is benchmarked against naive append-only, FIFO, reservoir sampling and the engine from `docs/EdgeAIMemoryArchitecture.md` on a labelled synthetic stream and on NASA CMAPSS FD001, with multi-seed bootstrap confidence intervals.

- `docs/PLAN.md`: the original implementation plan.
- `docs/EXPLAINED.md`: every component in plain words and math, and where it lives in code.
- `docs/RESULTS.md`: benchmark tables and findings (v2; v1 kept in `results/v1/`).
- `docs/IMPROVEMENT-ANALYSIS.md`: the diagnostics that motivated v2.
- `docs/research-*.md`: literature notes (machine-state mathematics, anomaly-detection evaluation, condition-monitoring features, CMAPSS practice).
- `docs/HOST-TELEMETRY.md`: uEngram on running Linux hosts (the Void cluster): the 1 Hz snapshot, the attribute schema with one owner per measured field, the per-signal mathematics, deployment, and cluster results; `docs/research-host-state.md` is its literature note.

## Environment

`/home` is small on the development machine, so the uv cache and venv live on `/home/kk/Backup`:

```bash
export UV_CACHE_DIR=/home/kk/Backup/.cache/uv UV_PROJECT_ENVIRONMENT=/home/kk/Backup/venvs/uengram
uv sync --extra cnn
```

Run everything from the repository root. Long runs are tee'd to `~/.local/state/uengram/`.

## Commands

```bash
mkdir -p ~/.local/state/uengram

# tests (CNN test needs --extra cnn)
uv run pytest -q

# smoke runs
uv run python -m bench.run_synth --quick --out /tmp/uengram_smoke.jsonl
uv run python -m bench.run_cmapss --quick --out /tmp/uengram_cmapss_smoke.jsonl

# full synthetic benchmark: 10 seeds x 3 encoders x 10 methods
uv run python -m bench.run_synth --seeds 0-9 --encoders dsp,ctx,cnn --jobs 2 --out results/synth_runs.jsonl 2>&1 | tee ~/.local/state/uengram/synth.log

# budget sweep
uv run python -m bench.run_synth --seeds 0-4 --encoders dsp --sweep-budgets 64,128,256,512 --out results/synth_sweep.jsonl 2>&1 | tee ~/.local/state/uengram/sweep.log

# CMAPSS FD001 (first run downloads ~12 MB into data/cmapss/)
uv run python -m bench.run_cmapss --seeds 0-4 --encoders dsp,cnn --out results/cmapss_runs.jsonl 2>&1 | tee ~/.local/state/uengram/cmapss.log

# tables
uv run python -m bench.report --out docs/RESULTS.md
```

Keep `--jobs` at 2 or below; the host has about 1 GB of RAM headroom.

## Host telemetry (cluster nodes)

```bash
# probe all sources on this node (needs CAP_PERFMON + CAP_DAC_READ_SEARCH, see docs/HOST-TELEMETRY.md)
python3 -m uengram.host probe
# record (a single file, or daily spool files), commission, replay, run, cluster view
python3 -m uengram.host record --spool ~/.local/state/uengram-host/spool
uv run python -m uengram.host commission --ticks ~/.local/state/uengram-host/spool/ticks-*.jsonl* --model model.pkl
uv run python -m uengram.host replay --model model.pkl --ticks more.jsonl --events events.jsonl
python3 -m uengram.host cluster --nodes msi-worker hp asus-worker lenovo-worker gigabyte-worker
# synthetic validation of the host pipeline
uv run python -m bench.run_host_synth --seeds 0-19 --jobs 2 --out results/host_synth.jsonl
# install the runit service on nodes (sudo via kepr); --campaign adds the campaign runner (msi only)
deploy/install-node.sh --numpy NODE...
# 7-day live validation campaign: record 3 days, commission, scheduled faults 4 days, report
python3 -m uengram.host campaign plan --dir ~/Backup/uengram-host/campaign
python3 -m uengram.host campaign status --dir ~/Backup/uengram-host/campaign
```

## Layout

```
src/uengram/   library: stats, metrics, synth, encoders (dsp, ctx, cnn), ledger, store, baselines, engine, lifecycle, cmapss
src/uengram/host/  host telemetry: schema, perf, sources, collector (stdlib); features, encoder, events, node, cluster, synth (numpy); campaign, validate, validate_variants
bench/         harness: provenance, evaluate, common, run_synth, run_cmapss, run_host_synth, report
deploy/        runit services (uengram-hostd, uengram-campaign) and the node installer
tests/         unit and behaviour tests
results/       committed run outputs (JSONL)
docs/          plan, explainer, results, improvement analysis, research notes, architecture notes
```
