# uEngram

A reference implementation and benchmark of a deterministic, fixed-budget memory for embedded sensor streams ("Mem0 for sensors"). Routine windows are absorbed into running Gaussian statistics at no extra write cost, rare anomalies are pinned with their raw window, wear drift is tracked with checkpoints, stale baselines are superseded, and eviction is Ward-merge / importance-based inside a hard budget. The engine works on a whitened latent space with exact chi-square gating; encoders are a DSP+PCA encoder and a 1D-CNN autoencoder. It is benchmarked against naive append-only, FIFO, reservoir sampling and the engine from `docs/EdgeAIMemoryArchitecture.md` on a labelled synthetic stream and on NASA CMAPSS FD001, with multi-seed bootstrap confidence intervals.

- `docs/PLAN.md`: the implementation plan.
- `docs/EXPLAINED.md`: every component in plain words and math, and where it lives in code.
- `docs/RESULTS.md`: benchmark tables and findings.

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

# full synthetic benchmark: 10 seeds x 2 encoders x 9 methods
uv run python -m bench.run_synth --seeds 0-9 --encoders dsp,cnn --jobs 2 --out results/synth_runs.jsonl 2>&1 | tee ~/.local/state/uengram/synth.log

# budget sweep
uv run python -m bench.run_synth --seeds 0-4 --encoders dsp --sweep-budgets 64,128,256,512 --out results/synth_sweep.jsonl 2>&1 | tee ~/.local/state/uengram/sweep.log

# CMAPSS FD001 (first run downloads ~12 MB into data/cmapss/)
uv run python -m bench.run_cmapss --seeds 0-4 --encoders dsp,cnn --out results/cmapss_runs.jsonl 2>&1 | tee ~/.local/state/uengram/cmapss.log

# tables
uv run python -m bench.report --out docs/RESULTS.md
```

Keep `--jobs` at 2 or below; the host has about 1 GB of RAM headroom.

## Layout

```
src/uengram/   library: stats, metrics, synth, encoders, ledger, store, baselines, engine, lifecycle, cmapss
bench/         harness: provenance, evaluate, common, run_synth, run_cmapss, report
tests/         unit and behaviour tests
results/       committed run outputs (JSONL)
docs/          plan, explainer, results, architecture notes
```
