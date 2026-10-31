# uEngram project rules

## GPU-suited models run on Google Colab through the `colab` CLI

This workstation has 4 CPU cores and no GPU. A model that is bigger and faster on a GPU MUST run on a
Colab GPU runtime driven by the `colab` CLI (`~/.local/bin/colab`), not on this machine.

**Run on Colab:** neural encoders and teachers where a GPU is materially faster: multi-seed or
multi-width CNN / CNN2 sweeps, MOMENT / Chronos and other foundation-model forward passes or
distillation, EngineAD-scale studies, any training whose CPU run here would take more than about 10
minutes. Reference points: the EngineAD seed study took 1,962 s on a T4; one small CNN fit on CMAPSS
takes about 20 s here, so it stays local.

**Stay local:** the engine, DSP / context / tabular encoders, benchmarks, tests, single small CNN fits,
and anything whose data cannot leave this machine. Do not start parallel torch processes on this
machine to compensate: on 4 cores they were slower than running sequentially.

### Procedure

1. `colab usage` (compute-unit balance) and `colab sessions` (reuse a live session instead of starting a second).
2. `colab new -s <name> --gpu T4`. Always pass `-s`. An unrecognized `--gpu` silently becomes A100; a 400
   means no entitlement for that accelerator, so use T4 or tell the user. Larger GPUs only when T4 memory
   or time is the proven limit.
3. Verify the device before any work: `colab exec -s <name>` with `nvidia-smi -L`, `torch.cuda.is_available()`
   and the python / torch / numpy / pandas versions. Record them with the result.
4. Send code with `colab upload`. Fetch datasets on the VM itself (not through this machine) and verify
   size and checksum against the committed manifest before use.
5. Start long jobs as a detached subprocess (`start_new_session=True`) that writes a log under `/content`.
   Check progress by reading the log tail with `colab exec`. Do not hold a blocking `exec` for the whole
   run, and do not poll in a tight loop.
6. Bring results back with `colab download` into `~/.local/state/uengram/<job>/`. Compare with any
   committed artifact first. Copy into `results/` only after that check, and record the script commit,
   device and library versions in `docs/reproducing-the-paper.md`. Never overwrite a committed result
   without stating the difference.
7. `colab stop -s <name>` when the job ends or fails. Sessions are billable and are not released by exiting.

### Limits

- Never run `colab repl`, `colab console`, `colab auth` or `colab drivemount` from an agent: they need a TTY.
- No secrets in uploaded files, scripts or argv.
- GPU results are reproducible statistically, not bit for bit. Say `device: cuda` in the artifact and in the
  docs, and do not claim exact reproduction of a CPU result.
- If the CLI cannot authenticate, no GPU is available or the balance is exhausted, stop and report it. Do not
  fall back silently to a multi-hour CPU run.
