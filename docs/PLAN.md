# uEngram reference implementation, benchmark and explainer

## Context
`/home/kk/projects/uEngram` started from a repository architecture note describing deterministic memory for embedded sensors: routine windows are absorbed, rare anomalies are pinned, wear drift is tracked, stale baselines are superseded and storage remains within a fixed budget. The goal is a Python/NumPy reference engine (`uengram`) with statistically correct state math, a DSP+PCA encoder and a Torch 1D-CNN variant. Benchmark against naive append-only, FIFO, reservoir and reference baselines on a synthetic stream with full ground truth and NASA CMAPSS FD001. Results get multi-seed confidence intervals. Also write a plain-language plus math explainer of every component.

Scope stays Python. No C++ port, no daemon, no SocketCAN.

### Corrections to the doc's engine (each is a deliberate decision used below)
1. The doc derives severity from ground-truth labels (`sev = 0.9 if labels[t]==2`), which is label leakage. Here severity is surprisal from the χ² tail; labels are never visible to any store.
2. Projecting onto the unit sphere discards magnitude, so a 2× scaled anomaly maps to the same point. Instead, work in a whitened Euclidean latent ℝ^d, where Gaussian/Mahalanobis statistics are well-defined. On a sphere the covariance is singular in the radial direction.
3. A fixed EWMA β alongside a mass counter is inconsistent. Use the adaptive step α_n = max(1/n, β_min): the exact mean early, exponential forgetting later.
4. Arbitrary cosine thresholds (0.90/0.75) are replaced by χ²_d quantile gates on a calibrated noise scale.
5. The additive utility `0.6·cos + 0.2·decay + 0.2·sev` mixes incommensurate scales. Replace it with a log-posterior: Gaussian log-likelihood plus a log-prior.
6. The ad-hoc density formula (`mean(max(sim,0))·log1p(mass)`) is replaced by Ward merge cost, which minimises the increase in within-cluster variance and so best preserves the distribution, plus a partitioned budget.
7. The doc has no transient-versus-regime distinction. Add M-of-N confirmation, as in radar track initiation.
8. Counting "zero flash writes" for RAM-only state is optimistic. All stores share a ledger that flushes dirty state periodically.
9. The doc trains its CNN on unscaled channels. Standardise per channel first.
10. The doc's benchmark is a single seed with oracle severity. Use 10 seeds, label-free stores, bootstrap CIs and paired differences.

## Approach

### Step 0: Repo and environment bootstrap
- `/home` has 1.8 GB free, so put the uv cache and venv on `/home/kk/Backup` (48 GB free, kk-writable). Every `uv` command in this plan runs with
  `export UV_CACHE_DIR=/home/kk/Backup/.cache/uv UV_PROJECT_ENVIRONMENT=/home/kk/Backup/venvs/uengram`.
- `git init -b main`. Add `.gitignore` covering `data/`, `__pycache__/`, `.pytest_cache/`, `*.pyc`, `.venv/`. Commit `docs/EdgeAIMemoryArchitecture.md` and `.gitignore` on `main` with `git commit -s` (subject `Add architecture notes`). Then `git switch -c feat/uengram-reference`. After each step below, commit with `git commit -s` (subject ≤50 chars, body wrapped at 72, no model names). This is the historical bootstrap plan; current development uses feature branches and pull requests.
- `.python-version` contains `3.12` (uv-managed 3.12.13 is already installed; torch has `cp312` CPU wheels).
- `pyproject.toml`:
  - `[project] name="uengram"`, `version="0.1.0"`, `requires-python=">=3.12,<3.15"`, `dependencies=["numpy>=2.0"]`.
  - `[project.optional-dependencies] cnn=["torch>=2.9,<3"]`.
  - `[dependency-groups] dev=["pytest>=8"]`.
  - `[tool.uv.sources] torch=[{index="pytorch-cpu", extra="cnn"}]` and `[[tool.uv.index]] name="pytorch-cpu"`, `url="https://download.pytorch.org/whl/cpu"`, `explicit=true`.
  - Build: hatchling, with `[tool.hatch.build.targets.wheel] packages=["src/uengram"]`.
  - `[tool.pytest.ini_options] testpaths=["tests"]`, `pythonpath=["src","."]`.
- Run `uv sync --extra cnn`. If it fails because torch has no wheel for the resolved Python, set `.python-version` to `3.13` and retry once.
- Layout: library in `src/uengram/`, harness in `bench/` (a package with `__init__.py`, run as `python -m bench.<mod>` from the repo root), tests in `tests/`, run outputs in `results/` (committed), docs in `docs/`.
- Hard limit: every code file stays ≤375 lines. Split a module into a sibling file when it would exceed that.
- All randomness uses `np.random.default_rng(seed)`. Never touch the global `np.random`.

### Step 1: Statistics primitives (`src/uengram/stats.py`) and evaluation metrics (`src/uengram/metrics.py`)
Write the tests first (`tests/test_stats.py`, `tests/test_metrics.py`). No scipy; only numpy and `math`.

`stats.py`:
- `chi2_logsf(x: float | np.ndarray, d: int) -> np.ndarray`: exact log survival function of χ²_d for even d, using
  Q(d/2, x/2) = e^{−x/2} Σ_{k=0}^{d/2−1} (x/2)^k / k!,
  evaluated in log space: `−x/2 + logsumexp_k(k·log(x/2) − lgamma(k+1))`. At x=0 it returns 0. Odd d raises `ValueError("d must be even for exact chi2 tail")`.
- `chi2_isf(alpha: float, d: int) -> float`: the x with Q = alpha, found by bisection on `chi2_logsf` over [0, 10·d + 200] to tolerance 1e-10.
- `surprisal(x, d) -> float`: `−chi2_logsf(x,d)/ln(10)` (decades), capped at 300.
- `robust_noise_sigma(z: np.ndarray, lag: int, groups: np.ndarray | None) -> np.ndarray`: the noise std per dimension, from a MAD of lag-`lag` successive differences. Δ = z[t+lag] − z[t], keeping only pairs where `groups[t]==groups[t+lag]` when groups is given. σ_j = 1.4826·median(|Δ_j − median(Δ_j)|)/√2. If fewer than 10 pairs exist, raise `ValueError`. Floor each σ_j at 1e-6. Rationale: differences cancel slow drift and regime levels, and the median ignores regime switches and anomaly jumps.
- `ew_update(mu, S, x, alpha) -> (mu, S)`: Finch's exact exponentially weighted mean/variance:
  δ = x − μ; μ ← μ + αδ; S ← (1−α)(S + αδ²).
  With α=1/n it reproduces the batch population variance.
- `shrunk_var(S, n_eff, sigma0_sq, kappa0) -> np.ndarray`: `(kappa0*sigma0_sq + n_eff*S)/(kappa0 + n_eff)`. This is a conjugate-prior shrinkage, so a new slot with n=1 still has a sane variance.
- `merge_moments(n1, mu1, var1, n2, mu2, var2) -> (n, mu, var)`: exact mixture moments, with n = n1+n2, μ = (n1μ1+n2μ2)/n and σ² = (n1(σ1²+μ1²)+n2(σ2²+μ2²))/n − μ², floored at 0.
- `bhattacharyya_diag(mu1, var1, mu2, var2) -> float`, with v̄ = (var1+var2)/2:
  D_B = ⅛Σ(μ1−μ2)²/v̄ + ½Σ ln(v̄/√(var1·var2)).
  For a batch, `bhattacharyya_pairwise(mu[n,d], var[n,d]) -> [n,n]` (diagonal set to +inf).
- `ward_cost_pairwise(n[n], mu[n,d], sigma0_sq[d]) -> [n,n]`: n_i n_j/(n_i+n_j) · Σ_j (μ_i−μ_j)²/σ0_j² (diagonal +inf).

`metrics.py`:
- `auroc(scores, labels_bool) -> float`: Mann–Whitney with average ranks for ties, (R_pos − n_pos(n_pos+1)/2)/(n_pos·n_neg). NaN if either class is empty.
- `average_precision(scores, labels_bool) -> float`: sort descending and compute Σ_k (R_k − R_{k−1})·P_k. Ties are broken by stable order after `np.argsort(-scores, kind="stable")`.
- `gauss_mixture_mmd2(a_w, a_mu, a_var, b_w, b_mu, b_var, h) -> float`: exact squared MMD between two weighted diagonal-Gaussian mixtures under the RBF kernel k(x,y) = exp(−‖x−y‖²/(2h²)). Point sets pass var=0. Weights are normalised to sum to 1. The pairwise expected kernel is
  Π_j (h²/(h²+v1_j+v2_j))^{1/2} · exp(−Σ_j (μ1_j−μ2_j)²/(2(h²+v1_j+v2_j))).
  Compute it in chunks of 512 rows so memory stays below ~100 MB.
- `median_heuristic(x[n,d]) -> float`: median pairwise Euclidean distance over at most 2000 rows (subsampled with rng seed 0), computed in chunks.
- `bootstrap_ci(values, n_boot=2000, seed=12345, level=0.95) -> (mean, lo, hi)`: percentile bootstrap of the mean. NaN entries are dropped first.
- `paired_bootstrap_diff(a, b, ...) -> (mean_diff, lo, hi)`: resample seed indices jointly.

Tests, all exact or with tight tolerances:
- d=2: `chi2_logsf(x,2) == −x/2`.
- `chi2_isf(0.001,16)` ≈ 39.252 (±0.001) and `chi2_isf(0.01,16)` ≈ 32.000 (±0.001), which are table values.
- `ew_update` with α=1/n over a sample matches `np.mean` and `np.var`.
- `merge_moments` matches the moments of the concatenated samples.
- `bhattacharyya_diag` returns 0 for identical inputs.
- `robust_noise_sigma` recovers σ=0.5 within 10% on a piecewise-constant mean (regime switches every 50 steps) plus N(0, 0.25) noise with 5% large jumps.
- `auroc` matches a brute-force pair count with ties counted as ½.
- `average_precision` on the hand case [pos,neg,pos] gives (1 + 2/3)/2.
- `gauss_mixture_mmd2` is 0 for identical mixtures, and for point sets it matches the V-statistic MMD² computed directly.

### Step 2: Synthetic generator (`src/uengram/synth.py`)
- Channels C=6: `["rpm","oil_psi","coolant_temp","vib_x","vib_y","vib_z"]`. K=64 samples per window, with in-window time `s = np.arange(K)/K`.
- Regimes 0 idle, 1 cruise, 2 heavy, driven by a Markov chain with `P = [[0.97,0.03,0],[0.015,0.97,0.015],[0,0.03,0.97]]`. The initial regime is 0.
  - rpm levels (0.8, 1.8, 2.6).
  - psi levels (2.0, 3.0, 3.5).
  - temp targets (0.6, 0.85, 1.0).
  - vibration amplitude (0.2, 0.35, 0.5).
  - vibration fundamental frequency f = 4·rpm cycles/window.
- Wear w(t): a linear ramp from 0 to 0.9 over t ∈ [0, t_m), with t_m = int(0.6·T). At t_m maintenance sets w=0.1, then it ramps linearly to 0.4 at T−1. Era is 0 before t_m and 1 after.
  - Effects of wear: vibration amplitude × (1+0.8w); an added 2f harmonic with amplitude 0.3w on vib_x and vib_y; psi −0.6w; temp +0.25w.
  - Maintenance (new pump): psi +0.5 in era 1.
- Coolant temperature level lags across windows: L_t = L_{t−1} + 0.5·(target_t − L_{t−1}). Within a window the signal is a linear ramp from L_{t−1} to L_t.
- Signals per window. rpm = level + 0.05·sin(2πs) + N(0,0.03). psi = level + N(0,0.05). temp = ramp + N(0,0.01). vib_x = A·sin(2πfs), vib_y = A·cos(2πfs), vib_z = 0.5A·sin(2π·2fs), each plus N(0,0.08).
- Anomaly types: 1 knock, 2 cavitation, 3 thermal, 4 seizure.
  - Events of type 1/2/3 start with per-window probabilities 0.003/0.003/0.002 when t ≥ 200 and no event is active. Each lasts L ~ Uniform{1,2,3} windows.
  - Type 4 is placed at exactly t ∈ {int(0.25T), int(0.5T), int(0.85T)}, with L=1, and overrides any other event.
  - Knock: vib_x += 1.5·sin(2π·25s) plus 3 impulses of +2.0 at random sample positions.
  - Cavitation: psi −1.2 and psi noise σ 0.3; vib_z noise σ ×3.
  - Thermal: temp += linear 0→0.4 across the window.
  - Seizure: rpm level → 0.3; every vib channel gets N(0, 4×0.08) added and its amplitude ×4.
- API:
  - `make_window(rng, regime, wear, era, temp_prev, temp_now, anomaly) -> np.ndarray[C,K] float32`.
  - `generate(T, seed, maint_frac=0.6) -> SynthStream`, a dataclass with `windows[T,C,K] float32` and int arrays `regime`, `era`, `anomaly`, `event_id` (−1 when nominal), plus float `wear`.
  - `make_probe(seed, T) -> SynthStream`, using rng `seed+20_000` and the final state (wear=w(T−1), era=1, steady-state temp per regime). It contains 30 nominal windows per regime and 30 per anomaly type 1–4 (210 windows).
- Tests (`tests/test_synth.py`) with `generate(4000, seed=3)`, t_m=2400:
  - shapes;
  - the same seed gives identical arrays;
  - exactly 3 seizure events;
  - `anomaly[:200]` is all 0;
  - the maintenance step is visible: mean `oil_psi` of nominal regime-1 windows in [t_m, t_m+500) minus that in [t_m−500, t_m) is > 0.5. The expected value is ≈ +0.5 + 0.6·(0.9·(t_m−250)/t_m − 0.1) ≈ +0.94.

### Step 3: Encoders (`src/uengram/encoders.py`, `src/uengram/encoder_cnn.py`)
Common protocol: `fit(train_windows[N,C,K], noise_lag:int, groups:np.ndarray|None) -> self`, `encode(windows) -> float64[N,d]`, attributes `d=16` and `sigma0: np.ndarray[d]`. d must be even for the exact χ² tail.
- `Whitener` (in `encoders.py`): `fit(F[N,p], d)` stores the mean m and the top-d eigenvectors/eigenvalues of `np.cov(F, rowvar=False)` from `np.linalg.eigh`, with eigenvalues floored at 1e-8·max. `transform(F) = (F−m) @ V_d / sqrt(λ_d)`. Training embeddings then have identity covariance.
- `DSPEncoder`: per-channel features, giving 6·C features:
  - mean;
  - `log(std+1e-6)`;
  - 4 log band energies. Take power `|rfft(x−mean)|²/K` on bins 1..K//2, split into 4 contiguous equal-count bands with `np.array_split`, and compute `log(sum+1e-8)`.

  Then robust-standardise each feature with the training median and IQR/1.349 (IQR floored at 1e-6). Then `Whitener` with d=16. Then `sigma0 = robust_noise_sigma(encode(train), noise_lag, groups)`.
- `CNNEncoder(seed)` (imports torch lazily inside the class, so the DSP path never needs torch):
  - Inputs are standardised per channel with the training mean/std over all samples (std floored at 1e-6).
  - Encoder: `Conv1d(C,32,5,stride=2,padding=2)–GELU–Conv1d(32,64,5,stride=2,padding=2)–GELU–AdaptiveAvgPool1d(1)–Flatten–Linear(64,16)`.
  - Decoder: `Linear(16,128)–GELU–Linear(128,C*K)` reshaped to (C,K). An MLP decoder makes it work for any K, both K=64 and K=16.
  - Loss: MSE on standardised input. AdamW (lr 1e-3, wd 1e-4), 30 epochs, batch 128. Set `torch.manual_seed(seed)`, `torch.set_num_threads(4)` and `torch.use_deterministic_algorithms(True)`; shuffle with `torch.Generator().manual_seed(seed)`.
  - No L2 normalisation. After training, fit `Whitener` on the training latents (d=16), then compute sigma0 as in the DSP encoder.
- Tests (`tests/test_encoders.py`):
  - DSP training embeddings have mean ≈0 (|·|<1e-6) and covariance ≈I (max abs deviation <1e-6).
  - On `generate(3000, seed=1)`, the DSP encoder separates anomalies: AUROC of the norm ‖z‖ for anomaly>0 vs 0 is >0.8.
  - CNN test uses `pytest.importorskip("torch")`: 2 epochs on 512 windows yield shape (N,16) and finite values.

### Step 4: Write ledger, store protocol and baselines
- `src/uengram/ledger.py`, `WriteLedger(flush_every=50)`. This accounting is applied to every store identically:
  - `write_now(item_id, nbytes)`: writes += 1, bytes += nbytes.
  - `mark_dirty(item_id, nbytes)`: put it in the dirty dict, overwriting.
  - `delete(item_id)`: writes += 1, bytes += 16 (a tombstone), and the id is removed from dirty.
  - `tick(t)`: when `(t+1) % flush_every == 0`, writes += len(dirty), bytes += the sum of dirty sizes, then clear.
  - Fields: `writes`, `bytes_written`.
- `src/uengram/store.py`:
  - `@dataclass IngestResult(item_id:int, score:float, flagged:bool|None, action:str)`.
  - `Protocol MemoryStore` with: `name`; `ingest(z, t, raw_idx) -> IngestResult`; `recall(q, t_now, k, mode) -> list[int]` where mode ∈ {"current","history"}; `items() -> ItemsView`, giving arrays `ids, mu[n,d], var[n,d], mass[n], t_last[n]` for MMD and accounting; `drain_events() -> list[tuple]` with tuples `("merge", src_id, dst_id)` or `("evict", id)`; `index_bytes() -> int`; `pinned_raw_bytes() -> int`; `ledger`.
  - `score` is the prequential novelty score computed before the update: larger means more novel.
  - Item ids are monotone ints from a per-store counter.
- `src/uengram/baselines.py`. Point stores use vector bytes = d·4+16 per item. They use Euclidean distance; `score` = min squared distance to the stored items (NaN when empty); `flagged=None`; recall = top-k smallest distance regardless of mode.
  - `NaiveStore`: a preallocated array that grows ×2; every insert calls `write_now`.
  - `FIFOStore(B)`: a ring that overwrites the oldest; emits `("evict", old_id)`; each insert calls `write_now`.
  - `ReservoirStore(B, seed)`: Algorithm R. At t ≥ B, replace a uniform index with probability B/(t+1) (emitting evict plus `write_now`); otherwise the window is discarded with no write. Its `item_id` is −1 when discarded.
- `src/uengram/reference_baseline.py`, `ReferenceBaseline(max_slots, sim_merge_thresh=0.90, ewma_beta=0.05, severity_mode)`. This is a historical reference baseline retained for comparison. Its ingest cases, `_evict_lowest_utility`, and `remember` utility are defined by the implementation and tests; the corrections below describe the differences from the original experiment.
  - Input z is L2-normalised internally, as it expects unit vectors.
  - `flash_writes` is replaced by ledger calls: append → `write_now`, EWMA → `mark_dirty`, pop → `delete` plus an evict event.
  - Ids are added.
  - `score = 1 − best_sim`.
  - `flagged = action=="PINNED_EPISODIC"`.

  `severity_mode="labelfree"` passes 0.1 always. `"oracle"` passes 0.9 when the harness label anomaly>0, else 0.1; the harness supplies `severity` via an extra kwarg `ingest(z,t,raw_idx,severity=...)`. Bytes per slot: d·4+32. The baseline's behavior is historical and is not the specification for the current bounded engine.
- Tests (`tests/test_baselines.py`):
  - FIFO holds exactly the last B ids.
  - Over 4000 seeded trials with T=100, B=10, the reservoir includes each index with frequency within 0.1±0.02.
  - Naive recall equals brute-force argmin.
  - A ledger with only `mark_dirty` calls over T=1000 and flush_every=50 counts ≤ 20·(#distinct ids) writes.

### Step 5: The uEngram engine (`src/uengram/engine.py`, `src/uengram/lifecycle.py`)
Data layout is struct-of-arrays with capacity `B_sem + B_epi`:
- `mu[B,d]`, `S[B,d]`, `n[B] int`;
- `kind[B] int8`: 0 empty, 1 candidate, 2 episodic, 3 semantic, 4 superseded;
- `t_created`, `t_last`;
- `surpr[B]` (decades);
- `ckpt_mu[B,d]`, `item_id[B]`, `superseded_by[B]` (−1);
- `exemplars: dict[item_id, list[int]]` holding `[first_raw_idx, latest_raw_idx]`;
- `lineage: list[dict]`, capped at 256 entries (ring).

Group budgets: kinds {1,2} ≤ `B_epi`; kinds {3,4} ≤ `B_sem`.

`@dataclass EngramConfig`:

| field | default | notes |
|---|---|---|
| `B_sem` | 192 | |
| `B_epi` | 64 | |
| `alpha_gate` | 1e-3 | |
| `beta_min` | 0.01 | |
| `kappa0` | 10.0 | |
| `m_promote` | 5 | |
| `n_promote` | 8 | |
| `consolidate_every` | 200 | |
| `db_merge` | 0.5 | |
| `t_dormant` | 2000 | |
| `theta_drift` | 3.0 | |
| `half_life` | 2000.0 | |
| `half_life_epi` | 20000.0 | |
| `flush_every` | 50 | |
| `use_slot_var` | True | ablation |
| `use_promotion` | True | ablation |

Constructor: `Engram(cfg, d, sigma0, raw_window_bytes)`. Precompute `tau = chi2_isf(alpha_gate, d)`, `sigma0_sq` and `n_eff_cap = (2−beta_min)/beta_min` (+inf when beta_min=0).

Gate variance per slot is `var_k = shrunk_var(S_k, n_eff_k, sigma0_sq, kappa0)`. For semantic slots `n_eff_k = min(n_k, n_eff_cap)`; for others `n_eff_k = n_k`. With `use_slot_var=False`, `var_k = sigma0_sq`.

`ingest(z, t, raw_idx)` runs in order:
1. Over gateable slots (kinds 1, 2, 3) compute the vector `D2_k = Σ(z−μ_k)²/var_k`. Set `score = min D2` (NaN if no slots).
2. If any `D2_k ≤ tau`, choose k* = argmin over those of `D2_k + Σ log var_k`; this is a Gaussian negative log-likelihood, so a broad slot does not steal points. Absorb:
   - Semantic: `alpha = max(1/(n+1), beta_min)`.
   - Candidate or episodic: `alpha = 1/(n+1)`.
   - Then `ew_update`, n += 1, t_last = t, `ledger.mark_dirty(id, 2·d·4+32)`.
   - Episodic or candidate: set `exemplars[id][1] = raw_idx` and `write_now(id, raw_window_bytes)`; `flagged=True`.
   - Candidate with n ≥ m_promote and t − t_created ≤ n_promote, and `use_promotion`: promote. Make semantic room (see eviction), set kind=3 and ckpt_mu=μ, `write_now`, action `"promote"`, `flagged=True`.
   - Otherwise action `"absorb_semantic"` (flagged=False) / `"absorb_episodic"` / `"absorb_candidate"` (flagged=True).
3. Otherwise create a new candidate:
   - Make episodic room first.
   - Set μ=z, S=0, n=1, `surpr = surprisal(score,d)` (0 if score is NaN), t_created = t_last = t, exemplars `[raw_idx, raw_idx]`.
   - `write_now(id, 2·d·4+32 + raw_window_bytes)`, action `"new_candidate"`, flagged=True.
   - With `use_promotion=False`, the new slot is created directly as kind 2.
4. Probation expiry: kind==1 with t − t_created > n_promote becomes kind 2 (`mark_dirty`).
5. If `(t+1) % consolidate_every == 0`, call `lifecycle.consolidate(self, t)`.
6. `ledger.tick(t)`.

Return `IngestResult`.

Eviction, in `lifecycle.py`:
- `make_semantic_room(eng, t)` when kinds {3,4} count == B_sem:
  - If any superseded slot exists, evict the one with the oldest t_last. Append a lineage record `{"type":"evict_superseded", ...}`, call `ledger.delete`, emit `("evict", id)`.
  - Otherwise Ward-merge the cheapest active semantic pair using `ward_cost_pairwise`. Merge src (smaller n) into dst (larger n) via `merge_moments` on `(n, μ, S)`: t_created=min, t_last=max, dst keeps ckpt_mu. Call `write_now(dst)` and `delete(src)` and emit `("merge", src, dst)`.
- `make_episodic_room(eng, t)` when kinds {1,2} count == B_epi: evict the kind-2 slot with the lowest
  `importance = surpr·(1+ln n)·2^{−(t−t_last)/half_life_epi}`.
  If there is no kind-2 slot, evict the kind-1 slot with the oldest t_created. Call `delete`, emit evict, and drop its exemplars.

`consolidate(eng, t)`:
- (a) Duplicate merge: while the minimum `bhattacharyya_pairwise` among active semantic slots (using their gate variances) is ≤ `db_merge`, merge that pair as above. Cap at 20 merges per call.
- (b) Drift checkpoints: for each active semantic slot with ‖(μ−ckpt_mu)/sigma0‖₂ ≥ theta_drift, append lineage `{"type":"checkpoint","item_id","t","mu","var","n"}`, call `write_now(id, 2·d·4+32)` and set ckpt_mu=μ.
- (c) Supersession: for each active semantic slot o with t − t_last[o] ≥ t_dormant, look for successors, which are active semantic slots s with t_created[s] > t_last[o]. If any exist, set kind[o]=4 and superseded_by[o] = the id of the successor with minimum Bhattacharyya distance. Append lineage `{"type":"supersede","old","new","t"}` and call `mark_dirty(o)`. An idle mode with no successor stays active.

`recall(q, t_now, k, mode)`:
- Per slot, `ll_i = −½(D2_i(q) + Σ log var_i)`.
- `"current"`: kinds {1,2,3}. Score = ll + log prior, where the semantic prior is `log n − ln2·(t_now−t_last)/half_life` and the episodic/candidate prior is `log n + log(1+surpr)`.
- `"history"`: kinds {1,2,3,4}, score = ll only.
- Return the top-k item ids, ties broken by lower id.

Also implement `items()`, `index_bytes()` = active slots·(2·d·4+32), `pinned_raw_bytes()` = Σ unique exemplars·raw_window_bytes, and `drain_events()`.

Tests (`tests/test_engine.py`). Use d=16, `sigma0 = 0.5·ones(16)`, default config unless stated, `raw_window_bytes=1536`, rng `default_rng(0)`, and noise N(0, 0.25·I). "Offset +a" means +a on dimension 0 only.
1. A stationary stream with T=5000 ends with exactly 1 semantic slot, and the flagged fraction over t ≥ 100 is ≤0.01.
2. One outlier at offset +20 at t=1000 gives action `new_candidate`. By t=1000+n_promote+1 it is kind 2, and it survives to T=3000.
3. A sporadic recurring outlier (offset +20 plus noise every 300 steps, 10 times, starting t=500) ends as a single kind-2 slot with n=10. It is never promoted, since the occurrences are 300 apart and M-of-N needs 5 within 8.
4. A persistent step change to offset +8 from t=2000 (T=5000) produces a `"promote"` action within t ∈ [2000, 2008]. The old semantic slot is kind 4 with superseded_by = the promoted id by t = 2000 + t_dormant + consolidate_every, and lineage contains a `"supersede"` record.
5. Budget invariant with B_sem=4, B_epi=4 and T=600 in 100 blocks of 6 steps. Block b first emits one isolated point at offset +(1000+40b), then 5 points at offset +40b plus noise. The isolated points overflow the episodic group; the 5-point runs promote and overflow the semantic group. Assert kinds {1,2} ≤ 4 and {3,4} ≤ 4 after every ingest. Also assert ≥1 `("merge", …)` event and ≥1 `("evict", …)` event in `drain_events()`.
6. Episodic eviction: build an engine with B_epi=2 and set two kind-2 slots directly (equal t_last=0, n=1, surpr 300 and 5). Calling `make_episodic_room(eng, t=10)` evicts the surpr-5 slot.
7. Slow drift: an offset moving linearly from 0 to +6 over T=10,000 steps leaves exactly 1 live semantic slot that is not superseded, with flagged fraction over t ≥ 100 ≤0.02. Lineage holds ≥3 `"checkpoint"` records (6/(3·0.5) = 4 expected).

### Step 6: Synthetic harness (`bench/provenance.py`, `bench/evaluate.py`, `bench/run_synth.py`)
- `Provenance`: harness-only and never visible to stores. Per item id it keeps:
  - label histograms `Counter` for the anomaly label and for (regime, era) of nominal members;
  - the set of anomaly `event_id`s;
  - mass-weighted mean RUL (used for CMAPSS).

  After every ingest, first `prov.apply(store.drain_events())` (merges and evictions triggered inside that ingest), then record the window's labels under `item_id`; skip recording when `item_id == −1`, which is a discarded reservoir window. `apply(events)`: a merge unions histograms and sets into dst and removes src; an evict removes the id. `majority_anomaly(id)` and `majority_regime_era(id)` give the majority by count.
- `bench/evaluate.py`, `evaluate_run(store, prov, stream, probe_z, probe_labels, scores, flags, h, latencies) -> dict` with keys:
  - Footprint: `items_end`, `index_bytes`, `pinned_raw_bytes`, `writes`, `bytes_written`, `write_reduction_pct` (vs T writes).
  - Retention: `event_recall`, the fraction of anomaly events whose event_id is in any live item's set. `type_recall` is over types 1–4: a type counts when some live item has majority anomaly == type. `seizure_recall` is the fraction of the 3 seizure events retained.
  - Detection, on t ≥ 200, positives anomaly>0: `auroc`, `ap` from `scores`. When flags is not None also `precision`, `recall`, `f1`, plus `false_flag_rate_era0` and `false_flag_rate_era1` over nominal windows.
  - Retrieval on the probe:
    - `stale_at1`: over nominal probe queries in "current" mode, the fraction whose top-1 item has majority era 0.
    - `regime_acc_at1`: the fraction whose top-1 majority (regime, era) == (r, 1).
    - `type_prec_at1`: over anomaly probe queries in "history" mode, top-1 majority anomaly == query type.
  - Distribution fidelity:
    - `mmd2_current`: live items with t_last ≥ T−2000, weights=mass, vs the last 2000 embeddings.
    - `mmd2_history`: all live items weighted by mass vs a uniform subsample of 2000 embeddings (rng seed 0).
    - Both use the same h for all methods in that (seed, encoder), computed by `median_heuristic` on the last 2000 embeddings. Point stores pass var=0 and mass=1. uEngram passes its gate variances and n.
    - The reference baseline stores unit-sphere centroids, which do not live in the reference space, so both MMD metrics are NaN for that method. RESULTS.md shows them as "n/a (reference sphere space)".
  - Latency: `lat_p50_us`, `lat_p99_us` from `time.perf_counter_ns` around `ingest`.
- `bench/run_synth.py` CLI:
  - Arguments: `--seeds 0-9`, `--T 20000`, `--encoders dsp,cnn`, `--methods naive,fifo,reservoir,reference_labelfree,reference_oracle,uengram,uengram_fixedvar,uengram_nopromo,uengram_frozen`, `--budget 256`, `--jobs 2`, `--out results/synth_runs.jsonl`.
  - Method construction:
    - Bounded stores use B=256 total: FIFO, reservoir and the reference baseline get `max_slots=256`; uEngram gets B_sem=192, B_epi=64.
    - `uengram_fixedvar`: `use_slot_var=False`.
    - `uengram_nopromo`: `use_promotion=False`.
    - `uengram_frozen`: `beta_min=0.0`.
  - Per seed: train each encoder on `generate(8000, seed+10_000)` (noise_lag=1, groups=None), encode the stream and probe, then run every method on the same embeddings.
  - Each output line is `{"dataset":"synth","seed","encoder","method","budget","metrics":{...}}`.
  - Use `multiprocessing.Pool(jobs)` over (seed, encoder). Print `[run_synth] seed=S enc=E method=M done in X.Xs` with `flush=True` after each method.
  - `--sweep-budgets 64,128,256,512` runs FIFO, reservoir and uEngram over those budgets. uEngram's split is B_epi = B//4, B_sem = B − B_epi. Output `results/synth_sweep.jsonl`.
  - Use `--quick` (T=3000, seeds 0, encoder dsp) for smoke runs.

### Step 7: CMAPSS FD001 (`src/uengram/cmapss.py`, `bench/run_cmapss.py`)
- `fetch(dest="data/cmapss")`: download `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip` (verified HTTP 200, 12,429,152 bytes) with `urllib.request` to `data/cmapss/cmapss.zip`; skip if present with that size.
- The archive layout is unverified; confirm it first. Find a member whose basename is `train_FD001.txt`. If none exists, open every member ending in `.zip` as a nested `zipfile.ZipFile(io.BytesIO(...))` and search there. Extract `train_FD001.txt`, `test_FD001.txt` and `RUL_FD001.txt` to `data/cmapss/`. If they are not found, raise `FileNotFoundError` naming the members seen.
- `load(path) -> np.ndarray[N,26]` via `np.loadtxt`. Columns: unit, cycle, op1–3, s1–s21. Keep sensors `s2,s3,s4,s7,s8,s9,s11,s12,s13,s14,s15,s17,s20,s21` (C=14).
- `windows(df, K=16) -> (W[N,14,16], unit[N], rul[N])`: stride 1 within a unit only. RUL is that of the window's last cycle, `max_cycle − cycle`, capped at 125. Test windows take the last K cycles per unit (every FD001 test unit has ≥31 cycles). Test truth is `RUL_FD001.txt`, uncapped.
- Unit test with an inline 2-unit text fixture written to `tmp_path`, with no network: window count, RUL values and cap.
- `bench/run_cmapss.py`:
  - Seeds `0-4` (encoder seed and reservoir seed). Encoders are trained on units 1–50 (noise_lag=16, groups=unit). The stream is units 51–100 in order, with t a global window index.
  - Methods: `naive,fifo,reservoir,reference_labelfree,uengram,uengram_frozen`, all with B=256 as in Step 6. `reference_oracle` is omitted because there are no anomaly labels.
  - Engine config overrides: `t_dormant=400`; everything else is default. CLI flags `--seeds`, `--encoders`, `--methods`, `--jobs 2`, `--out`, and `--quick` (seed 0, dsp, units 51–60 only).
  - Metrics:
    - Footprint as in Step 6.
    - `auroc`/`ap` of score, with positives RUL ≤ 20 and negatives RUL > 75; windows in between are excluded.
    - `rul_rmse`, `rul_mae`: for each test unit, run `recall(q, t_now=T−1, k=5, mode="history")` and predict the mass-weighted mean of the provenance mean RUL over the returned items. Compare against `RUL_FD001`.
    - `eol_retention`: the fraction of stream units with any live item holding a window of that unit with RUL ≤ 20. Provenance tracks a set of (unit) ids for windows with RUL ≤ 20.
    - `mmd2_current` / `mmd2_history` as in Step 6.
  - Output `results/cmapss_runs.jsonl` with the same line schema, `dataset:"cmapss"`.

### Step 8: Report (`bench/report.py`)
- Read the three jsonl files. For each (dataset, encoder, method), compute `bootstrap_ci` per metric over seeds.
- Write `docs/RESULTS.md`:
  - One table per (dataset, encoder) with rows = methods and cells `mean [lo, hi]`, 3 significant figures.
  - Columns for synthetic: items_end, writes, write_reduction_pct, event_recall, seizure_recall, type_recall, auroc, ap, f1, false_flag_rate_era1, stale_at1, type_prec_at1, mmd2_current, lat_p99_us.
  - Columns for CMAPSS: items_end, writes, auroc, ap, rul_rmse, eol_retention, mmd2_current.
  - A paired-difference table: uEngram minus each other method (`paired_bootstrap_diff`) for event_recall, auroc, stale_at1, mmd2_current and writes, marked `*` when the CI excludes 0.
  - The budget sweep table.
  - A fixed note that `reference_oracle` uses ground-truth labels and is not a deployable configuration.
- Below the generated tables, the implementer writes a `## Findings` section by hand from the actual numbers. It includes where uEngram loses (for example, AUROC vs naive kNN, or recall vs naive which is unbounded). No claims beyond the tables.

### Step 9: Explainer (`docs/EXPLAINED.md`) and `README.md`
`docs/EXPLAINED.md` has one section per component, each with five subsections:
1. **Plain words**: an analogy such as the mechanic, habit memory vs flashbulb memory, or a notebook with limited pages.
2. **Why it exists**.
3. **The math**: every symbol defined and formulas exactly as implemented.
4. **Why this function and not the obvious alternative**.
5. **Where in code**: `path:symbol`.

Components, in order:
1. Windows and channels.
2. DSP encoder.
3. CNN encoder.
4. Whitening.
5. Noise scale σ0 via MAD of successive differences.
6. Memory slots and the state machine: candidate → semantic/episodic, semantic → superseded, plus lineage. Include a mermaid `stateDiagram-v2`.
7. Novelty gate: Mahalanobis, the χ²_d quantile, the exact even-d tail, surprisal.
8. Absorption: Finch EW mean/variance, α_n = max(1/n, β_min), variance shrinkage κ0.
9. M-of-N promotion.
10. Consolidation: Bhattacharyya duplicates, drift checkpoints, supersession.
11. Eviction: Ward cost and episodic importance.
12. Retrieval as a log-posterior.
13. Write ledger.
14. Baselines, including the 10 numbered corrections to the doc's engine from the Context section.
15. Metrics and statistics: AUROC vs AP under imbalance, mixture MMD, percentile and paired bootstrap.

`README.md`: one-paragraph description, the env exports from Step 0, and the exact commands from Verification.

## Critical files & anchors
- `docs/EdgeAIMemoryArchitecture.md`: repository architecture summary; implementation behavior is defined by source code and tests.
- `bench/run_synth.py`: synthetic generator and lifecycle benchmark; do not replace its deterministic seed handling with global process state.
- `src/uengram/engine.py` / `lifecycle.py`: all state math. The probation expiry and supersession rules are the subtle parts, covered by engine tests 2–4.
- `src/uengram/stats.py:chi2_logsf`: every threshold and surprisal depends on it, verified against table values.

## Verification
All commands run from `/home/kk/projects/uEngram` with `export UV_CACHE_DIR=/home/kk/Backup/.cache/uv UV_PROJECT_ENVIRONMENT=/home/kk/Backup/venvs/uengram`. Long runs are tee'd to `~/.local/state/uengram/` (`mkdir -p` first).

1. After each of steps 1–5 and 7: `uv run pytest -q tests/<that step's test files>` passes. Before the final commit: `uv run pytest -q` passes everything, with the CNN test not skipped since `--extra cnn` is installed.
2. Smoke: `uv run python -m bench.run_synth --quick --out /tmp/uengram_smoke.jsonl` finishes, printing one `done` line per method. In the output, `uengram` has `items_end ≤ 256`, `writes < 3000` and finite `auroc`. The `naive` line has `items_end == 3000`.
3. Full synthetic: `uv run python -m bench.run_synth --seeds 0-9 --encoders dsp,cnn --jobs 2 --out results/synth_runs.jsonl 2>&1 | tee ~/.local/state/uengram/synth.log`. The output has 180 lines (10 seeds × 2 encoders × 9 methods).
4. Budget sweep: `uv run python -m bench.run_synth --seeds 0-4 --encoders dsp --sweep-budgets 64,128,256,512 --out results/synth_sweep.jsonl 2>&1 | tee ~/.local/state/uengram/sweep.log`.
5. CMAPSS smoke: `uv run python -m bench.run_cmapss --quick --out /tmp/uengram_cmapss_smoke.jsonl`. The first run downloads about 12 MB into `data/cmapss/`. Expect 6 lines with finite `rul_rmse`. Then the full run: `uv run python -m bench.run_cmapss --seeds 0-4 --encoders dsp,cnn --out results/cmapss_runs.jsonl 2>&1 | tee ~/.local/state/uengram/cmapss.log`, which yields 60 lines.
6. `uv run python -m bench.report --out docs/RESULTS.md` writes the tables.
7. Expected behavioural evidence in `docs/RESULTS.md`; the Findings section must report whatever is actually observed:
   - uEngram `stale_at1` is lower than FIFO, reservoir and the reference baseline (the supersession works).
   - `seizure_recall` is higher than FIFO's.
   - `write_reduction_pct` is >90% for uEngram.
   - The `uengram_frozen` ablation shows a higher `false_flag_rate_era1` than `uengram`.

   If an expectation fails, report it as a finding. Do not retune constants to force it, except as below.

## Assumptions & contingencies
- B = 256 total items for every bounded method; synthetic T = 20,000; 10 synthetic seeds and 5 CMAPSS seeds. These are user-overridable scale choices.
- If `uv sync --extra cnn` fails because of disk or wheel availability even after the 3.13 retry, run everything with `--encoders dsp`. Mark the CNN columns "not run (reason)" in `docs/RESULTS.md` and keep `encoder_cnn.py` with its importorskip test.
- If the full synthetic run takes more than 90 minutes at `--jobs 2`, rerun with `--seeds 0-4` and state that in RESULTS.md. Do not raise `--jobs` above 2: this host has about 1 GB of RAM available.
- If the smoke run shows uEngram `false_flag_rate_era0 > 0.05` with the DSP encoder, the only allowed adjustment is `alpha_gate` 1e-3 → 1e-4, applied to all uEngram variants and recorded in RESULTS.md.
- uEngram on CMAPSS uses `t_dormant=400`, because each engine lives about 190 windows.
