# NASA CMAPSS FD001: RUL and health-monitoring practice

## What the benchmark represents and standard preparation

NASA describes each engine as a separate multivariate time series, with 24 sensor measurements plus three operating settings, noise, and unit-to-unit initial wear/manufacturing variation that is normal rather than fault. FD001 has 100 run-to-failure training engines and 100 truncated test engines, one operating condition (sea level) and one fault mode (HPC degradation); NASA supplies one true RUL per test engine. [NASA Open Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

The common 14-sensor subset is **s2,s3,s4,s7,s8,s9,s11,s12,s13,s14,s15,s17,s20,s21** (sensor columns numbered in original data; remove constant/near-constant channels and retain these useful ones). Preprocessing is not fully standardized across papers: z-score `(x−μ)/σ` and min-max scaling both appear. Fit scaling on training data only; apply unchanged to validation/test. A frequent sequence setup is sliding windows of length 30, stride 1; many studies use clipped piecewise target `RUL(t)=min(Rmax,T−t)`, with 125 or 130 cycles common. Lai et al. explicitly use window 30, cap 125, z-score and report 125 works well on FD001; Pasa et al. test 130/100/80 and call 130 common. [Lai et al. (2022), methods/results](https://arxiv.org/html/2212.05772v1); [Pasa et al. (2019), PDF](https://www.papers.phmsociety.org/index.php/phmconf/article/download/786/phmc_19_786)

Operating-condition normalization is **not necessary for FD001's single condition**, but is important for FD002/FD004 (six regimes): cluster/assign by the three operating settings, then normalize each sensor using per-condition training statistics. Pasa et al. show operating-condition variation can overwhelm degradation signal; Lai et al. likewise cluster settings then z-score within condition. [Pasa et al.](https://www.papers.phmsociety.org/index.php/phmconf/article/download/786/phmc_19_786); [Lai et al.](https://arxiv.org/html/2212.05772v1)

## Reported FD001 benchmark values

Scores below are reported values from Lai et al.’s comparative table (test FD001, RMSE and PHM08 score); methods in that table cite their original work. They are useful reference points, **not a controlled re-run**: preprocessing, cap, feature subset, window, and evaluation implementations differ among papers. Original “winner” wording must be handled carefully: Wang et al.’s similarity-based method won the PHM08 competition for its dataset; it is not a like-for-like FD001 score. [Lai et al., Table 3](https://arxiv.org/html/2212.05772v1); [Wang et al. (2008), DOI](https://doi.org/10.1109/PHM.2008.4711421)

| Method (paper/source) | FD001 RMSE | PHM08 score | Paper |
|---|---:|---:|---|
| Deep LSTM + feedforward (Zheng et al. 2017) | 16.14 | 338 | [DOI](https://doi.org/10.1109/ICPHM.2017.7998311); reported in [Lai Table 3](https://arxiv.org/html/2212.05772v1) |
| CNN + FNN (Babu et al.; table labels DCNN, 2017) | 12.61 | 273 | [Springer 2016 chapter](https://link.springer.com/chapter/10.1007/978-3-319-32025-0_14); [Lai Table 3](https://arxiv.org/html/2212.05772v1) |
| 2-D deep CNN (Li, Ding & Sun 2018) | ~12.6* | — | [Reliability Engineering & System Safety](https://doi.org/10.1016/j.ress.2017.11.021) |
| Directed acyclic graph CNN-LSTM (2019) | 11.96 | 229 | [Lai Table 3](https://arxiv.org/html/2212.05772v1) |
| Attention LSTM (2020) | 14.53 | 322 | [Lai Table 3](https://arxiv.org/html/2212.05772v1) |
| Distributed attention TCN (2020) | 11.78 | 229 | [Lai Table 3](https://arxiv.org/html/2212.05772v1) |
| Multi-dimensional attention + LSTM, feature attention | 11.71 | 223 | [Lai et al. 2022](https://arxiv.org/html/2212.05772v1) |
| Multi-dimensional attention + LSTM, feature + time attention | **11.43** | **209** | [Lai et al. 2022](https://arxiv.org/html/2212.05772v1) |
| Classical MLP / SVR / RVR / CNN (PHM08 challenge test—not FD001) | not reported | 3212 / 15886 / 8242 / 2056 | [Hou et al. 2020, Table 4](https://pmc.ncbi.nlm.nih.gov/articles/PMC7416243/) |
| Linear regression, random forest, SVR FD001 comparison | **No comparable numeric values verified here** | — | Avoid filling with numbers from incompatible splits/feature protocols. |

*The ~12.6 shorthand is the commonly cited value, while the comparative source lists the related DCNN benchmark at 12.61; paper configurations and dataset naming can vary. The table values above anchor the numerics to the source where they were actually visible.*

## Health index: make health a trajectory, not a raw point

A health index (HI) compresses multiple sensor measurements into a scalar ordered along degradation. A supervised construction can map features `x` to a continuous target such as `HI*=clip(1−RUL/Rmax,0,1)` (healthy near 0, failure near 1), using linear regression or another regressor; fit the mapping on run-to-failure training units, then smooth/regularize the HI over time. A PCA HI instead projects standardized, selected sensor features onto a principal degradation direction; orient the sign so degradation increases, and check monotonicity/consistency before mapping HI to RUL. Pasa et al. describe the common two-stage sensor→HI→RUL design and smoothing/monotonicity; Alomari et al. apply rolling statistical features and PCA; the cited literature documents both approaches. [Pasa et al.](https://www.papers.phmsociety.org/index.php/phmconf/article/download/786/phmc_19_786); [Alomari et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10439180/)

Crucially, align/normalize each engine to its own early-life baseline (e.g., first healthy cycles or unit intercept) before interpreting absolute levels. NASA says initial wear/manufacturing variation differs by unit and is normal; without per-unit baseline handling, a naturally high/low sensor offset can masquerade as advanced/healthy degradation. Do not normalize away the later within-engine trend. [NASA Open Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

## Unsupervised degradation detection and feature signal

I found **no verified published FD001 unsupervised late-life AUROC or detection-delay result** in the opened primary sources. Mark such figures UNVERIFIED rather than infer them from supervised RUL regression or anomaly scores. Note what evidence does support: Hou et al. cite an LSTM-autoencoder-derived unsupervised HI in prior work, but their own reported experiment is supervised RUL and does not establish an AUROC/delay for an RUL<20–30 detector. [Hou et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7416243/)

Why generic pointwise kNN is a poor RUL model: sensor observations have noise and engine-specific starting offsets, whereas degradation is a gradual **within-engine temporal trajectory**. Euclidean proximity between two raw points may match different life stages because of offset/noise; it discards direction and rate. Prefer trailing-window statistics—level/mean, slope or first differences, dispersion, and optionally filtered/latent HI—after per-engine baseline and operating-condition treatment. This is a methodological inference grounded in NASA’s explicit offset/noise statement and papers’ use of temporal windows, trend features, and smoothing. [NASA](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data); [Alomari et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10439180/); [Pasa et al.](https://www.papers.phmsociety.org/index.php/phmconf/article/download/786/phmc_19_786)

## Evaluation protocol and score

Train on the 100 complete FD001 training engines; predict one RUL for the last available cycle of each of the 100 test engines and compare against the corresponding values in `RUL_FD001.txt` (do not score every overlapping window as independent test units). Report RMSE and the PHM08 asymmetric score, with split and RUL cap stated. NASA confirms the supplied test truth vector. [NASA Open Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

Let `d_i = predicted_RUL_i − true_RUL_i`. The PHM08 score is

`S = Σ_i [ exp(−d_i/13) − 1, if d_i < 0;  exp(d_i/10) − 1, if d_i ≥ 0 ]`.

Thus overestimating remaining life (late warning) is penalized more steeply. This exact convention and constants are stated by Lai et al. [Lai et al., metric definition](https://arxiv.org/html/2212.05772v1).