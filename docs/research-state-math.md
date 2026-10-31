# Modeling machine state and choosing update rules

## 1. A machine is a hybrid dynamical system

A useful decomposition is a discrete mode (idle, loaded, cruise, etc.) plus a continuous physical state (temperature, pressure, speed, wear):

\[
s_t\in\{1,\ldots,K\},\quad x_{t+1}=A_{s_t}x_t+B_{s_t}u_t+w_t,\quad y_t=C_{s_t}x_t+v_t,
\]
with mode transition probabilities \(P(s_t=j\mid s_{t-1}=i)=\Pi_{ij}\). Switching state-space models combine a Markov chain with mode-specific continuous state dynamics; Ghahramani & Hinton give a variational treatment, and Fox et al. develop nonparametric switching linear dynamical systems (SLDS) allowing an unknown number of modes. A sticky HDP-HMM adds persistence to discourage rapid redundant switching. [Ghahramani & Hinton (2000)](https://direct.mit.edu/neco/article/12/4/831/6358/Variational-Learning-for-Switching-State-Space); [Fox et al. (2008), SLDS](https://proceedings.neurips.cc/paper/2008/file/950a4152c2b4aa3ad78bdd6b366cc179-Paper.pdf); [Fox et al. (2009), sticky HDP-HMM](https://arxiv.org/abs/0905.2592)

For an HMM with emission likelihood \(L_t(j)=p(y_t\mid s_t=j)\), the online forward filter is
\[
\tilde p_t(j)=L_t(j)\sum_i\Pi_{ij}p_{t-1}(i),\qquad p_t(j)=\tilde p_t(j)/\sum_\ell\tilde p_t(\ell).
\]
For an SLDS, each mode also carries a continuous-state Kalman posterior; the likelihood is the mode-specific predictive density and continuous posteriors are mixed/conditioned on mode. This preserves uncertainty and temporal transition probabilities. uEngram’s hard nearest-slot Mahalanobis gate instead makes a one-step accept/reject decision against a static slot: it has no transition prior, posterior probabilities, or explicit dynamics. [Harvey, Koopman & Penzer, state-space/Kalman formulation](https://www.stat.berkeley.edu/~brill/Stat248/messyts.pdf)

## 2. Abrupt steps: detect a change; do not smear it into an old state

For observations with pre-change density \(f_0\) and post-change density \(f_1\), Page’s one-sided CUSUM is
\[
S_t=\max(0,S_{t-1}+\log[f_1(y_t)/f_0(y_t)]),\quad \text{alarm if }S_t\ge h.
\]
It accumulates weak evidence until a threshold is crossed; reset/reinitialize the affected baseline after a confirmed step. Page introduced continuous inspection/CUSUM; Moustakides proved Page’s stopping rule optimal for Lorden’s minimax criterion under the independent-observation model (not universal optimality for arbitrary streams). [Page (1954), DOI](https://doi.org/10.1093/biomet/41.1-2.100); [Moustakides (1986), primary theorem](https://projecteuclid.org/journals/annals-of-statistics/volume-14/issue-4/Optimal-Stopping-Times-for-Detecting-Changes-in-Distributions/10.1214/aos/1176350164.full)

Page-Hinkley is a cumulative-mean-shift monitor: for target mean \(\mu_0\), set \(d_t=y_t-\mu_0-\delta\), \(C_t=C_{t-1}+d_t\), \(m_t=\min(m_{t-1},C_t)\), and alarm if \(C_t-m_t>\lambda\) (reverse sign for downward shifts). It is a convenient mean-specific CUSUM, not a full regime model. [Page-Hinkley implementation reference](https://riverml.xyz/0.13.0/api/drift/PageHinkley)

Bayesian Online Changepoint Detection (BOCPD) tracks run length \(r_t\), time since last change. With hazard \(H(r)\) and run-specific predictive likelihood \(\pi_t(r)=p(y_t\mid r_{t-1}=r, y_{1:t-1})\), unnormalized masses update as
\[
P(r_t=0,y_{1:t})=\sum_r P(r,y_{1:t-1})\pi_t(r)H(r),\quad
P(r_t=r+1,y_{1:t})=P(r,y_{1:t-1})\pi_t(r)[1-H(r)],
\]
then normalize. [Adams & MacKay (2007), equations and algorithm](https://arxiv.org/html/0710.3742v1)

ADWIN maintains an adaptive window and tests every eligible split: for old/new parts \(W_0,W_1\), shrink/drop old data when \(|\bar x_0-\bar x_1|>\epsilon\). For bounded [0,1] data, their Hoeffding threshold is \(\epsilon=\sqrt{(1/(2m))\ln(4/\delta') }\), with \(m^{-1}=n_0^{-1}+n_1^{-1}\) (paper’s multiple-test adjustment defines \(\delta'\)); it supplies false-positive/negative guarantees under stated assumptions. [Bifet & Gavaldà (2007), ADWIN](https://www.cs.upc.edu/~gavalda/papers/adwin06.pdf)

Detection delay means samples from a true change until alarm; false-alarm rate is spurious alarms during stability. Lowering thresholds/raising hazard improves speed at cost of false alarms; higher thresholds suppress false alarms but allow more post-change data to contaminate the old estimate. For maintenance, use an explicit reset/segment boundary (or a new regime posterior), not merely a faster EWMA.

## 3. Transition ramps and residual-based fault detection

For a first-order thermal/pressure lag toward setpoint \(u_t\),
\[
x_t=x_{t-1}+a(u_t-x_{t-1})+w_t,\quad 0<a<1.
\]
A step from \(u_0\) to \(u_1\) gives \(x_t-u_1=(1-a)^t(x_0-u_1)\): a predictable exponential ramp, not a new stationary class at every intermediate value. Estimate/condition on the operating input and lag; do not train a stationary-state covariance by absorbing the whole transient.

Model-based fault detection creates an expected measurement \(\hat y_t\) from the process model and monitors residual/innovation \(r_t=y_t-\hat y_t\), usually normalized by its predicted covariance. Analytical redundancy means comparing measured signals with model-implied relationships; parity equations eliminate unknown state/input terms, while observers generate residuals. A normal regime transition is explained by the model and yields small residuals; a fault is a model mismatch and persists in residuals (assuming the model captures the transition and the fault is observable). [Willsky (1984), analytical redundancy and robust failure detection](https://willsky.lids.mit.edu/publ_pdfs/43_pub_IEEE.pdf); [Isermann, *Fault-Diagnosis Systems*](https://link.springer.com/book/10.1007/3-540-30368-5)

## 4. Slow drift: local level and exact EWMA/Kalman equivalence

For slowly changing wear or baseline, use the local-level model
\[
\ell_t=\ell_{t-1}+w_t,\quad y_t=\ell_t+v_t,\quad w_t\sim N(0,q_w),\;v_t\sim N(0,r_v),
\]
where \(q=q_w/r_v\) is the process/measurement noise ratio. The scalar Kalman update is \(\hat\ell_t=\hat\ell_{t|t-1}+K_t(y_t-\hat\ell_{t|t-1})\). At steady state, this is EWMA: \(\hat\ell_t=(1-K)\hat\ell_{t-1}+Ky_t\). Solving the Riccati equation gives the exact smoothing constant
\[
K=\frac{\sqrt{q^2+4q}-q}{2},\qquad q=\frac{K^2}{1-K}.
\]
Thus higher process noise relative to sensor noise means faster adaptation; q→0 gives K→0, q→∞ gives K→1. This is an optimal model-based gain, unlike arbitrary fixed alpha. [Harvey, Koopman & Penzer, local-level model and EWMA](https://www.stat.berkeley.edu/~brill/Stat248/messyts.pdf); [Muth (1960), original EWMA result](https://doi.org/10.1080/01621459.1960.10482064)

## 5. Progressive degradation and RUL

Degradation should be a latent health/damage trajectory, not a collection of stationary operating slots. Examples include a monotone damage state \(d_{t+1}=d_t+g(d_t,u_t;\theta)+\eta_t\), with failure at \(d\ge d_{crit}\), or a parametric exponential/power law \(d(t)=d_0+ct^b\) / \(d(t)=d_0+c(e^{bt}-1)\). Bayesian prognostics maintain posterior uncertainty over damage, parameters, and failure time; a particle filter propagates particles through the degradation dynamics, weights by sensor likelihood, resamples, then estimates RUL from first-passage times to the failure boundary. NASA’s C-MAPSS damage-propagation paper describes simulated turbofan degradation and run-to-failure data; algorithm choice depends on data/physics and assumptions, rather than one universally best method. [Saxena et al. (2008), NASA DASHlink C-MAPSS damage propagation](https://c3.ndc.nasa.gov/dashlink/resources/740); [An, Kim & Choi (2015), prognostics algorithm selection](https://doi.org/10.1016/j.ress.2014.10.006)

## 6. Few-sample Gaussian slots need predictive uncertainty

A slot estimated from few samples has uncertain mean and covariance. In one dimension, Normal-Inverse-Gamma prior \((\mu,\sigma^2)\) gives a Student-t posterior predictive, not a plug-in Gaussian:
\[
\mu,\sigma^2\mid D\sim NIG(\mu_n,\lambda_n,\alpha_n,\beta_n),\quad
x_{new}\mid D\sim t_{2\alpha_n}\!\left(\mu_n,\frac{\beta_n(1+1/\lambda_n)}{\alpha_n}\right),
\]
where the second argument is the t scale-squared. Multivariate normal-inverse-Wishart analogously gives a multivariate Student-t predictive. [Murphy (2007) conjugate Gaussian derivation](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf); [NIG posterior predictive formula](https://gnpalencia.org/cprior/formulas_models_normal.html)

If using sample mean \(\bar x\) and unbiased covariance \(S\) from n independent p-dimensional normal observations, the exact region for a *new observation* is
\[
(x_{new}-\bar x)'S^{-1}(x_{new}-\bar x)\le \frac{n+1}{n}\frac{p(n-1)}{n-p}F_{p,n-p;1-\alpha},\quad n>p.
\]
This is a prediction region (not the confidence region for the population mean). It accounts for finite-sample covariance/mean uncertainty and is heavier-tailed / more conservative than the asymptotic \(\chi^2_p\) gate; it approaches the chi-square form as n grows. [IIT Kanpur, Hotelling T² distribution derivation](https://home.iitk.ac.in/~shalab/multivariate/Chap5-multivariate-HotellingT2Distribution.pdf). For diagonal slots, use the corresponding dimension-wise predictive t scores or an explicitly justified approximation; chi-square calibration requires a known covariance and Gaussian predictive vector.

## 7. Robust updates: gate innovations before learning

With predicted residual \(e_t=y_t-\hat y_{t|t-1}\) and covariance \(S_t\), use normalized innovation squared \(d_t^2=e_t'S_t^{-1}e_t\); update normally only when consistent with the predictive gate, otherwise quarantine/downweight it and evaluate as a fault/change. This prevents a single impulse or transition mismatch from both shifting the mean and inflating variance. Robust M-estimation applies bounded influence, e.g. \(\psi_c(u)=\max(-c,\min(c,u))\), so a scalar location update is \(\mu_t=\mu_{t-1}+\eta s\psi_c((y_t-\mu_{t-1})/s)\); covariance updates should similarly use bounded weights. Minimum Covariance Determinant (MCD) estimates location/scatter from the h-subset with smallest covariance determinant, then reweights by robust distance; it resists masking/outliers, but is a batch initialization/consolidation estimator rather than a cheap per-sample recursion. [Huber (1964), robust estimation](https://doi.org/10.1214/aoms/1177703732); [Hubert, Debruyne & Rousseeuw, MCD](https://arxiv.org/html/1709.07045v1); [Bar-Shalom, Li & Kirubarajan, tracking/innovation-based consistency](https://doi.org/10.1002/0471221279)

## Recommended mapping

| State type | Recommended model | Update / step rule | Detection statistic |
|---|---|---|---|
| Stationary regime | Mode-specific emission + HMM/SLDS state | Forward posterior; mode-conditioned Kalman update | Mode posterior / predictive likelihood |
| Regime switch | Markov mode transition; CUSUM or BOCPD | Transition matrix; reset/new segment after evidence | CUSUM log-likelihood sum; BOCPD \(P(r_t=0\mid y_{1:t})\) |
| Transition ramp | First-order lag state-space response | \(x_t=x_{t-1}+a(u_t-x_{t-1})\) | Model residual normalized by innovation covariance |
| Slow wear | Local-level / health-index process | Kalman gain \(K=(\sqrt{q^2+4q}-q)/2\) | Drift in latent level or residual CUSUM |
| Maintenance step | Changepoint + post-maintenance baseline/health reset | Segment, reinitialize operating state; retain asset history | BOCPD/CUSUM plus maintenance event record |
| Transient fault | Nominal dynamics plus fault residual model | Gate/quarantine outlier; do not update normal slot | Whitened innovation / parity residual |
| Progressive degradation | Latent damage model + Bayesian particle filter | Propagate, likelihood-weight, resample; first passage to failure | Damage posterior, RUL quantiles, likelihood residual |

**Practical implication:** split the representation into mode, within-mode physical dynamics, slow health, and fault evidence. One Gaussian slot with a single flexible variance cannot both absorb legitimate thermal ramps and preserve sensitivity to real faults.