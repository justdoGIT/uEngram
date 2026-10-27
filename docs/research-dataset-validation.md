# Research note: validating and splitting host telemetry datasets

## Question

Can existing online tools create trustworthy training and evaluation sets for
uEngram host telemetry, or should the host package implement its own validator?

## Findings

### 1. Chronological splitting is mandatory

Host measurements are ordered in time. A random split can place measurements
from one operating episode in both training and evaluation. The model then
sees a near-duplicate future episode during training, which makes the score
look better than the real deployment result. This is a form of **data
leakage**: information from the answer period accidentally influences the
learning period.

The official
[scikit-learn `TimeSeriesSplit` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
provides chronological folds and a `gap` parameter. It also says that samples
should be equally spaced when fold durations are meant to be comparable. The
host collector can have missing ticks and long pauses, so applying
`TimeSeriesSplit` directly would hide an important quality problem. The
native `dataset split` command therefore uses timestamps, reports gaps, and
excludes windows around boundaries instead of pretending that row numbers are
uniform time.

The official
[scikit-learn common-pitfalls guide](https://scikit-learn.org/stable/common_pitfalls.html)
identifies leakage from fitting preprocessing on all data. For uEngram, every
normalizer, encoder, threshold, and memory baseline must be fitted on the
commissioning portion only. Calibration may select thresholds; test and
holdout must remain untouched until the final report.

### 2. Schema and distribution validation are useful, but heavy

[TensorFlow Data Validation](https://www.tensorflow.org/tfx/tutorials/tfx/penguin_tfdv)
provides statistics, schema inference, and anomaly checks for typed datasets.
It can find missing values, changed types, and distribution shifts. This is a
strong offline quality gate for a large data platform, but its TFX dependency
stack is too large for the small host package and is not needed to check the
few fixed telemetry fields already defined in `uengram.host.schema`.

[ pandera dataframe schemas ](https://pandera.readthedocs.io/en/stable/dataframe_schemas.html)
provides expressive column types, nullability, and checks. It is a good
choice for a pandas-based research notebook. It is not a runtime dependency
here because host collection currently uses JSONL, NumPy, and a restricted
node environment. The native validator keeps the same important invariants
without requiring pandas.

### 3. Experiment versioning is separate from runtime validation

[DVC's data and model versioning examples](https://doc.dvc.org/example-scenarios/versioning-data-and-models)
show how Git can track dataset metadata while large data remains external.
That is useful for recording the manifest hash, split file, model hash, and
metrics for repeatable research. DVC does not decide whether a window is
causal, whether a kernel changed, or whether an incident label is valid, so it
complements rather than replaces `dataset validate`.

[sktime's forecasting examples](https://www.sktime.org/en/latest/examples/01_forecasting.html)
are useful for offline temporal forecasting experiments and backtesting. They
are not required for the embedded export format and should not be introduced
as a deployment dependency.

## Decision for uEngram

Implement a small repository-native gate first:

1. Validate manifest format, row counts, finite numeric fields, time ordering,
   window length, kernel consistency, model-kernel agreement, label references,
   and prediction/label separation.
2. Emit causal label phases: `baseline`, `pre_incident`, `incident`, and
   `recovery`. A phase helps analysis distinguish an early warning from the
   fault itself; it does not create physical ground truth.
3. Generate chronological `commission`, `calibration`, `test`, and `holdout`
   assignments using the common time range across nodes.
4. Exclude boundary guard intervals and incident events that would otherwise
   span multiple usable splits.
5. Keep raw telemetry and the split metadata separate, so the same dataset can
   be re-evaluated with a different model without rewriting source data.

External tools remain recommended for offline analysis:

| Tool | Best use | Why it is not the host runtime gate |
|---|---|---|
| scikit-learn `TimeSeriesSplit` | Quick chronological folds | Equal-spacing assumption and no host-specific label checks |
| TensorFlow Data Validation | Large pipeline schema/drift checks | Too heavy for a restricted node |
| Pandera | Notebook/dataframe contracts | Requires a dataframe stack not needed by JSONL export |
| DVC | Dataset/model/metric versioning | Does not validate telemetry semantics |
| sktime | Forecasting backtests | Scope is broader than export integrity |

## Remaining limitations

The validator cannot prove that a label describes a real mechanical failure.
It can only prove that the label is well formed and attached to the correct
node and time interval. Independent operator, maintenance, or test-rig
records are still required. A single host also cannot establish cross-host
causality; cluster campaigns should provide synchronized clocks and common
split boundaries. Finally, a distribution can be valid while still being
unrepresentative, so later work should add offline drift reports and
cross-kernel stratified metrics.
