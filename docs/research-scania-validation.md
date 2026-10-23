# SCANIA Component X validation report

This report records the first validation on a dataset that was not used to
select the uEngram configuration. The frozen procedure is
[validation-scania-protocol.md](validation-scania-protocol.md).

## Data and integrity

- Source: SCANIA Component X, SND/researchdata record `2024-34/3`.
- License: CC BY 4.0.
- Archive: `/home/kk/Backup/data/uengram/scania/ComponentX.zip`.
- The archive's bundled `sha256.txt` verified every extracted member.
- The validation stream contains 1,212,827 rows and 105 anonymized features:
  812,712 commissioning rows, 203,888 calibration rows, and 196,227 held-out
  validation rows from 5,046 vehicles.
- Final validation labels contain 4,910 class-0 vehicles, 16 class-1, 14
  class-2, 30 class-3, and 76 class-4 vehicles.

Training vehicles marked `in_study_repair == 0` were split deterministically by
`vehicle_id % 5`; validation rows were never used for fitting or calibration.
The detector and alarm configuration are unchanged from the protocol.

## Results

The primary result is the uncorroborated AND/control policy. Metrics are taken
at each vehicle's final observed validation row because SCANIA labels are
coarse time-to-repair classes, not exact failure intervals.

| Policy | Final-row alarm vehicles | Class-0 alarm rate | Class 1–4 recall |
|---|---:|---:|---:|
| AND/control | 978 / 5,046 | 19.287% | 22.794% |
| OR, no evidence control | 978 / 5,046 | 19.287% | 22.794% |
| OR, missingness tripwire | 1,049 / 5,046 | 20.713% | 23.529% |

The no-evidence OR control matched AND in active and opened arrays and in all
reported metrics. The missingness tripwire observed 3,937 missing-feature rows;
its OR run produced 129 `error`-triggered episodes, compared with 1,057 total
AND episodes and 1,176 total OR episodes.

The final-row novelty ranking was unchanged by the tripwire: AUROC 0.5410 and
average precision 0.0321. Class-specific final-row recall was 18.75%, 28.57%,
20.00%, and 23.68% for classes 1–4 under AND, and 18.75%, 28.57%, 20.00%, and
25.00% under the missingness tripwire.

An independent recomputation in `results/scania_validation_independent.json`
reconstructed the policies from the saved scores and matched the report.

## Interpretation and limits

The SCANIA validation labels establish relative proximity to a workshop repair,
not an exact failure time. Therefore this report makes no claim about hours of
early warning, false alarms per week, or compressor-domain transfer.

SCANIA also provides no independent physical threshold or alarm channel. The
OR evidence source is raw missingness, selected before scoring as a data-quality
tripwire. Its small recall increase is accompanied by a higher healthy alarm
rate; it is not a component-failure detector. The OR rule is implemented for
systems that can provide real independent evidence, but this SCANIA run does
not supply such evidence.
