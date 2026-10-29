# Paper corrections

## October 2026 claim audit

- Regenerated synthetic d=36 and SKAB dimension-sweep results at HEAD after finding the committed artifacts stale.
- Reinterpreted the CMAPSS horizon metric as alarm-onset timing, not independently established early warning.
- Clarified that 12 of 42 OS-Metrics events begin before the test split and are not scored there.
- Corrected the MetroPT-3 lead-time sign convention: negative values mean alarms precede report start.
- Clarified that write reduction counts write operations; total bytes written rise by 37% because raw windows are pinned.
- Corrected Table A10's averaging: it averaged two alarm policies.
- Withdrew the retired NASA IMS “63.8-hour early warning” claim. The evaluation was invalid: its script used hardcoded degradation indices and an assumed healthy prefix.
- Removed the EngineAD result from the paper because the artifact was produced on a Colab GPU and cannot be reproduced locally.

## October 2026 standalone rewrite

- Rewrote the paper as a self-contained article with Related Work, Method, Experimental Setup, Results, Limitations, Conclusion and Data and Code Availability sections. The paper no longer cites file paths, line numbers, commit ids, module names, result-file names or internal version labels. These details live in `docs/reproducing-the-paper.md` and `docs/results-ledger.md`.
- Removed the 43-table results ledger from the paper. It is generated into `docs/results-ledger.md`.
- Removed the competitive-admission comparison (16-slot budget, event recall 0.760 to 0.851). It was cited from an older note with no committed result file. A rerun with admission disabled (`episodic_admission_ratio` 0 against 1, five seeds, budget 16, DSP encoder) gave identical results (event retention 0.9015, write reduction 86.669%), so the comparison cannot support a claim.
- Removed the unmeasured "30-minute CPU cap" wording for the SKAB teacher gate: the code enforces no such cap.
- Table 7 (CNN readouts) was regenerated from the released pipeline: 146 of 150 per-seed values identical, four differ by less than 4e-6.
- Added the synthetic comparison table (Table 4) from the committed d=16 and d=36 result files, with event retention, write reduction and AP per method.
- Figure 6 panel B now reads its CNN row from the SKAB sensor-fusion result file instead of hard-coded values. The EngineAD figure was removed with the EngineAD result.
