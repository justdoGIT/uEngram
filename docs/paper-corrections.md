# Paper corrections

## October 2026 claim audit

- Regenerated synthetic d=36 and SKAB dimension-sweep results at HEAD after finding the committed artifacts stale.
- Reinterpreted the CMAPSS horizon metric as alarm-onset timing, not independently established early warning.
- Clarified that 12 of 42 OS-Metrics events begin before the test split and are not scored there.
- Corrected the MetroPT-3 lead-time sign convention: negative values mean alarms precede report start.
- Clarified that write reduction counts write operations; total bytes written rise by 37% because raw windows are pinned.
- Corrected Table A10's averaging: it averaged two alarm policies.
- Withdrew the retired NASA IMS “63.8-hour early warning” claim. The evaluation was invalid: its script used hardcoded degradation indices and an assumed healthy prefix.
- EngineAD: reported with a Colab GPU provenance note. The artifact's `script_sha256` field holds the label `colab-gpu-sweep-v1`, not a hash, so the committed writer script is identified by that label only. Medians now use the primary first-point-label convention (0.539, 0.556, 0.537; earlier text used the any-positive variant: 0.538, 0.556, 0.536).

## October 2026 standalone rewrite

- Rewrote the paper as a self-contained article with Related Work, Method, Experimental Setup, Results, Limitations, Conclusion and Data and Code Availability sections. The paper no longer cites file paths, line numbers, commit ids, module names, result-file names or internal version labels. These details live in `docs/reproducing-the-paper.md` and `docs/results-ledger.md`.
- Removed the 43-table results ledger from the paper. It is generated into `docs/results-ledger.md`.
- Removed the competitive-admission comparison (16-slot budget, event recall 0.760 to 0.851) because it was cited from an older note with no committed result file. **Superseded:** the stated reason for distrusting it, a rerun showing identical results with admission disabled, was itself wrong; see the improvement pass below. The comparison is restored in Section 5.1 from a committed artifact.
- Removed the unmeasured "30-minute CPU cap" wording for the SKAB teacher gate: the code enforces no such cap.
- Table 7 (CNN readouts) was regenerated from the released pipeline: 146 of 150 per-seed values identical, four differ by less than 4e-6.
- Added the synthetic comparison table (Table 4) from the committed d=16 and d=36 result files, with event retention, write reduction and AP per method.
- Figure 7 panel B (formerly Figure 6) now reads its CNN row from the SKAB sensor-fusion result file instead of hard-coded values. The EngineAD figure is kept as Figure 5.

## October 2026 improvement pass

- Added 95% percentile-bootstrap intervals (10,000 resamples over the 10 synthetic seeds) to Section 5.1 and 5.2, including a paired d=16-to-d=36 difference, so the width regression is shown to not be a seed artifact.
- Added Section 5.10 and Table 10: μEngram against two external streaming baselines, half-space trees and MemStream [6], on MetroPT-3 and SKAB, under the same frozen-alarm-selection protocol as the rest of the paper. New script `bench.skab_baselines`; new artifact `results/skab_baselines.json`.
- While regenerating the MetroPT-3 baseline numbers, found that `results/metropt_runs.jsonl` (the CNN-encoder row cited in Table 5) and the DSP uEngram row of `results/v2_development.jsonl` were stale relative to the current slot-admission code — the same class of staleness fixed earlier for the d=36 and SKAB-sweep files, here hiding in a committed lifecycle result rather than an obviously out-of-date one. Regenerated both at HEAD. MetroPT-3's primary result changed from "3.6 of 4 events (range 2–4); 10.2 false alarms per week" to "4 of 4 events in every seed; 12.6 false alarms per week"; the pre-report AUROC figures (0.07, 0.35), which come from a separate frozen-score artifact, did not move beyond rounding.
- Found that `bench.alarm_components` (writer of `results/v8_component_profile.json`) reads a live score cache under `~/.local/state/uengram/v2scores` rather than a committed input, and that cache was stale for the same reason. Regenerated it; the paper's cited pre-report AUROC values were unchanged at the reported precision.
- Regenerated `results/v2_metropt2.jsonl` at HEAD: event counts (0 of 3 January, 0 or 2 of 2 April) are unchanged; the April less-conservative-rule false-alarm rate moved from 14.9 to 17.8 per week, and Table 5's MetroPT-2 row was updated.
- Corrected Table 2's slot-budget entry: the 48/208 semantic/episodic split applies to the synthetic and CMAPSS experiments; the MetroPT, SKAB and SCANIA alarm experiments use 192/64 (`EngramConfig` defaults, unmodified by those runners). The earlier text stated 48/208 for every experiment.
- Verified the dataset split dates, window lengths and clean-negative exclusion buffers in Table 2 against the loaders; no other discrepancies found.
- Split `scripts/build_paper_pdf.py` (372 lines) into the entry point and a new `scripts/paper_render_lib.py`, both under the file-size limit, with no change to the built output (HTML byte-identical before and after).
- Trimmed the abstract from 250 to 247 words and added the baseline-comparison result to it.
- Reported results for the two newly measured ablations, with committed artifacts: competitive admission (`bench.admission_ablation`, `results/admission_ablation.json`) and noise whitening (`bench.whitening_ablation`, `results/whitening_ablation.json`). Competitive admission raises event retention at every saturating budget (for example 0.901 against 0.778 at 16 slots) and has no effect at 256, where the bank never saturates; whitening has a small effect on the synthetic stream and cuts SKAB detections from 5 to 2 of 34 when disabled.
- **Correction of an error made during this pass.** A first version of both ablations reported "identical results" and was about to be written into the paper. It was wrong: `bench.run_synth.main` runs workers in a spawn `multiprocessing` pool even with `--jobs 1`, and spawned workers re-import their modules, so a monkeypatch applied in the parent (the ablation switch) never reached them. Both arms ran the unpatched code. This was caught by re-running the arms in-process through `run_synth.run_one`, which showed a large admission effect (retention 0.194 against 0.009 at 4 slots). The final scripts call `run_one` in-process and document why. The same flaw invalidated the earlier "identical results at 16 slots" rerun cited above.
- Reworked the alarm-tuning result in Section 5.5 after regenerating `results/v7_alarm_definition_sweep_metropt3.jsonl` at HEAD: the earlier statement that no setting meets all three operational targets no longer holds. One setting of 1,302 does (4 of 4 events, 1.21% clean alarm time, 0.58 false alarms per week), through a corroboration vote that is true for 94.8% of clean time, and the paper now says so and cautions that one pass among 1,300 combinations is consistent with chance.

## October 2026 capsule restart-recovery study

- Added Section 5.14 and Tables 14 to 16 (restart recovery, SIGKILL campaigns, cost of durability), the Sensor Memory Capsule description in Section 3.5, and Tables A49 to A55 in the generated ledger.
- **Correction of an error made during this study.** A first version of Section 5.14 reported the capsule at "five to six times" the memory-only restart time and attributed it to durable commits. All timings came from the system temp directory, which is tmpfs on every host checked, where fsync is nearly free; they measured CPU work in checkpoint serialization. On tmpfs the capsule costs the same under full and normal synchronization. The cost on real disks, measured afterwards, is 8 to 490 ms per window under full synchronization. The artifacts now record the filesystem backing the temp directory so this cannot recur unnoticed.
- Removed the "ranked history" comparison between arms. The Lite-only value of 1 came from a stand-in store that deliberately returns only its newest record; the real EdgeMemory Lite library returned all four records.
- Replaced a count-based comparison of lifecycle records with a comparison of idempotency-key sets after the first aggregation flagged that counts disagreed: a restart without detector state emits two spurious records and omits one, which a count of 4 against 3 hid.
- Replaced an untested explanation for the lost alarm-open with a measured one (a warm-up ablation); it supports, without proving, that a cold restart re-baselines on already-drifted data.
- While collecting on-disk results, stale tmpfs artifacts from earlier runs were rsynced back into the new result directories. They were identified by the absent `storage` field, moved to the trash, and the runner now collects from a per-run subdirectory.
- The first SIGKILL campaigns used a recovery check that restored the final checkpoint; a cadence above one made that check wrong, so all campaigns were rerun with a check that recovers from the final on-disk state. Reported figures are from the reruns.
- Limits stated in the paper: SIGKILL and in-process reopening are not power loss, so power cuts, torn writes and the durability of relaxed synchronization are untested; the alarm records come from a fixed threshold rather than the calibrated policy; one host (`gigabyte-worker`) was unreachable and is not covered.
