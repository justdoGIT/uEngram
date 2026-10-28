from pathlib import Path
import json
import statistics


def _json(repo: Path, name: str):
    return json.loads((repo / "results" / name).read_text())


def _fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(title, outcome, source, columns, rows, note):
    return {"title": title, "outcome": outcome, "source": source,
            "columns": columns, "rows": rows, "note": note}


def build(repo: Path) -> list[dict]:
    selection = _json(repo, "v8_interaction_selection.json")
    effects = _json(repo, "v8_interaction_effects.json")
    profile = _json(repo, "v8_component_profile.json")
    replay_path = repo / "results" / "v8_interaction_replay_metropt2.jsonl"
    replay = [json.loads(line) for line in replay_path.read_text().splitlines() if line.strip()]
    sweep_path = repo / "results" / "v8_interaction_sweep_metropt3.jsonl"

    rules = selection["rows_meeting_budget"]
    evaluated = selection["design_points"]
    within_budgets = selection["rows_within_both_budgets"]
    selection_rows = []
    for label, key in (("Default", "default_row"), ("Fallback selected", "selected"),
                       ("Most early warnings", "most_early_warnings_within_budgets")):
        item = selection[key]
        metrics = item["metrics"]
        selection_rows.append([
            label, str(item["index"]), _fmt(metrics["events_detected"], 0),
            _fmt(metrics["alarm_time_share_clean"], 4),
            _fmt(metrics["false_alarm_per_week"], 3),
            str(item["satisfies_recall_and_budget"]).lower(),
        ])
    selected_events = selection["selected"]["metrics"]["events_detected"]
    selection_note = (
        f"No design point met the recall-and-budget rule ({rules} rows) among {evaluated} evaluated; "
        f"{within_budgets} were within both budgets. Selection used the {selection['used']} rule; "
        f"its selected point detected {selected_events:.0f} events but failed the recall-and-budget rule."
    )
    tables = [_table(
        "MetroPT3 design selection and rule failure", "negative",
        "results/v8_interaction_selection.json",
        ["Point", "Index", "Events detected", "Clean alarm-time share", "False alarms/week", "Meets recall + budget"],
        selection_rows, selection_note,
    )]

    # Pool the reported median effects by parameter to keep the table compact while
    # retaining every tested level in the contributing counts.
    by_parameter = {}
    for key, result in effects.items():
        parameter = key.split("=", 1)[0]
        by_parameter.setdefault(parameter, []).append(result)
    effect_rows = []
    for parameter, items in by_parameter.items():
        effect_rows.append([
            parameter, str(sum(item["pairs"] for item in items)),
            _fmt(statistics.median(item["median"]["events_detected"] for item in items), 2),
            _fmt(statistics.median(item["median"]["alarm_time_share_clean"] for item in items), 4),
            _fmt(statistics.median(item["median"]["false_alarm_per_week"] for item in items), 3),
            _fmt(statistics.median(item["median"]["events_warned_before_report"] for item in items), 2),
        ])
    tables.append(_table(
        "Interaction effects by design parameter", "mixed",
        "results/v8_interaction_effects.json",
        ["Parameter", "Pairs represented", "Median Δ events", "Median Δ clean share", "Median Δ false alarms/week", "Median Δ early warnings"],
        effect_rows,
        f"Medians pool the per-level paired effects across {len(effects)} reported comparisons; positive and negative shifts coexist across parameter settings.",
    ))

    component_rows = []
    for stream, item in profile.items():
        rates = item["rates"]
        states = item["states"]
        sep = item["separability"]
        state2 = sep.get("state2", {})
        nov = state2.get("novelty", {})
        health = state2.get("health", {})
        clustering = item["clustering"]
        component_rows.append([
            stream, str(states["test"][0]), str(states["test"][1]), str(states["test"][2]),
            _fmt(rates["event/state2"]["novelty_above"], 3),
            _fmt(rates["clean/state2"]["novelty_above"], 3),
            _fmt(nov.get("clean_vs_event"), 3),
            _fmt(health.get("clean_vs_event"), 3),
            _fmt(clustering["observed_openings_per_week"], 3),
            _fmt(clustering["predicted_triggers_per_week_k3"]["counter"]["markov"], 3),
        ])
    tables.append(_table(
        "Component profile: state rates, separability, and clustering", "mixed",
        "results/v8_component_profile.json",
        ["Stream", "Test state 0 count", "Test state 1 count", "Test state 2 count",
         "Event novelty-above rate in state 2", "Clean novelty-above rate in state 2",
         "Novelty clean/event separation", "Health clean/event separation",
         "Observed openings per week", "Counter Markov triggers per week (k=3)"],
        component_rows,
        "State-2 signal rates and clean/event separability describe component discrimination; Markov trigger estimates track the observed clustering burden.",
    ))

    replay_rows = []
    for item in replay:
        m = item["metrics"]
        replay_rows.append([
            item["stream"], item["replay_of"], str(item["index"]),
            _fmt(m["events_detected"], 0), _fmt(m["alarm_time_share_clean"], 4),
            _fmt(m["false_alarm_per_week"], 3),
            f"{sum(bool(event['detected']) for event in item['events'])}/{len(item['events'])}",
        ])
    cases_per_stream = {}
    for item in replay:
        cases_per_stream[item["stream"]] = max(
            cases_per_stream.get(item["stream"], 0), len(item["events"])
        )
    total_cases = sum(cases_per_stream.values())
    detected_per_design = {
        item["replay_of"]: sum(bool(event["detected"]) for event in item["events"])
        for item in replay
    }
    total_detected = sum(detected_per_design.values())
    tables.append(_table(
        "MetroPT2 replay of selected and default MetroPT3 designs", "negative",
        "results/v8_interaction_replay_metropt2.jsonl",
        ["Stream", "Replay", "Index", "Events detected", "Clean alarm-time share", "False alarms/week", "Events detected / total"],
        replay_rows,
        f"Neither design detected an event: {total_detected} of {total_cases} distinct MetroPT2 events under both designs.",
    ))

    # The sweep has 1,728 rows; summarize each persistence strategy with medians.
    groups = {}
    n_sweep = 0
    with sweep_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            n_sweep += 1
            groups.setdefault(item["levels"]["persistence"], []).append(item["metrics"])
    sweep_rows = []
    for persistence, items in groups.items():
        sweep_rows.append([
            persistence, str(len(items)),
            _fmt(statistics.median(row["events_detected"] for row in items), 2),
            _fmt(statistics.median(row["alarm_time_share_clean"] for row in items), 4),
            _fmt(statistics.median(row["false_alarm_per_week"] for row in items), 3),
            _fmt(statistics.median(row["events_warned_before_report"] for row in items), 2),
        ])
    tables.append(_table(
        "MetroPT3 interaction sweep by persistence", "mixed",
        "results/v8_interaction_sweep_metropt3.jsonl",
        ["Persistence", "Design points", "Median events detected", "Median clean alarm-time share", "Median false alarms/week", "Median early warnings"],
        sweep_rows,
        f"The {n_sweep} MetroPT3 design points are aggregated by persistence rather than emitted individually; medians show the broad trade-off across the sweep.",
    ))
    return tables
