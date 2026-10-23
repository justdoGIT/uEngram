"""Archive native compressor readings around model-window decisions.

This is an offline evidence recorder. It preserves source rows and timestamps;
post-decision rows are explicitly marked and are never represented as available
when the detector opened an alarm.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np


SOURCE_PATHS = {
    "metropt3": "data/metropt3/MetroPT3(AirCompressor).csv",
    "metropt2_jan": "data/metropt2/dataset_train.csv",
    "metropt2_apr": "data/metropt2/MetroPT2.csv",
}


def _timestamp(value):
    value = value.strip().replace(" ", "T")
    # Preserve the original string in the archive; parse only for interval tests.
    return np.datetime64(value, "ns")


def _iso(value):
    return np.datetime_as_string(value, unit="ns")


def _row_timestamp(header):
    for candidate in ("timestamp", "Timestamp", "time", "Time", "date", "Date"):
        if candidate in header:
            return header.index(candidate)
    # MetroPT-3 has an unnamed index then timestamp as its second column.
    if len(header) > 1 and header[0] == "":
        return 1
    raise ValueError(f"could not identify timestamp column in {header!r}")


def _source_path(stream_name, source_path):
    path = source_path or SOURCE_PATHS.get(stream_name)
    if path is None:
        raise ValueError(f"no native CSV source configured for stream {stream_name!r}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"native source CSV not found: {path}")
    return os.path.abspath(path)


def _context(stream, ep, test_indices, radius):
    pos = int(ep["open_index"])
    if pos < 0 or pos >= len(test_indices):
        raise IndexError(f"open_index {pos} outside test stream of length {len(test_indices)}")
    center = int(test_indices[pos])
    members, missing_offsets = [], []
    for offset in range(-radius, radius + 1):
        index = center + offset
        if 0 <= index < len(stream.W) and stream.segment[index] == stream.segment[center]:
            members.append(index)
        else:
            missing_offsets.append(offset)
    return center, members, missing_offsets


def _same_second(value, second):
    return value.astype("datetime64[s]") == second.astype("datetime64[s]")


def _expected_final(stream, center):
    expected = {}
    if getattr(stream, "W", None) is not None and np.ndim(stream.W) == 3:
        # Compressor streams store channels x samples; last source sample in window.
        names = getattr(stream, "extra", {}).get("analog_channels")
        if names is None:
            from uengram.metropt import ANALOG
            names = ANALOG
        expected.update({str(name): float(stream.W[center, i, -1])
                         for i, name in enumerate(names) if i < stream.W.shape[1]})
        cond_names = ("COMP", "DV_eletric", "MPG", "LPS")
        if getattr(stream, "cond", None) is not None:
            expected.update({name: float(stream.cond[center, i, -1])
                             for i, name in enumerate(cond_names) if i < stream.cond.shape[1]})
    return expected


def archive_native(stream, episodes, path, run, radius=2, *, source_path=None,
                   trigger_type="predicted_alarm", provenance=None, progress_interval=30):
    """Write native source rows in the union of same-segment model context windows.

    ``open_index`` is a position in ``flatnonzero(stream.test)``. CSV is scanned
    once; rows are emitted only if their timestamp lies in requested intervals.
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")
    episodes = list(episodes)
    if not episodes:
        raise ValueError("native snapshot requires at least one episode")
    source = _source_path(stream.name, source_path)
    test_indices = np.flatnonzero(stream.test)
    contexts, requests = [], []
    for episode_id, episode in enumerate(episodes):
        center, members, missing_offsets = _context(stream, episode, test_indices, radius)
        start = np.datetime64(stream.start[members[0]], "s").astype("datetime64[ns]")
        end = (np.datetime64(stream.end[members[-1]], "s") + np.timedelta64(1, "s")
               - np.timedelta64(1, "ns")).astype("datetime64[ns]")
        contexts.append((center, members, missing_offsets))
        requests.append({"episode_id": episode_id, "start": start, "end": end,
                         "model_start": np.datetime64(stream.start[members[0]], "s"),
                         "model_end": np.datetime64(stream.end[members[-1]], "s"),
                         "missing_offsets": missing_offsets, "center": center, "members": members})
    # Merge overlapping/touching intervals so filtering and missingness use the union.
    ordered = sorted(requests, key=lambda x: x["start"])
    merged = []
    for req in ordered:
        if merged and req["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], req["end"])
            merged[-1]["episode_ids"].append(req["episode_id"])
        else:
            merged.append({"start": req["start"], "end": req["end"],
                           "episode_ids": [req["episode_id"]]})
    interval_starts = [item["start"] for item in merged]
    digest = hashlib.sha256()
    selected, header, timestamp_index = [], None, None
    last_progress = time.monotonic()
    with open(source, "rb") as raw:
        line_no = 0
        while True:
            line = raw.readline()
            if not line:
                break
            digest.update(line)
            line_no += 1
            if line_no == 1:
                header = next(csv.reader([line.decode("utf-8-sig").rstrip("\r\n")]))
                timestamp_index = _row_timestamp(header)
                continue
            parts = next(csv.reader([line.decode("utf-8").rstrip("\r\n")]))
            if not parts or len(parts) != len(header):
                continue
            stamp = _timestamp(parts[timestamp_index])
            interval_index = bisect_right(interval_starts, stamp) - 1
            if interval_index >= 0 and stamp <= merged[interval_index]["end"]:
                selected.append((line_no - 2, stamp, parts, hashlib.sha256(line).hexdigest()))
            now = time.monotonic()
            if progress_interval and now - last_progress >= progress_interval:
                print(f"[native_snapshots] scanned {line_no - 1:,} source rows; "
                      f"selected {len(selected):,}", file=sys.stderr, flush=True)
                last_progress = now
    source_hash = digest.hexdigest()
    if header is None:
        raise ValueError("source CSV is empty")

    # Native decision is the exact source sample in the model endpoint's rounded second
    # whose modeled channels equal the final cached model observation.
    episode_records = []
    for episode_id, (center, members, missing_offsets) in enumerate(contexts):
        expected = _expected_final(stream, center)
        end_second = np.datetime64(stream.end[center], "s")
        matching = []
        for item in selected:
            rowno, stamp, parts, row_hash = item
            if not _same_second(stamp, end_second):
                continue
            matches = True
            for name, value in expected.items():
                if name not in header:
                    matches = False
                    break
                try:
                    observed = float(parts[header.index(name)])
                except ValueError:
                    matches = False
                    break
                if not (np.float32(observed) == np.float32(value)
                        or (np.isnan(observed) and np.isnan(value))):
                    matches = False
                    break
            if matches and expected:
                matching.append(item)
        decision = matching[0][1] if matching else None
        request = requests[episode_id]
        available = [x for x in selected if request["start"] <= x[1] <= request["end"]]
        causal = [bool(x[1] <= decision) if decision is not None else False for x in available]
        rec = dict(episodes[episode_id])
        stamps = [x[1] for x in available]
        context_gaps = []
        if len(stamps) > 1:
            deltas = np.diff(np.asarray(stamps, dtype="datetime64[ns]").astype(np.int64)) / 1e9
            typical = float(np.median(deltas[deltas > 0])) if np.any(deltas > 0) else None
            threshold = max(1.0, 1.5 * typical) if typical is not None else None
            if threshold is not None:
                for gap_index in np.flatnonzero(deltas > threshold):
                    context_gaps.append({"after": _iso(stamps[gap_index]),
                                         "before": _iso(stamps[gap_index + 1]),
                                         "seconds": float(deltas[gap_index])})
        missing_pre_windows = [offset for offset in missing_offsets if offset < 0]
        missing_post_windows = [offset for offset in missing_offsets if offset > 0]
        missing_pre_source = bool(not available or (
            available[0][1].astype("datetime64[s]") > request["model_start"]))
        missing_post_source = bool(any(i > center for i in members) and (
            not available or available[-1][1].astype("datetime64[s]") < request["model_end"]))
        rec.update({"episode_id": episode_id, "trigger_type": trigger_type,
                    "input_index": center, "input_end_cached_seconds": str(stream.end[center]),
                    "decision_timestamp_native": _iso(decision) if decision is not None else None,
                    "causal_boundary_known": decision is not None,
                    "causal_boundary_uncertainty": None if decision is not None else
                        "no native row matched all modeled final-sample values in the cached endpoint second",
                    "matching_final_native_rows": len(matching),
                    "context_model_indices": members,
                    "missing_requested_pre_model_offsets": missing_pre_windows,
                    "missing_requested_post_model_offsets": missing_post_windows,
                    "context_start": _iso(request["start"]),
                    "context_end": _iso(request["end"]),
                    "context_row_count": len(available),
                    "causal_row_count": sum(causal) if decision is not None else None,
                    "post_decision_row_count": len(causal) - sum(causal) if decision is not None else None,
                    "context_gaps": context_gaps,
                    "model_final_sample_match": bool(matching),
                    "model_final_channels_checked": sorted(expected),
                    "missing_context": len(available) == 0,
                    "missing_pre_context": bool(missing_pre_windows) or missing_pre_source,
                    "missing_post_context": bool(missing_post_windows) or missing_post_source,
                    "availability_semantics": "only native rows at or before the exact matched decision sample are known causal"})
        if provenance is not None:
            rec["trigger_provenance"] = provenance
        episode_records.append(rec)
    ncol = len(header)
    # Raw strings preserve original representations; numeric array is convenience, NaN only
    # represents source cells that are blank/non-numeric, with raw_values retained verbatim.
    raw_values = np.asarray([x[2] for x in selected], dtype=str).reshape((-1, ncol))
    numeric = np.full(raw_values.shape, np.nan, dtype=np.float64)
    for row in range(len(selected)):
        for col in range(ncol):
            try:
                numeric[row, col] = float(raw_values[row, col])
            except ValueError:
                pass
    timestamps = np.asarray([_iso(x[1]) for x in selected], dtype=str)
    row_numbers = np.asarray([x[0] for x in selected], dtype=np.int64)
    row_hashes = np.asarray([x[3] for x in selected], dtype=str)
    causal_mask = np.zeros((len(episodes), len(selected)), dtype=bool)
    causal_known = np.zeros((len(episodes), len(selected)), dtype=bool)
    episode_map = np.full((len(episodes), len(selected)), -1, dtype=np.int64)
    model_window_map = np.full((len(episodes), len(selected)), -1, dtype=np.int64)
    for i, rec in enumerate(episode_records):
        decision = (_timestamp(rec["decision_timestamp_native"])
                    if rec["decision_timestamp_native"] is not None else None)
        request = requests[i]
        for j, item in enumerate(selected):
            if request["start"] <= item[1] <= request["end"]:
                episode_map[i, j] = i
                if decision is not None:
                    causal_known[i, j] = True
                    causal_mask[i, j] = item[1] <= decision
        for model_index in contexts[i][1]:
            model_start = np.datetime64(stream.start[model_index], "s").astype("datetime64[ns]")
            model_end = (np.datetime64(stream.end[model_index], "s") + np.timedelta64(1, "s")
                         - np.timedelta64(1, "ns")).astype("datetime64[ns]")
            for j, item in enumerate(selected):
                if model_start <= item[1] <= model_end:
                    model_window_map[i, j] = model_index
    manifest = {"stream": stream.name, "trigger_type": trigger_type, "source_path": source,
                "source_sha256": source_hash, "source_row_indexing": "zero-based data-row index",
                "header": header, "units": {name: "unknown (not assigned by source metadata)" for name in header},
                "timestamp_column": header[timestamp_index], "timestamp_precision": "source text preserved; parsed at nanoseconds",
                "sampling_diagnostics": _sampling_diagnostics(selected),
                "requested_intervals": [{"episode_id": r["episode_id"], "start": _iso(r["start"]),
                                         "end": _iso(r["end"])} for r in requests],
                "merged_intervals": [{"start": _iso(m["start"]), "end": _iso(m["end"]),
                                      "episode_ids": m["episode_ids"]} for m in merged],
                "code_sha256": _code_sha256(),
                "run": run, "provenance": provenance, "episodes": episode_records}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, row_number=row_numbers, row_sha256=row_hashes,
                        timestamp=timestamps, header=np.asarray(header), raw_values=raw_values,
                        numeric_values=numeric, episode_causal_mask=causal_mask,
                        episode_causal_known=causal_known,
                        episode_row_map=episode_map, episode_model_window_map=model_window_map,
                        manifest=json.dumps(manifest, default=str))
    with open(path, "rb") as archive:
        archive_hash = hashlib.file_digest(archive, "sha256").hexdigest()
    return {"path": path, "sha256": archive_hash, "source_sha256": source_hash,
            "bytes": os.path.getsize(path), "episodes": len(episodes),
            "native_rows": len(selected), "missing_context_episodes": sum(r["missing_context"] for r in episode_records),
            "model_final_sample_mismatches": sum(not r["model_final_sample_match"] for r in episode_records)}


def _code_sha256():
    with open(__file__, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()

def _sampling_diagnostics(selected):
    if len(selected) < 2:
        return {"observations": len(selected), "positive_delta_count": 0,
                "median_interval_seconds": None, "max_interval_seconds": None,
                "gaps_over_60_seconds": 0}
    stamps = np.asarray([x[1] for x in selected], dtype="datetime64[ns]").astype(np.int64)
    delta = np.diff(stamps) / 1e9
    positive = delta[delta > 0]
    return {"observations": len(selected), "positive_delta_count": int(len(positive)),
            "median_interval_seconds": float(np.median(positive)) if len(positive) else None,
            "max_interval_seconds": float(np.max(positive)) if len(positive) else None,
            "gaps_over_60_seconds": int(np.sum(positive > 60))}


def _read_episodes(path):
    with open(path) as fh:
        content = fh.read()
    try:
        value = json.loads(content)
        return value.get("episodes", value) if isinstance(value, dict) else value
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        if len(rows) == 1 and isinstance(rows[0], dict) and "episodes" in rows[0]:
            return rows[0]["episodes"]
        return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episodes-json")
    source.add_argument("--results")
    parser.add_argument("--out", required=True)
    parser.add_argument("--method")
    parser.add_argument("--policy")
    parser.add_argument("--run-json", help="JSON file containing method/config/model provenance")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--source-csv")
    parser.add_argument("--trigger-type", default="predicted_alarm")
    parser.add_argument("--provenance", help="JSON object or free-text provenance")
    args = parser.parse_args(argv)
    try:
        provenance = json.loads(args.provenance) if args.provenance else None
    except json.JSONDecodeError:
        provenance = args.provenance
    if args.results:
        with open(args.results) as fh:
            candidates = [json.loads(line) for line in fh if line.strip()]
        matches = [r for r in candidates if r.get("stream") == args.stream
                   and (not args.method or r.get("method") == args.method)
                   and (not args.policy or r.get("policy") == args.policy)]
        if len(matches) != 1:
            raise SystemExit(f"expected exactly one matching result record, found {len(matches)}")
        result = matches[0]
        episodes = result.get("details", {}).get("episodes", [])
        run = {k: result.get(k) for k in ("protocol", "stream", "method", "encoder", "seed", "policy", "config")}
    else:
        episodes = _read_episodes(args.episodes_json)
        run = {"stream": args.stream, "episode_source": os.path.abspath(args.episodes_json)}
    if args.run_json:
        with open(args.run_json) as fh:
            run.update(json.load(fh))
    from bench.streams import load_stream
    stream = load_stream(args.stream)
    info = archive_native(stream, episodes, args.out, run, radius=args.radius,
                          source_path=args.source_csv, trigger_type=args.trigger_type,
                          provenance=provenance)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
