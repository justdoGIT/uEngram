#!/usr/bin/env bash
# Run the capsule resilience experiment on fleet hosts and collect JSON artifacts.
#
# usage: scripts/capsule_target_run.sh [host ...]    (default: msi hp asus-worker lenovo-worker)
# "msi" runs locally through uv; any other host gets a minimal tree (src, bench, one SMD machine)
# rsynced to a scratch directory and runs with its system python3 + numpy.
# Artifacts: $OUT/<label>/<crash>-<n>.json, OUT defaults to ~/.local/state/uengram/capsule-targets.
# Copy them into results/capsule_resilience/ and aggregate with bench.capsule_resilience_summary.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out=${OUT:-$HOME/.local/state/uengram/capsule-targets}
machine='machine-1-1'
windows=64
repeats=5
remote=.local/state/uengram-capsule-run
crashes=(after_wal after_ingest after_checkpoint)

stamp() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

run_local() {
    local label=$1 crash=$2 n=$3
    (cd "$repo" && uv run python -m bench.capsule_resilience \
        --stream smd --machine "$machine" --windows "$windows" --crash "$crash" \
        --target-label "$label" --require-target-label \
        --out "$out/$label/$crash-$n.json" >/dev/null)
}

sync_remote() {
    local host=$1
    ssh -o BatchMode=yes "$host" "mkdir -p '$remote/data/smd/train' '$remote/data/smd/test' '$remote/data/smd/test_label'"
    rsync -a --delete --exclude '__pycache__' "$repo/src" "$repo/bench" "$host:$remote/"
    local kind
    for kind in train test test_label; do
        rsync -a "$repo/data/smd/$kind/$machine.txt" "$host:$remote/data/smd/$kind/"
    done
}

run_remote() {
    local host=$1 label=$2 crash=$3 n=$4
    ssh -o BatchMode=yes "$host" "cd '$remote' && mkdir -p results && PYTHONPATH=src python3 -m bench.capsule_resilience \
        --stream smd --machine '$machine' --windows '$windows' --crash '$crash' \
        --target-label '$label' --require-target-label \
        --out 'results/$crash-$n.json' >/dev/null"
}

hosts=("$@")
[ "${#hosts[@]}" -gt 0 ] || hosts=(msi hp asus-worker lenovo-worker)

for host in "${hosts[@]}"; do
    label=${host%-worker}
    mkdir -p "$out/$label"
    if [ "$host" = msi ]; then
        stamp "$label: local run"
    else
        stamp "$label: syncing tree"
        sync_remote "$host"
    fi
    for crash in "${crashes[@]}"; do
        for n in $(seq 1 "$repeats"); do
            if [ "$host" = msi ]; then
                run_local "$label" "$crash" "$n"
            else
                run_remote "$host" "$label" "$crash" "$n"
            fi
        done
        stamp "$label: $crash x$repeats done"
    done
    if [ "$host" != msi ]; then
        rsync -a "$host:$remote/results/" "$out/$label/"
    fi
    stamp "$label: collected $(find "$out/$label" -name '*.json' | wc -l) artifacts"
done
