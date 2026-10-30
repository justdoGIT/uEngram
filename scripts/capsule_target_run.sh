#!/usr/bin/env bash
# Run a capsule experiment on fleet hosts and collect JSON artifacts.
#
# usage: [EXPERIMENT=resilience|cost] [STORAGE=tmpfs|disk] [OUT=dir] \
#        scripts/capsule_target_run.sh [host ...]    (default hosts: msi hp asus-worker lenovo-worker)
#
# "msi" runs locally through uv; any other host gets a minimal tree (src, bench, one SMD machine)
# rsynced to a scratch directory and runs with its system python3 + numpy.
#
# STORAGE=tmpfs uses the system temp directory, which is tmpfs on these hosts, so fsync is nearly
# free. STORAGE=disk points TMPDIR at a directory on the host's own disk. Each artifact records the
# filesystem that actually backed its temp directory; trust that field, not the name of the mode.
#
# resilience: $OUT/<label>/<crash>-<n>.json   (aggregate with bench.capsule_resilience_summary)
# cost:       $OUT/<label>/cost-<storage>.json
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
experiment=${EXPERIMENT:-resilience}
storage=${STORAGE:-tmpfs}
out=${OUT:-$HOME/.local/state/uengram/capsule-targets}
machine='machine-1-1'
windows=64
repeats=${REPEATS:-5}
cost_windows=${COST_WINDOWS:-256}
cost_repeats=${COST_REPEATS:-5}
remote_limit=${REMOTE_TIMEOUT:-1800} # seconds; a slow disk must not hang the batch
remote=.local/state/uengram-capsule-run
results_dir="results/$experiment-$storage" # one directory per run, so stale files are never collected
local_disk_tmp="$HOME/.local/state/uengram/capsule-tmp"
crashes=(after_wal after_ingest after_checkpoint)

case "$experiment" in resilience | cost) ;; *) echo "EXPERIMENT must be resilience or cost" >&2; exit 2 ;; esac
case "$storage" in tmpfs | disk) ;; *) echo "STORAGE must be tmpfs or disk" >&2; exit 2 ;; esac

stamp() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

if [ "$storage" = disk ]; then
    mkdir -p "$local_disk_tmp"
    local_env=(env "TMPDIR=$local_disk_tmp")
    remote_env="mkdir -p tmp && TMPDIR=\$PWD/tmp "
else
    local_env=(env)
    remote_env=""
fi

run_local() { # module arguments
    (cd "$repo" && "${local_env[@]}" uv run python -m "$@" >/dev/null)
}

run_remote() { # host, then module arguments
    local host=$1 quoted
    shift
    quoted=$(printf '%q ' "$@")
    ssh -o BatchMode=yes "$host" "cd '$remote' && mkdir -p '$results_dir' && ${remote_env}PYTHONPATH=src timeout $remote_limit python3 -m $quoted>/dev/null"
}

sync_remote() {
    local host=$1 kind
    ssh -o BatchMode=yes "$host" "mkdir -p '$remote/data/smd/train' '$remote/data/smd/test' '$remote/data/smd/test_label'"
    rsync -a --delete --exclude '__pycache__' "$repo/src" "$repo/bench" "$host:$remote/"
    for kind in train test test_label; do
        rsync -a "$repo/data/smd/$kind/$machine.txt" "$host:$remote/data/smd/$kind/"
    done
}

hosts=("$@")
[ "${#hosts[@]}" -gt 0 ] || hosts=(msi hp asus-worker lenovo-worker)

for host in "${hosts[@]}"; do
    label=${host%-worker}
    mkdir -p "$out/$label"
    if [ "$host" = msi ]; then
        stamp "$label: local $experiment on $storage"
    else
        stamp "$label: syncing tree"
        sync_remote "$host"
    fi
    if [ "$experiment" = cost ]; then
        args=(bench.capsule_cost_study --machine "$machine" --windows "$cost_windows" --repeats "$cost_repeats" --label "$label")
        if [ "$host" = msi ]; then
            run_local "${args[@]}" --out "$out/$label/cost-$storage.json"
        else
            run_remote "$host" "${args[@]}" --out "$results_dir/cost-$storage.json"
        fi
        stamp "$label: cost study done"
    else
        for crash in "${crashes[@]}"; do
            args=(bench.capsule_resilience --stream smd --machine "$machine" --windows "$windows"
                --crash "$crash" --target-label "$label" --require-target-label)
            for n in $(seq 1 "$repeats"); do
                if [ "$host" = msi ]; then
                    run_local "${args[@]}" --out "$out/$label/$crash-$n.json"
                else
                    run_remote "$host" "${args[@]}" --out "$results_dir/$crash-$n.json"
                fi
            done
            stamp "$label: $crash x$repeats done"
        done
    fi
    if [ "$host" != msi ]; then
        rsync -a "$host:$remote/$results_dir/" "$out/$label/"
    fi
    stamp "$label: collected $(find "$out/$label" -name '*.json' | wc -l) artifacts"
done
