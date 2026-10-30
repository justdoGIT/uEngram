#!/usr/bin/env bash
# Run the capsule power-loss crash-consistency campaign and write one JSON per cell.
#
# usage: [OUT=dir] [JOBS=n] scripts/capsule_power_loss_campaign.sh
#
# Cells: three SMD machines x {FULL, NORMAL} x checkpoint interval {1, 16} on a 48-window prefix; the
# two negative controls on one machine; and one 400-window NORMAL run, long enough for SQLite to
# auto-checkpoint its WAL, to show how much a NORMAL-sync crash can lose. Needs gcc (the I/O shim is
# built on first use) and the SMD data. Results are storage-independent: they come from a recorded
# operation log, not from timing.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out=${OUT:-$repo/results/capsule_power_loss}
jobs=${JOBS:-3}
machines=(machine-1-1 machine-2-1 machine-3-1)
mkdir -p "$out"

cells=()
for machine in "${machines[@]}"; do
    for sync in FULL NORMAL; do
        for every in 1 16; do
            cells+=("$machine $sync $every 48 60 8 none")
        done
    done
done
cells+=("machine-1-1 FULL 1 48 60 8 sync-off")
cells+=("machine-1-1 FULL 1 48 60 8 no-journal")
cells+=("machine-1-1 NORMAL 1 400 80 4 none")
cells+=("machine-1-1 FULL 1 400 80 4 none")

run_cell() { # machine sync every windows spread combos control
    local machine=$1 sync=$2 every=$3 windows=$4 spread=$5 combos=$6 control=$7
    local name="${machine}_${sync,,}_every${every}_w${windows}"
    [ "$control" = none ] || name="${name}_control-${control}"
    (cd "$repo" && uv run python -m bench.capsule_power_loss --machine "$machine" --windows "$windows" \
        --sync "$sync" --checkpoint-every "$every" --spread "$spread" --combos-per-point "$combos" \
        --max-boundaries 150 --control "$control" --out "$out/$name.json" >/dev/null)
    printf '[%s] done %s\n' "$(date +%H:%M:%S)" "$name"
}
export -f run_cell
export repo out

printf '%s\n' "${cells[@]}" | xargs -P "$jobs" -L 1 bash -c 'run_cell "$@"' _
printf '[%s] %s cells written to %s\n' "$(date +%H:%M:%S)" "${#cells[@]}" "$out"
