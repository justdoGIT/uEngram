#!/bin/sh
# Controlled, temporary faults for the uengram-host validation campaign.
#   fault-inject.sh check                          busy share over 5 s, MemAvailable MB (Slurm is asked on msi)
#   fault-inject.sh spin FRACTION SECONDS          nice-19 busy loops on FRACTION of the CPUs
#   fault-inject.sh mem GB SECONDS          (root) GB working set inside a 1 GiB memory cgroup (swap thrash)
#   fault-inject.sh oom                     (root) one OOM kill inside a 256 MiB cgroup without swap
#   fault-inject.sh leak MBPS SECONDS       (root) MBPS MiB allocated and touched per second, never freed,
#                                                  in a cgroup bounded at 1.25x the total, no swap
#   fault-inject.sh diskread MBPS SECONDS   (root) O_DIRECT sequential reads of the first disk, MBPS per
#                                                  second (0 = as fast as possible); read-only
#   fault-inject.sh powercap FRACTION SECONDS (root) package PL1 = FRACTION of TDP while all CPUs spin
#   fault-inject.sh netem LOSS DELAY HOST PORT MBPS SECONDS (root) egress loss % and delay ms on the
#                                                  interface towards HOST while streaming to it
#   fault-inject.sh netrecv PORT SECONDS           plain TCP sink
#   fault-inject.sh netsend HOST PORT MBPS SECONDS plain TCP stream of zeros (0 = unpaced)
# Memory faults are confined to cgroups under /sys/fs/cgroup/guards and removed afterwards.
# powercap and netem undo themselves on exit, and a detached timer (unpowercap,
# unnetem) undoes them 60 s after their end even if this shell or its ssh is killed.
set -eu
cg=/sys/fs/cgroup/guards
rapl=/sys/class/powercap/intel-rapl:0
user=kk

# NAME MEM_BYTES SWAP_BYTES CMD...: run CMD as $user inside a fresh memory cgroup;
# prints the exit status (an OOM kill, 137, is the expected end of the oom fault).
in_cgroup() {
	d="$cg/$1"
	mkdir -p "$d"
	if [ ! -e "$d/memory.max" ]; then
		echo "memory controller not enabled under $cg" >&2
		exit 1
	fi
	echo "$2" >"$d/memory.max"
	echo "$3" >"$d/memory.swap.max"
	shift 3
	rc=0
	(
		echo 0 >"$d/cgroup.procs" # moves this subshell into the cgroup
		exec setpriv --reuid="$user" --regid="$user" --init-groups "$@"
	) || rc=$?
	echo "fault process exit status $rc"
	rmdir "$d"
}

timed() { # SECONDS CMD...: timeout ending the command is the expected outcome
	rc=0
	timeout "$@" || rc=$?
	if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ]; then
		exit "$rc"
	fi
}

busy_share() { # share of CPU time not idle/iowait over 5 s
	a="$(awk '/^cpu / { print $5 + $6, $2 + $3 + $4 + $5 + $6 + $7 + $8 + $9 }' /proc/stat)"
	sleep 5
	b="$(awk '/^cpu / { print $5 + $6, $2 + $3 + $4 + $5 + $6 + $7 + $8 + $9 }' /proc/stat)"
	echo "$a $b" | awk '{ t = $4 - $2; printf "%.3f", (t > 0) ? 1 - ($3 - $1) / t : 0 }'
}

spin_all() { # FRACTION SECONDS: nice-19 busy loops on FRACTION of the CPUs
	n="$(awk -v f="$1" -v c="$(nproc)" 'BEGIN { n = int(f * c + 0.5); print n < 1 ? 1 : n }')"
	dur="$2"
	set --
	i=0
	while [ "$i" -lt "$n" ]; do
		timed "$dur" nice -n 19 sh -c 'while :; do :; done' &
		set -- "$@" "$!"
		i=$((i + 1))
	done
	wait "$@" # only the loops: a bare wait would also wait for a detached restore timer (60 s longer)
}

clear_netem() { # DEV: remove our netem if it is still there; the kernel's default root qdisc returns
	if tc qdisc show dev "$1" root | grep -q netem; then
		tc qdisc del dev "$1" root
	fi
}

case "$1" in
check)
	avail="$(awk '/^MemAvailable:/ { print int($2 / 1024) }' /proc/meminfo)"
	echo "busy=$(busy_share) mem_avail_mb=$avail"
	;;
spin)
	spin_all "$2" "$3"
	;;
mem)
	in_cgroup uengram-mem 1073741824 max python3 -c "
import time
b = [bytearray(64 << 20) for _ in range(int($2 * 16))]
t = time.time()
while time.time() - t < $3:
    for x in b:
        for i in range(0, len(x), 4096):
            x[i] = 1
"
	;;
oom)
	in_cgroup uengram-oom 268435456 0 python3 -c "b = [bytearray(64 << 20) for _ in range(16)]"
	;;
leak)
	in_cgroup uengram-leak "$(($2 * $3 * 1310720 + 67108864))" 0 python3 -c "
import time
b, t = [], time.time()
while time.time() - t < $3:
    b.append(b'\x01' * ($2 << 20))
    time.sleep(1)
"
	;;
powercap)
	old="$(cat "$rapl/constraint_0_power_limit_uw")"
	cap="$(awk -v f="$2" -v t="$(cat "$rapl/constraint_0_max_power_uw")" 'BEGIN { printf "%d", f * t }')"
	setsid sh "$0" unpowercap "$old" "$(($3 + 60))" </dev/null >/dev/null 2>&1 &
	trap 'echo "$old" >"$rapl/constraint_0_power_limit_uw"' EXIT
	trap 'exit 1' INT TERM HUP
	echo "$cap" >"$rapl/constraint_0_power_limit_uw"
	echo "package PL1 $old -> $cap uW"
	spin_all 1.0 "$3"
	;;
unpowercap)
	sleep "$3"
	echo "$2" >"$rapl/constraint_0_power_limit_uw"
	;;
netem)
	dev="$(ip route get "$4" | awk '{ for (i = 1; i < NF; i++) if ($i == "dev") { print $(i + 1); exit } }')"
	setsid sh "$0" unnetem "$dev" "$(($7 + 60))" </dev/null >/dev/null 2>&1 &
	trap 'clear_netem "$dev"' EXIT
	trap 'exit 1' INT TERM HUP
	tc qdisc replace dev "$dev" root netem delay "${3}ms" loss "${2}%"
	echo "netem on $dev: delay ${3}ms loss ${2}%"
	setpriv --reuid="$user" --regid="$user" --init-groups sh "$0" netsend "$4" "$5" "$6" "$7"
	;;
unnetem)
	sleep "$3"
	clear_netem "$2"
	;;
diskread)
	disk="$(lsblk -dno NAME,TYPE | awk '$2 == "disk" { print $1; exit }')"
	if [ "$2" -eq 0 ]; then
		timed "$3" dd if="/dev/$disk" of=/dev/null bs=1M iflag=direct status=none
	else
		end=$(($(date +%s) + $3))
		skip=0
		while [ "$(date +%s)" -lt "$end" ]; do
			dd if="/dev/$disk" of=/dev/null bs=1M count="$2" skip="$skip" iflag=direct status=none
			skip=$((skip + $2))
			sleep 1
		done
	fi
	;;
netrecv)
	python3 -c "
import socket, time
s = socket.create_server(('0.0.0.0', $2), reuse_port=True)
s.settimeout(60)
c, _ = s.accept()
c.settimeout(10)
end = time.time() + $3 + 30
while time.time() < end:
    try:
        if not c.recv(1 << 20):
            break
    except TimeoutError:
        break
"
	;;
netsend)
	python3 -c "
import socket, time
c = socket.create_connection(('$2', $3), timeout=30)
buf = bytes(1 << 20)
rate, t0 = $4, time.time()
sent = 0
while time.time() - t0 < $5:
    c.sendall(buf)
    sent += 1
    if rate > 0:
        time.sleep(max(0.0, t0 + sent / rate - time.time()))
c.close()
"
	;;
*)
	echo "unknown fault $1" >&2
	exit 2
	;;
esac
