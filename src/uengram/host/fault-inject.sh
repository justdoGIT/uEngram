#!/bin/sh
# Controlled, temporary faults for the uengram-host validation campaign.
#   fault-inject.sh check                          busy share over 5 s, MemAvailable MB (Slurm is asked on msi)
#   fault-inject.sh spin FRACTION SECONDS          nice-19 busy loops on FRACTION of the CPUs
#   fault-inject.sh mem GB SECONDS          (root) GB working set inside a 1 GiB memory cgroup (swap thrash)
#   fault-inject.sh oom                     (root) one OOM kill inside a 256 MiB cgroup without swap
#   fault-inject.sh diskread MBPS SECONDS   (root) O_DIRECT sequential reads of the first disk, MBPS per
#                                                  second (0 = as fast as possible); read-only
#   fault-inject.sh netrecv PORT SECONDS           plain TCP sink
#   fault-inject.sh netsend HOST PORT MBPS SECONDS plain TCP stream of zeros (0 = unpaced)
# Memory faults are confined to cgroups under /sys/fs/cgroup/guards and removed afterwards.
set -eu
cg=/sys/fs/cgroup/guards
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

case "$1" in
check)
	avail="$(awk '/^MemAvailable:/ { print int($2 / 1024) }' /proc/meminfo)"
	echo "busy=$(busy_share) mem_avail_mb=$avail"
	;;
spin)
	n="$(awk -v f="$2" -v c="$(nproc)" 'BEGIN { n = int(f * c + 0.5); print n < 1 ? 1 : n }')"
	i=0
	while [ "$i" -lt "$n" ]; do
		timed "$3" nice -n 19 sh -c 'while :; do :; done' &
		i=$((i + 1))
	done
	wait
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
