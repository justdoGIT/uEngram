#!/bin/sh
# Power and scheduler knobs and a fixed reference build, for uengram.host.tune.
#   knob.sh get pl1|epp                     current value (pl1: uW, then the TDP in uW; epp: cpu0's)
#   knob.sh set pl1|epp VALUE ORIG HOLD_S   (root) set VALUE; a detached timer restores ORIG after HOLD_S
#   knob.sh restore pl1|epp ORIG            (root) set ORIG and cancel the timer
#   knob.sh scx start NAME HOLD_S           (root) attach /usr/bin/scx_NAME; a timer detaches it after HOLD_S
#   knob.sh scx stop                        (root) detach it and wait until sched_ext is disabled
#   knob.sh gen DIR FILES FUNCS             deterministic C sources (FILES files of FUNCS functions)
#   knob.sh build DIR JOBS                  (root) compile them as kk with JOBS parallel gcc -O2;
#                                           prints wall_s energy_j busy_s child_s ncpu throttle rc
#   knob.sh idle                            busy share over 5 s
#   knob.sh clean DIR                       remove the generated sources (only u[0-9][0-9][0-9].c) and DIR
# The timers are what make a knob temporary even if the ssh session or the
# orchestrator dies; the orchestrator restores and cancels them at the end.
set -eu
rapl=/sys/class/powercap/intel-rapl:0
user=kk
run=/run

epp_files() { ls /sys/devices/system/cpu/cpu[0-9]*/cpufreq/energy_performance_preference; }

write_knob() { # KNOB VALUE
	case "$1" in
	pl1)
		[ "$(cat "$rapl/enabled")" = 1 ] || {
			echo "RAPL power limit disabled by firmware" >&2
			exit 1
		}
		echo "$2" >"$rapl/constraint_0_power_limit_uw"
		;;
	epp) for f in $(epp_files); do echo "$2" >"$f"; done ;;
	*)
		echo "unknown knob $1" >&2
		exit 2
		;;
	esac
}

cancel_timer() { # NAME
	if [ -r "$run/uengram-knob-$1.pid" ]; then
		kill "$(cat "$run/uengram-knob-$1.pid")" 2>/dev/null || :
		rm -f "$run/uengram-knob-$1.pid"
	fi
}

arm_timer() { # NAME SECONDS ACTION...: run `knob.sh ACTION...` after SECONDS unless cancelled
	name="$1"
	cancel_timer "$name"
	setsid sh "$0" after "$@" </dev/null >/dev/null 2>&1 &
	echo $! >"$run/uengram-knob-$name.pid"
}

scx_state() { cat /sys/kernel/sched_ext/state; }

wait_scx() { # STATE: up to 10 s
	i=0
	while [ "$(scx_state)" != "$1" ]; do
		i=$((i + 1))
		[ "$i" -le 100 ] || {
			echo "sched_ext did not become $1" >&2
			exit 1
		}
		sleep 0.1
	done
}

throttle_sum() {
	cat /sys/devices/system/cpu/cpu*/thermal_throttle/core_throttle_count \
		/sys/devices/system/cpu/cpu*/thermal_throttle/package_throttle_count 2>/dev/null |
		awk '{ s += $1 } END { print s + 0 }'
}

cpu_jiffies() { awk '/^cpu / { print $5 + $6, $2 + $3 + $4 + $5 + $6 + $7 + $8 + $9 }' /proc/stat; }

gen() { # DIR FILES FUNCS: every function differs (constants from its index), so gcc does real work
	mkdir -p "$1"
	awk -v dir="$1" -v files="$2" -v funcs="$3" 'BEGIN {
		for (f = 0; f < files; f++) {
			out = sprintf("%s/u%03d.c", dir, f)
			printf "#include <stdint.h>\n" > out
			for (g = 0; g < funcs; g++) {
				k = f * funcs + g
				printf "uint64_t f%d_%d(uint64_t x, const uint32_t *a, int n) {\n", f, g >> out
				printf "\tuint64_t s = %du;\n\tfor (int i = 0; i < n; i++) {\n\t\tswitch ((x ^ a[i]) %% 7u) {\n", k + 1 >> out
				for (c = 0; c < 7; c++)
					printf "\t\tcase %d: s = s * %du + (a[i] >> %d) + %du; break;\n", c, 2 * ((k + c) % 97) + 3, (k + c) % 13, (k * 31 + c) % 1009 >> out
				printf "\t\t}\n\t\tx = (x << %d) ^ (x >> %d) ^ s;\n\t}\n\treturn s ^ x;\n}\n", k % 11 + 1, k % 7 + 3 >> out
			}
			close(out)
		}
	}'
}

build() { # DIR JOBS
	e0="$(cat "$rapl/energy_uj")"
	j0="$(cpu_jiffies)"
	th0="$(throttle_sum)"
	t0="$(date +%s.%N)"
	rc=0
	setpriv --reuid="$user" --regid="$user" --init-groups \
		sh -c "cd '$1' && ls ./*.c | xargs -P '$2' -n 1 gcc -O2 -c -o /dev/null" || rc=$?
	t1="$(date +%s.%N)"
	e1="$(cat "$rapl/energy_uj")"
	j1="$(cpu_jiffies)"
	th1="$(throttle_sum)"
	# `times` must run in this shell: a pipeline or $(...) forks, and a fork's child times start at 0
	times >"$run/uengram-times.$$"
	child="$(awk 'NR == 2 { split($1, u, /[ms]/); split($2, s, /[ms]/); print 60 * (u[1] + s[1]) + u[2] + s[2] }' \
		"$run/uengram-times.$$")"
	rm -f "$run/uengram-times.$$"
	echo "$t0 $t1 $e0 $e1 $(cat "$rapl/max_energy_range_uj") $j0 $j1 $(getconf CLK_TCK) $th0 $th1 $child $(nproc) $rc" |
		awk '{ de = $4 - $3; if (de < 0) de += $5
		       printf "wall_s=%.3f energy_j=%.3f busy_s=%.3f child_s=%.3f ncpu=%d throttle=%d rc=%d\n",
		              $2 - $1, de / 1e6, (($9 - $7) - ($8 - $6)) / $10, $13, $14, $12 - $11, $15 }'
}

case "$1" in
get)
	case "$2" in
	pl1) echo "$(cat "$rapl/constraint_0_power_limit_uw") $(cat "$rapl/constraint_0_max_power_uw")" ;;
	epp) cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference ;;
	esac
	;;
set)
	write_knob "$2" "$3"
	arm_timer "$2" "$5" restore "$2" "$4"
	;;
restore)
	cancel_timer "$2"
	write_knob "$2" "$3"
	;;
after) # NAME SECONDS ACTION...: the detached timer
	sleep "$3"
	rm -f "$run/uengram-knob-$2.pid"
	shift 3
	sh "$0" "$@"
	;;
scx)
	case "$2" in
	start)
		[ "$(scx_state)" = disabled ] || {
			echo "a sched_ext scheduler is already attached" >&2
			exit 1
		}
		setsid "/usr/bin/scx_$3" </dev/null >"$run/uengram-scx.log" 2>&1 &
		echo $! >"$run/uengram-scx-proc.pid"
		wait_scx enabled
		arm_timer scx "$4" scx stop
		cat /sys/kernel/sched_ext/root/ops
		;;
	stop)
		cancel_timer scx
		if [ -r "$run/uengram-scx-proc.pid" ]; then
			kill "$(cat "$run/uengram-scx-proc.pid")" 2>/dev/null || :
			rm -f "$run/uengram-scx-proc.pid"
		fi
		wait_scx disabled
		;;
	esac
	;;
gen) gen "$2" "$3" "$4" ;;
build) build "$2" "$3" ;;
clean)
	rm -f "$2"/u[0-9][0-9][0-9].c
	rmdir "$2"
	;;
idle)
	a="$(cpu_jiffies)"
	sleep 5
	echo "$a $(cpu_jiffies)" | awk '{ t = $4 - $2; printf "%.3f\n", (t > 0) ? 1 - ($3 - $1) / t : 0 }'
	;;
*)
	sed -n '2,14p' "$0"
	exit 2
	;;
esac
