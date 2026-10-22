#!/usr/bin/env bash
# Install uengram-hostd on fleet nodes: sync the package, install the runit
# service, optionally install python3-numpy and enable the service.
#   deploy/install-node.sh [--numpy] [--enable] NODE...
# sudo runs with the password from kepr on stdin (never in argv or files).
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
numpy=0
enable=0
nodes=()
for a in "$@"; do
	case "$a" in
	--numpy) numpy=1 ;;
	--enable) enable=1 ;;
	-h | --help)
		sed -n '2,5p' "$0"
		exit 0
		;;
	*) nodes+=("$a") ;;
	esac
done
[ "${#nodes[@]}" -gt 0 ] || {
	echo "usage: $0 [--numpy] [--enable] NODE..." >&2
	exit 2
}

remote_sudo() { # NODE CMD: run CMD as root on NODE
	kepr get KK_SUDO_PASSWORD --quiet | ssh -o BatchMode=yes "$1" "sudo -S -p '' sh -c '$2'"
}

for n in "${nodes[@]}"; do
	echo "== $n"
	ssh -o BatchMode=yes "$n" 'mkdir -p ~/.local/share/uengram-host/src ~/.local/state/uengram-host'
	rsync -a --delete --exclude __pycache__ "$repo/src/uengram" "$n:.local/share/uengram-host/src/"
	scp -q "$repo/deploy/runit/uengram-hostd/run" "$n:/tmp/uengram-hostd.run"
	remote_sudo "$n" 'install -D -m 0755 /tmp/uengram-hostd.run /etc/sv/uengram-hostd/run && rm -f /tmp/uengram-hostd.run'
	if [ "$numpy" = 1 ]; then
		remote_sudo "$n" 'xbps-install -y python3-numpy'
	fi
	if [ "$enable" = 1 ]; then
		remote_sudo "$n" '[ -e /var/service/uengram-hostd ] || ln -s /etc/sv/uengram-hostd /var/service/'
	fi
	ssh -o BatchMode=yes "$n" 'python3 -c "import numpy" 2>/dev/null && echo "numpy: yes" || echo "numpy: no (run mode needs it)"'
done
