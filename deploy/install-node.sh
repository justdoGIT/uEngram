#!/usr/bin/env bash
# Install uengram-hostd on fleet nodes: sync the package, install the runit
# service, optionally install python3-numpy and enable the service.
#   deploy/install-node.sh [--numpy] [--enable] [--campaign] NODE...
# --campaign also installs (and with --enable enables) uengram-campaign, the
# validation-campaign runner; install it on one node only (msi).
# sudo runs with the password from kepr on stdin (never in argv or files).
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
numpy=0
enable=0
campaign=0
nodes=()
for a in "$@"; do
	case "$a" in
	--numpy) numpy=1 ;;
	--enable) enable=1 ;;
	--campaign) campaign=1 ;;
	-h | --help)
		sed -n '2,7p' "$0"
		exit 0
		;;
	*) nodes+=("$a") ;;
	esac
done
[ "${#nodes[@]}" -gt 0 ] || {
	echo "usage: $0 [--numpy] [--enable] [--campaign] NODE..." >&2
	exit 2
}

remote_sudo() { # NODE CMD: run CMD as root on NODE
	kepr get KK_SUDO_PASSWORD --quiet | ssh -o BatchMode=yes "$1" "sudo -S -p '' sh -c '$2'"
}

install_service() { # NODE NAME: copy deploy/runit/NAME/run to /etc/sv/NAME/run (and enable)
	scp -q "$repo/deploy/runit/$2/run" "$1:/tmp/$2.run"
	remote_sudo "$1" "install -D -m 0755 /tmp/$2.run /etc/sv/$2/run && rm -f /tmp/$2.run"
	if [ "$enable" = 1 ]; then
		remote_sudo "$1" "[ -e /var/service/$2 ] || ln -s /etc/sv/$2 /var/service/"
	fi
}

for n in "${nodes[@]}"; do
	echo "== $n"
	ssh -o BatchMode=yes "$n" 'mkdir -p ~/.local/share/uengram-host/src ~/.local/state/uengram-host'
	rsync -a --delete --exclude __pycache__ "$repo/src/uengram" "$n:.local/share/uengram-host/src/"
	if [ "$numpy" = 1 ]; then
		remote_sudo "$n" 'xbps-install -y python3-numpy'
	fi
	install_service "$n" uengram-hostd
	if [ "$campaign" = 1 ]; then
		install_service "$n" uengram-campaign
	fi
	ssh -o BatchMode=yes "$n" 'python3 -c "import numpy" 2>/dev/null && echo "numpy: yes" || echo "numpy: no (run mode needs it)"'
done
