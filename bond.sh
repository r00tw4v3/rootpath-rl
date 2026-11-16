#!/usr/bin/env bash
# ==============================================================================
# Multi-WAN Resilient Policy-Based Routing & Nexthop Bonding
# Supported interfaces: usb0 (101), wlan1 (102), wlan0 (103), eth0 (104)
# Fully idempotent, auto-detecting, health-checking multi-interface balancer.
# ==============================================================================

set -euo pipefail

# Require root
if [[ $EUID -ne 0 ]]; then
    echo "[!] This script must be run as root (e.g. sudo bash bond.sh)" >&2
    exit 1
fi

echo "=================================================================="
echo " [Multi-WAN Bonding] Initializing Resilient Policy-Based Routing"
echo "=================================================================="

# Enable IP forwarding and set loose reverse-path filtering for multi-path
sysctl -qw net.ipv4.ip_forward=1
sysctl -qw net.ipv4.conf.all.rp_filter=2
sysctl -qw net.ipv4.conf.default.rp_filter=2

# Candidate interfaces and static table mappings
CANDIDATES=("usb0" "wlan1" "wlan0" "eth0")
declare -A TABLE_MAP=(
    ["usb0"]="101"
    ["wlan1"]="102"
    ["wlan0"]="103"
    ["eth0"]="104"
)

# Clean up dangling old rules for managed tables
for dev in "${CANDIDATES[@]}"; do
    tid="${TABLE_MAP[$dev]}"
    # Remove existing ip rules targeting this table
    while ip rule show | grep -E "lookup ${tid}\b" >/dev/null 2>&1; do
        ip rule del table "$tid" 2>/dev/null || break
    done
    # Flush existing table routes
    ip route flush table "$tid" 2>/dev/null || true
    # Set loose rp_filter on interface if it exists
    if [[ -d "/proc/sys/net/ipv4/conf/${dev}" ]]; then
        sysctl -qw "net.ipv4.conf.${dev}.rp_filter=2"
    fi
done

# Arrays for verified active interfaces
ACTIVE_DEVS=()
declare -A DEV_IPS=()
declare -A DEV_GWS=()

echo "[*] Auditing candidate interfaces..."

for dev in "${CANDIDATES[@]}"; do
    # Check if interface exists and is UP in kernel
    if ! ip link show "$dev" >/dev/null 2>&1; then
        echo "  [-] Interface $dev: not present"
        continue
    fi

    # Extract IPv4 address
    ip_addr=$(ip -4 -o addr show dev "$dev" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -n1 || true)
    if [[ -z "$ip_addr" ]]; then
        echo "  [-] Interface $dev: present but no IPv4 assigned"
        continue
    fi

    # Extract Gateway (check ip route first, fallback to nmcli)
    gw=$(ip route show dev "$dev" 2>/dev/null | awk '/^default via/ {print $3}' | head -n1 || true)
    if [[ -z "$gw" ]]; then
        gw=$(nmcli -g IP4.GATEWAY device show "$dev" 2>/dev/null | head -n1 || true)
    fi

    if [[ -z "$gw" ]]; then
        echo "  [-] Interface $dev (IP: $ip_addr): no default gateway found"
        continue
    fi

    # Live connectivity health check (max 4 seconds timeout)
    echo -n "  [*] Interface $dev (IP: $ip_addr, GW: $gw): verifying internet... "
    ext_ip=$(curl -s --interface "$dev" --max-time 4 https://ifconfig.me 2>/dev/null || \
             curl -s --interface "$dev" --max-time 4 http://ifconfig.me 2>/dev/null || \
             curl -s --interface "$dev" --max-time 4 https://icanhazip.com 2>/dev/null || true)

    if [[ -n "$ext_ip" ]]; then
        echo "ONLINE (Public: $ext_ip)"
        ACTIVE_DEVS+=("$dev")
        DEV_IPS["$dev"]="$ip_addr"
        DEV_GWS["$dev"]="$gw"
    else
        echo "DEAD / NO ROUTE (Excluded from load-balancer)"
    fi
done

echo "------------------------------------------------------------------"
echo "[*] Active and verified interfaces: ${#ACTIVE_DEVS[@]}"

if [[ ${#ACTIVE_DEVS[@]} -eq 0 ]]; then
    echo "[!] CRITICAL: No active internet interfaces verified! Preserving fallback default route." >&2
    exit 1
fi

NEXTHOP_ARGS=""

for dev in "${ACTIVE_DEVS[@]}"; do
    tid="${TABLE_MAP[$dev]}"
    ip_addr="${DEV_IPS[$dev]}"
    gw="${DEV_GWS[$dev]}"

    # Extract device subnet/scope
    subnet=$(ip -4 -o addr show dev "$dev" | awk '{print $4}' | head -n1)

    echo "  [+] Configuring Table $tid for $dev ($ip_addr -> GW: $gw)"
    # Populate isolated table
    ip route flush table "$tid" 2>/dev/null || true
    ip route add "$subnet" dev "$dev" proto kernel scope link src "$ip_addr" table "$tid" 2>/dev/null || true
    ip route add default via "$gw" dev "$dev" table "$tid"

    # Add policy routing rule
    ip rule add from "$ip_addr" table "$tid" priority 10000

    NEXTHOP_ARGS="${NEXTHOP_ARGS} nexthop via ${gw} dev ${dev} weight 1"
done

echo "[*] Applying default routing policy..."
if [[ ${#ACTIVE_DEVS[@]} -eq 1 ]]; then
    single_dev="${ACTIVE_DEVS[0]}"
    single_gw="${DEV_GWS[$single_dev]}"
    echo "  [>] Single live link active ($single_dev). Setting direct default route via $single_gw."
    ip route replace default via "$single_gw" dev "$single_dev"
else
    echo "  [>] Multiple live links active (${#ACTIVE_DEVS[@]}). Applying multipath nexthop load balancing."
    # shellcheck disable=SC2086
    ip route replace default scope global $NEXTHOP_ARGS
fi

# Flush route cache
ip route flush cache

echo "=================================================================="
echo " [Multi-WAN Bonding] Configuration Complete & Verified Active"
echo " Current Default Route:"
ip route show default
echo " Active Policy Rules:"
ip rule show | grep -E "lookup (101|102|103|104)" || true
echo "=================================================================="
