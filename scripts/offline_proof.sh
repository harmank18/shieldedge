#!/usr/bin/env bash
# ==============================================================================
# Prove that ShieldEdge runs with no network access.
#
#   1. records that the internet is unreachable (ping 8.8.8.8 must fail)
#   2. captures all traffic on every interface with tcpdump while one full
#      analysis (detector + Llama via ExecuTorch) runs
#   3. lists the sockets owned by the process while it runs
# Output: results/<device>/offline_proof.txt and offline_capture.pcap
#
# Disconnect Wi-Fi/Ethernet first (e.g. `nmcli radio wifi off`). Needs sudo for tcpdump.
# Usage: ./scripts/offline_proof.sh
# ==============================================================================
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
# shellcheck disable=SC1091
source venv/bin/activate
SLUG="$(python -c 'from shieldedge.device import device_slug; print(device_slug())')"
OUT="results/$SLUG"
mkdir -p "$OUT"
LOG="$OUT/offline_proof.txt"

{
  echo "ShieldEdge offline proof - $(date -Is)"
  echo "Device: $(python -c 'from shieldedge.device import device_name; print(device_name())')"
  echo
  echo "## Network state"
  ip -brief addr
  echo
  echo "## ping 8.8.8.8 (expected to FAIL)"
  if ping -c 2 -W 2 8.8.8.8; then echo "!! internet is reachable - disconnect it for a valid proof"; else echo "-> unreachable (good)"; fi
} > "$LOG" 2>&1

sudo tcpdump -i any -n -w "$OUT/offline_capture.pcap" not host 127.0.0.1 and not host ::1 >/dev/null 2>&1 &
TCPDUMP=$!
sleep 1

python shieldedge.py --sms "Dear customer, your SBI account will be blocked in 24 hours. Update KYC at http://sbi-kyc-verify.top" >> "$LOG" 2>&1 &
APP=$!
sleep 3
{
  echo
  echo "## Sockets held by ShieldEdge / llama_main during inference (expected: none)"
  ss -tunap 2>/dev/null | grep -E "python|llama_main" || echo "(none)"
} >> "$LOG"
wait $APP
sleep 1
sudo kill "$TCPDUMP" 2>/dev/null; wait "$TCPDUMP" 2>/dev/null

{
  echo
  echo "## Packets captured on all interfaces during the run (expected: 0)"
  sudo tcpdump -n -r "$OUT/offline_capture.pcap" 2>/dev/null | wc -l
} >> "$LOG"
echo "Wrote $LOG and $OUT/offline_capture.pcap"
tail -n 4 "$LOG"
