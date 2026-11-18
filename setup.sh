#!/usr/bin/env bash
# ShieldEdge setup for a Linux laptop (x86_64 or arm64). For a Raspberry Pi use scripts/setup_pi.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
command -v python3 >/dev/null || { echo "Python 3 is required"; exit 1; }
[[ -d venv ]] || python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "from shieldedge.storage import AlertLogger; AlertLogger(); print('Local alert database ready (hashes only).')"
echo "Done. Try: source venv/bin/activate && python shieldedge.py --url http://sbi-net-banking-update.xyz/login --fast"
echo "For the on-device Llama step see executorch/export_llama.sh and docs/PI_DEPLOYMENT.md."
