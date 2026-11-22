#!/usr/bin/env bash
# ==============================================================================
# ShieldEdge - Raspberry Pi 5 setup (Ubuntu 24.04 / Raspberry Pi OS / Debian 12+ arm64)
#
# What it does
#   1. checks the board is 64-bit Arm, installs build and runtime packages
#   2. creates the Python venv and installs ShieldEdge's requirements
#   3. builds the ExecuTorch llama_main runner natively for aarch64
#      (ExecuTorch v1.6.0, XNNPACK backend with Arm KleidiAI kernels)
#   4. copies llama_main + libllama_runner.so into ./executorch/
#   5. checks that the .pte program and tokenizer were copied from the laptop
#
# The .pte program does NOT need to be exported on the Pi: copy it from the
# laptop with scripts/copy_to_pi.sh (it is architecture-independent).
#
# Usage (on the Pi, from the repo root):   ./scripts/setup_pi.sh
#   SKIP_BUILD=1 ./scripts/setup_pi.sh     # Python part only
# Build time on a Pi 5 (8 GB, Ubuntu 24.04) was 18 min with 4 jobs. Use a fan/heatsink.
# ==============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ET_DIR="${EXECUTORCH_DIR:-$HOME/executorch}"
ET_TAG="${ET_TAG:-v1.6.0}"
cd "$REPO"

echo "== [1/5] System checks =="
ARCH="$(uname -m)"
MODEL="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
MEM_GB="$(awk '/MemTotal/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)"
echo "Board: $MODEL | arch: $ARCH | RAM: ${MEM_GB} GB | OS: $(. /etc/os-release && echo "$PRETTY_NAME")"
if [[ "$ARCH" != "aarch64" ]]; then
  echo "WARNING: expected aarch64 (64-bit OS). A 32-bit OS cannot run the Llama runner." >&2
fi
awk -v m="$MEM_GB" 'BEGIN { if (m < 3.5) print "WARNING: less than 4 GB RAM - Llama needs ~1.1 GB free." }'

sudo apt-get update
sudo apt-get install -y git cmake ninja-build build-essential python3-venv python3-dev python3-pip \
  libzbar0 sqlite3 time iputils-ping iproute2 tcpdump

echo "== [2/5] Python environment =="
[[ -d venv ]] || python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  echo "== [3/5] Building ExecuTorch $ET_TAG llama_main (this takes a while) =="
  if [[ ! -d "$ET_DIR/.git" ]]; then
    git clone --branch "$ET_TAG" --depth 1 https://github.com/pytorch/executorch.git "$ET_DIR"
  fi
  cd "$ET_DIR"
  git submodule sync && git submodule update --init --recursive --depth 1
  # Build-time Python helpers used by ExecuTorch's code generation.
  pip install pyyaml ruamel.yaml tomli setuptools
  # The CMake build also reads PyTorch's C++ headers and torchgen (build time only;
  # ShieldEdge never imports torch at run time). CPU wheel, same version as the export.
  python -c "import torch" 2>/dev/null ||
    pip install "torch==2.15.0" --index-url https://download.pytorch.org/whl/cpu ||
    pip install torch
  export PYTHON_EXECUTABLE="$(command -v python)"
  # Leave one core free so the Pi stays responsive and does not overheat.
  export CMAKE_BUILD_PARALLEL_LEVEL="${JOBS:-3}"
  if ! make llama-cpu; then
    echo "Build failed. If the error mentions missing Python modules run:" >&2
    echo "  cd $ET_DIR && ./install_requirements.sh   and re-run this script." >&2
    exit 1
  fi
  cd "$REPO"

  echo "== [4/5] Installing runner into ./executorch =="
  cp "$ET_DIR/cmake-out/examples/models/llama/llama_main" executorch/llama_main
  LIB="$(find "$ET_DIR/cmake-out" -name 'libllama_runner.so' | head -n1 || true)"
  [[ -n "$LIB" ]] && cp "$LIB" executorch/
  file executorch/llama_main
fi

echo "== [5/5] Model files =="
missing=0
for f in executorch/llama3_2_1b_int4.pte models/Llama-3.2-1B-Instruct/original/tokenizer.model; do
  if [[ -f "$f" ]]; then echo "  ok      $f ($(du -h "$f" | cut -f1))"; else echo "  MISSING $f"; missing=1; fi
done
if [[ $missing == 1 ]]; then
  echo "Copy them from the laptop:  ./scripts/copy_to_pi.sh <user>@<pi-address>  (run on the laptop)"
fi

echo
echo "Setup finished. Next:"
echo "  source venv/bin/activate"
echo "  python shieldedge.py --url http://sbi-net-banking-update.xyz/login"
echo "  ./scripts/benchmark_device.sh        # all Pi numbers for the report"
