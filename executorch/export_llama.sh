#!/usr/bin/env bash
# ==============================================================================
# ShieldEdge - export Llama 3.2 1B-Instruct to an ExecuTorch .pte program
#
# Default: INT4 weights / INT8 dynamic activations ("8da4w", group size 128),
#          4-bit embeddings (group size 32), KV cache + SDPA custom op,
#          XNNPACK delegate. This is the configuration used for all reported
#          ShieldEdge Llama numbers.
#
# The .pte is architecture-independent: export once on the laptop and copy it
# to the Raspberry Pi 5. Only the llama_main runner must be built per device
# (see scripts/setup_pi.sh).
#
# Usage:
#   EXECUTORCH_DIR=~/executorch ./executorch/export_llama.sh          # INT4 (default)
#   QUANT=bf16 EXECUTORCH_DIR=~/executorch ./executorch/export_llama.sh  # bf16 baseline
#
# Tested with ExecuTorch 1.6.0, torch 2.15, torchao 0.18 (x86_64, Ubuntu 24.04).
# Requires the Meta Llama 3.2 licence and the original checkpoint in
# models/Llama-3.2-1B-Instruct/original/ (consolidated.00.pth, params.json,
# tokenizer.model).
# ==============================================================================
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXECUTORCH_DIR="${EXECUTORCH_DIR:-$HOME/executorch}"
VENV_DIR="${VENV_DIR:-${BASE_DIR}/venv}"
MODEL_DIR="${MODEL_DIR:-${BASE_DIR}/models/Llama-3.2-1B-Instruct/original}"
OUTPUT_DIR="${BASE_DIR}/executorch"
QUANT="${QUANT:-int4}"
MAX_SEQ="${MAX_SEQ:-1024}"

for f in consolidated.00.pth params.json tokenizer.model; do
  [[ -f "${MODEL_DIR}/${f}" ]] || { echo "Missing ${MODEL_DIR}/${f}"; exit 1; }
done
[[ -d "${EXECUTORCH_DIR}" ]] || { echo "ExecuTorch source not found at ${EXECUTORCH_DIR}"; exit 1; }

# shellcheck disable=SC1091
[[ -f "${VENV_DIR}/bin/activate" ]] && source "${VENV_DIR}/bin/activate"

case "${QUANT}" in
  int4) CONFIG="examples/models/llama/config/llama_xnnpack.yaml"; NAME="llama3_2_1b_int4" ;;
  bf16) CONFIG="examples/models/llama/config/llama_bf16.yaml";    NAME="llama3_2_1b_bf16" ;;
  *) echo "QUANT must be int4 or bf16"; exit 1 ;;
esac

echo "=== Exporting Llama 3.2 1B-Instruct (${QUANT}) with ${CONFIG} ==="
cd "${EXECUTORCH_DIR}"
python -m extension.llm.export.export_llm \
  --config "${CONFIG}" \
  +base.model_class="llama3_2" \
  +base.checkpoint="${MODEL_DIR}/consolidated.00.pth" \
  +base.params="${MODEL_DIR}/params.json" \
  +export.max_seq_length="${MAX_SEQ}" \
  +export.max_context_length="${MAX_SEQ}" \
  +export.output_name="${NAME}.pte"

# export_llm writes into the current directory; move the program next to the runner.
mv -f "${EXECUTORCH_DIR}/${NAME}.pte" "${OUTPUT_DIR}/${NAME}.pte"
echo "=== Export complete ==="
ls -lh "${OUTPUT_DIR}/${NAME}.pte"
sha256sum "${OUTPUT_DIR}/${NAME}.pte"
