#!/bin/bash
# ==============================================================================
# ShieldEdge - ExecuTorch Llama 3.2 1B-Instruct Model Export Script
# Target: On-Device CPU inference with XNNPACK delegation
# ==============================================================================

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXECUTORCH_DIR="/home/harman/executorch"
VENV_DIR="${BASE_DIR}/venv"
MODEL_DIR="${BASE_DIR}/models/Llama-3.2-1B-Instruct"
OUTPUT_DIR="${BASE_DIR}/executorch"

echo "=== Activating ShieldEdge Virtual Environment ==="
source "${VENV_DIR}/bin/activate"

echo "=== Exporting Llama 3.2 1B-Instruct to ExecuTorch (.pte) ==="
cd "${EXECUTORCH_DIR}"
PYTHONPATH="${EXECUTORCH_DIR}" python examples/models/llama/export_llama.py \
  --model llama3_2 \
  --checkpoint "${MODEL_DIR}/original/consolidated.00.pth" \
  --params "${MODEL_DIR}/original/params.json" \
  -o "${OUTPUT_DIR}" \
  -n llama3_2_1b \
  --max_seq_length 512 \
  --max_context_length 512 \
  -kv \
  -d bf16 \
  --xnnpack

echo "=== Export Complete ==="
ls -lh "${OUTPUT_DIR}/llama3_2_1b.pte"
