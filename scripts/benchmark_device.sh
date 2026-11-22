#!/usr/bin/env bash
# ==============================================================================
# Collect every number the report needs on the current device (laptop or Pi):
#   * detector quality + per-input latency    (models/evaluate_models.py)
#   * detector latency benchmark               (shieldedge.py --benchmark)
#   * Llama via ExecuTorch, 10 runs            (scripts/benchmark_llm.py)
#   * Llama explanation quality/injection test (scripts/prompt_eval.py)
#   * storage footprint, CPU temperature/throttling (Pi)
# Output: results/<device>/ plus the shared files in results/.
# Usage: ./scripts/benchmark_device.sh
# ==============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
# shellcheck disable=SC1091
source venv/bin/activate
unset PYTHONPATH  # avoid ROS/other site-packages leaking in
SLUG="$(python -c 'from shieldedge.device import device_slug; print(device_slug())')"
OUT="results/$SLUG"
mkdir -p "$OUT"
echo "Writing results to $OUT"

python -c 'import json; from shieldedge.device import device_info; print(json.dumps(device_info(), indent=2))' | tee "$OUT/device.json"

if [[ -f data/processed/url_test.csv ]]; then
  python models/evaluate_models.py | tee "$OUT/evaluate_models.txt"
  cp results/benchmarks.md "$OUT/benchmarks.md"
else
  echo "(data/processed missing - run the training scripts first to evaluate on this device; skipping)"
fi
python shieldedge.py --benchmark | tee "$OUT/detector_latency.txt"
python scripts/benchmark_llm.py --runs 10 | tee "$OUT/llm_benchmark.txt"
python scripts/prompt_eval.py | tee "$OUT/prompt_eval.txt"

{
  echo "## Storage footprint"
  du -sh models/*.joblib executorch/*.pte executorch/llama_main 2>/dev/null
  echo "## Thermals after the run"
  if command -v vcgencmd >/dev/null; then
    vcgencmd measure_temp
    vcgencmd get_throttled   # 0x0 = never throttled
  else
    # Ubuntu on the Pi has no vcgencmd: read the same values from sysfs.
    [[ -r /sys/class/thermal/thermal_zone0/temp ]] &&
      awk '{printf "temp=%.1f C\n", $1/1000}' /sys/class/thermal/thermal_zone0/temp
    T="$(find /sys/devices/platform -name get_throttled 2>/dev/null | head -n1)"
    [[ -n "$T" ]] && echo "throttled=0x$(cat "$T" 2>/dev/null)   (0x0 = never throttled)"
  fi
} | tee "$OUT/system.txt"
echo "Done. Commit $OUT and results/llm_benchmark_${SLUG}_*.{json,md}."
