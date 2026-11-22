#!/usr/bin/env bash
# Copy the files that are not in git (Llama program + tokenizer) from the laptop to the Pi.
# Usage (on the laptop):  ./scripts/copy_to_pi.sh pi@raspberrypi.local [~/shieldedge]
set -euo pipefail
TARGET="${1:?usage: $0 <user>@<pi-host> [remote repo path]}"
REMOTE_DIR="${2:-~/shieldedge}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ssh "$TARGET" "mkdir -p $REMOTE_DIR/executorch $REMOTE_DIR/models/Llama-3.2-1B-Instruct/original"
rsync -ah --progress "$REPO/executorch/llama3_2_1b_int4.pte" "$TARGET:$REMOTE_DIR/executorch/"
rsync -ah --progress "$REPO/models/Llama-3.2-1B-Instruct/original/tokenizer.model" \
  "$TARGET:$REMOTE_DIR/models/Llama-3.2-1B-Instruct/original/"
echo "Done. On the Pi: cd $REMOTE_DIR && ./scripts/setup_pi.sh"
