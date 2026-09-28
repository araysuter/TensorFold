#!/bin/bash
# Run from the fork's virtual environment. Caddy/llama.cpp must not own port 8080.
set -euo pipefail
umask 077
MODEL_PATH="$(hf download ukisai/Swift-1.5-4bit-MLX \
  --revision 82276731e47e1db4ac502f24c63cfaf886639be9 --quiet)"
KEY_FILE="${STUDIO_API_KEY_FILE:-$HOME/.config/tensorfold/api-key}"
if [[ ! -s "$KEY_FILE" ]]; then
  mkdir -p "$(dirname "$KEY_FILE")"
  cp "$HOME/.config/llama.cpp/api-key" "$KEY_FILE"
  chmod 600 "$KEY_FILE"
fi
exec tensorfold serve "$MODEL_PATH" \
  --name swift-1.5 \
  --context 131072 \
  --parallel 2 \
  --conversation-slots 8 \
  --checkpoint-slots 2 \
  --prompt-cache-gib 8 \
  --spill-gib 128 \
  --drafter z-lab/Qwen3.8-27B-DFlash2 \
  --snapshot-dir "$HOME/.cache/tensorfold/studio/prefix-snapshots" \
  --api-key-file "$KEY_FILE" \
  --host 127.0.0.1 \
  --port "${STUDIO_PORT:-8080}"
