#!/bin/bash
# Use the pinned native runtime and the existing downloaded checkpoints.
set -euo pipefail
umask 077
NATIVE="${TENSORFOLD_NATIVE_BIN:-$HOME/ai/tensorfold-runtime/1.1.1/bin/tensorfold-native}"
HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
MODEL_PATH="$HUB_CACHE/models--ukisai--Swift-1.5-4bit-MLX/snapshots/82276731e47e1db4ac502f24c63cfaf886639be9"
DRAFTER_PATH="$HUB_CACHE/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
KEY_FILE="${STUDIO_API_KEY_FILE:-$HOME/.config/tensorfold/api-key}"
HOST_ARGS=()
PUBLIC_HOST="${STUDIO_ALLOWED_HOST:-}"
if [[ -z "$PUBLIC_HOST" && -s "$HOME/.config/tensorfold/studio-host" ]]; then
  IFS= read -r PUBLIC_HOST < "$HOME/.config/tensorfold/studio-host" || true
fi
if [[ -n "$PUBLIC_HOST" ]]; then HOST_ARGS=(--allowed-host "$PUBLIC_HOST"); fi
for required in "$NATIVE" "$MODEL_PATH/config.json" "$DRAFTER_PATH/config.json" "$KEY_FILE"; do
  if [[ ! -s "$required" ]]; then
    echo "Missing native runtime, pinned checkpoint, drafter or API key; see docs/studio/README.md." >&2
    exit 1
  fi
done
exec "$NATIVE" serve "$MODEL_PATH" \
  --name swift-1.5 \
  --context 131072 \
  --parallel 2 \
  --prompt-cache-gib "${STUDIO_PROMPT_CACHE_GIB:-8}" \
  --drafter "$DRAFTER_PATH" \
  --snapshot-dir none \
  --temperature 1 --top-k 20 --top-p 0.95 \
  --dashboard \
  --api-key-file "$KEY_FILE" \
  "${HOST_ARGS[@]}" \
  --host 127.0.0.1 \
  --port "${STUDIO_PORT:-8080}"
