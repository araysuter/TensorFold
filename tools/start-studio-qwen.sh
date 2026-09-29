#!/bin/bash
# Run from the fork's virtual environment. Caddy/llama.cpp must not own port 8080.
set -euo pipefail
umask 077
# 58 GiB process budget leaves up to 55 GiB for MLX (subject to Metal's cap).
export TENSORFOLD_MEMORY_LIMIT_GB="${TENSORFOLD_MEMORY_LIMIT_GB:-58}"
MODEL_PATH="$(hf download Vontra/Qwen3.8-27B-MLX-4bit \
  --revision 70ae7fac63274ff2eac54152031433374cb80f2f --quiet)"
KEY_FILE="${STUDIO_API_KEY_FILE:-$HOME/.config/tensorfold/api-key}"
if [[ ! -s "$KEY_FILE" ]]; then
  mkdir -p "$(dirname "$KEY_FILE")"
  cp "$HOME/.config/llama.cpp/api-key" "$KEY_FILE"
  chmod 600 "$KEY_FILE"
fi
exec tensorfold serve "$MODEL_PATH" \
  --name qwen3.8-27b \
  --context 131072 \
  --parallel 2 \
  --conversation-slots 8 \
  --checkpoint-slots 2 \
  --prompt-cache-gib 8 \
  --spill-gib 128 \
  --clear-cache-on-exit \
  --drafter z-lab/Qwen3.8-27B-DFlash2 \
  --snapshot-dir "$HOME/.cache/tensorfold/studio-qwen/prefix-snapshots" \
  --api-key-file "$KEY_FILE" \
  --host 127.0.0.1 \
  --port "${STUDIO_PORT:-8080}"
