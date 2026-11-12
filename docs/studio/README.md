# Studio native serving

For Swift and Ornith behind one API with one engine resident at a time, use the
[Studio gateway](../../tools/studio-gateway/README.md). It owns model switching,
SSD conversation snapshots, and idle unloading. The single-model launcher below
remains available for direct Swift serving and historical benchmark reproduction.

The current upstream engine is native TensorFold 1.1.1. Serving uses the installed Zig/Metal binary, not the frozen Python engine or MLX. The Python virtual environment remains useful for benchmark tokenization and clients.

`tools/start-studio.sh` serves the pinned Swift 1.5 affine 4-bit checkpoint with the pinned 4-bit DFlash2 drafter, 131,072 total context tokens, two concurrent streams and a 5 GiB retained-prompt budget. It listens on loopback port 8080 and keeps the existing restricted-permission API-key file. `STUDIO_PORT` selects an isolated benchmark port, and `TENSORFOLD_NATIVE_BIN` selects another native executable. The public hostname comes from `STUDIO_ALLOWED_HOST` or the private `$HOME/.config/tensorfold/studio-host` file.

Install the official `tensorfold-1.1.1-macos-arm64.tar.gz` release under `$HOME/ai/tensorfold-runtime/1.1.1`, preserving its directory structure, and verify its companion SHA-256 checksum before running it. The drafter stays in the Hugging Face cache at its pinned revision. The alternate Qwen launcher preserves its existing checkpoint revision.

Swift needs the same 161 RMSNorm scale conversions that the old fork applied during loading. Prepare a separate native-compatible checkpoint once; the converter leaves the downloaded snapshot intact, preserves all other tensor bytes and records the converted vectors in `swift-native.json`. The launcher refuses an unprepared checkpoint. Native inference still uses the upstream kernels unchanged.

```sh
python3 tools/prepare-swift-native.py \
  "$HOME/.cache/huggingface/hub/models--ukisai--Swift-1.5-4bit-MLX/snapshots/82276731e47e1db4ac502f24c63cfaf886639be9" \
  "$HOME/ai/models/swift-1.5-native-82276731"
```

```sh
bash tools/start-studio.sh
```

The native `/dashboard` and authenticated `/stats`, `/memory` and `/v1/models` endpoints replace the Python-only Studio monitoring routes. `/health` returns status only when keys are configured. Clients send a Bearer key or `x-api-key` header. The final source-built binary admits a 5 GiB cache with two streams; an explicit 8 GiB budget exceeded its startup memory guard. The saved cold benchmark used 8 GiB and had zero cache hits. Memory admission and retained-cache capacity come from the native engine; the removed Python process/MLX budget variables do not impose native limits.

Python-only conversation inventory, checkpoint-slot, spill and clear-on-exit flags are no longer passed. Native retained prompts are in memory by default; a process exit releases them. Preserve the published native cache limits rather than enabling over-cap operation to reproduce the former disk spill settings.

Use the existing tmux AI session: window 0 for the server and window 1 for checks. Stop the model process once and wait for shutdown before running GPU tests. Keep the existing tunnel and collector processes running. The collector accepts native aggregate status and labels its memory counters as process measurements. The former per-conversation inventory belongs to the historical Python server.

Run the current upstream host and Metal checks using Zig 0.17.0 and Xcode with its Metal toolchain:

```sh
zig build test test-kernel-contract test-golden fuzz-cortheon -Dcpu=apple_m1 --summary all
```

For the historical client benchmark, run one attempt at every 20K, 40K, 60K, 80K, 100K and 128K context with concurrency 1 and 2 and a 1,024-token output limit. Keep the original published solo reference unchanged. Native API prefill duration excludes queue time; derived queue-plus-prefill measurements must add the matching request's queued duration from the local completion log. Publish only sanitized counters, never raw server logs or local output paths.
