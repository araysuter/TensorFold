# Studio native serving

The current upstream engine is native TensorFold 1.1.1. Serving uses the installed Zig/Metal binary, not the frozen Python engine or MLX. The Python virtual environment remains useful for benchmark tokenization and clients.

`tools/start-studio.sh` serves the pinned Swift 1.5 affine 4-bit checkpoint with the pinned 4-bit DFlash2 drafter, 131,072 total context tokens, two concurrent streams and an 8 GiB retained-prompt budget. It listens on loopback port 8080 and keeps the existing restricted-permission API-key file. `STUDIO_PORT` selects an isolated benchmark port, and `TENSORFOLD_NATIVE_BIN` selects another native executable. The public hostname comes from `STUDIO_ALLOWED_HOST` or the private `$HOME/.config/tensorfold/studio-host` file.

Install the official `tensorfold-1.1.1-macos-arm64.tar.gz` release under `$HOME/ai/tensorfold-runtime/1.1.1`, preserving its directory structure, and verify its companion SHA-256 checksum before running it. The launcher expects the model and drafter in the Hugging Face cache at their pinned revisions; it refuses missing inputs. The alternate Qwen launcher preserves its existing checkpoint revision.

```sh
bash tools/start-studio.sh
```

The native `/dashboard` and authenticated `/stats`, `/memory` and `/v1/models` endpoints replace the Python-only Studio monitoring routes. `/health` returns status only when keys are configured. Clients send a Bearer key or `x-api-key` header. Memory admission and retained-cache capacity come from the native engine; the removed Python process/MLX budget variables do not impose native limits.

Python-only conversation inventory, checkpoint-slot, spill and clear-on-exit flags are no longer passed. Native retained prompts are in memory by default; a process exit releases them. Preserve the published native cache limits rather than enabling over-cap operation to reproduce the former disk spill settings.

Use the existing tmux AI session: window 0 for the server and window 1 for checks. Stop the model process once and wait for shutdown before running GPU tests. Keep the existing tunnel and collector processes running. The collector accepts native aggregate status and labels its memory counters as process measurements. The former per-conversation inventory belongs to the historical Python server.

Run the current upstream host and Metal checks using Zig 0.17.0 and Xcode with its Metal toolchain:

```sh
zig build test test-kernel-contract test-golden fuzz-cortheon -Dcpu=apple_m1 --summary all
```

For the historical client benchmark, run one attempt at every 20K, 40K, 60K, 80K, 100K and 128K context with concurrency 1 and 2 and a 1,024-token output limit. Keep the original published solo reference unchanged. Native API prefill duration excludes queue time; derived queue-plus-prefill measurements must add the matching request's queued duration from the local completion log. Publish only sanitized counters, never raw server logs or local output paths.
