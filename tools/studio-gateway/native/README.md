# Native conversation snapshots

This package adds private session control to a pinned native TensorFold engine.
It serves Swift-1.5 and Ornith mixed 2-bit with their respective DFlash2 drafters.
The gateway serializes requests and runs exactly one backend process.

Build on an M4 or newer Apple Silicon Mac with Xcode/Metal and Zig 0.17.0:

```sh
bash tools/studio-gateway/native/build.sh /path/to/native-source /path/to/zig
```

The script pins `BobClawblaw/TensorFold@9328aff8a6c470b026aee7cefe53a0527de44814`,
applies the independently quantized shared-expert kernel fix and the bounded
recurrent scratch fix, applies the session RPC patch, runs CPU snapshot tests,
and builds `zig-out/native/bin/tensorfold-native`. It never starts a server.
The default host CPU target is `apple_m4`, as tested on the Studio. Set
`TENSORFOLD_NATIVE_CPU=apple_m1` to build host instructions compatible with M1
through M5; this package does not claim model qualification on those other chips.
The copied kernel patch is self-contained in this folder. No private dataset,
model tensors, credentials, or machine-specific paths are needed in this package.

Use the already normalized native Swift checkpoint. Its RMSNorm vectors include
the historical `+1` conversion; this fork reads those vectors directly. Do not
apply that conversion again. Ornith reads its ordinary MLX mixed-precision checkpoint.

## Backend contract

The gateway creates a fresh private mode-0700 flat directory and an ephemeral
backend bearer key file. The public API keys are not the backend key. Pass:

- `TF_SESSION_ROOT=/absolute/private/session-directory`
- `TF_SESSION_BUILD_ID=SHA256_OF_EXECUTABLE` (64 hexadecimal characters)
- `TF_PROMPT_ROWS=128`
- `--host 127.0.0.1 --api-key-file /private/backend-key --context 131072 --parallel 1`
- `--prompt-cache-gib 9` for Swift; `--prompt-cache-gib 4` for Ornith.
- `--prompt-cache-over-cap --learn-conversations --learn-dir /private/session-directory/bootstrap`
- The model's own `--drafter ... --drafter-bits 4`.

The native `--learn-conversations` startup probe supplies the checkpoint, drafter,
GPU arithmetic, machine and source identity. Session mode folds the executable
SHA256 into that identity, closes the automatic disk store, and limits the RAM
snapshot store to one entry. It performs no background learned-cache writes.
The bootstrap directory may remain empty until gateway shutdown removes it.

One live 128K runner and one snapshot of that same active conversation can coexist
in RAM. Thus the snapshot budget is additional to the runner's own cache. The
70% admission rule must be explicitly overridden for Ornith; actual total process
memory and 128K behavior must be measured on each machine. No inactive model or
inactive conversation snapshot is retained in RAM.

All three authenticated routes are disabled without `TF_SESSION_ROOT`. They require
bearer authentication even if the rest of the native server was started without it.
The gateway must never proxy these routes from public clients.

| Request | Body | Effect |
| --- | --- | --- |
| `POST /v1/internal/session/save` | `{"path":"/private/session-directory/opaque.tfs"}` | Save the latest complete cached state atomically. |
| `POST /v1/internal/session/load` | Same | Validate and import into an empty RAM store, without a prefill fallback. |
| `POST /v1/internal/session/clear` | `{}` | Release the RAM snapshot; subsequent cold requests reset the live runner. |

Responses contain `object`, `tokens`, `bytes`, and a 16-digit `fingerprint`. An
unused empty conversation saves as `tokens:0, bytes:0`, without creating a file.
A used conversation whose save returns zero must not be marked safely parked.
A load into a nonempty store fails: clear first. Missing/corrupt files, incompatible
model/build identities, token layouts and cache-budget violations return HTTP 409
with `snapshot_error`; the gateway must fail closed. Files must be direct children
with ordinary nonhidden `.tfs` filenames. The private root must be canonical and
must not contain symlink parent directories.

The file contains the complete native family image: attention K/V rows, GDN
recurrent states and convolution tails, drafter tap ring and committed end,
plus the cache's token IDs and canonical decoded spans. Writes and reads use
16 MiB staging pieces; checksums cover all payload and metadata. Only a fully
verified import becomes visible to inference. The first request after import
must match its exact token/mode prefix. A mismatch is an error, never an implicit
cold prefill. The gateway may explicitly start a new cold conversation after a
preflight miss, but must report it as cold rather than restored.

For the preflight, send the actual generation body to `POST /v1/imprint/resume`.
Its `ram_tokens` reports whether the canonical rendered prefix matches. This
checks without generating any tokens. The gateway can then intentionally clear
and cold-run a branched/shortened conversation, or refuse it.

Control calls wait until prior HTTP handlers finish queueing next-turn warm work,
then until the engine and warm queue are idle. A second concurrent private control
fails immediately. The gateway serializes all of these operations.

## Verification

Run the entire CPU snapshot/cache regression graph, without a GPU or model:

```sh
bash tools/studio-gateway/native/test.sh /path/to/patched-native-source /path/to/zig
```

This uses `zig/src/core/session_file.zig` as its test root. With no test filter,
Zig also runs its imported prompt-cache, learned-state, row-store and fault tests;
the final candidate passed **107/107**. The exact command and scope are recorded
in `validation.json`. This is a scoped CPU receipt: the full repository
`test test-kernel-contract test-golden fuzz-cortheon` and plain-build qualification
were not run for this Studio service patch. The same lean checker reported
444 inherited findings on the base and candidate, with zero new findings.

The CPU tests cover multi-part state/token/mode round trips, single-slot memory,
wrong-model and wrong-prefix rejection, corrupt-payload rejection, missing-file
failure, empty saves, and path confinement. Deployment additionally needs real
Swift and Ornith checks: compare deterministic token IDs for a fresh continuation
and a save/clear/load continuation, require positive cached token counts, verify
immediate control after streaming `[DONE]`, then repeat across engine stop/start.
At 128K, measure process footprint and verify exact restore without paging-driven
memory exhaustion. The format supports 128K; short test success alone does not
prove 128K performance on a particular machine.
