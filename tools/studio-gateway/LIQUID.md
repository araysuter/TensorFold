# Liquid concurrent profile

Liquid LFM2.5-24B-A2B uses a separate qualified native DSpark build. Swift and
Arete retain their existing snapshot-capable engines. A source synchronization
of the fork does not imply that an upstream release binary supports DSpark.

## Reproduce the runtime

The experimental runtime is pinned at
[`2d5af28e8e82941cbfc261a9694a3dbafa54cca3`](https://github.com/araysuter/TensorFold/tree/2d5af28e8e82941cbfc261a9694a3dbafa54cca3).
It extends upstream DSpark PR #1394 at `22b19797ec1048bb2f50e307aae3339077ba68be`
with the family prefill steps from upstream PR #3058 at
`652490cdb6ff36c51a6074006bf551c51d28fa18`. Owned DSpark features open before
the first prompt chunk. Providers that require an existing target cache keep
their original opening order. No GPU kernels, precision or arithmetic change.

Build in a separate checkout, using Zig 0.17.0, the macOS SDK and Metal toolchain:

```sh
git clone --branch asher/codex/liquid-dspark-runtime \
  https://github.com/araysuter/TensorFold.git liquid-runtime
cd liquid-runtime
git checkout 2d5af28e8e82941cbfc261a9694a3dbafa54cca3
zig build test-lfm2-dspark test-lfm2 test-shared-lane-regressions-host test-golden -Dcpu=apple_m1
zig build native -Doptimize=ReleaseFast -Dcpu=apple_m1
```

Structured output and required tool calls also need `libtfgrammar.dylib` in the
binary distribution's `lib` directory. Follow the pinned checkout's grammar
build instructions; do not advertise required tool calls without testing them.

Use these immutable weight revisions, outside the source checkout:

- Target: [`LiquidAI/LFM2.5-24B-A2B-MLX-4bit`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit/tree/667408d494bde191c2cb9dd4f4549827816b9275), revision `667408d494bde191c2cb9dd4f4549827816b9275`.
- Drafter: [`LiquidAI/LFM2.5-24B-A2B-DSpark`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-DSpark/tree/e052e5b1dd103c38cdd4f8ddc068d7dea4586f55), revision `e052e5b1dd103c38cdd4f8ddc068d7dea4586f55`.

## Configure the gateway

Add these fields to a model profile using that executable and target:

```json
{
  "id": "lfm2.5-24b-a2b",
  "stateless": true,
  "context_window": 96000,
  "max_concurrent": 8,
  "memory_required_gib": 18.5,
  "memory_per_request_gib": 2.875,
  "request_reserve_gib": 8
}
```

The executable argv must include `--context 96000 --parallel 8
--prompt-cache-gib 0 --drafter /absolute/path/to/drafter --drafter-bits 8`,
alongside the normal model, loopback port and credential-file arguments.
This is a profile fragment, not a complete gateway config. Keep credentials and
machine paths in a private config outside this repository.

The 96,000-token limit includes input and output. The per-user budget reserves
a full window, even for a shorter request. On the tested 64 GiB M5 Max, the
known target KV plus drafter context/staging arrays total about 2.786 GiB per
full window. The 2.875 GiB setting adds per-user slack; the separate 8 GiB host
reserve is retained. These measurements do not qualify another model or chip.
The admission guard checks current memory and pressure; another application can
still consume memory after admission, so this is not an OS-enforced allocation.

Liquid requests send their full history each turn and do not create SSD context
snapshots. Swift and Arete retain the shared four-context LRU. A model switch
waits for all active requests to finish and keeps only one model process loaded.

## Measure

Set `TF_BENCH_AUTH` privately, then run from the gateway source checkout:

```sh
python3 tools/bench_studio_concurrency.py http://127.0.0.1:8080 lfm2.5-24b-a2b \
  --api-key-env TF_BENCH_AUTH --context 96000 --lengths 4000,8000,32000 \
  --levels 1,2,4,8 --tokens 256 --output concurrency.json
```

The client first obtains plain solo references, then checks each drafted
concurrent result token-for-token. It exits nonzero on mismatches, incomplete
streams, transport failures or refused users. Synthetic token arrays measure
speed and isolation, not answer quality. One run per cell gives a snapshot;
small differences need repeated, alternated trials before claiming improvement.
