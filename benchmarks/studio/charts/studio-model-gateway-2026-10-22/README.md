# Two-model Studio gateway validation

Tested October 22, 2026 on one M5 Max Mac Studio, 40 GPU cores and 64 GiB unified
memory, running macOS 27.0.1. This is a receipt for this machine and these pinned
checkpoints, not qualification of other GPUs or a model-quality leaderboard.

The native runtime is the pinned Ornith port plus the published shared-expert
and session patches. Its executable SHA256 is
`a3bbc06661eb375a33cf2e0e0de261d8875c85f657636dafa82430c18a445bb8`.
Both models used their own DFlash2 drafter, 131,072 total context tokens, one
stream and 128-token prompt chunks. Swift's snapshot budget was 9 GiB; Ornith's
was 4 GiB. The explicit cache-cap override was enabled.

## Near-128K capacity

Each model received one cold synthetic prompt of exactly **130,944 tokens**,
then saved its complete state, switched to the other model, and restored its
conversation. Both continuations reused **130,950 cached tokens** with drafting
enabled. These are capacity and restoration checks; repeated filler is not a
representative prompt-processing or reasoning benchmark.

| Model | Footprint with large saved state | Kernel lifetime peak | SSD snapshot |
| --- | ---: | ---: | ---: |
| Swift-1.5 affine 4-bit | 34.3 GiB | 42.6 GiB | 8.23 GiB |
| Ornith-1.5-122B mixed 2-bit | 47.2 GiB | 48.3 GiB | 3.23 GiB |

The footprint includes model, drafter, live execution state, and the retained
copy of the active context. Peaks include temporary allocations. The
[kernel samples](context-memory.json) observed at most one native model process
over 524 samples and about 4 MiB of host swap-outs during that interval. Host
swap activity is not attributed to an individual model; existing swap was not
cleared. The [capacity receipt](context-capacity.json) records exact byte counts,
cache hits, hashes, timings, and memory before and after restoration.

At these sizes, five full saved contexts per model need roughly 58 GiB of SSD
space together, in addition to model weights, drafters, and build files. Only
the selected model's current context stays in RAM. All five may be on SSD while
that model is unloaded.

## Correctness and lifecycle

The [short real-model smoke receipt](gateway-smoke.json) checks both model swaps,
positive cached-token reuse, full output-token hashes against fresh plain
decoding, streaming completion followed immediately by a model switch, owner
and friend isolation, authentication, internal-route confinement, and stats.
Tiny synthetic replies in this receipt are not representative decode-speed
measurements.

The supervisor's **45/45** CPU integration tests cover authentication and queued
revocation, real child processes, serialization, bounded context eviction,
corruption and failed saves, timeouts, cancellation, crash supervision, and
shutdown cleanup. The [native receipt](../../../../tools/studio-gateway/native/validation.json)
records **107/107** transitive snapshot/cache tests, a successful native build,
and zero new lean findings over 444 inherited findings. Full repository and
other-platform qualification were not run for this Studio integration.

The [real idle check](idle-resume.json) used an accelerated two-second timeout.
The engine unloaded, both models remained discoverable without loading either,
and a full-history continuation resumed from SSD with its drafter and positive
cached-token counts. The deployed idle timeout is 1,800 seconds (30 minutes).

The [HTTPS deployment check](https-api.json) verified both existing client keys,
both model generations with their drafters, refusal of unauthenticated requests
and private control routes, and native counters reaching the existing metrics
collector. It used the named client User-Agent recorded in the receipt; the
existing proxy rejected Python's default User-Agent with error 1010.

Reproduce the checks against an isolated gateway, with its private key files:

```sh
python3 -m unittest discover -s tools/studio-gateway/tests -v
bash tools/studio-gateway/native/test.sh /path/to/native-source /path/to/zig
python3 tools/studio-gateway/tests/studio_smoke.py --run \
  --base-url http://127.0.0.1:8082 --key-file "$HOME/.config/tensorfold/api-key" \
  --output /private/tmp/gateway-smoke.json
python3 tools/studio-gateway/tests/studio_capacity.py --run \
  --key-file "$HOME/.config/tensorfold/api-key" --output /private/tmp/gateway-capacity.json
```

The capacity client uses the isolated backend at port 18082 and the gateway's
private cache root. Keep an inference test in the existing Studio `AI` tmux
session. Never run these clients against an active user's conversation cache.

## Quality recommendation

The [sanitized GPQA summary](gpqa-summary.json) retains our adaptive-budget
results: Swift **175/189 (92.6%)**, Ornith **157/189 (83.1%)** on the versioned
clean subset, counting unfinished answers as incorrect. Original-set results
are 176/198 and 159/198. Those evaluations retained earlier completed answers
and extended only cutoffs; they are not fresh uniform 96K runs. Early Ornith
answers came from Python MLX before its native/DFlash2 continuation. No gated
questions, answer keys, or response traces are published here.
