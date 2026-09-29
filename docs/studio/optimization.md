# Swift 1.5 performance candidate

Branch: `asher/codex/swift-runner-speed`. This incorporates all 55 commits between
the fork's `024f8af1274cfeb98e2355adf218b27dbd490818` and upstream
`9cd52ab4daba68ddd09be89be8f23ad43175e821` (TensorFold 0.5.0). The upstream merge
and the additional performance work are separate commits for comparison.

`tools/start-studio.sh` is unchanged: the pinned Swift snapshot, DFlash2 drafter,
two compute lanes, context limit, conversation slots, cache budgets, API key and
port are the existing configuration. The Swift normalization compatibility fix,
monitoring, SSE keepalives and shutdown cache policy remain in the fork.

## Changes to measure

Upstream now interleaves decode rounds between prompt chunks, reduces unnecessary
evaluation of intermediate prompt hidden states, and materializes DFlash's bounded
prompt context each chunk. The merge carries those changes into the Studio scheduler
and keeps its per-chunk rates and conversation states. Responses API requests also
retain authentication and conversation headers through the internal chat adapter;
reading and deleting stored responses require the configured key.

The additional prefill kernel reads the runner's tiled Q4 weights directly. Previously,
each large prompt chunk converted each packed weight back to MLX's layout before its
matmul. This candidate changes the weight addresses in MLX's M5 TensorOps loader and
retains its dequantization, tile arithmetic and reduction order. It keeps one packed
weight layout, without a second permanent copy. Small verification windows and
unsupported shapes or quantization formats use the existing path.

This path requires an M5 GPU, BF16 activations, Q4/group-64 weights, and the recognized
MLX 0.32.2 or 0.32.3 header. A small startup comparison checks the bits against native
MLX for both 32-column and 64-column layouts, including a partial prompt tile. Missing
support, compilation errors or a mismatch disable the new path. Startup reports
`[tensorfold] tiled prompt matmul: on` or the reason it stayed off. Snapshot identities
include whether the path is enabled; restarting with another path cannot reuse its
prefix snapshots.

The DFlash change selects accepted verification rows before concatenating hidden
states from the tapped layers. During prefill it selects only the portion of a chunk
that survives the drafter's context window. It preserves the retained rows, their
positions, draft search and target verification rules. This aims to reduce copying
and temporary allocations during both prefill and decode.

Prompt chunks also retain their existing MLX token arrays through the target forward,
instead of synchronously reading them back into Python and uploading them again.
Token dtype, ordering and the cache's prompt position are preserved. The isolated
host conversion step for 2,048 tokens fell from about 84 to 1.4 microseconds; its
contribution to whole-model latency is small.

For concurrent M5 requests, eligible DFlash lattices use the trained eight-position
block when at least two short-context lattices can share a batch. This can combine
otherwise unequal block lengths and keep two blocks inside one 16-row TensorOps
tile. Solo calls and lattices with more than 64 newly supplied context rows retain
the existing block choice, as do non-M5 decoders. Tree-node budgets and complete
target verification are unchanged.
`TF_DRAFT_BATCH_BLOCK=0` disables this batching change independently.

No tests, model runs or benchmarks were executed on the development machine.
Whole-model throughput and representative output parity still need the full Studio
benchmark and workload checks below.

## Studio validation

On September 29, 2026, the Studio (M5 Max, 64 GiB, macOS 27.0, MLX 0.32.2,
mlx-lm 0.31.3) passed all **261** focused tests in **57.39 seconds**, including all
12 tiled-prefill checks. The startup comparison enabled the kernel. The initial
compilation failures were fixed by composing MLX's weight loader (Metal disallows
class inheritance) and passing the dimension buffer's values into the helper
(its original constant-address-space references cannot bind a device buffer).

The final expanded suite, including the concurrent-drafter qualification checks,
tree search and draft vocabulary coverage, passed **279 tests in 15.28 seconds**.

Targeted GPU probes compared the previous untile-plus-native matmul with the direct
tiled reader at 129, 256, 512 and 1,024 prompt rows. Both large Swift MLP projections
(17,408 by 5,120 and its reverse) showed about **6–30% faster kernel execution**;
this includes removal of the weight conversion and is not a whole-model speedup.
The probes checked exact native-MLX output bits before timing. Wider 128-row or
128-column TensorOps tiles were generally slower, so the 64-by-64 layout is retained.
Small projections showed smaller and more variable differences.

Further probes at the normal 2,048-row chunk size showed smaller gains, including
small regressions in some measurements. Combining gate/up prefill
projections gave about 0–1% difference. Changing the decode pipeline interval from
four layers gave no consistent improvement, and adding native prefill pipelining
saved only about 0.3% at 2,048 rows. Those additional changes were left out.
After retaining GPU prompt arrays, another 56 family, prompt-fill, Studio, spill and
cancellation checks passed. A coding smoke request with 1,226 prompt tokens produced
the same 18 generated tokens with drafting on and off (target hash `c7c468890b2d`).
Its roughly 879 cold input tokens/s and 113 draft decode tokens/s are short-workload
observations, not comparisons with the historical long-context benchmark.

A controlled local-HTTP ablation sent identical 4,096-token prompts with drafting
enabled but only one generated token, six requests per mode, all with zero cached
tokens. After excluding each mode's first warmup request, median server-reported
prefill time was **4.1603 s** with `TF_TILED_PREFILL=0` and **4.0952 s** with it
enabled: about **1.6% faster whole-model prefill** for this workload. All six
target-token hashes matched between modes. This isolates the tiled reader; both
modes include the upstream integration, DFlash copy changes and GPU token arrays.
It does not measure long-context decode or the total improvement over the old fork.

Three short coding tasks (2,048 input tokens, up to 256 generated tokens) checked
solo and two-request runs. Always shrinking draft blocks to eight slightly hurt
solo throughput, so that change was rejected. The shared-only candidate reproduced
all nine baseline token hashes and reply texts. Its two-request aggregate throughput
(output tokens divided by the slower request's elapsed time, including prefill)
changed by **-0.4%, +2.8%, and +1.3%** across the three tasks. Solo decode remained
about 97, 106 and 118 tokens/s. These are small diagnostic runs; request order can
strongly skew an individual request's reported decode rate. The full context and
concurrency sweep is still needed to qualify the shared batching gain.

A final two-request sampling check (temperature 0.7, top-k 20, top-p 0.95,
seed 17, 128 generated tokens per request) also reproduced both undrafted
target-token hashes and reply texts with drafting enabled.

Use the Studio's normal environment and launch commands. Focused regressions cover partial GPU
tiles, native-MLX bit parity, avoiding the weight conversion, selected DFlash rows and
positions, authenticated Responses requests, and monitoring through chunked prefill:

```bash
python -m pytest -q tests/test_tiled_prefill.py tests/test_lane_qmm.py \
  tests/test_qwen_family.py tests/test_prompt_fill.py tests/test_studio.py \
  tests/test_responses_api.py tests/test_server_openai_compat.py \
  tests/test_checkpoint_spill.py tests/test_cancellation.py \
  tests/test_dflash_tree_search.py tests/test_dflash_draft_vocab.py
```

Use a separate client machine for throughput measurements, as in the existing
[benchmark instructions](../../benchmarks/studio/README.md). Keep other clients idle,
use a fresh results directory, and compare the same model, prompt, context size,
generation limit, sampling and cache state. Start with 20K and 40K, then the full
context sweep once parity and memory behavior are satisfactory:

```bash
python benchmarks/studio/benchmark.py --contexts 20000 40000 \
  --reps 3 --out benchmark-results-swift-candidate
python benchmarks/studio/plot_run.py benchmark-results-swift-candidate \
  --title 'TensorFold · Swift 1.5 performance candidate'
```

For an ablation, restart the same performance commit with `TF_TILED_PREFILL=0` in its
environment. The default is `1`. This disables only the direct tiled prefill path;
the DFlash copy reduction remains active. The first parent of the performance commit
is the upstream integration baseline, with neither additional optimization.

Record the tested commit, MLX/mlx-lm and macOS versions, the startup kernel status and
prompt chunk size, server-reported prefill/decode rates, time to first token, accepted
draft rate, cached tokens, process footprint and swap. Compare one active request and
two concurrent requests separately. The upstream scheduling change can affect their
latency differently. Check representative coding replies, tools and resumed chats
for target-token parity as well as throughput.

The saved Swift 20K solo requests in `studio-benchmark-resumed/results.json` recorded
about 878–881 input tokens/s including queue/prefill time, 86–89 decode tokens/s and
22–24% draft acceptance. Those files are historical measurements, not a live reading
of the Studio. [TensorFold's website](https://tensorfold.dev/) currently describes its
120–124 tokens/s Qwen3.8/DFlash2 result as short thinking replies on an M5 Max with
128 GB, using earlier builds. It is a different model and workload; it is not a Swift
long-context baseline or a promised speed for this branch.

[Apple's MLX M5 article](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
describes prefill as compute-bound and decode as limited by memory bandwidth, with
TensorOps acceleration requiring macOS 26.2 or later. The direct weight reader targets
an avoidable layout conversion around that accelerated prefill matmul. DFlash already
provides speculative decoding; this branch preserves complete target prompt processing.
