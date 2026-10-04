# Long prompt processing on the Studio

The M5 path now sends each prompt chunk through MLX's fused D256 attention instead of
splitting it into 128-query calls. It also uses a packed DeltaNet recurrence after
startup verifies both its output and its final state against the installed MLX-LM kernel.
Both paths are enabled by default on the qualified M5/MLX runtime. Decode kernels and
the 2,048-token prompt chunk remain the same.

The final branch includes upstream TensorFold 0.6.0 and keeps the fork's Studio monitoring,
authentication, Swift loader, spill and cache cleanup options. Its prompt scheduler admits
short prompts alongside long ones. The recorded experiment uses the pinned Swift 1.5 checkpoint,
the existing DFlash2 drafter, MLX 0.32.2 and MLX-LM 0.31.3 on the 40-core M5 Max with
64 GB RAM. The final Swift launcher uses 48 GiB for the process, including 45 GiB for
MLX, after the memory trials below. No system memory settings were raised.

## Reproduce

Use the existing `AI` tmux session. Stop the current server once and wait for it to exit.
Run the test server in window 0, on a free local port:

```bash
cd ~/ai/tensorfold-studio
source .venv/bin/activate
STUDIO_PORT=18080 bash tools/start-studio.sh
```

In window 1, run two uncached trials per context:

```bash
cd ~/ai/tensorfold-studio
.venv/bin/python benchmarks/studio/prefill.py \
  --url http://127.0.0.1:18080 --label optimized --reps 2 \
  --out /tmp/prefill-optimized
```

For the baseline, stop that server and restart with both paths disabled:

```bash
TF_NATIVE_PREFILL_ATTENTION=0 TF_PACKED_PREFILL_GDN=0 \
  STUDIO_PORT=18080 bash tools/start-studio.sh
```

Run the same client command with `--label baseline --out /tmp/prefill-baseline`.
The benchmark uses identical prompt IDs and greedy sampling across configurations.
Every trial starts with a different token and must report zero cached tokens. It saves
server prompt-processing time, time to first token, 64-token reply hashes, GPU utilization
samples, MLX peak memory and swap readings. The key is read locally and is never saved.

These are synthetic raw completions with one active request. They measure throughput;
they do not establish coding quality or performance for every prompt, model or concurrency level.

## Cold API results

Median of two trials per size, using identical input IDs, seed 17, temperature 0 and 64
reply tokens. All twelve requests reported zero cached prompt tokens. The baseline disables
both new paths on the initial performance branch and installed runtime. The final optimized
run includes upstream 0.6.0; all six reply hashes matched the optimized 0.5.0 run exactly.

| Prompt | Baseline | Optimized | Less prompt time | Prompt throughput |
| --- | ---: | ---: | ---: | ---: |
| 20,000 tokens | 22.19 s | 20.63 s | 7.0% | 901 → 969 tokens/s |
| 60,000 tokens | 83.81 s | 73.16 s | 12.7% | 716 → 820 tokens/s |
| 128,000 tokens | 267.33 s | 197.02 s | 26.3% | 479 → 650 tokens/s |

The 128K throughput improvement is 35.7%. GPU utilization sampled once per second averaged
99.3%, 98.9% and 98.9% for the final runs, including request startup and completion;
the median was 100% at all three sizes. Maximum observed MLX allocation was 29.73 GiB,
versus 30.84 GiB in the baseline. Every before/after swap reading was zero.
Raw results, utilization samples and replies are in `baseline.json`, `optimized.json`
(initial 0.5.0 run) and `optimized-060.json` (final run).

Three of the six 64-token replies matched the baseline hash exactly. The other three
diverged: whole-chunk attention changes floating-point reduction order. This path still
computes full causal attention; no prompt tokens, layers or attention heads were removed.
These measurements do not prove equivalent answer quality. Use `TF_NATIVE_PREFILL_ATTENTION=0`
when the old prompt's exact output is required.

The packed recurrence was separately checked bit for bit, including its FP32 final state.
The initial kernel, snapshot, family prefill and cancellation suite passed 64 tests.
The 0.6.0 candidate passed 479 tests with two skips, adding the scheduler, prompt memory,
checkpoint, Studio/API and Swift loader checks. Snapshot identities include both prefill modes.

`validation.json` records two concurrent 20K-token requests with temperature 0.7 and seed 17:
both 128-token replies matched exactly with drafting on and off. A 20,047-token chat then
produced the same 128-token hash when resumed from its 20,024-token cached prefix. Prompt
work fell from 21.36 s to 0.082 s. This validates resumption for that case, rather than
measuring cold-prompt throughput. Reproduce with a fresh prefix tag:

```bash
.venv/bin/python benchmarks/studio/validate_prefill.py \
  --url http://127.0.0.1:18080 --out /tmp/prefill-validation.json
```

The final `validation-060.json` reruns these checks on 0.6.0: the concurrent hashes again
match, and a 20,064-token chat resumes from 20,041 tokens with identical output. Prompt work
takes 21.30 s cold and 0.083 s resumed.

## Queued requests

In one paired trial, a 1K-token request was submitted one second after a 20K-token request.
Upstream's scheduler reduced the short request's time to first token from 21.03 s to 1.92 s
(about 11x sooner). Both requests' 32-token replies matched the old scheduler exactly.
All requests were uncached. The last request finished about 23 s after the pair started in
both modes; the scheduler reduces the short request's wait by sharing the GPU with it.
See `queue-050.json` and `queue-060.json`. Reproduce against each running server with:

```bash
.venv/bin/python benchmarks/studio/queue_prefill.py \
  --url http://127.0.0.1:18080 --label optimized-060 --out /tmp/prefill-queue.json
```

## Memory and 128K cached conversations

The repeated cold API table above used the original effective 51.8 GiB process /
48.8 GiB MLX budgets. Those runs used zero swap. A later 128K system-message chat
retained several large checkpoints and increased the system's swap usage to 208.56 MB;
its cold and resumed replies matched (`cache-128k-060-original.json`). Resumption used
126,976 cached tokens and took 4.525 seconds. This exposed a memory case the raw
completion benchmark did not cover.

Reducing the process cap to 48 GiB (45 GiB MLX) preserved the original 128K raw
completion's exact reply hash and took 196.93 seconds (`optimized-48g.json`). There
were no new swap-outs. A 128K system-message chat then resumed from 124,928 tokens in
8.347 seconds with identical output (`cache-128k-48g.json`). The 49 GiB / 46 GiB trial
also retained 124,928 tokens and resumed in 8.111 seconds, again with identical output
and no new swap-outs (`cache-128k-49g.json`).

The initial validation script demanded a cache hit within one 2K chunk of the prefix's
end. These two smaller-budget runs failed that coverage assertion while retaining 97.6%
of the prefix; their saved token hashes and message text matched exactly. The published
script checks meaningful prefix reuse and exact output equality, and records coverage
separately. Cache retention depends on the budget and other requests.

Swap readings are system-wide. The already allocated 208.56 MB remained after the
larger-budget test, so later trials record `vm_stat` counters as well as the allocation:
unchanged swap-out counters show no additional swapping in the 48 and 49 GiB trials.
The 50 GiB trial also resumed from 124,928 tokens in 8.356 seconds with identical
output, but swap-outs increased by 256 pages (4 MB) during resumption, and global
swap allocation rose to 212.56 MB (`cache-128k-50g.json`). It offered no cache benefit.
The final Swift launcher therefore defaults to 48 GiB / 45 GiB MLX. These measurements
do not establish that every workload will avoid swap.

## Other trials

`prefill-sweep.json` records the preliminary 32,768-token full-model sweep. A single trial
per configuration found:

| Configuration | Prompt time | Tokens/s | MLX peak |
| --- | ---: | ---: | ---: |
| Existing attention, 2K chunks | 43.82 s | 748 | 23.09 GiB |
| Native attention, 2K chunks | 37.57 s | 872 | 23.08 GiB |
| Native attention, 4K chunks | 39.71 s | 825 | 25.33 GiB |
| Native attention, 8K chunks | 43.28 s | 757 | 29.54 GiB |
| Native attention + packed DeltaNet, 512-token chunks | 39.20 s | 836 | 22.03 GiB |
| Native attention + packed DeltaNet, 1K chunks | 35.97 s | 911 | 21.96 GiB |
| Native attention + packed DeltaNet, 2K chunks | 36.33 s | 902 | 23.09 GiB |

The 1K versus 2K difference was about 1%, so the existing 2K chunk remains the default.
The larger chunks used more memory and were slower. Isolated attention measurements
are in `attention-profile.json`; isolated recurrence measurements are in `gdn-profile.json`.
The recurrence outputs and FP32 states matched bit for bit in those measurements.

The final paired manual full-model trial (`long-chunks.json`) used 128K tokens: 1K chunks
took 202.71 s and 2K chunks took 203.73 s, a 0.5% difference in a single pair. Both selected
the same first token and used no swap. This does not establish a useful throughput gain,
so the existing 2K default stays. These manual sweeps use a different synthetic prompt and
prefill driver from the API comparison above and should not be combined with its medians.

`threadgroups.json` compares 1, 2, 4, 8 and 16 SIMD groups per packed recurrence threadgroup
on the actual 16-key-head / 48-value-head shapes. After two warmup trials, eight alternating
measurements per choice retained two groups: one group offered only a 0.5% difference for
2K BF16 rows and lost on the other cases; larger groups were slower. Every choice preserved
both output and state bits.

`dense-profile.json` tests dequantizing tiled weights into BF16 matrices for prefill.
Including conversion, the large projections were generally slower or essentially tied
with the existing tiled QMM. Keeping full BF16 weights gave some small kernel gains,
but expands weight storage and changed output bits for the MLP down projection. This
path was not enabled.

An isolated MLX 0.32.3 installation passed the same 63 focused tests and three 32K full-model
trials (`mlx-0323-profile.json`, median 36.53 seconds). That offered no clear speed gain over
the 36.33-second MLX 0.32.2 trial, so the Studio's installed runtime remains 0.32.2.

Whole-chunk fused attention can round differently from the old split calls. Snapshot
identities include both active prefill modes, so checkpoints from another mode are not reused.
The native attention path is restricted to MLX 0.32.2/0.32.3 on M5; other devices/runtimes
retain bounded attention. Unsupported recurrence shapes keep the installed MLX-LM kernel,
and a failed startup comparison disables the packed path.

The baseline is fork main `25f8b0f`, including upstream 0.5.0 (`9cd52ab`). The final branch
also merges upstream 0.6.0 (`c464617`), which became visible during testing, and resolves its
parser and scheduler moves while preserving the fork's Studio authentication and monitoring.
