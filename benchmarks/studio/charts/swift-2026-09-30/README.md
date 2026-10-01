# Swift benchmark figures · September 30, 2026

This run uses exactly **one trial at each context/concurrency setting**: 20K, 40K,
60K, 80K, 100K and 128K input tokens, with one request or two simultaneous requests.
There are 12 batches and 18 measured requests; all succeeded with zero cached tokens. The client keeps the original
1,024-token output limit, greedy sampling, reasoning off and drafting on. Replies
can finish early; all 18 replies in this run reached the full 1,024-token limit.

`swift-updated-benchmark.svg` / `.png` / `.pdf` use the established four-panel layout
for generation per request, effective prompt throughput, time to first token and
combined end-to-end output throughput. Solo points have no repeated-trial range.
Paired points are the median of the two users, with whiskers showing their range;
these whiskers do not represent repeated trials.

`swift-solo-comparison.svg` / `.png` / `.pdf` use the established blue/pink comparison
layout. Blue is the original published Swift dataset (three solo trials at 20K/40K/60K,
one at 80K/100K/128K); pink is this single-trial September 30 run. The original
128K point uses the higher-memory rerun selected for the first published chart. This is a historical
comparison, not a controlled A/B. The new client uses local HTTP on the Studio;
the earlier client used the public endpoint. Model/tokenizer revision, sampling,
drafting and output limit match.

The tested source is performance branch commit `e2601a41679c6c0b7fff9e4f59355e1e415f87c0`,
including upstream TensorFold 0.6.0, M5 fused prompt attention and packed DeltaNet.
The Swift launcher uses 48 GiB process / 45 GiB MLX, 131,072-token context and two
lanes. MLX remains 0.32.2 and MLX-LM 0.31.3. The existing `AI` tmux session runs
both the server and benchmark client; all groups run once, with no automatic retries.

`updated-results.json` preserves the new measured requests and batches.
`previous-results.json` preserves the original published reference dataset.
`environment.json` records the runtime and benchmark provenance.
`summary.csv` and `solo-comparison.csv` contain the plotted measurements.
No API credentials are saved in these artifacts.

Generation rate is reported by the server and includes time sharing the GPU with
other work. Effective prompt throughput is uncached input tokens divided by
server-reported queue plus prefill time, rather than isolated GPU prefill.
Combined output throughput is total output tokens divided by the entire batch's
wall time, including queueing and prefill. Synthetic prompts measure throughput,
not coding quality.

This run took 33.3 minutes across its 12 batches. These are single-trial
observations for the new configuration. The long paired requests were subject to
memory admission and include queueing. Exact original-to-current changes are in
`solo-comparison.csv`.

To run the same client once at every setting against a freshly started test server:

```bash
cd ~/ai/tensorfold-studio
.venv/bin/python benchmarks/studio/benchmark.py \
  --base-url http://127.0.0.1:18080/v1 \
  --api-key-file "$HOME/.config/tensorfold/api-key" \
  --contexts 20000 40000 60000 80000 100000 128000 \
  --concurrency 1 2 --reps 1 --output-tokens 1024 \
  --out /tmp/swift-single-trial-new
```

Stop the current server once and wait for graceful shutdown before starting the
test server in tmux window 0 with `STUDIO_PORT=18080 bash tools/start-studio.sh`.
Run the client in window 1. Use a new output directory each time.

To regenerate both figures from saved results, without inference:

```bash
.venv-benchmark/bin/python benchmarks/studio/charts/swift-2026-09-30/render_visualizations.py
```
