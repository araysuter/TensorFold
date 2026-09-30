# Swift benchmark figures · September 29, 2026

`swift-updated-benchmark.svg` / `.png` / `.pdf` reproduce the established
four-panel layout for the updated configuration. The sheet combines all four
completed client runs: 44 successful requests across 30 batches, with zero cached
tokens. It retains the original measured output lengths, including early finishes
in some paired 20K and 40K replies.

`swift-solo-comparison.svg` / `.png` / `.pdf` compare one request in each
configuration. Blue is the previous configuration; pink is the updated one.
The previous series selects exactly the data used by `plot_results.py` for the
published Swift chart: the resumed run plus the 100K run, with both original
128K groups replaced by the higher-memory rerun. This is a historical comparison,
not an isolated or controlled measurement of a single optimization.

The current configuration has three trials through 100K solo and through 80K
paired, and one trial at 100K paired and 128K. The historical solo series has
three trials at 20K, 40K and 60K, and one at the larger contexts. Points are
medians; whiskers show observed minimum and maximum, not confidence intervals.

The effective prompt throughput metric is uncached input tokens divided by
server-reported queue plus prefill time. Combined output throughput is total
output tokens divided by the batch's full wall time, including queueing and
prefill. Per-request generation rate is reported by the server. Do not sum
per-request rates to estimate aggregate throughput.

`updated-results.json` and `previous-results.json` preserve the selected raw
records, input configurations, original source paths and SHA-256 file digests.
`summary.csv` contains medians, ranges and sample counts; `solo-comparison.csv`
contains the solo medians and percent changes. Both configurations use the same
tokenizer revision, greedy sampling, drafting enabled, reasoning disabled, a
1,024-token output limit and the same API URL.

To reproduce the two figures from the repository root using the existing
benchmark environment (no model loading or inference):

```bash
.venv-benchmark/bin/python benchmarks/studio/charts/swift-2026-09-29/render_visualizations.py
```

The renderer uses the saved combined JSON records when the original client-run
directories are unavailable, so the figures also reproduce from a fresh clone.
