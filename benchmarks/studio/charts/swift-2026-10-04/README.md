# Swift performance on current main

October 4, 2026: TensorFold 0.6.5 on an M5 Max with 40 GPU cores and 64 GiB, MLX / mlx-metal 0.32.3 and MLX-LM 0.32.0. Tested main `39f4a51839ba3f903c02ca99ed8e1c48606c214e` includes all upstream changes through `609ca419` and our M5 prompt optimizations. Model and drafter revisions, memory budgets and authentication checks are recorded in `environment.json`.

The standard Swift benchmark completed 18 successful uncached requests in 12 batches, exactly one attempt per context/concurrency setting. Each reply reached the 1,024-token output limit. Six context sizes run from 20K to 128K, with one and two simultaneous submissions. Sampling is greedy, reasoning is off, and drafting is on.

At 128K solo, effective prompt throughput is 644.7 tokens/s versus 482.4 in the original published benchmark (+33.7%); first-token wait is 198.5 versus 265.4 seconds. Generation is 50.3 versus 50.5 tokens/s (-0.4%). Compared with the September 30 optimized build, current solo prompt throughput is 0.3–0.8% lower across the measured sizes. These single attempts do not demonstrate an additional prefill speed improvement from the runtime upgrade.

The blue/orange figure compares one versus two users. The blue/pink figure compares current solo performance with the **original published** Swift benchmark, not the September 30 optimized run. The original reference retains three solo trials at 20K/40K/60K and one at 80K/100K/128K; its 128K point is the higher-memory rerun selected for the first published chart. Historical configurations and public-versus-local HTTP differ, so this is not a controlled A/B or a coding-quality measurement.

Prompt throughput uses uncached tokens divided by queue plus prefill time. Paired per-request rates are summarized by their median and observed range; combined output throughput uses total output divided by batch wall time. Whiskers show observed ranges rather than confidence intervals. The two users may queue, and their generation rates must not be added.

The full Studio host suite passed: 5,054 tests, 609 skips and 6 subtests; zero failures. Both seeded 20K concurrent requests matched with drafting enabled versus disabled. Cold and resumed chat output matched, with 20,038 cached prompt tokens reducing prompt time from 21.3 to 0.08 seconds. See `tests.json` and `validation.json`.

GPU device utilization sampled approximately once per second over the whole benchmark window had a median of 100% and mean of 97.6%. This window includes preflight, gaps, prefill and decoding; it is not a tensor-core occupancy measurement. The Studio reported zero swap usage before and after the run. See `gpu-utilization.jsonl` and `environment.json`.

The native prompt-attention path changes floating-point reduction order relative to the older bounded path; historical short greedy outputs were not all identical. Packed DeltaNet is enabled only after its startup bitwise check. Current drafted/undrafted and cache checks establish compatibility within the tested optimized configuration.

`updated-results.json` contains the new raw measurements with its local output-directory path replaced by a portable label; `previous-results.json` is the original reference. `summary.csv` and `solo-comparison.csv` contain the derived measurements. Both figures are available as PNG, SVG and PDF, and `swift-performance.html` is the editable Page visualization source. Run `render_visualizations.py` from a Python environment with Matplotlib to regenerate the figures without running any models.
