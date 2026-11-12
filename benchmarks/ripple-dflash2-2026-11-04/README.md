# Ripple DFlash2 on the 64 GiB M5 Max

The selected configuration is the original affine mixed-4bit Ripple target,
mixed-4/8-bit DFlash2, four-token verification and 512-token prompt chunks.
It uses the isolated MLX verification dispatcher described in the
[runbook](../../tools/ripple/DFLASH2.md). The target's precision is unchanged.

| Public short prompt | Sampling | Plain tok/s | DFlash2 tok/s | Target tokens |
| --- | --- | --- | --- | --- |
| Hash-table explanation | Greedy | 12.31 | 16.29 | Identical |
| Hash-table explanation | Temperature 0.7, seed 17 | 12.26 | 16.11 | Identical |
| Merge-intervals code | Greedy | 12.24 | 22.26 | Identical |
| Merge-intervals code | Temperature 0.7, seed 17 | 12.30 | 21.44 | Identical |

These are steady generation rates with a 256-token output budget, stopping
earlier at EOS. Each cell ran once. They are separate workloads, not a pooled
score. Initial model loading and compilation are separate costs.

The selected 512-row configuration measured 315.5 tokens/sec prompt processing
at 8K and 304.3 at 32K. At 130,944 input tokens plus 64 output tokens, it measured
255.1 prompt tokens/sec and 12.4 generation tokens/sec, with 49.10 GiB peak MLX
allocation. Both the cold and restored full-window token hashes match the
earlier plain-profile capacity receipt. That earlier run measured 255.4 and
9.55 tokens/sec respectively. This is a separate capacity comparison, not an
interleaved performance experiment.

The full-window probe passed 23 API checks, including restoration of 131,007
cached tokens after switching through Swift; only 16 new prompt tokens needed
processing. The rollout's HTTPS/auth/cache/streaming/native-profile and
concurrent-versus-solo checks also passed. At most one resident target process
was sampled. These checks do not establish model-task accuracy.

Larger blocks, adaptive blocks and an 8-bit drafter did not beat the selected
setting. Larger prompt chunks and allocator reuse gave no material gain.
The original stock-arithmetic speculative runs failed token identity, and
2,048-row prefill failed a 32K output comparison. Those failures are preserved,
not counted as qualified speedups.

- [Sanitized comparison table](comparison.md) and [machine-readable comparison](comparison.json).
- [Full API, capacity and deployment summary](../../tools/ripple/validation-dflash2-2026-11-04.json).
- [Architecture oracle](../../tools/ripple/dflash-reference-validation.json) and
  [projection checks](../../tools/ripple/dflash-projection-validation.json), with explicit numerical limits.
- [Raw results archive](raw-results.tar.gz) and [member checksums](raw-index.json).

The raw archive contains only the public benchmark prompts/token arrays,
measurements, and the synthetic API probe. It excludes private configuration,
keys, snapshot files and the private reply capture. Extract it to a new local
directory to rerun the comparison tool against its JSON members:

```sh
python3 -c 'import tarfile; tarfile.open("benchmarks/ripple-dflash2-2026-11-04/raw-results.tar.gz").extractall("/tmp/ripple-raw-results", filter="data")'
python3 tools/ripple/compare_dflash.py \
  --receipt /tmp/ripple-raw-results/raw/selected_fixed4.json --label selected_fixed4 \
  --output /tmp/ripple-selected-comparison.json --markdown /tmp/ripple-selected-comparison.md
```

The archive preserves each attempt's source hashes. Core behavior was tested
from `b64cd6aac21d824a07900799cd21d9893b7f9a52`; the full-window preparation used
the same inference implementation before the final operator admission was
narrowed to the selected 512-row setting. The production HTTPS/concurrency
checks used the published commit. Receipt/documentation commits add no inference
changes. The deployed native Swift/Ornith binary and their profile settings are
unchanged.
